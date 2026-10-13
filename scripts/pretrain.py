"""
Action-Conditioned World Model Pre-training
============================================
Trains the LeJEPA dynamics model: an encoder Encθ that learns to predict the
future latent state of a patient's ECG given a pathology transition vector at.

Architecture (Figure 1 of the paper):
  ht      = Encθ(Xt)
  ht1     = Encθ(Xt+1)             ← target (same encoder, no stop-grad)
  at_emb  = Projω(at.float())
  ht1_hat = Dynϕ(ht, at_emb)
  L = (1-λ) * MSE(ht1_hat, ht1) + λ * (SIGReg(Ht) + SIGReg(Ht+1))

The OnlineProbe monitors downstream Triage AUROC on a frozen linear head
during pre-training without leaking label supervision into the encoder.
The _DecoyProbe checks, on validation pairs, whether Dynϕ's MSE ranks the true
transition above admissible decoy transitions (eval/decoy_*), what CEM planning needs.

The encoder is cfg.backbone: resnet1d (default) or vit1d (src/models/vit1d.py, sized by cfg.vit).

Example
-------
HYDRA_FULL_ERROR=1 python scripts/pretrain.py ++max_epochs=1 ++batch_size=64 \\
    ++num_workers=0 ++use_wandb=false

ViT encoder (see configs/pretrain.yaml for sizes, memory and speed)
-------------------------------------------------------------------
python scripts/pretrain.py ++backbone=vit1d ++vit.depth=4 ++vit.dim=128 \\
    ++ckpt_path=checkpoints/pretrain_vit_.ckpt ++ckpt_path_auroc=checkpoints/pretrain_vit_auroc_.ckpt

Resuming an interrupted run
----------------------------
python scripts/pretrain.py \\
    ++resume_ckpt=runs/runs/20260707/232210/dc470fe3e905/checkpoints/last.ckpt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from hydra.utils import get_original_cwd
import lance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from functools import partial
from torch.utils.data import DataLoader, Dataset, Subset
from lightning.pytorch.loggers import WandbLogger

import torchmetrics
from torchmetrics.utilities import dim_zero_cat
import stable_pretraining as spt

from dataset.dataset import MIMICLanceDataset
from models.backbones import build_backbone
from models.dynamics import ActionProjector, DynamicsPredictor, SlicedEppsPulley
from utils import check_tcp


class _MultilabelAUROC(torchmetrics.classification.MultilabelAUROC):
    """Macro AUROC over the classes with both outcomes, as scripts/evaluate_inverse.py
    computes it; torchmetrics' own macro average scores a class with no positives as 0."""

    def __init__(self, num_labels: int):
        super().__init__(num_labels=num_labels, average=None)

    def update(self, preds, target):
        super().update(preds, target.long())

    def compute(self):
        target = dim_zero_cat(self.target)
        n_pos  = target.sum(0)
        return super().compute()[(n_pos > 0) & (n_pos < len(target))].mean()


def _transition_priors(lance_path, pairs_path, pair_types, action_dim):
    """Laplace-smoothed per-class P(onset | yt=0) and P(resolution | yt=1) over the train split's pairs, from
    labels only: the base_rate prior that scripts/cem_planning.py samples its first proposals from."""
    type_filter = " OR ".join(f"pair_type = '{t}'" for t in pair_types)
    pairs = (lance.dataset(pairs_path)
             .to_table(filter=f"(fold <= 17) AND ({type_filter})", columns=["idx_t", "idx_t1"]).to_pandas())
    idx = np.unique(np.concatenate([pairs["idx_t"].values, pairs["idx_t1"].values]))
    labels = (lance.dataset(lance_path).take(idx.tolist(), columns=["icd"])
              .column("icd").combine_chunks().flatten().to_numpy(zero_copy_only=False)
              .reshape(len(idx), action_dim))
    yt  = labels[np.searchsorted(idx, pairs["idx_t"].values)]
    yt1 = labels[np.searchsorted(idx, pairs["idx_t1"].values)]
    p_ons = (((yt == 0) & (yt1 == 1)).sum(0) + 1) / ((yt == 0).sum(0) + 2)
    p_res = (((yt == 1) & (yt1 == 0)).sum(0) + 1) / ((yt == 1).sum(0) + 2)
    return torch.tensor(p_ons, dtype=torch.float32), torch.tensor(p_res, dtype=torch.float32)


class _DecoyProbe(pl.Callback):
    """Can the forward energy E(a) = mean_D ||Dynϕ(ht, Projω(a)) - ht1||^2 tell the true transition from decoys?
    On validation pairs, the true transition is ranked against admissible decoys (onsets only on absent labels,
    resolutions only on present ones, as scripts/cem_planning.py searches):

      eval/decoy_same    active pairs; n_decoys decoys with the true numbers of onsets and resolutions, on
                         other labels: does E know which labels changed?
      eval/decoy_extra   all pairs; n_decoys copies of the truth plus one extra flip: does flipping more lower E,
                         the way plain-E CEM was exploited?
      eval/decoy_zero    active pairs; the zero action: does the truth beat doing nothing?
      eval/decoy_prior   all pairs; n_decoys draws from the train base-rate prior given yt, CEM's first proposal

    Each is the fraction of decoys with a higher energy than the truth (ties 1/2, decoys equal to the truth
    excluded), averaged over pairs: a within-pair AUROC, chance 0.5. Each is repeated against the pair's own ht
    instead of ht1 (eval/decoy_*_nochange): that keeps the patient and removes the change, so a model that ranks
    as well there ranks from ht alone, not from evidence in the outcome (anatomie/decoy_objectives.py's "none").

      eval/mse_ratio     E(truth) / mean ||ht1 - ht||^2: val prediction error relative to doing nothing, unlike
                         val/pred_loss comparable across backbones and λ (embedding scales differ)"""

    FAMILIES = ("same", "extra", "zero", "prior")

    def __init__(self, n_decoys: int, prior_ons: torch.Tensor, prior_res: torch.Tensor, seed: int):
        self._k     = n_decoys
        self._prior = (prior_ons, prior_res)
        self._seed  = seed
        names = [f"eval/decoy_{f}{s}" for f in self.FAMILIES for s in ("", "_nochange")]
        self._metrics = {n: torchmetrics.MeanMetric() for n in names + ["e_true", "e_nothing"]}

    def on_validation_epoch_start(self, trainer, pl_module):
        # The val batches are the same every epoch (fixed Subset, no shuffle), so reseeding scores the same decoys
        self._gen = torch.Generator(device=pl_module.device).manual_seed(self._seed)

    def _subsets(self, allowed, counts):
        """n_decoys random subsets of each row's allowed labels, counts[i] labels each.
        allowed: (B, C) bool, counts: (B,) -> (B, n_decoys, C) float"""
        scores = torch.rand(allowed.shape[0], self._k, allowed.shape[1], generator=self._gen, device=allowed.device)
        scores = scores.masked_fill(~allowed.unsqueeze(1), -1.0)   # disallowed labels rank last
        rank   = scores.argsort(-1, descending=True).argsort(-1)
        return (rank < counts.view(-1, 1, 1)).float()

    def _candidates(self, yt, at):
        """(B, 2 + 3*n_decoys, C): [truth, same-count decoys, extra-flip decoys, zero action, prior decoys]."""
        B, C = yt.shape
        same  = self._subsets(yt == 0, (at == 1).sum(1)) - self._subsets(yt == 1, (at == -1).sum(1))
        # One label the truth leaves unchanged, flipped the only admissible way
        one   = torch.ones(B, dtype=torch.long, device=yt.device)
        extra = at.unsqueeze(1) + self._subsets(at == 0, one) * (1.0 - 2.0 * yt).unsqueeze(1)
        p_ons, p_res = (p.to(yt.device).expand(B, self._k, C).contiguous() for p in self._prior)
        prior = (torch.bernoulli(p_ons, generator=self._gen) * (1.0 - yt).unsqueeze(1)
                 - torch.bernoulli(p_res, generator=self._gen) * yt.unsqueeze(1))
        return torch.cat([at.unsqueeze(1), same, extra, torch.zeros_like(at).unsqueeze(1), prior], dim=1)

    @torch.no_grad()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not isinstance(outputs, dict):
            return
        yt, at  = batch["yt"].float(), batch["at"].float()
        ht, ht1 = outputs["embedding"].float(), outputs["ht1"].float()
        cand    = self._candidates(yt, at)                          # (B, N, C)
        B, N, C = cand.shape
        K       = self._k

        # fp32: per-pair energy gaps sit near zero, where fp16 rounding would decide them
        with torch.autocast(device_type=ht.device.type, enabled=False):
            pred = pl_module.projector["pred"](ht.repeat_interleave(N, dim=0),
                                               pl_module.projector["proj"](cand.view(B * N, C))).view(B, N, -1)

        active = (at != 0).any(1)
        valid  = (cand != at.unsqueeze(1)).any(-1).float()          # a decoy equal to the truth doesn't count
        rows   = {"same": active, "extra": torch.ones_like(active), "zero": active, "prior": torch.ones_like(active)}
        cols   = {"same": slice(1, K + 1), "extra": slice(K + 1, 2 * K + 1),
                  "zero": slice(2 * K + 1, 2 * K + 2), "prior": slice(2 * K + 2, N)}
        for suffix, target in (("", ht1), ("_nochange", ht)):
            E = (pred - target.unsqueeze(1)).pow(2).mean(-1)         # (B, N)
            w = (E > E[:, :1]).float() + 0.5 * (E == E[:, :1]).float()
            for fam in self.FAMILIES:
                v    = valid[:, cols[fam]]
                keep = rows[fam] & (v.sum(1) > 0)
                if keep.any():
                    rate = (w[:, cols[fam]] * v).sum(1) / v.sum(1).clamp(min=1)
                    self._metrics[f"eval/decoy_{fam}{suffix}"].to(w.device).update(rate[keep])
            if suffix == "":
                self._metrics["e_true"].to(E.device).update(E[:, 0])
        self._metrics["e_nothing"].to(ht.device).update((ht1 - ht).pow(2).mean(-1))

    def on_validation_epoch_end(self, trainer, pl_module):
        for name, metric in self._metrics.items():
            if name.startswith("eval/"):
                pl_module.log(name, metric.compute(), prog_bar=name == "eval/decoy_same", sync_dist=True)
        pl_module.log("eval/mse_ratio", self._metrics["e_true"].compute() / self._metrics["e_nothing"].compute(),
                      sync_dist=True)
        for metric in self._metrics.values():
            metric.reset()


def _make_loader(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=shuffle,
        pin_memory=True,
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="pretrain")
def main(cfg):
    pair_types = tuple(cfg.pair_types)

    # Train: longitudinal pairs (Xt, Xt+1, yt, at)
    train_ds = MIMICLanceDataset(
        cfg.lance_path,
        split="train",
        mode="pair",
        pairs_path=cfg.pairs_path,
        pair_types=pair_types,
        train_frac=cfg.train_frac,
        cache=cfg.cache,
    )
    # Val: longitudinal pairs — same format as train, gives a real dynamics val/loss
    val_ds = MIMICLanceDataset(
        cfg.lance_path,
        split="val",
        mode="pair",
        pairs_path=cfg.pairs_path,
        pair_types=pair_types,
        cache=cfg.cache,
    )
    # The pairs table is stored patient by patient, so in-order val batches repeat the same
    # ECGs and inflate SIGReg; one fixed shuffle keeps the batches identical across epochs.
    val_ds = Subset(val_ds, torch.randperm(len(val_ds), generator=torch.Generator().manual_seed(cfg.seed)).tolist())

    num_workers = 0 if cfg.cache else cfg.num_workers
    train_loader = _make_loader(train_ds, cfg.batch_size, num_workers, shuffle=True)
    val_loader   = _make_loader(val_ds,   cfg.batch_size, num_workers, shuffle=False)
    data_module  = spt.data.DataModule(train=train_loader, val=val_loader)

    backbone  = build_backbone(cfg)
    projector = ActionProjector(action_dim=cfg.action_dim, embed_dim=cfg.embedding_dim)
    predictor = DynamicsPredictor(embed_dim=cfg.embedding_dim, hidden_dim=cfg.predictor_hidden_dim)
    sigreg    = SlicedEppsPulley(num_slices=cfg.n_slices, t_max=cfg.t_max, n_points=cfg.n_points)

    # Bundle all extra modules so spt.Module keeps them on the right device
    # and includes their parameters in the optimizer.
    extra = nn.ModuleDict({"proj": projector, "pred": predictor, "sigreg": sigreg})

    def forward(self, batch, stage):
        is_train = stage == "fit"
        prefix   = "train" if is_train else "val"

        # Both train and val use pair batches: {"xt", "xt1", "yt", "at"}
        xt, xt1, yt, at = batch["xt"], batch["xt1"], batch["yt"], batch["at"]

        ht      = self.backbone(xt)                          # (B, D)
        ht1     = self.backbone(xt1)                         # (B, D)
        at_emb  = self.projector["proj"](at.float())         # (B, D)
        ht1_hat = self.projector["pred"](ht, at_emb)         # (B, D)

        pred_loss = F.mse_loss(ht1_hat, ht1)
        reg_loss  = self.projector["sigreg"](ht) \
                  + self.projector["sigreg"](ht1)
        loss = (1.0 - cfg.lambda_reg) * pred_loss + cfg.lambda_reg * reg_loss

        self.log(f"{prefix}/loss",      loss,      on_step=is_train, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log(f"{prefix}/pred_loss", pred_loss, on_step=is_train, on_epoch=True,                 sync_dist=True)
        self.log(f"{prefix}/reg_loss",  reg_loss,  on_step=is_train, on_epoch=True,                 sync_dist=True)
        # Gradient-magnitude ratio — target ≈ (1-λ)/λ for equal contribution.
        # Log once per epoch (on_step=False) to avoid overhead.
        if is_train:
            self.log("train/loss_ratio", reg_loss / (pred_loss + 1e-8), on_step=False, on_epoch=True, sync_dist=True)

        # "embedding" + "label" feed the OnlineProbe, "embedding" + "ht1" the _DecoyProbe
        return {"embedding": ht, "ht1": ht1, "label": yt, "loss": loss}

    module = spt.Module(
        backbone=backbone,
        projector=extra,        # reuse the projector slot for the extra modules
        forward=forward,
        hparams=cfg,
        optim={
            "optimizer": partial(
                torch.optim.AdamW,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            ),
            "scheduler": "LinearWarmupCosineAnnealing",
        },
    )

    auroc_probe = spt.callbacks.OnlineProbe(
        module,
        name="auroc",
        input="embedding",
        target="label",
        probe=nn.Linear(cfg.embedding_dim, 76),
        loss=nn.BCEWithLogitsLoss(),
        metrics=_MultilabelAUROC(num_labels=76),
    )

    logger    = False
    callbacks = [auroc_probe]
    if cfg.decoy_probe_decoys > 0:
        prior_ons, prior_res = _transition_priors(cfg.lance_path, cfg.pairs_path, pair_types, cfg.action_dim)
        callbacks.append(_DecoyProbe(cfg.decoy_probe_decoys, prior_ons, prior_res, seed=cfg.seed))
    if cfg.use_wandb:
        if check_tcp():
            logger = WandbLogger(project=cfg.wandb_project)
            callbacks.append(pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"))
        else:
            print("WARNING: wandb unreachable (TCP check failed) — running without logger")

    # Unlike standard SSL, the dynamics objective already conditions on the
    # transition vector at = yt+1 - yt (label-derived), so it isn't label-blind
    # the way an encoder-only SSL loss would be. We stop on the downstream
    # signal we actually care about — the online linear probe's AUROC — rather
    # than the SSL pred/SIGReg loss, which is only a proxy for it.
    # callbacks.append(pl.pytorch.callbacks.EarlyStopping(
    #     monitor="eval/auroc__MultilabelAUROC_epoch", patience=20, mode="max", check_finite=False,
    # ))
    # Disabled: val/pred_loss tracks the embedding scale while SIGReg spreads the collapsed initial embedding
    # (ViT1d per-dim variance 0.004 at init -> 1.8 at epoch 3 -> 1.3 at epoch 20), so its minimum sits at epoch 0.
    # callbacks.append(pl.pytorch.callbacks.EarlyStopping(
    #     monitor="val/pred_loss", patience=20, mode="min", check_finite=False,
    # ))

    if cfg.ckpt_path:
        ckpt = Path(get_original_cwd()) / cfg.ckpt_path
        callbacks.append(pl.pytorch.callbacks.ModelCheckpoint(
            monitor="val/loss",
            dirpath=str(ckpt.parent),
            filename=ckpt.stem,
            mode="min",
            save_weights_only=True,
        ))

    if cfg.ckpt_path_auroc:
        ckpt_auroc = Path(get_original_cwd()) / cfg.ckpt_path_auroc
        callbacks.append(pl.pytorch.callbacks.ModelCheckpoint(
            monitor="eval/auroc__MultilabelAUROC_epoch",
            dirpath=str(ckpt_auroc.parent),
            filename=ckpt_auroc.stem,
            mode="max",
            save_weights_only=True,
        ))

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        num_sanity_val_steps=1,
        callbacks=callbacks,
        precision="16-mixed",
        logger=logger,
        sync_batchnorm=True,
    )

    spt.set(cache_dir=cfg.spt_runs_dir)

    resume_ckpt = None
    if cfg.resume_ckpt:
        print('Resuming from checkpoint:', cfg.resume_ckpt)
        resume_ckpt = Path(cfg.resume_ckpt).expanduser()
        if not resume_ckpt.is_absolute():
            resume_ckpt = Path(get_original_cwd()) / resume_ckpt
    else:
        # Manager restores a wandb run id from a `wandb_resume.json` sidecar
        # left in the CWD by the previous invocation (legacy fallback, used
        # even when cache_dir is active). Clear it on a non-resume run so we
        # don't silently reattach to a stale wandb run.
        (Path(get_original_cwd()) / "wandb_resume.json").unlink(missing_ok=True)

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        seed=cfg.seed,
        ckpt_path=str(resume_ckpt) if resume_ckpt else None,
        weights_only=cfg.resume_weights_only,
    )
    manager()


if __name__ == "__main__":
    main()

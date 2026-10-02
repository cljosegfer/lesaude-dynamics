"""
Joint Forward + Inverse Dynamics Pre-training
==============================================
Trains one encoder Encθ under both world-model objectives at once: the forward
dynamics model of scripts/pretrain.py and the inverse dynamics model of
scripts/inverse_pretrain.py. The two heads share the encoder and nothing else.

Architecture:
  ht      = Encθ(Xt)
  ht1     = Encθ(Xt+1)                         ← same encoder, no stop-grad
  ht1_hat = Dynϕ(ht, Projω(at.float()))        ← forward head
  onset_logits, res_logits = Invψ(ht1 - ht)    ← inverse head
  L_dyn = MSE(ht1_hat, ht1)
  L_inv = BCE(onset, at==1) + BCE(res, at==-1)
  L = (1-λ) * (w_dyn * L_dyn + w_inv * L_inv) + λ * (SIGReg(Ht) + SIGReg(Ht+1))

inv_weight=0 recovers the scripts/pretrain.py objective and dyn_weight=0 the
scripts/inverse_pretrain.py one, on the same pairs and optimizer.

_CEMProbe checks online what CEM planning needs from Dynϕ: whether the true
transition has a lower energy than admissible decoys (val/cem_auroc,
val/cem_extra_auroc).

The forward head keeps scripts/pretrain.py's state_dict keys (projector.proj,
projector.pred), so scripts/cem_planning.py loads joint checkpoints unchanged;
the inverse head lives at projector.inv, which scripts/evaluate_inverse.py
picks up automatically.

Example
-------
HYDRA_FULL_ERROR=1 python scripts/joint_pretrain.py ++max_epochs=1 ++batch_size=64 \\
    ++num_workers=0 ++use_wandb=false ++train_frac=0.01

Resuming an interrupted run
----------------------------
python scripts/joint_pretrain.py \\
    ++resume_ckpt=runs/runs/20260707/232210/dc470fe3e905/checkpoints/last.ckpt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from hydra.utils import get_original_cwd
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
from models.resnet1d import ResNet1d
from models.dynamics import ActionPredictor, ActionProjector, DynamicsPredictor, SlicedEppsPulley
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


class _DirectAUROC(pl.Callback):
    """Accumulates pre-computed probabilities across the validation epoch and
    reports macro AUROC at epoch end — no linear probe, no optimizer."""

    def __init__(self, name: str, probs_key: str, target_key: str, num_labels: int):
        self._name       = name
        self._probs_key  = probs_key
        self._target_key = target_key
        self._metric     = _MultilabelAUROC(num_labels=num_labels)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not isinstance(outputs, dict):
            return
        probs  = outputs[self._probs_key].detach().float()
        target = outputs[self._target_key].detach()
        self._metric = self._metric.to(probs.device)
        self._metric.update(probs, target)

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.log(self._name, self._metric.compute(), prog_bar=True, sync_dist=True)
        self._metric.reset()


class _CEMProbe(pl.Callback):
    """Checks on validation pairs what CEM planning needs from Dynϕ: the energy
    E(a) = mean_D ||Dynϕ(ht, Projω(a)) - ht1||^2 of the true transition against
    n_decoys decoys per pair, all admissible given yt (onsets only on absent labels,
    resolutions only on present ones), as CEM searches:

      eval/cem_auroc        active pairs; decoys with the same number of onsets and
                            resolutions on other labels: does E know which labels changed?
      eval/cem_extra_auroc  all pairs; the true transition plus one extra flip: does
                            flipping more lower E, the way plain-E CEM was exploited?

    Each is the fraction of decoys with a higher energy than the true transition
    (ties 1/2), averaged over pairs: a within-pair AUROC with chance at 0.5."""

    def __init__(self, n_decoys: int, seed: int):
        self._n_decoys = n_decoys
        self._seed     = seed
        self._metrics  = {
            "eval/cem_auroc":       torchmetrics.MeanMetric(),
            "eval/cem_extra_auroc": torchmetrics.MeanMetric(),
        }

    def on_validation_epoch_start(self, trainer, pl_module):
        # The val loader isn't shuffled, so every epoch scores the same decoys
        self._generator = torch.Generator(device=pl_module.device).manual_seed(self._seed)

    def _subsets(self, allowed, counts):
        """n_decoys random subsets of each row's allowed labels, counts[i] labels each.
        allowed: (B, C) bool, counts: (B,) -> (B, n_decoys, C) bool"""
        scores = torch.rand(allowed.shape[0], self._n_decoys, allowed.shape[1],
                            generator=self._generator, device=allowed.device)
        scores = scores.masked_fill(~allowed.unsqueeze(1), -1.0)   # disallowed labels rank last
        rank   = scores.argsort(-1, descending=True).argsort(-1)
        return rank < counts.view(-1, 1, 1)

    def _candidates(self, yt, at):
        """(B, 1 + 2*n_decoys, C): the true transition, the same-count decoys, the extra-flip decoys."""
        same_count = (self._subsets(yt == 0, (at == 1).sum(1)).float()
                      - self._subsets(yt == 1, (at == -1).sum(1)).float())
        # One label the true transition leaves unchanged, flipped the only admissible way
        extra = self._subsets(at == 0, torch.ones(len(at), dtype=torch.long, device=at.device)).float()
        extra_flip = at.unsqueeze(1) + extra * (1.0 - 2.0 * yt).unsqueeze(1)
        return torch.cat([at.unsqueeze(1), same_count, extra_flip], dim=1)

    @torch.no_grad()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not isinstance(outputs, dict):
            return
        yt, at  = batch["yt"].float(), batch["at"].float()
        ht, ht1 = outputs["embedding"].float(), outputs["ht1"].float()
        actions = self._candidates(yt, at)   # (B, N, C)
        B, N, C = actions.shape

        # fp32: per-pair energy gaps sit near zero, where fp16 rounding would decide them
        with torch.autocast(device_type=ht.device.type, enabled=False):
            pred   = pl_module.projector["pred"](ht.repeat_interleave(N, dim=0),
                                                 pl_module.projector["proj"](actions.view(B * N, C)))
            energy = (pred.view(B, N, -1) - ht1.unsqueeze(1)).pow(2).mean(-1)   # (B, N)

        e_true, e_decoy = energy[:, :1], energy[:, 1:]
        wins = (e_decoy > e_true).float() + 0.5 * (e_decoy == e_true).float()
        wins = torch.where((actions[:, 1:] == actions[:, :1]).all(-1), 0.5, wins)   # decoy is the truth

        K      = self._n_decoys
        active = (at != 0).any(1)
        if active.any():
            self._metrics["eval/cem_auroc"].to(wins.device).update(wins[active, :K].mean(1))
        self._metrics["eval/cem_extra_auroc"].to(wins.device).update(wins[:, K:].mean(1))

    def on_validation_epoch_end(self, trainer, pl_module):
        for name, metric in self._metrics.items():
            pl_module.log(name, metric.compute(), prog_bar=True, sync_dist=True)
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="joint_pretrain")
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
    # Val: longitudinal pairs — same format as train, gives real dynamics and inverse val losses
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

    backbone  = ResNet1d(in_channels=12, embedding_dim=cfg.embedding_dim)
    projector = ActionProjector(action_dim=cfg.action_dim, embed_dim=cfg.embedding_dim)
    predictor = DynamicsPredictor(embed_dim=cfg.embedding_dim, hidden_dim=cfg.predictor_hidden_dim)
    inverse   = ActionPredictor(
        embed_dim=cfg.embedding_dim,
        hidden_dim=cfg.predictor_hidden_dim,
        action_dim=cfg.action_dim,
    )
    sigreg    = SlicedEppsPulley(num_slices=cfg.n_slices, t_max=cfg.t_max, n_points=cfg.n_points)

    # Bundle all extra modules so spt.Module keeps them on the right device
    # and includes their parameters in the optimizer. "proj"/"pred" match
    # scripts/pretrain.py's keys, so forward-model tools load joint checkpoints.
    extra = nn.ModuleDict({"proj": projector, "pred": predictor, "inv": inverse, "sigreg": sigreg})

    def forward(self, batch, stage):
        is_train = stage == "fit"
        prefix   = "train" if is_train else "val"
        # Validation's headline loss goes to eval/, next to the probe AUROCs; its components stay in val/
        head     = "train" if is_train else "eval"

        xt, xt1, yt, at = batch["xt"], batch["xt1"], batch["yt"], batch["at"]

        ht  = self.backbone(xt)   # (B, D)
        ht1 = self.backbone(xt1)  # (B, D)

        # Forward dynamics (scripts/pretrain.py): predict ht1 from ht and at
        at_emb   = self.projector["proj"](at.float())    # (B, D)
        ht1_hat  = self.projector["pred"](ht, at_emb)    # (B, D)
        dyn_loss = F.mse_loss(ht1_hat, ht1)

        # Inverse dynamics (scripts/inverse_pretrain.py): recover at from the displacement
        onset_logits, res_logits = self.projector["inv"](ht1 - ht)

        onset_target = (at == 1).float()    # (B, 76)
        res_target   = (at == -1).float()   # (B, 76)

        onset_pw = torch.full((cfg.action_dim,), cfg.onset_pos_weight,      device=xt.device)
        res_pw   = torch.full((cfg.action_dim,), cfg.resolution_pos_weight, device=xt.device)
        onset_loss = F.binary_cross_entropy_with_logits(onset_logits, onset_target, pos_weight=onset_pw)
        res_loss   = F.binary_cross_entropy_with_logits(res_logits,   res_target,   pos_weight=res_pw)
        inv_loss   = onset_loss + res_loss

        reg_loss  = self.projector["sigreg"](ht) + self.projector["sigreg"](ht1)
        pred_loss = cfg.dyn_weight * dyn_loss + cfg.inv_weight * inv_loss
        loss      = (1.0 - cfg.lambda_reg) * pred_loss + cfg.lambda_reg * reg_loss

        log_kw = dict(on_step=is_train, on_epoch=True, sync_dist=True)
        self.log(f"{head}/loss",         loss,       prog_bar=True, **log_kw)
        self.log(f"{prefix}/dyn_loss",   dyn_loss,                  **log_kw)
        self.log(f"{prefix}/inv_loss",   inv_loss,                  **log_kw)
        self.log(f"{prefix}/onset_loss", onset_loss,                **log_kw)
        self.log(f"{prefix}/res_loss",   res_loss,                  **log_kw)
        self.log(f"{prefix}/reg_loss",   reg_loss,                  **log_kw)
        # Gradient-magnitude ratio — target ≈ (1-λ)/λ for equal contribution.
        if is_train:
            self.log("train/loss_ratio", reg_loss / (pred_loss + 1e-8),
                     on_step=False, on_epoch=True, sync_dist=True)

        # Contextualized next-state prediction (mirrors evaluate_inverse.py):
        # P(yt+1_i=1) = yt_i*(1-res_prob_i) + (1-yt_i)*onset_prob_i
        onset_probs = torch.sigmoid(onset_logits)
        res_probs   = torch.sigmoid(res_logits)
        ctx_preds   = yt * (1.0 - res_probs) + (1.0 - yt) * onset_probs  # (B, 76) in [0,1]
        yt1         = (yt + at).clamp(0.0, 1.0)                           # (B, 76) true next labels

        # "embedding" + "label" feed the OnlineProbe, "embedding" + "ht1" the _CEMProbe;
        # the rest feed the _DirectAUROC callbacks
        return {
            "embedding":        ht,
            "ht1":              ht1,
            "label":            yt,
            "onset_probs":      onset_probs,
            "onset_label":      onset_target,
            "res_probs":        res_probs,
            "resolution_label": res_target,
            "ctx_preds":        ctx_preds,
            "yt1":              yt1,
            "loss":             loss,
        }

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
    ctx_probe        = _DirectAUROC("eval/ctx_auroc",       "ctx_preds",   "yt1",              76)
    onset_probe      = _DirectAUROC("val/onset_auroc",      "onset_probs", "onset_label",      cfg.action_dim)
    resolution_probe = _DirectAUROC("val/resolution_auroc", "res_probs",   "resolution_label", cfg.action_dim)
    cem_probe        = _CEMProbe(n_decoys=cfg.cem_probe_decoys, seed=cfg.seed)

    logger    = False
    callbacks = [auroc_probe, ctx_probe, onset_probe, resolution_probe, cem_probe]
    if cfg.use_wandb:
        if check_tcp():
            logger = WandbLogger(project=cfg.wandb_project)
            callbacks.append(pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"))
        else:
            print("WARNING: wandb unreachable (TCP check failed) — running without logger")

    # As in scripts/pretrain.py: stop on the online linear probe's AUROC, the
    # downstream signal we care about for the encoder, rather than the SSL loss.
    callbacks.append(pl.pytorch.callbacks.EarlyStopping(
        monitor="eval/auroc__MultilabelAUROC_epoch", patience=20, mode="max", check_finite=False,
    ))

    # One weights-only checkpoint per selection rule: validation loss (the rule
    # behind both reference checkpoints), the online probe AUROC, and the
    # inverse head's contextualized AUROC.
    for path, monitor, mode in (
        (cfg.ckpt_path,       "eval/loss",                          "min"),
        (cfg.ckpt_path_auroc, "eval/auroc__MultilabelAUROC_epoch", "max"),
        (cfg.ckpt_path_ctx,   "eval/ctx_auroc",                     "max"),
    ):
        if path:
            ckpt = Path(get_original_cwd()) / path
            callbacks.append(pl.pytorch.callbacks.ModelCheckpoint(
                monitor=monitor,
                dirpath=str(ckpt.parent),
                filename=ckpt.stem,
                mode=mode,
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

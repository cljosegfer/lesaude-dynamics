"""
Action-Conditioned World Model Pre-training — Displacement Variant
====================================================================
Variant of scripts/pretrain.py where the latent displacement (ht1 - ht) is
predicted from the action alone, so the predictor can only account for the
part of the change the transition explains. cfg.action_model picks the
predictor g:
  linear  W·at, one fixed vector per label: a transition moves the embedding
          by the sum of its labels' vectors. Displacements and actions both
          add up across consecutive stays, so this form is exact on
          multistep pairs.
  mlp     ActionProjector, which can also model label interactions.
Neither has biases, so a stable pair (at = 0) is predicted not to move.

Architecture:
  ht     = Encθ(Xt)
  ht1    = Encθ(Xt+1)              ← target (same encoder, no stop-grad)
  d      = ht1 - ht                ← observed displacement
  d_hat  = g(at)                   ← predicted from the action alone
  L = (1-λ) * ||d - d_hat||² / ||d||² + λ * (SIGReg(Ht) + SIGReg(Ht+1))

The prediction term is a normalized MSE over the batch: the error of g(at)
divided by the error of predicting no movement. It is scale-free, so
shrinking every displacement (collapsing a patient's ECGs onto one point)
no longer lowers it; only aligning displacements with the action does.
1.0 = no better than "nothing changed".

Monitors:
  - OnlineProbe: downstream Triage AUROC on a frozen linear head.
  - {train,val}/pair_dist: squared distance between a pair's two ECGs over
    that between ECGs of different pairs (0 = same point, 1 = unrelated).
    Train falling far below val means the encoder is memorizing training
    patients.
Early stopping and the main checkpoint follow val/pred_loss.

Example
-------
HYDRA_FULL_ERROR=1 python scripts/pretrain_displacement.py ++max_epochs=1 ++batch_size=64 \\
    ++num_workers=0 ++use_wandb=false

# The two displacement predictors
python scripts/pretrain_displacement.py ++action_model=linear
python scripts/pretrain_displacement.py ++action_model=mlp

Resuming an interrupted run
----------------------------
python scripts/pretrain_displacement.py \\
    ++resume_ckpt=runs/runs/20260707/232210/dc470fe3e905/checkpoints/last.ckpt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from hydra.utils import get_original_cwd
import torch
import torch.nn as nn
import lightning as pl
from functools import partial
from torch.utils.data import DataLoader, Dataset, Subset
from lightning.pytorch.loggers import WandbLogger

import torchmetrics
from torchmetrics.utilities import dim_zero_cat
import stable_pretraining as spt

from dataset.dataset import MIMICLanceDataset
from models.resnet1d import ResNet1d
from models.dynamics import ActionProjector, SlicedEppsPulley
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="pretrain_displacement")
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
    # ECGs, which inflates SIGReg and puts the same patient on both sides of pair_dist's
    # baseline; one fixed shuffle keeps the batches identical across epochs.
    val_ds = Subset(val_ds, torch.randperm(len(val_ds), generator=torch.Generator().manual_seed(cfg.seed)).tolist())

    num_workers = 0 if cfg.cache else cfg.num_workers
    train_loader = _make_loader(train_ds, cfg.batch_size, num_workers, shuffle=True)
    val_loader   = _make_loader(val_ds,   cfg.batch_size, num_workers, shuffle=False)
    data_module  = spt.data.DataModule(train=train_loader, val=val_loader)

    backbone = ResNet1d(in_channels=12, embedding_dim=cfg.embedding_dim)
    if cfg.action_model == "linear":
        action = nn.Linear(cfg.action_dim, cfg.embedding_dim, bias=False)
        out_layer = action
    elif cfg.action_model == "mlp":
        action = ActionProjector(action_dim=cfg.action_dim, embed_dim=cfg.embedding_dim, bias=False)
        out_layer = action.net[-1]
    else:
        raise ValueError(f"Unknown action_model: {cfg.action_model!r} (expected linear | mlp)")
    # Start from "no action effect" (pred_loss exactly 1): a random g(at) predicts worse than
    # no movement, and above 1 the normalized loss rewards inflating every displacement.
    nn.init.zeros_(out_layer.weight)
    sigreg = SlicedEppsPulley(num_slices=cfg.n_slices, t_max=cfg.t_max, n_points=cfg.n_points)

    # Bundle all extra modules so spt.Module keeps them on the right device
    # and includes their parameters in the optimizer.
    extra = nn.ModuleDict({"action": action, "sigreg": sigreg})

    def forward(self, batch, stage):
        is_train = stage == "fit"
        prefix   = "train" if is_train else "val"

        # Both train and val use pair batches: {"xt", "xt1", "yt", "at"}
        xt, xt1, yt, at = batch["xt"], batch["xt1"], batch["yt"], batch["at"]

        ht     = self.backbone(xt)                              # (B, D)
        ht1    = self.backbone(xt1)                             # (B, D)
        h0, h1 = ht.float(), ht1.float()                        # fp32: displacements can be small
        d      = h1 - h0                                        # (B, D) observed displacement
        d_hat  = self.projector["action"](at.float()).float()   # (B, D) predicted from the action alone

        # The denominator keeps its gradient: that is what makes the loss scale-free
        pred_loss = (d - d_hat).pow(2).mean() / (d.pow(2).mean() + 1e-8)
        reg_loss  = self.projector["sigreg"](ht) \
                  + self.projector["sigreg"](ht1)
        loss = (1.0 - cfg.lambda_reg) * pred_loss + cfg.lambda_reg * reg_loss

        # Unrelated baseline: the next-state side rolled by one row. Batches are shuffled,
        # so the rolled partner is almost always another patient.
        with torch.no_grad():
            pair_dist = d.pow(2).mean() / (h1.roll(1, dims=0) - h0).pow(2).mean()

        self.log(f"{prefix}/loss",      loss,      on_step=is_train, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log(f"{prefix}/pred_loss", pred_loss, on_step=is_train, on_epoch=True,                 sync_dist=True)
        self.log(f"{prefix}/reg_loss",  reg_loss,  on_step=is_train, on_epoch=True,                 sync_dist=True)
        self.log(f"{prefix}/pair_dist", pair_dist, on_step=is_train, on_epoch=True,                 sync_dist=True)
        # Gradient-magnitude ratio — target ≈ (1-λ)/λ for equal contribution.
        # Log once per epoch (on_step=False) to avoid overhead.
        if is_train:
            self.log("train/loss_ratio", reg_loss / (pred_loss + 1e-8), on_step=False, on_epoch=True, sync_dist=True)

        # Expose ht + yt for the OnlineProbe (reads "embedding" and "label")
        return {"embedding": ht, "label": yt, "loss": loss}

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
    if cfg.use_wandb:
        if check_tcp():
            logger = WandbLogger(project=cfg.wandb_project)
            callbacks.append(pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"))
        else:
            print("WARNING: wandb unreachable (TCP check failed) — running without logger")

    # Stop on the dynamics on new patients: val/pred_loss is the displacement error relative
    # to predicting no movement. val/loss is dominated by λ·SIGReg once the encoder starts
    # memorizing training patients, and the probe AUROC can keep rising while the dynamics
    # overfit.
    callbacks.append(pl.pytorch.callbacks.EarlyStopping(
        monitor="val/pred_loss", patience=20, mode="min", check_finite=False,
    ))

    if cfg.ckpt_path:
        ckpt = Path(get_original_cwd()) / cfg.ckpt_path
        callbacks.append(pl.pytorch.callbacks.ModelCheckpoint(
            monitor="val/pred_loss",
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

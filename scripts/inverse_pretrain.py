"""
Inverse Dynamics World Model Pre-training
==========================================
Trains an inverse dynamics model: given two consecutive ECG embeddings,
predict the pathology transition vector at that occurred between them.

Architecture:
  ht      = Encθ(Xt)
  ht1     = Encθ(Xt+1)
  d       = ht1 - ht                           ← displacement
  onset_logits, res_logits = InvPredψ(d)
  L = (1-λ) * (BCE(onset, at==1) + BCE(res, at==-1)) + λ * (SIGReg(Ht) + SIGReg(Ht+1))

Only cross_stay pairs are used: last ECG of stay k → first ECG of stay k+1.

Example
-------
HYDRA_FULL_ERROR=1 python scripts/inverse_pretrain.py ++max_epochs=1 ++batch_size=64 \\
    ++num_workers=0 ++use_wandb=false

Resuming an interrupted run
----------------------------
python scripts/inverse_pretrain.py \\
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
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.loggers import WandbLogger

import torchmetrics
import stable_pretraining as spt

from dataset.dataset import MIMICLanceDataset
from models.resnet1d import ResNet1d
from models.dynamics import ActionPredictor, SlicedEppsPulley
from utils import check_tcp


class _MultilabelAUROC(torchmetrics.classification.MultilabelAUROC):
    def update(self, preds, target):
        super().update(preds, target.long())


class _DirectAUROC(pl.Callback):
    """Accumulates pre-computed probabilities across the validation epoch and
    reports macro AUROC at epoch end — no linear probe, no optimizer."""

    def __init__(self, name: str, probs_key: str, target_key: str, num_labels: int):
        self._name       = name
        self._probs_key  = probs_key
        self._target_key = target_key
        self._metric     = _MultilabelAUROC(num_labels=num_labels, average="macro")

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


@hydra.main(version_base="1.3", config_path="../configs", config_name="inverse_pretrain")
def main(cfg):
    pair_types = tuple(cfg.pair_types)

    train_ds = MIMICLanceDataset(
        cfg.lance_path,
        split="train",
        mode="pair",
        pairs_path=cfg.pairs_path,
        pair_types=pair_types,
        train_frac=cfg.train_frac,
        cache=cfg.cache,
    )
    val_ds = MIMICLanceDataset(
        cfg.lance_path,
        split="val",
        mode="pair",
        pairs_path=cfg.pairs_path,
        pair_types=pair_types,
        cache=cfg.cache,
    )

    num_workers = 0 if cfg.cache else cfg.num_workers
    train_loader = _make_loader(train_ds, cfg.batch_size, num_workers, shuffle=True)
    val_loader   = _make_loader(val_ds,   cfg.batch_size, num_workers, shuffle=False)
    data_module  = spt.data.DataModule(train=train_loader, val=val_loader)

    backbone  = ResNet1d(in_channels=12, embedding_dim=cfg.embedding_dim)
    predictor = ActionPredictor(
        embed_dim=cfg.embedding_dim,
        hidden_dim=cfg.predictor_hidden_dim,
        action_dim=cfg.action_dim,
    )
    sigreg = SlicedEppsPulley(num_slices=cfg.n_slices, t_max=cfg.t_max, n_points=cfg.n_points)

    extra = nn.ModuleDict({"pred": predictor, "sigreg": sigreg})

    def forward(self, batch, stage):
        is_train = stage == "fit"
        prefix   = "train" if is_train else "val"

        xt, xt1, yt, at = batch["xt"], batch["xt1"], batch["yt"], batch["at"]

        ht  = self.backbone(xt)   # (B, D)
        ht1 = self.backbone(xt1)  # (B, D)
        d   = ht1 - ht            # (B, D) displacement

        onset_logits, res_logits = self.projector["pred"](d)

        onset_target = (at == 1).float()    # (B, 76)
        res_target   = (at == -1).float()   # (B, 76)

        onset_pw = torch.full((cfg.action_dim,), cfg.onset_pos_weight,      device=xt.device)
        res_pw   = torch.full((cfg.action_dim,), cfg.resolution_pos_weight, device=xt.device)
        onset_loss = F.binary_cross_entropy_with_logits(onset_logits, onset_target, pos_weight=onset_pw)
        res_loss   = F.binary_cross_entropy_with_logits(res_logits,   res_target,   pos_weight=res_pw)
        pred_loss  = onset_loss + res_loss
        reg_loss   = self.projector["sigreg"](ht) + self.projector["sigreg"](ht1)
        loss       = (1.0 - cfg.lambda_reg) * pred_loss + cfg.lambda_reg * reg_loss

        log_kw = dict(on_step=is_train, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}/loss",        loss,        prog_bar=True, **log_kw)
        self.log(f"{prefix}/pred_loss",   pred_loss,                  **log_kw)
        self.log(f"{prefix}/onset_loss",  onset_loss,                 **log_kw)
        self.log(f"{prefix}/res_loss",    res_loss,                   **log_kw)
        self.log(f"{prefix}/reg_loss",    reg_loss,                   **log_kw)
        if is_train:
            self.log("train/loss_ratio", reg_loss / (pred_loss + 1e-8),
                     on_step=False, on_epoch=True, sync_dist=True)

        # Contextualized next-state prediction (mirrors evaluate_inverse.py):
        # P(yt+1_i=1) = yt_i*(1-res_prob_i) + (1-yt_i)*onset_prob_i
        onset_probs = torch.sigmoid(onset_logits)
        res_probs   = torch.sigmoid(res_logits)
        ctx_preds   = yt * (1.0 - res_probs) + (1.0 - yt) * onset_probs  # (B, 76) in [0,1]
        yt1         = (yt + at).clamp(0.0, 1.0)                           # (B, 76) true next labels

        return {
            "embedding":        ht,
            "label":            yt,
            "displacement":     d,
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
        projector=extra,
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

    auroc_probe       = _DirectAUROC("val/ctx_auroc",        "ctx_preds",   "yt1",              76)
    onset_probe       = _DirectAUROC("val/onset_auroc",      "onset_probs", "onset_label",      cfg.action_dim)
    resolution_probe  = _DirectAUROC("val/resolution_auroc", "res_probs",   "resolution_label", cfg.action_dim)

    logger    = False
    callbacks = [auroc_probe, onset_probe, resolution_probe]
    if cfg.use_wandb:
        if check_tcp():
            logger = WandbLogger(project=cfg.wandb_project)
            callbacks.append(pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"))
        else:
            print("WARNING: wandb unreachable (TCP check failed) — running without logger")

    callbacks.append(pl.pytorch.callbacks.EarlyStopping(
        monitor="val/loss", patience=20, mode="min", check_finite=False,
    ))

    if cfg.ckpt_path:
        ckpt = Path(get_original_cwd()) / cfg.ckpt_path
        callbacks.append(pl.pytorch.callbacks.ModelCheckpoint(
            monitor="val/loss",
            dirpath=str(ckpt.parent),
            filename=ckpt.stem,
            mode="min",
            save_weights_only=True,
        ))

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        num_sanity_val_steps=1,
        log_every_n_steps=1,
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

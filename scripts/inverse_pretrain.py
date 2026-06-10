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

        return {
            "embedding":        ht,
            "label":            yt,
            "displacement":     d,
            "onset_label":      onset_target,
            "resolution_label": res_target,
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

    auroc_probe = spt.callbacks.OnlineProbe(
        module,
        name="auroc",
        input="embedding",
        target="label",
        probe=nn.Linear(cfg.embedding_dim, 76),
        loss=nn.BCEWithLogitsLoss(),
        metrics=_MultilabelAUROC(num_labels=76, average="macro"),
    )
    onset_probe = spt.callbacks.OnlineProbe(
        module,
        name="onset_auroc",
        input="displacement",
        target="onset_label",
        probe=nn.Linear(cfg.embedding_dim, cfg.action_dim),
        loss=nn.BCEWithLogitsLoss(),
        metrics=_MultilabelAUROC(num_labels=cfg.action_dim, average="macro"),
    )
    resolution_probe = spt.callbacks.OnlineProbe(
        module,
        name="resolution_auroc",
        input="displacement",
        target="resolution_label",
        probe=nn.Linear(cfg.embedding_dim, cfg.action_dim),
        loss=nn.BCEWithLogitsLoss(),
        metrics=_MultilabelAUROC(num_labels=cfg.action_dim, average="macro"),
    )

    logger    = False
    callbacks = [auroc_probe, onset_probe, resolution_probe]
    if cfg.use_wandb:
        if check_tcp():
            logger = WandbLogger(project=cfg.wandb_project)
            callbacks.append(pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"))
        else:
            print("WARNING: wandb unreachable (TCP check failed) — running without logger")

    callbacks.append(pl.pytorch.callbacks.EarlyStopping(
        monitor="val/loss", patience=20, mode="min",
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

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        seed=cfg.seed,
    )
    manager()


if __name__ == "__main__":
    main()

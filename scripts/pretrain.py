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

Example
-------
HYDRA_FULL_ERROR=1 python scripts/pretrain.py ++max_epochs=1 ++batch_size=64 \\
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
from models.dynamics import ActionProjector, DynamicsPredictor, SlicedEppsPulley
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="pretrain")
def main(cfg):
    # Train: longitudinal pairs (Xt, Xt+1, yt, at)
    train_ds = MIMICLanceDataset(
        cfg.lance_path,
        split="train",
        mode="pair",
        pairs_path=cfg.pairs_path,
        train_frac=cfg.train_frac,
        cache=cfg.cache,
    )
    # Val: longitudinal pairs — same format as train, gives a real dynamics val/loss
    val_ds = MIMICLanceDataset(
        cfg.lance_path,
        split="val",
        mode="pair",
        pairs_path=cfg.pairs_path,
        cache=cfg.cache,
    )

    num_workers = 0 if cfg.cache else cfg.num_workers
    train_loader = _make_loader(train_ds, cfg.batch_size, num_workers, shuffle=True)
    val_loader   = _make_loader(val_ds,   cfg.batch_size, num_workers, shuffle=False)
    data_module  = spt.data.DataModule(train=train_loader, val=val_loader)

    backbone  = ResNet1d(in_channels=12, embedding_dim=cfg.embedding_dim)
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
        metrics=_MultilabelAUROC(num_labels=76, average="macro"),
    )

    logger    = False
    callbacks = [auroc_probe]
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

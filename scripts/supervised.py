"""
Supervised Classification Experiment
=====================================
Trains a 1D ResNet end-to-end for multi-label ICD-10 classification on
MIMIC-IV-ECG, replicating the supervised baseline from the paper.

Example
-------
HYDRA_FULL_ERROR=1 python scripts/supervised.py \
    ++max_epochs=100 ++batch_size=256 ++embedding_dim=256
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from hydra.utils import get_original_cwd
import torch
import torch.nn.functional as F
import lightning as pl
from functools import partial
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.loggers import WandbLogger

import torchmetrics
import stable_pretraining as spt


class _MultilabelAUROC(torchmetrics.classification.MultilabelAUROC):
    def update(self, preds, target):
        super().update(preds, target.long())

from dataset.dataset import MIMICLanceDataset
from models.resnet1d import ResNet1d
from utils import check_tcp


def _make_loader(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        persistent_workers=num_workers > 0,  # keeps workers alive between epochs
        prefetch_factor=4 if num_workers > 0 else None,  # batches queued per worker
        drop_last=shuffle,
        pin_memory=True,
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="supervised")
def main(cfg):
    train_ds = MIMICLanceDataset(cfg.lance_path, split="train", mode="monitoring", train_frac=cfg.train_frac, cache=cfg.cache)
    val_ds = MIMICLanceDataset(cfg.lance_path, split="val", mode="triage", cache=cfg.cache)

    # cached datasets live entirely in RAM — spawn workers would replicate them
    num_workers = 0 if cfg.cache else cfg.num_workers
    train_loader = _make_loader(train_ds, cfg.batch_size, num_workers, shuffle=True)
    val_loader = _make_loader(val_ds, cfg.batch_size, num_workers, shuffle=False)
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)

    backbone = ResNet1d(in_channels=12, embedding_dim=cfg.embedding_dim)
    projector = torch.nn.Linear(cfg.embedding_dim, 76)

    def forward(self, batch, stage):
        batch["embedding"] = self.backbone(batch["waveform"])
        batch["logits"] = self.projector(batch["embedding"])
        batch["loss"] = F.binary_cross_entropy_with_logits(
            batch["logits"], batch["label"]
        )
        is_train = stage == "fit"
        self.log(
            "train/loss" if is_train else "val/loss",
            batch["loss"],
            on_step=is_train,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return batch

    module = spt.Module(
        backbone=backbone,
        projector=projector,
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
        probe=torch.nn.Linear(cfg.embedding_dim, 76),
        loss=torch.nn.BCEWithLogitsLoss(),
        metrics=_MultilabelAUROC(num_labels=76, average="macro"),
    )

    logger = False
    callbacks = [auroc_probe]
    if cfg.use_wandb:
        if check_tcp():
            logger = WandbLogger(project=cfg.wandb_project)
            callbacks.append(pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"))
        else:
            print("WARNING: wandb unreachable (TCP check failed) — running without logger")
    callbacks.append(pl.pytorch.callbacks.EarlyStopping(
        monitor='val/loss', 
        patience=20, 
        mode='min', 
        ))
    if cfg.ckpt_path:
        ckpt = Path(get_original_cwd()) / cfg.ckpt_path
        callbacks.append(pl.pytorch.callbacks.ModelCheckpoint(
            monitor='val/loss', 
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

    # Disable spt's cache_dir run-dir management so that ModelCheckpoint.dirpath
    # is respected as-is. Without this, spt forcibly redirects every
    # ModelCheckpoint to ~/.cache/stable-pretraining/runs/<date>/<time>/<hash>/checkpoints/.
    # spt.set(cache_dir=None)
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

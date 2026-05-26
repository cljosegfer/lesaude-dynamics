"""
Supervised Classification Experiment
=====================================
Trains a 1D ResNet end-to-end for multi-label ICD-10 classification on
MIMIC-IV-ECG, replicating the supervised baseline from the paper.

Example
-------
HYDRA_FULL_ERROR=1 python scripts/supervised.py \
    ++fold=1 ++max_epochs=100 ++batch_size=256 ++embedding_dim=256
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
import torch
import torch.nn.functional as F
import lightning as pl
from functools import partial
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt

from dataset.eval_dataset import MIMICLanceEvalDataset
from models.resnet1d import ResNet1d


def _make_loader(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        drop_last=shuffle,
        pin_memory=True,
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="supervised")
def main(cfg):
    train_ds = MIMICLanceEvalDataset(cfg.lance_path, fold=cfg.fold, split="train", mode="monitoring", train_frac=cfg.train_frac, cache=cfg.cache)
    val_ds = MIMICLanceEvalDataset(cfg.lance_path, fold=cfg.fold, split="val", mode="triage", cache=cfg.cache)

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

    lr_monitor = pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step")
    logger = WandbLogger(project=cfg.wandb_project)
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        num_sanity_val_steps=1,
        callbacks=[lr_monitor],
        precision="16-mixed",
        logger=logger,
        sync_batchnorm=True,
        enable_checkpointing=False,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        seed=cfg.seed,
    )
    manager()
    manager.validate()


if __name__ == "__main__":
    main()

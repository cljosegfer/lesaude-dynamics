"""
Test-set Bootstrap Evaluation on CSN
======================================
Loads a finetuned/supervised checkpoint and reports macro AUROC with 95%
bootstrap CI on the CSN held-out test fold.

Example
-------
HYDRA_FULL_ERROR=1 python scripts/evaluate_csn.py \
    ++evaluate.ckpt_path=checkpoints/finetune_csn_.ckpt
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.csn_dataset import CSNLanceDataset
from models.resnet1d import ResNet1d


def _load_model(ckpt_path: str, embedding_dim: int, num_classes: int, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    backbone = ResNet1d(in_channels=12, embedding_dim=embedding_dim)
    backbone.load_state_dict(
        {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    )

    projector = torch.nn.Linear(embedding_dim, num_classes)
    projector.load_state_dict(
        {k[len("projector."):]: v for k, v in sd.items() if k.startswith("projector.")}
    )

    backbone.to(device).eval()
    projector.to(device).eval()
    return backbone, projector


@torch.no_grad()
def _run_inference(backbone, projector, loader, device):
    all_preds, all_targets = [], []
    for batch in tqdm(loader, desc="Inference"):
        x = batch["waveform"].to(device)
        emb = backbone(x)
        logits = projector(emb)
        all_preds.append(F.sigmoid(logits).cpu().float().numpy())
        all_targets.append(batch["label"].cpu().float().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def _bootstrap_auroc(preds, targets, n_bootstrap: int, seed: int):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(preds)
    scores = []
    n_classes = targets.shape[1]
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
        idx = rng.integers(0, n, size=n)
        t, p = targets[idx], preds[idx]
        per_class = [
            roc_auc_score(t[:, c], p[:, c])
            for c in range(n_classes)
            if 0 < t[:, c].sum() < len(idx)
        ]
        if per_class:
            scores.append(np.mean(per_class))
    scores = np.array(scores)
    return scores.mean(), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


@hydra.main(version_base="1.3", config_path="../configs", config_name="finetune_csn")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    num_classes = len(json.load(open(cfg.vocabulary_path)))
    print(num_classes, cfg.evaluate.ckpt_path)
    backbone, projector = _load_model(cfg.evaluate.ckpt_path, cfg.embedding_dim, num_classes, device)

    ds = CSNLanceDataset(cfg.lance_path, split="test")
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.evaluate.num_workers,
        multiprocessing_context="spawn" if cfg.evaluate.num_workers > 0 else None,
        pin_memory=True,
    )
    print(f"Test set: {len(ds)} samples  ({num_classes} Dx classes)")

    preds, targets = _run_inference(backbone, projector, loader, device)

    mean, lo, hi = _bootstrap_auroc(preds, targets, cfg.evaluate.n_bootstrap, cfg.evaluate.seed)
    print(f"Macro AUROC: {mean:.4f}  (95% CI: {lo:.4f} – {hi:.4f})")


if __name__ == "__main__":
    main()

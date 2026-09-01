"""
Test-set Bootstrap Evaluation on CPSC2018
======================================
Loads a finetuned/supervised checkpoint and reports macro AUROC with 95%
bootstrap CI on the CPSC2018 held-out test fold.

Each Lance row is one 5000-sample window, and multiple windows can share the
same parent record (and therefore the same Dx labels) — see
scripts/build_lance_cpsc.py's docstring. Scoring every window as an
independent example would let correlated observations from the same record
inflate/distort the AUROC and its bootstrap CI, and wouldn't be comparable to
CSN's one-score-per-patient convention. Instead, a record's window
predictions are mean-pooled into a single score *before* computing AUROC —
record_id is fetched directly from the Lance dataset (matching CPSCLanceDataset's
row order) rather than added to the dataset's {"waveform","label"} contract.

Example
-------
HYDRA_FULL_ERROR=1 python scripts/evaluate_cpsc.py \
    ++evaluate.ckpt_path=checkpoints/finetune_cpsc_supervised.ckpt
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
import lance
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.cpsc_dataset import CPSCLanceDataset
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


def _aggregate_by_record(preds: np.ndarray, targets: np.ndarray, record_ids: list[str]):
    """Mean-pool window-level predictions into one row per record_id. Targets
    are also mean-pooled and asserted to land exactly on 0/1 — every window of
    a record carries the same dx labels by construction (build_lance_cpsc.py
    propagates labels record->window), so this is a sanity check, not a
    reduction that changes the labels."""
    n_classes = preds.shape[1]
    df_p = pd.DataFrame(preds, columns=[f"p{i}" for i in range(n_classes)])
    df_p["record_id"] = record_ids
    agg_preds = df_p.groupby("record_id", sort=True).mean().values

    df_t = pd.DataFrame(targets, columns=[f"t{i}" for i in range(n_classes)])
    df_t["record_id"] = record_ids
    agg_targets = df_t.groupby("record_id", sort=True).mean().values
    assert np.all((agg_targets == 0) | (agg_targets == 1)), (
        "Windows from the same record disagree on labels — expected identical dx across all "
        "windows of one record."
    )
    return agg_preds, agg_targets


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


@hydra.main(version_base="1.3", config_path="../configs", config_name="finetune_cpsc")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    num_classes = len(json.load(open(cfg.vocabulary_path)))
    print(num_classes, cfg.evaluate.ckpt_path)
    backbone, projector = _load_model(cfg.evaluate.ckpt_path, cfg.embedding_dim, num_classes, device)

    ds = CPSCLanceDataset(cfg.lance_path, split="test")
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.evaluate.num_workers,
        multiprocessing_context="spawn" if cfg.evaluate.num_workers > 0 else None,
        pin_memory=True,
    )
    print(f"Test set: {len(ds)} windows  ({num_classes} Dx classes)")

    preds, targets = _run_inference(backbone, projector, loader, device)

    # shuffle=False -> loader yields ds[0], ds[1], ... in order, and
    # CPSCLanceDataset indexes via self.rows[i], so ds.rows gives the exact
    # Lance row order the predictions came out in.
    record_ids = lance.dataset(cfg.lance_path).take(
        ds.rows.tolist(), columns=["record_id"]
    ).to_pydict()["record_id"]

    agg_preds, agg_targets = _aggregate_by_record(preds, targets, record_ids)
    print(f"Aggregated to {len(agg_preds)} records (mean-pooled window predictions)")

    mean, lo, hi = _bootstrap_auroc(agg_preds, agg_targets, cfg.evaluate.n_bootstrap, cfg.evaluate.seed)
    print(f"Macro AUROC (record-level): {mean:.4f}  (95% CI: {lo:.4f} – {hi:.4f})")


if __name__ == "__main__":
    main()

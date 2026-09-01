"""
Distribution Shift Evaluation: MIMIC -> CPSC2018
==============================================
Applies a MIMIC-trained/finetuned classifier (76-dim sigmoid output, one
column per 3-digit ICD-10 code), *without any further finetuning*, to
CPSC2018 waveforms. Scoring is restricted to the ICD-10 clusters that have a
genuine CPSC2018 counterpart (src/dataset/icd10_crosswalk.py +
src/dataset/snomed_to_icd10_cpsc.csv), using CPSC2018's own diagnosis
labels, OR-reduced onto those clusters, as ground truth.

This is the transfer/generalization stress test of papel/experiments.tex's
"Distribution Shift" study, mirroring scripts/evaluate_distshift.py's
MIMIC->CSN protocol.

Each Lance row is one 5000-sample window, and multiple windows can share a
parent record (see scripts/build_lance_cpsc.py). As in evaluate_cpsc.py, a
record's window-level predictions are mean-pooled into one score per record
before computing AUROC, rather than scoring windows independently.

Example
-------
# Inverse-dynamics MIMIC-finetuned classifier (the config default)
HYDRA_FULL_ERROR=1 python scripts/evaluate_distshift_cpsc.py

# Compare a different MIMIC-trained checkpoint
HYDRA_FULL_ERROR=1 python scripts/evaluate_distshift_cpsc.py \\
    ++evaluate.ckpt_path=archive/supervised_0.ckpt

# Include the low-confidence crosswalk rows too (no-op here — CPSC2018's
# crosswalk currently only uses "high" confidence rows)
HYDRA_FULL_ERROR=1 python scripts/evaluate_distshift_cpsc.py ++min_confidence=low
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
# icd10_crosswalk.py's project_csn_labels is dataset-agnostic in body despite
# its CSN-flavored name (it just OR-reduces multi-hot labels across groups of
# column indices) — aliased here rather than renaming the shared module and
# risking scripts/evaluate_distshift.py's existing CSN usage.
from dataset.icd10_crosswalk import build_intersection, project_csn_labels as project_labels
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
    """Returns MIMIC-space 76-dim sigmoid predictions and raw 9-dim CPSC2018 labels."""
    all_preds, all_cpsc_labels = [], []
    for batch in tqdm(loader, desc="Inference"):
        x = batch["waveform"].to(device)
        emb = backbone(x)
        logits = projector(emb)
        all_preds.append(F.sigmoid(logits).cpu().float().numpy())
        all_cpsc_labels.append(batch["label"].cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_cpsc_labels)


def _aggregate_by_record(preds: np.ndarray, targets: np.ndarray, record_ids: list[str]):
    """Mean-pool window-level predictions/targets into one row per record_id
    (same rationale as evaluate_cpsc.py's helper). Targets are asserted to
    land exactly on 0/1 since every window of a record shares the same
    (already OR-reduced) ground truth."""
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


def _per_class_auroc(preds, targets):
    from sklearn.metrics import roc_auc_score

    scores = []
    for c in range(targets.shape[1]):
        if 0 < targets[:, c].sum() < len(targets):
            scores.append(roc_auc_score(targets[:, c], preds[:, c]))
        else:
            scores.append(float("nan"))
    return np.array(scores)


def _bootstrap_macro_auroc(preds, targets, n_bootstrap: int, seed: int):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(preds)
    n_classes = targets.shape[1]
    scores = []
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="distshift_cpsc")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mimic_vocab = json.load(open(cfg.mimic_vocabulary_path))
    cpsc_vocab = json.load(open(cfg.cpsc_vocabulary_path))
    xwalk = build_intersection(
        mimic_vocab, cpsc_vocab, min_confidence=cfg.min_confidence,
        crosswalk_path=str(Path(__file__).parent.parent / "src/dataset/snomed_to_icd10_cpsc.csv"),
    )
    icd10_codes = xwalk["icd10_codes"]
    if not icd10_codes:
        raise RuntimeError(f"No ICD-10 clusters matched at min_confidence={cfg.min_confidence!r}")
    print(f"Matched {len(icd10_codes)} ICD-10 clusters at min_confidence={cfg.min_confidence!r}: {icd10_codes}")

    backbone, projector = _load_model(cfg.evaluate.ckpt_path, cfg.embedding_dim, len(mimic_vocab), device)
    print(f"backbone: {cfg.evaluate.ckpt_path}")

    ds = CPSCLanceDataset(cfg.cpsc_lance_path, split="test")
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.evaluate.num_workers,
        multiprocessing_context="spawn" if cfg.evaluate.num_workers > 0 else None,
        pin_memory=True,
    )
    print(f"CPSC2018 test set: {len(ds)} windows")

    preds_76, cpsc_labels = _run_inference(backbone, projector, loader, device)

    preds_k = preds_76[:, xwalk["mimic_indices"]]
    targets_k = project_labels(cpsc_labels, xwalk["csn_groups"])

    # shuffle=False -> loader yields ds[0], ds[1], ... in order, and
    # CPSCLanceDataset indexes via self.rows[i], so ds.rows gives the exact
    # Lance row order the predictions came out in.
    record_ids = lance.dataset(cfg.cpsc_lance_path).take(
        ds.rows.tolist(), columns=["record_id"]
    ).to_pydict()["record_id"]
    preds_k, targets_k = _aggregate_by_record(preds_k, targets_k, record_ids)
    print(f"Aggregated to {len(preds_k)} records (mean-pooled window predictions)")

    per_class = _per_class_auroc(preds_k, targets_k)
    print("\nPer-code AUROC (point estimate, full test set):")
    for code, auroc, n_pos in zip(icd10_codes, per_class, targets_k.sum(axis=0)):
        print(f"  {code}: {auroc:.4f}  (n_positive={int(n_pos)})")

    mean, lo, hi = _bootstrap_macro_auroc(preds_k, targets_k, cfg.evaluate.n_bootstrap, cfg.evaluate.seed)
    print(f"\nMacro AUROC ({len(icd10_codes)} matched classes): {mean:.4f}  (95% CI: {lo:.4f} - {hi:.4f})")


if __name__ == "__main__":
    main()

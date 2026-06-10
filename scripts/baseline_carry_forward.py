"""
Carry-Forward Baselines
========================
Evaluates three baselines that use only the current label yt — no model, no ECG signal.

  1. Onset structural      — score: (1 - yt_i),  target: (at_i == +1)
     Rationale: onset can only happen from yt=0; (1-yt) = 1 for all true onset events.
     AUROC ≈ fraction of onset-negatives where yt=1 (disease already present → can't onset).

  2. Resolution structural — score: yt_i,         target: (at_i == -1)
     Rationale: resolution can only happen from yt=1; yt = 1 for all true resolution events.
     AUROC ≈ fraction of resolution-negatives where yt=0 (disease absent → can't resolve).

  3. Contextualized carry-forward — score: yt_i, target: yt+1_i
     Rationale: most diagnoses persist across hospital stays; predict no change.

These numbers bound the floor of model performance: any model that uses the ECG
displacement should beat all three, otherwise it is adding noise rather than signal.

Example
-------
HYDRA_FULL_ERROR=1 python scripts/baseline_carry_forward.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import MIMICLanceDataset


def _collect(loader):
    yt_list, at_list = [], []
    for batch in tqdm(loader, desc="Loading"):
        yt_list.append(batch["yt"].float().numpy())
        at_list.append(batch["at"].float().numpy())
    yt = np.concatenate(yt_list)   # (N, 76)
    at = np.concatenate(at_list)   # (N, 76)
    return yt, at


def _bootstrap_auroc(preds, targets, n_bootstrap, seed, desc):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n, n_classes = targets.shape
    scores = []
    for _ in tqdm(range(n_bootstrap), desc=desc):
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="inverse_pretrain")
def main(cfg):
    ds = MIMICLanceDataset(
        cfg.lance_path,
        split="test",
        mode="pair",
        pairs_path=cfg.pairs_path,
        pair_types=tuple(cfg.pair_types),
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
        pin_memory=False,
    )
    print(f"Test pairs: {len(ds)}")

    yt, at = _collect(loader)
    yt1 = np.clip(yt + at, 0.0, 1.0)   # (N, 76) true future labels

    # onset_targets = (at == 1.0).astype(np.float32)   # (N, 76)
    # res_targets   = (at == -1.0).astype(np.float32)  # (N, 76)

    # onset_baseline = (1.0 - yt)   # high score where disease absent → onset possible
    # res_baseline   = yt           # high score where disease present → resolution possible
    ctx_baseline   = yt           # carry current state forward unchanged

    n_bootstrap = cfg.evaluate.n_bootstrap
    seed        = cfg.evaluate.seed

    print()
    # mean, lo, hi = _bootstrap_auroc(onset_baseline, onset_targets, n_bootstrap, seed, "Onset bootstrap")
    # print(f"Onset structural      AUROC: {mean:.4f}  (95% CI: {lo:.4f}–{hi:.4f})")

    # mean, lo, hi = _bootstrap_auroc(res_baseline, res_targets, n_bootstrap, seed, "Resolution bootstrap")
    # print(f"Resolution structural AUROC: {mean:.4f}  (95% CI: {lo:.4f}–{hi:.4f})")

    mean, lo, hi = _bootstrap_auroc(ctx_baseline, yt1, n_bootstrap, seed, "Contextualized bootstrap")
    print(f"Carry-forward         AUROC: {mean:.4f}  (95% CI: {lo:.4f}–{hi:.4f})")


if __name__ == "__main__":
    main()

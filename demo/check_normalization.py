#!/usr/bin/env python3
"""
Check whether waveforms stored in the Lance dataset are already z-score normalized.

If the per-lead mean ≈ 0 and per-lead std ≈ 1 before any processing, the normalization
applied in dataset.py (lines 126–128) is idempotent and harmless.

Usage
-----
python scripts/check_normalization.py
python scripts/check_normalization.py --config configs/data.yaml --n-samples 2000 --seed 42
"""

import argparse

import lance
import numpy as np
import yaml

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PERCENTILES = [1, 5, 25, 50, 75, 95, 99]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def fmt_row(values):
    return "  ".join(f"{v:+.4f}" for v in values)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    lance_path = cfg["lance_path"]

    ds = lance.dataset(lance_path)
    n_total = ds.count_rows()
    n = min(args.n_samples, n_total)
    print(f"Dataset : {lance_path}")
    print(f"Rows    : {n_total:,}  |  sampling {n:,} (seed={args.seed})\n")

    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(n_total, size=n, replace=False))

    table = ds.take(indices.tolist(), columns=["waveform"])
    waveforms = (
        table.column("waveform").combine_chunks().flatten()
        .to_numpy(zero_copy_only=False)
        .reshape(n, 5000, 12)
        .astype(np.float32)
    )

    # Per-lead stats for each ECG: shapes (N, 12)
    means = waveforms.mean(axis=1)   # mean over 5000 time steps
    stds  = waveforms.std(axis=1)    # std  over 5000 time steps

    # ── Global summary ────────────────────────────────────────────────────────
    all_means = means.ravel()
    all_stds  = stds.ravel()

    pct_labels = "  ".join(f"p{p:02d}" for p in PERCENTILES)
    print(f"{'Per-lead MEANS across all samples':}")
    print(f"  overall  mean={all_means.mean():+.4f}  std={all_means.std():.4f}")
    print(f"  {pct_labels}")
    print(f"  {fmt_row(np.percentile(all_means, PERCENTILES))}\n")

    print(f"{'Per-lead STDS across all samples':}")
    print(f"  overall  mean={all_stds.mean():+.4f}  std={all_stds.std():.4f}")
    print(f"  {pct_labels}")
    print(f"  {fmt_row(np.percentile(all_stds, PERCENTILES))}\n")

    # ── Per-lead breakdown ────────────────────────────────────────────────────
    header = f"{'Lead':<6}  {'mean(μ)':>9}  {'std(μ)':>9}  {'mean(σ)':>9}  {'std(σ)':>9}"
    print(header)
    print("-" * len(header))
    for lead_idx, lead in enumerate(LEADS):
        m = means[:, lead_idx]
        s = stds[:, lead_idx]
        print(f"{lead:<6}  {m.mean():>+9.4f}  {m.std():>9.4f}  {s.mean():>9.4f}  {s.std():>9.4f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    mean_of_means = abs(all_means.mean())
    mean_of_stds  = abs(all_stds.mean() - 1.0)
    print()
    if mean_of_means < 0.05 and mean_of_stds < 0.05:
        print("VERDICT: likely pre-normalized  (|mean(μ)| < 0.05 and |mean(σ)−1| < 0.05)")
        print("         The z-score step in dataset.py is idempotent on these data.")
    else:
        print("VERDICT: NOT pre-normalized")
        print(f"         |mean(μ)|={mean_of_means:.4f},  |mean(σ)−1|={mean_of_stds:.4f}")
        print("         The z-score step in dataset.py is doing real work.")


if __name__ == "__main__":
    main()

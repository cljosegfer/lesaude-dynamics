#!/usr/bin/env python3
"""
Exploratory: how many cross-stay pairs would a multi-step pairing scheme add,
and how does the stable/active label split shift as the offset grows?

Current build_pair_index() only pairs consecutive stays per patient:
    stay N (first ECG) -> stay N+1 (first ECG)          [offset k=1]

This script asks: if we additionally pair stay N -> stay N+k for k=2,3,4,...,
how fast does the pair count grow, and does the fraction of pairs with an
"active" transition (at least one ICD label flips between the two stays,
at = clip(yt1 - yt, -1, 1) != 0) hold up as k grows? Since all such pairs stay
within one patient, they never cross a fold boundary, so no leakage risk from
adding offsets — this is purely a "how much more data, and is it still useful
data" question.

For a patient with L stays (first-ECG-of-stay count), pairing at offset k
yields max(0, L - k) pairs (indices i, i+k over the time-ordered stay list).

Usage:
  python demo/cross_stay_multistep_pairs.py
  python demo/cross_stay_multistep_pairs.py --config configs/data.yaml --k-max 10
"""

import argparse

import lance
import numpy as np
import pandas as pd
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--k-max", type=int, default=10)
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_multistep_pairs(stays: pd.DataFrame, k_max: int) -> pd.DataFrame:
    """For every patient and every offset k=1..k_max, pair first-ECG-of-stay i -> i+k."""
    idx_t_parts, idx_t1_parts, k_parts = [], [], []
    for _, group in stays.groupby("subject_id", sort=False):
        idx = group["lance_idx"].values
        max_k = min(k_max, len(idx) - 1)
        for k in range(1, max_k + 1):
            idx_t_parts.append(idx[:-k])
            idx_t1_parts.append(idx[k:])
            k_parts.append(np.full(len(idx) - k, k, dtype=np.int16))

    return pd.DataFrame({
        "idx_t": np.concatenate(idx_t_parts),
        "idx_t1": np.concatenate(idx_t1_parts),
        "offset_k": np.concatenate(k_parts),
    })


def main():
    args = parse_args()
    config = load_config(args.config)

    ds = lance.dataset(config["lance_path"])
    meta = ds.to_table(columns=["subject_id", "ecg_no_within_stay", "ecg_time", "fold"]).to_pandas()
    meta["lance_idx"] = meta.index

    # First ECG of each stay, time-ordered within patient, train fold only.
    stays = meta[(meta["ecg_no_within_stay"] == 0) & (meta["fold"] <= 17)].sort_values(
        ["subject_id", "ecg_time"]
    )
    stay_counts = stays.groupby("subject_id").size()

    print(f"Train patients with >=1 stay: {len(stay_counts):,}")
    print(
        f"Stays per patient (train): mean={stay_counts.mean():.2f}  "
        f"median={stay_counts.median():.0f}  max={stay_counts.max()}"
    )

    pairs = build_multistep_pairs(stays, args.k_max)

    # Fetch ICD labels once for every row referenced at any offset, deduplicated.
    unique_idx = np.unique(np.concatenate([pairs["idx_t"].values, pairs["idx_t1"].values]))
    labels_table = ds.take(unique_idx.tolist(), columns=["icd"])
    labels = (
        labels_table.column("icd").combine_chunks().flatten()
        .to_numpy(zero_copy_only=False)
        .reshape(len(unique_idx), 76)
        .astype(np.int16)
    )
    pos_t = np.searchsorted(unique_idx, pairs["idx_t"].values)
    pos_t1 = np.searchsorted(unique_idx, pairs["idx_t1"].values)
    at = np.clip(labels[pos_t1] - labels[pos_t], -1, 1)
    pairs["active"] = np.abs(at).sum(axis=1) > 0

    rows = []
    for k in range(1, args.k_max + 1):
        cum = pairs[pairs["offset_k"] <= k]
        pairs_at_k = int((pairs["offset_k"] == k).sum())
        cumulative_pairs = len(cum)
        n_active = int(cum["active"].sum())
        n_stable = cumulative_pairs - n_active
        stable_pct = 100 * n_stable / cumulative_pairs if cumulative_pairs else float("nan")
        active_pct = 100 * n_active / cumulative_pairs if cumulative_pairs else float("nan")
        rows.append((k, pairs_at_k, cumulative_pairs, f"{stable_pct:.1f}% / {active_pct:.1f}%"))

    table = pd.DataFrame(
        rows, columns=["offset_k", "pairs_at_k", "cumulative_pairs", "cumulative_stable/active"]
    )
    baseline = table.loc[0, "cumulative_pairs"]  # k=1 pairs == current cross_stay pair count
    table["pct_vs_k1_only"] = (table["cumulative_pairs"] / baseline - 1) * 100

    print("\n=== train ===")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

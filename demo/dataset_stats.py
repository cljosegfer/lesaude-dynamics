#!/usr/bin/env python3
"""
Training-set characterization for MIMIC-IV-ECG.

Reports, for the train split (folds 0-17):
  - number of exams (rows in the main Lance dataset)
  - for each pair type (within_stay, cross_stay):
      - number of pairs
      - number of stable pairs (at == 0, i.e. yt1 == yt) vs. active pairs (at != 0 somewhere)
      - mean number of non-zero action columns (||a_t||_0), averaged over active pairs only

Usage:
  python scripts/dataset_stats.py
  python scripts/dataset_stats.py --config configs/data.yaml
"""

import argparse

import lance
import numpy as np
import yaml


TRAIN_FOLD_FILTER = "fold <= 17"
PAIR_TYPES = ("within_stay", "cross_stay")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    ds = lance.dataset(config["lance_path"])
    pairs_ds = lance.dataset(config["pairs_path"])

    # Number of exams in the training split.
    fold = ds.to_table(columns=["fold"]).to_pandas()["fold"]
    n_exams = int((fold <= 17).sum())
    print(f"Train exams: {n_exams:,}")

    pairs_df = pairs_ds.to_table(filter=TRAIN_FOLD_FILTER).to_pandas()[
        ["idx_t", "idx_t1", "pair_type"]
    ]

    # Fetch ICD labels once for every row referenced by a training pair, deduplicated.
    unique_idx = np.unique(np.concatenate([pairs_df["idx_t"].values, pairs_df["idx_t1"].values]))
    labels_table = ds.take(unique_idx.tolist(), columns=["icd"])
    labels = (
        labels_table.column("icd").combine_chunks().flatten()
        .to_numpy(zero_copy_only=False)
        .reshape(len(unique_idx), 76)
        .astype(np.int16)
    )
    idx_to_pos = {int(idx): pos for pos, idx in enumerate(unique_idx)}

    header = f"\n{'pair_type':<14}{'pairs':>12}{'stable':>12}{'active':>12}{'mean ||a||_0 (active)':>24}"
    print(header)
    for pair_type in PAIR_TYPES:
        sub = pairs_df[pairs_df["pair_type"] == pair_type]
        n_pairs = len(sub)
        if n_pairs == 0:
            print(f"{pair_type:<14}{0:>12}{0:>12}{0:>12}{'--':>24}")
            continue

        pos_t = sub["idx_t"].map(idx_to_pos).values
        pos_t1 = sub["idx_t1"].map(idx_to_pos).values
        at = np.clip(labels[pos_t1] - labels[pos_t], -1, 1)

        l0 = np.abs(at).sum(axis=1)  # number of flipped label columns per pair
        active_mask = l0 > 0
        n_active = int(active_mask.sum())
        n_stable = n_pairs - n_active
        mean_l0_active = float(l0[active_mask].mean()) if n_active > 0 else float("nan")

        print(f"{pair_type:<14}{n_pairs:>12,}{n_stable:>12,}{n_active:>12,}{mean_l0_active:>24.3f}")


if __name__ == "__main__":
    main()

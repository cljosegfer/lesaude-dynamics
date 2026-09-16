#!/usr/bin/env python3
"""
Build a multi-step cross-stay pair index for MIMIC-IV-ECG.

Generalizes build_pair_index()'s (scripts/build_lance.py) cross-stay pairing
— stay N -> stay N+1 (offset k=1) only — to every offset k=1..--k-max:
stay N -> stay N+k, for every hop distance up to k-max, all pooled into one
Lance dataset. Every row has pair_type="cross_stay" (there is no within_stay
in this file), so it's a drop-in replacement for pairs.lance via
MIMICLanceDataset(pairs_path=...) with the default pair_types — no changes
needed downstream.

All folds are included (not just train), same as pairs.lance, so val/test
splits still work if ever needed for pair-mode evaluation.

Output schema (identical to pairs.lance, plus one extra column):
  idx_t      int64   — row index into mimic_iv_ecg.lance (Xt)
  idx_t1     int64   — row index into mimic_iv_ecg.lance (Xt+k)
  subject_id int32
  fold       int8
  pair_type  string  — always "cross_stay"
  offset_k   int16   — hop distance between the two stays (1..k_max)

Usage:
  python scripts/build_pairs_multistep.py
  python scripts/build_pairs_multistep.py --config configs/data.yaml --k-max 12 \
      --out-path /path/to/pairs_multistep_k12.lance
"""

import argparse
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import yaml
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument(
        "--out-path",
        default=None,
        help="Output .lance path. Defaults to pairs_multistep_k<k-max>.lance next to lance_path.",
    )
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_multistep_pairs(meta: pd.DataFrame, k_max: int) -> pd.DataFrame:
    """
    For every patient, take the time-ordered first-ECG-of-stay rows (dropping
    anomalous ecg_no_within_stay < 0 first, matching build_pair_index), and
    pair stay i -> stay i+k for every k = 1..k_max.
    """
    idx_t_parts, idx_t1_parts, k_parts, subj_parts, fold_parts = [], [], [], [], []

    for subj_id, group in tqdm(meta.groupby("subject_id", sort=False), desc="Building multistep pairs"):
        group = group[group["ecg_no_within_stay"] >= 0].sort_values("ecg_time")
        first_idx = group.loc[group["ecg_no_within_stay"] == 0, "lance_idx"].values
        L = len(first_idx)
        if L < 2:
            continue

        fold_val = int(group["fold"].values[0])
        max_k = min(k_max, L - 1)
        for k in range(1, max_k + 1):
            t, t1 = first_idx[:-k], first_idx[k:]
            b = len(t)
            idx_t_parts.append(t)
            idx_t1_parts.append(t1)
            k_parts.append(np.full(b, k, dtype=np.int16))
            subj_parts.append(np.full(b, int(subj_id), dtype=np.int32))
            fold_parts.append(np.full(b, fold_val, dtype=np.int8))

    return pd.DataFrame({
        "idx_t": np.concatenate(idx_t_parts),
        "idx_t1": np.concatenate(idx_t1_parts),
        "subject_id": np.concatenate(subj_parts),
        "fold": np.concatenate(fold_parts),
        "offset_k": np.concatenate(k_parts),
    })


def main():
    args = parse_args()
    config = load_config(args.config)

    lance_path = Path(config["lance_path"])
    out_path = (
        Path(args.out_path) if args.out_path
        else lance_path.parent / f"pairs_multistep_k{args.k_max}.lance"
    )

    print(f"Source : {lance_path}")
    print(f"Output : {out_path}")

    ds = lance.dataset(str(lance_path))
    meta = ds.to_table(columns=["subject_id", "ecg_no_within_stay", "ecg_time", "fold"]).to_pandas()
    meta["lance_idx"] = meta.index

    pairs_df = build_multistep_pairs(meta, args.k_max)

    table = pa.table({
        "idx_t": pa.array(pairs_df["idx_t"].values, type=pa.int64()),
        "idx_t1": pa.array(pairs_df["idx_t1"].values, type=pa.int64()),
        "subject_id": pa.array(pairs_df["subject_id"].values, type=pa.int32()),
        "fold": pa.array(pairs_df["fold"].values, type=pa.int8()),
        "pair_type": pa.array(["cross_stay"] * len(pairs_df), type=pa.string()),
        "offset_k": pa.array(pairs_df["offset_k"].values, type=pa.int16()),
    })

    lance.write_dataset(table, str(out_path), mode="overwrite")

    print(f"\nMultistep pair index: {len(pairs_df):,} total pairs -> {out_path}")
    print("\nPairs per offset k:")
    print(pairs_df["offset_k"].value_counts().sort_index().rename("pairs").to_string())


if __name__ == "__main__":
    main()

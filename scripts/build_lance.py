#!/usr/bin/env python3
"""
Convert MIMIC-IV-ECG HDF5 + CSV files into a unified Lance dataset.

Outputs (written to --out-dir, defaulting to the same directory as the source files):
  mimic_iv_ecg.lance      — main dataset (800035 rows)
  pairs.lance             — precomputed pair index (within-stay + cross-stay)
  icd_vocabulary_76.json  — list[str] of 76 ICD-10 3-digit codes (index → code)

Usage:
  python scripts/build_lance.py
  python scripts/build_lance.py --config configs/data.yaml --out-dir /scratch/josefernandes/
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import lance
import yaml
from tqdm import tqdm


BATCH_SIZE = 2000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to the same directory as the source files.",
    )
    p.add_argument(
        "--pairs-only",
        action="store_true",
        help="Skip main dataset write; only regenerate pairs.lance (fast, ~10s).",
    )
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _make_schema():
    return pa.schema([
        pa.field("study_id", pa.int32()),
        pa.field("subject_id", pa.int32()),
        pa.field("ecg_time", pa.float64()),
        pa.field("ecg_no_within_stay", pa.int16()),
        pa.field("fold", pa.int8()),
        pa.field("icd_raw", pa.list_(pa.string())),
        pa.field("icd", pa.list_(pa.int8(), 76)),
        pa.field("waveform", pa.list_(pa.float16(), 60000)),
    ])


def _to_fixed_list_float16(arr2d: np.ndarray, list_size: int) -> pa.Array:
    """Convert (B, list_size) float16 numpy array → FixedSizeListArray<float16>."""
    flat = pa.array(arr2d.flatten(), type=pa.float16())
    return pa.FixedSizeListArray.from_arrays(flat, list_size)


def _to_fixed_list_int8(arr2d: np.ndarray, list_size: int) -> pa.Array:
    """Convert (B, list_size) int8 numpy array → FixedSizeListArray<int8>."""
    flat = pa.array(arr2d.flatten(), type=pa.int8())
    return pa.FixedSizeListArray.from_arrays(flat, list_size)


def build_main_dataset(df: pd.DataFrame, f_wave, f_label, vocabulary: list[str], out_path: Path):
    schema = _make_schema()
    n = len(df)
    first_write = True

    for start in tqdm(range(0, n, BATCH_SIZE), desc="Writing main Lance dataset"):
        batch_df = df.iloc[start : start + BATCH_SIZE]
        h5_idx = batch_df["h5_index"].values

        # Sort h5_idx before reading — HDF5 fancy indexing is much faster on sorted indices.
        sort_order = np.argsort(h5_idx)
        inv_order = np.argsort(sort_order)
        sorted_h5_idx = h5_idx[sort_order]

        waveforms = f_wave["waveforms"][sorted_h5_idx]  # (B, 5000, 12) float16
        ecg_times = f_wave["ecg_time"][sorted_h5_idx]   # (B,)
        labels = f_label["icd"][sorted_h5_idx]           # (B, 76) int8

        # Restore CSV order
        waveforms = waveforms[inv_order]
        ecg_times = ecg_times[inv_order]
        labels = labels[inv_order]

        # icd_raw: list of active ICD code strings per ECG
        icd_raw = [[vocabulary[j] for j in np.where(row)[0]] for row in labels]

        # Flatten waveforms: (B, 5000, 12) → (B, 60000)
        waveforms_flat = waveforms.reshape(len(batch_df), -1)

        table = pa.table(
            {
                "study_id": pa.array(batch_df["study_id"].values, type=pa.int32()),
                "subject_id": pa.array(batch_df["subject_id"].values, type=pa.int32()),
                "ecg_time": pa.array(ecg_times, type=pa.float64()),
                "ecg_no_within_stay": pa.array(
                    batch_df["ecg_no_within_stay"].values.astype(np.int16), type=pa.int16()
                ),
                "fold": pa.array(batch_df["fold"].values.astype(np.int8), type=pa.int8()),
                "icd_raw": pa.array(icd_raw, type=pa.list_(pa.string())),
                "icd": _to_fixed_list_int8(labels, 76),
                "waveform": _to_fixed_list_float16(waveforms_flat, 60000),
            },
            schema=schema,
        )

        mode = "create" if first_write else "append"
        lance.write_dataset(table, str(out_path), mode=mode)
        first_write = False


def build_pair_index(lance_path: Path, out_path: Path):
    """
    Construct within-stay and cross-stay pairs.

    Within-stay:  consecutive ECGs within the same hospital stay, detected by
                  ecg_no_within_stay increasing (study_id is unique per ECG in
                  MIMIC-IV-ECG, so it cannot be used for stay boundary detection).
    Cross-stay:   last ECG of stay N → first ECG of stay N+1 for the same patient.
                  Detected when ecg_no_within_stay resets (next value ≤ current).
    """
    ds = lance.dataset(str(lance_path))

    # Read only lightweight metadata columns — no waveforms.
    meta = ds.to_table(
        columns=["subject_id", "ecg_no_within_stay", "ecg_time", "fold"]
    ).to_pandas()
    meta["lance_idx"] = meta.index  # true row position in the Lance dataset

    idx_t_parts: list[np.ndarray] = []
    idx_t1_parts: list[np.ndarray] = []
    subj_parts: list[list[int]] = []
    fold_parts: list[list[int]] = []
    type_parts: list[list[str]] = []

    for subj_id, group in tqdm(
        meta.groupby("subject_id", sort=False), desc="Building pair index"
    ):
        if len(group) < 2:
            continue

        group = group.sort_values("ecg_time")
        idx = group["lance_idx"].values
        enws = group["ecg_no_within_stay"].values
        fold_val = int(group["fold"].values[0])
        b = len(idx) - 1

        # ecg_no_within_stay increases within a stay; resets at stay boundary.
        same_stay = enws[:-1] < enws[1:]
        pair_types = ["within_stay" if s else "cross_stay" for s in same_stay]

        idx_t_parts.append(idx[:-1])
        idx_t1_parts.append(idx[1:])
        subj_parts.append([int(subj_id)] * b)
        fold_parts.append([fold_val] * b)
        type_parts.append(pair_types)

    idx_t_all = np.concatenate(idx_t_parts)
    idx_t1_all = np.concatenate(idx_t1_parts)
    subj_all = [v for part in subj_parts for v in part]
    fold_all = [v for part in fold_parts for v in part]
    type_all = [v for part in type_parts for v in part]

    pair_table = pa.table({
        "idx_t": pa.array(idx_t_all, type=pa.int64()),
        "idx_t1": pa.array(idx_t1_all, type=pa.int64()),
        "subject_id": pa.array(subj_all, type=pa.int32()),
        "fold": pa.array(fold_all, type=pa.int8()),
        "pair_type": pa.array(type_all, type=pa.string()),
    })

    lance.write_dataset(pair_table, str(out_path), mode="overwrite")

    n_within = sum(1 for t in type_all if t == "within_stay")
    n_cross = len(type_all) - n_within
    print(
        f"Pair index: {len(type_all):,} total  "
        f"({n_within:,} within-stay, {n_cross:,} cross-stay)"
    )


def verify(lance_path: Path, pairs_path: Path, df: pd.DataFrame, f_wave, f_label):
    print("\nRunning verification checks...")

    ds = lance.dataset(str(lance_path))
    pairs_ds = lance.dataset(str(pairs_path))

    assert ds.count_rows() == len(df), (
        f"Row count mismatch: Lance={ds.count_rows()}, CSV={len(df)}"
    )
    print(f"  [OK] Main dataset row count: {ds.count_rows():,}")

    pair_count = pairs_ds.count_rows()
    assert pair_count > 0, "Pair index is empty"
    print(f"  [OK] Pair index row count: {pair_count:,}")

    # Sample 5 random rows and compare waveforms against HDF5
    rng = np.random.default_rng(42)
    sample_csv_rows = rng.integers(0, len(df), size=5)
    sample_h5_idx = df.iloc[sample_csv_rows]["h5_index"].values
    lance_rows = ds.take(sample_csv_rows.tolist(), columns=["waveform"]).to_pydict()

    for i, (csv_row, h5_idx) in enumerate(zip(sample_csv_rows, sample_h5_idx)):
        expected = f_wave["waveforms"][int(h5_idx)]  # (5000, 12) float16
        got = np.array(lance_rows["waveform"][i], dtype=np.float16).reshape(5000, 12)
        assert np.array_equal(expected, got), f"Waveform mismatch at CSV row {csv_row}"
    print("  [OK] Waveform integrity (5 random samples)")

    print("Verification passed.\n")


def main():
    args = parse_args()
    config = load_config(args.config)

    data_dir = Path(config["data_dir"])
    out_dir = Path(args.out_dir) if args.out_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    waveform_path = data_dir / "mimic_iv_ecg_waveforms.h5"
    label_path = data_dir / "mimic_iv_ecg_icd.h5"
    metadata_path = data_dir / "metadata.csv"
    lance_path = out_dir / "mimic_iv_ecg.lance"
    pairs_path = out_dir / "pairs.lance"
    vocab_path = out_dir / "icd_vocabulary_76.json"

    print(f"Source  : {data_dir}")
    print(f"Output  : {out_dir}")

    df = pd.read_csv(metadata_path)
    print(f"Metadata: {len(df):,} records")

    if args.pairs_only:
        # Regenerate pairs.lance only — no need to open HDF5 files.
        print("--pairs-only: skipping main dataset write.")
        build_pair_index(lance_path, pairs_path)
        # Lightweight verify: just check row counts.
        ds = lance.dataset(str(lance_path))
        pairs_ds = lance.dataset(str(pairs_path))
        print(f"  [OK] Main dataset row count: {ds.count_rows():,}")
        print(f"  [OK] Pair index row count: {pairs_ds.count_rows():,}")
        return

    # Large read-ahead cache for sequential HDF5 access
    f_wave = h5py.File(waveform_path, "r", rdcc_nbytes=1024 * 1024 * 512)
    f_label = h5py.File(label_path, "r")

    # Step 1 — ICD vocabulary
    vocabulary = [
        v.decode() if isinstance(v, bytes) else str(v) for v in f_label["vocabulary"][:]
    ]
    with open(vocab_path, "w") as fp:
        json.dump(vocabulary, fp)
    print(f"Vocabulary: {len(vocabulary)} codes → {vocab_path}")

    # Step 2 — main Lance dataset
    build_main_dataset(df, f_wave, f_label, vocabulary, lance_path)

    # Step 3 — pair index
    build_pair_index(lance_path, pairs_path)

    # Step 4 — verify
    verify(lance_path, pairs_path, df, f_wave, f_label)

    f_wave.close()
    f_label.close()

    ds = lance.dataset(str(lance_path))
    import shutil
    lance_size_gb = sum(
        f.stat().st_size for f in lance_path.rglob("*") if f.is_file()
    ) / 1e9
    print(f"Lance dataset size on disk: {lance_size_gb:.1f} GB")
    print(f"Done. Main rows: {ds.count_rows():,}")


if __name__ == "__main__":
    main()

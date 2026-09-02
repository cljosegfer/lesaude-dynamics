#!/usr/bin/env python3
"""
Convert the raw PTB-XL WFDB records (500Hz copy) into a Lance dataset,
matching MIMIC-IV-ECG's normalization convention so the pretrained encoder
sees the same input distribution it was trained on.

Same preprocessing as build_lance_csn.py/build_lance_cpsc.py:
  - Read physical-unit signal via wfdb.rdrecord(...).p_signal (mV)
  - Truncate to the first 5000 samples if longer, zero-pad the end if shorter
    (defensive only -- the pre-flight scan confirmed all 21,799 records are
    already exactly 5000 samples @ 500Hz)
  - Per-lead z-score normalization (robust_z_score), cast to float16

Two structural differences from CSN/CPSC:
  - Labels come from ptbxl_database.csv's `scp_codes` column (a dict-literal
    string of code:likelihood pairs), not a WFDB header comment line. Any
    code present in the dict counts as positive regardless of its listed
    likelihood (0 in PTB-XL's own convention means "asserted without a
    likelihood value", not "absent") -- the same "any listed code =
    positive" treatment CSN/CPSC give their Dx comments. Per user decision,
    the label vocabulary is built from the raw ~71 observed SCP codes (repo
    convention: vocabulary = everything actually observed), not reduced to
    the 5-class diagnostic-superclass grouping PTB-XL papers typically use.
  - Folds use PTB-XL's own official `strat_fold` column (1-10) directly --
    confirmed by the pre-flight scan to be well-balanced (~2173-2198/fold)
    -- instead of a synthetic seeded permutation. Recommended split
    (PhysioNet project page / Wagner et al.): fold<=8 train, fold==9 val,
    fold==10 test.

Gotcha found during the pre-flight scan: PTB-XL's headers name the augmented
leads "AVR"/"AVL"/"AVF" (uppercase, no lowercase "a"), unlike MIMIC/CSN/
CPSC's "aVR"/"aVL"/"aVF" -- lead reordering below matches case-insensitively
(21,799/21,799 records confirmed already in canonical order once compared
case-insensitively, so this is defensive, not corrective).

Outputs (written to --out-dir, defaulting to the PTB-XL data dir):
  ptbxl.lance                — main dataset
  dx_vocabulary_ptbxl.json   — list[str] of distinct SCP codes (index -> code)

Usage:
  python scripts/build_lance_ptbxl.py
  python scripts/build_lance_ptbxl.py --raw-dir /path/to/ptbxl/raw --out-dir /path/to/out
"""

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import lance
import wfdb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dataset.lance_utils import to_fixed_list_float16, to_fixed_list_int8


TARGET_LEN = 5000
CHANNELS = 12
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEADS_UPPER = [lead.upper() for lead in LEADS]
EPSILON = 1e-6
BATCH_SIZE = 2000

DEFAULT_RAW_DIR = (
    "/snfs2/josefernandes/datasets/lesaude/ptbxl-monolith/raw/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
DEFAULT_OUT_DIR = "/snfs2/josefernandes/datasets/lesaude/ptbxl-monolith"

_REL_PATH_CACHE: dict[str, str] = {}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return p.parse_args()


def robust_z_score(signal: np.ndarray) -> np.ndarray:
    """Per-lead z-score. Matches demo/preprocess_waveforms0.py exactly, copied
    verbatim from build_lance_csn.py/build_lance_cpsc.py, so PTB-XL waveforms
    share MIMIC's normalization convention."""
    signal = np.nan_to_num(signal)
    mean = signal.mean(axis=0)
    std = signal.std(axis=0)
    normalized = np.zeros_like(signal)
    for lead in range(signal.shape[1]):
        if std[lead] > EPSILON:
            normalized[:, lead] = (signal[:, lead] - mean[lead]) / std[lead]
    return normalized


def load_metadata(db_csv: Path) -> list[dict]:
    with open(db_csv, newline="") as f:
        return list(csv.DictReader(f))


def parse_scp_codes(raw: str) -> list[str] | None:
    try:
        codes = ast.literal_eval(raw)
        return sorted(codes.keys())
    except Exception:
        return None


def scan_records(raw_dir: Path, rows: list[dict]):
    """Pass 1 (fast, header-only): validate each row's header/leads and parse
    scp_codes, skip malformed rows. Mirrors CSN/CPSC's scan_headers()."""
    record_ids, scp_raw_list, folds = [], [], []
    skipped = []
    for row in tqdm(rows, desc="Scanning records"):
        ecg_id = row["ecg_id"]
        rel_path = row["filename_hr"].strip()
        try:
            header = wfdb.rdheader(str(raw_dir / rel_path))
            sig_name_upper = [s.upper() for s in header.sig_name]
            if set(LEADS_UPPER) - set(sig_name_upper):
                raise ValueError(f"missing leads: {set(LEADS_UPPER) - set(sig_name_upper)}")
            scp_codes = parse_scp_codes(row["scp_codes"])
            if scp_codes is None:
                raise ValueError(f"unparsable scp_codes: {row['scp_codes']!r}")
            fold = int(row["strat_fold"])
        except Exception as e:
            skipped.append((ecg_id, str(e)))
            continue
        record_ids.append(ecg_id)
        scp_raw_list.append(scp_codes)
        folds.append(fold)
    return record_ids, scp_raw_list, folds, skipped


def read_waveform(raw_dir: Path, record_id: str) -> np.ndarray:
    """Read + crop/pad + normalize a single record's signal. Returns (5000, 12) float16."""
    rel_path = _REL_PATH_CACHE[record_id]
    record = wfdb.rdrecord(str(raw_dir / rel_path))
    signal = record.p_signal  # (length, n_leads), physical units (mV)

    sig_name_upper = [s.upper() for s in record.sig_name]
    if sig_name_upper != LEADS_UPPER:
        order = [sig_name_upper.index(lead) for lead in LEADS_UPPER]
        signal = signal[:, order]

    length = signal.shape[0]
    if length > TARGET_LEN:
        signal = signal[:TARGET_LEN, :]
    elif length < TARGET_LEN:
        signal = np.pad(signal, ((0, TARGET_LEN - length), (0, 0)), "constant")

    return robust_z_score(signal).astype(np.float16)


def _make_schema(n_classes: int):
    return pa.schema(
        [
            pa.field("record_id", pa.string()),
            pa.field("fold", pa.int8()),
            pa.field("scp_raw", pa.list_(pa.string())),
            pa.field("scp", pa.list_(pa.int8(), n_classes)),
            pa.field("waveform", pa.list_(pa.float16(), TARGET_LEN * CHANNELS)),
        ]
    )


def build_lance(raw_dir: Path, record_ids, scp_raw_list, folds, vocabulary, out_path: Path):
    n_classes = len(vocabulary)
    code_to_idx = {c: i for i, c in enumerate(vocabulary)}
    schema = _make_schema(n_classes)
    n = len(record_ids)
    first_write = True

    for start in tqdm(range(0, n, BATCH_SIZE), desc="Writing Lance dataset"):
        end = min(start + BATCH_SIZE, n)
        batch_ids = record_ids[start:end]
        batch_scp_raw = scp_raw_list[start:end]

        batch_waveforms = np.stack([read_waveform(raw_dir, rid) for rid in batch_ids])  # (B,5000,12)

        batch_scp = np.zeros((end - start, n_classes), dtype=np.int8)
        for i, codes in enumerate(batch_scp_raw):
            for c in codes:
                batch_scp[i, code_to_idx[c]] = 1

        table = pa.table(
            {
                "record_id": pa.array(batch_ids, type=pa.string()),
                "fold": pa.array(folds[start:end], type=pa.int8()),
                "scp_raw": pa.array(batch_scp_raw, type=pa.list_(pa.string())),
                "scp": to_fixed_list_int8(batch_scp, n_classes),
                "waveform": to_fixed_list_float16(
                    batch_waveforms.reshape(end - start, -1), TARGET_LEN * CHANNELS
                ),
            },
            schema=schema,
        )

        mode = "create" if first_write else "append"
        lance.write_dataset(table, str(out_path), mode=mode)
        first_write = False


def verify(out_path: Path, raw_dir: Path, expected_n: int):
    print("\nRunning verification checks...")
    ds = lance.dataset(str(out_path))
    assert ds.count_rows() == expected_n, (
        f"Row count mismatch: Lance={ds.count_rows()}, expected={expected_n}"
    )
    print(f"  [OK] Row count: {ds.count_rows():,}")

    rng = np.random.default_rng(42)
    sample_rows = rng.integers(0, expected_n, size=5).tolist()
    lance_rows = ds.take(sample_rows, columns=["record_id", "waveform"]).to_pydict()
    for i, row in enumerate(sample_rows):
        rid = lance_rows["record_id"][i]
        expected = read_waveform(raw_dir, rid)
        got = np.array(lance_rows["waveform"][i], dtype=np.float16).reshape(TARGET_LEN, CHANNELS)
        assert np.array_equal(expected, got), f"Waveform mismatch at row {row} ({rid})"
    print("  [OK] Waveform integrity (5 random samples, recomputed from source)")
    print("Verification passed.\n")


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_csv = raw_dir / "ptbxl_database.csv"
    rows = load_metadata(db_csv)
    print(f"Loaded {len(rows):,} rows from {db_csv}")
    _REL_PATH_CACHE.update({row["ecg_id"]: row["filename_hr"].strip() for row in rows})

    record_ids, scp_raw_list, folds, skipped = scan_records(raw_dir, rows)
    print(f"Valid records: {len(record_ids):,}  (skipped {len(skipped)})")
    for ecg_id, err in skipped:
        print(f"  SKIP ecg_id={ecg_id}: {err}")

    vocabulary = sorted({c for codes in scp_raw_list for c in codes})
    vocab_path = out_dir / "dx_vocabulary_ptbxl.json"
    with open(vocab_path, "w") as f:
        json.dump(vocabulary, f)
    print(f"Vocabulary: {len(vocabulary)} SCP codes -> {vocab_path}")

    n = len(record_ids)
    folds = np.array(folds, dtype=np.int8)

    lance_path = out_dir / "ptbxl.lance"
    build_lance(raw_dir, record_ids, scp_raw_list, folds, vocabulary, lance_path)

    verify(lance_path, raw_dir, n)

    lance_size_gb = sum(f.stat().st_size for f in lance_path.rglob("*") if f.is_file()) / 1e9
    n_train = int((folds <= 8).sum())
    n_val = int((folds == 9).sum())
    n_test = int((folds == 10).sum())
    print(f"Lance dataset size on disk: {lance_size_gb:.2f} GB")
    print(f"Splits (strat_fold<=8/==9/==10): train={n_train:,}  val={n_val:,}  test={n_test:,}")
    print(f"Done. {n:,} rows, {len(vocabulary)} SCP classes.")


if __name__ == "__main__":
    main()

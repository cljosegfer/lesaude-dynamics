#!/usr/bin/env python3
"""
Convert the raw CSN (Chapman-Shaoxing-Ningbo / PhysioNet "ecg-arrhythmia")
WFDB records into a Lance dataset, matching MIMIC-IV-ECG's normalization
convention so the pretrained encoder sees the same input distribution.

Preprocessing mirrors demo/preprocess_waveforms0.py — the recovered
MIMIC raw-WFDB -> HDF5 stage that scripts/build_lance.py's HDF5 inputs
originally came from (build_lance.py itself takes HDF5 as already given):
  - Read physical-unit signal via wfdb.rdrecord(...).p_signal (mV)
  - Truncate to the first 5000 samples if longer, zero-pad the end if shorter
  - Per-lead z-score normalization (robust_z_score): NaN->0 first, then
    (x - mean) / std per lead, flat leads (std ~ 0) left at 0
  - Cast to float16

Unlike MIMIC, there's no separate raw->HDF5 hop here: CSN is small enough
(45,152 records, 5.1 GB raw) to go directly from WFDB to Lance in one pass.
Also unlike MIMIC, CSN has one ECG per patient (no longitudinal stays), so
there's no pairs.lance / fold-must-follow-patient concern — folds are
assigned per record.

Diagnoses are SNOMED-CT codes (not ICD-10). The vocabulary is built from
whatever codes are actually observed in the data (94 distinct codes as of
the 2026-08-20 download) rather than a curated subset, mirroring how
MIMIC's icd_vocabulary_76.json was built from icd_raw.

Outputs (written to --out-dir, defaulting to the CSN data dir):
  chapman_shaoxing_ningbo.lance   — main dataset
  dx_vocabulary_csn.json          — list[str] of distinct SNOMED-CT Dx codes (index -> code)

Usage:
  python scripts/build_lance_csn.py
  python scripts/build_lance_csn.py --raw-dir /path/to/WFDBRecords/.. --out-dir /path/to/out
"""

import argparse
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
EPSILON = 1e-6
BATCH_SIZE = 2000
N_FOLDS = 20  # train: fold<=17, val: fold==18, test: fold==19 (matches MIMICLanceDataset)

DEFAULT_RAW_DIR = (
    "/snfs2/josefernandes/datasets/lesaude/csn-monolith/raw/"
    "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
)
DEFAULT_OUT_DIR = "/snfs2/josefernandes/datasets/lesaude/csn-monolith"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=0, help="Fold-assignment RNG seed.")
    return p.parse_args()


def robust_z_score(signal: np.ndarray) -> np.ndarray:
    """Per-lead z-score. Matches demo/preprocess_waveforms0.py exactly so CSN
    waveforms share MIMIC's normalization (that script's output is what
    scripts/build_lance.py's source HDF5 actually contains)."""
    signal = np.nan_to_num(signal)
    mean = signal.mean(axis=0)
    std = signal.std(axis=0)
    normalized = np.zeros_like(signal)
    for lead in range(signal.shape[1]):
        if std[lead] > EPSILON:
            normalized[:, lead] = (signal[:, lead] - mean[lead]) / std[lead]
    return normalized


def parse_dx(comments: list[str]) -> list[str]:
    for c in comments:
        if c.startswith("Dx:"):
            return [x.strip() for x in c[len("Dx:") :].split(",") if x.strip()]
    return []


def scan_headers(hea_files: list[Path]):
    """Pass 1 (fast, header-only): collect record ids + Dx codes, skip malformed."""
    record_ids, dx_raw_list = [], []
    skipped = []
    for hea in tqdm(hea_files, desc="Scanning headers"):
        try:
            header = wfdb.rdheader(str(hea.with_suffix("")))
            if header.sig_len != TARGET_LEN:
                pass  # length mismatches are handled (crop/pad) in pass 2, not fatal here
            if set(LEADS) - set(header.sig_name):
                raise ValueError(f"missing leads: {set(LEADS) - set(header.sig_name)}")
            dx_codes = parse_dx(header.comments)
        except Exception as e:
            skipped.append((hea.name, str(e)))
            continue
        record_ids.append(hea.stem)
        dx_raw_list.append(dx_codes)
    return record_ids, dx_raw_list, skipped


def read_waveform(raw_dir: Path, record_id: str, sig_name: list[str] | None = None) -> np.ndarray:
    """Read + crop/pad + normalize a single record's signal. Returns (5000, 12) float16."""
    record = wfdb.rdrecord(str(_record_path(raw_dir, record_id)))
    signal = record.p_signal  # (length, n_leads), physical units (mV)

    if record.sig_name != LEADS:
        order = [record.sig_name.index(lead) for lead in LEADS]
        signal = signal[:, order]

    length = signal.shape[0]
    if length > TARGET_LEN:
        signal = signal[:TARGET_LEN, :]
    elif length < TARGET_LEN:
        signal = np.pad(signal, ((0, TARGET_LEN - length), (0, 0)), "constant")

    return robust_z_score(signal).astype(np.float16)


_RECORD_PATH_CACHE: dict[str, Path] = {}


def _record_path(raw_dir: Path, record_id: str) -> Path:
    if not _RECORD_PATH_CACHE:
        for hea in raw_dir.rglob("*.hea"):
            _RECORD_PATH_CACHE[hea.stem] = hea.with_suffix("")
    return _RECORD_PATH_CACHE[record_id]


def _make_schema(n_classes: int):
    return pa.schema(
        [
            pa.field("record_id", pa.string()),
            pa.field("fold", pa.int8()),
            pa.field("dx_raw", pa.list_(pa.string())),
            pa.field("dx", pa.list_(pa.int8(), n_classes)),
            pa.field("waveform", pa.list_(pa.float16(), TARGET_LEN * CHANNELS)),
        ]
    )


def build_lance(raw_dir: Path, record_ids, dx_raw_list, folds, vocabulary, out_path: Path):
    n_classes = len(vocabulary)
    code_to_idx = {c: i for i, c in enumerate(vocabulary)}
    schema = _make_schema(n_classes)
    n = len(record_ids)
    first_write = True

    for start in tqdm(range(0, n, BATCH_SIZE), desc="Writing Lance dataset"):
        end = min(start + BATCH_SIZE, n)
        batch_ids = record_ids[start:end]
        batch_dx_raw = dx_raw_list[start:end]

        batch_waveforms = np.stack([read_waveform(raw_dir, rid) for rid in batch_ids])  # (B,5000,12)

        batch_dx = np.zeros((end - start, n_classes), dtype=np.int8)
        for i, codes in enumerate(batch_dx_raw):
            for c in codes:
                batch_dx[i, code_to_idx[c]] = 1

        table = pa.table(
            {
                "record_id": pa.array(batch_ids, type=pa.string()),
                "fold": pa.array(folds[start:end], type=pa.int8()),
                "dx_raw": pa.array(batch_dx_raw, type=pa.list_(pa.string())),
                "dx": to_fixed_list_int8(batch_dx, n_classes),
                "waveform": to_fixed_list_float16(
                    batch_waveforms.reshape(end - start, -1), TARGET_LEN * CHANNELS
                ),
            },
            schema=schema,
        )

        mode = "create" if first_write else "append"
        lance.write_dataset(table, str(out_path), mode=mode)
        first_write = False


def verify(out_path: Path, raw_dir: Path, record_ids: list[str], expected_n: int):
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

    hea_files = sorted(raw_dir.rglob("*.hea"))
    print(f"Found {len(hea_files):,} header files under {raw_dir}")

    record_ids, dx_raw_list, skipped = scan_headers(hea_files)
    print(f"Valid records: {len(record_ids):,}  (skipped {len(skipped)})")
    for name, err in skipped:
        print(f"  SKIP {name}: {err}")

    vocabulary = sorted({c for codes in dx_raw_list for c in codes})
    vocab_path = out_dir / "dx_vocabulary_csn.json"
    with open(vocab_path, "w") as f:
        json.dump(vocabulary, f)
    print(f"Vocabulary: {len(vocabulary)} SNOMED-CT Dx codes -> {vocab_path}")

    n = len(record_ids)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    folds = np.empty(n, dtype=np.int8)
    folds[perm] = (np.arange(n) % N_FOLDS).astype(np.int8)

    lance_path = out_dir / "chapman_shaoxing_ningbo.lance"
    build_lance(raw_dir, record_ids, dx_raw_list, folds, vocabulary, lance_path)

    verify(lance_path, raw_dir, record_ids, n)

    lance_size_gb = sum(f.stat().st_size for f in lance_path.rglob("*") if f.is_file()) / 1e9
    n_train = int((folds <= 17).sum())
    n_val = int((folds == 18).sum())
    n_test = int((folds == 19).sum())
    print(f"Lance dataset size on disk: {lance_size_gb:.2f} GB")
    print(f"Splits: train={n_train:,}  val={n_val:,}  test={n_test:,}")
    print(f"Done. {n:,} rows, {len(vocabulary)} Dx classes.")


if __name__ == "__main__":
    main()

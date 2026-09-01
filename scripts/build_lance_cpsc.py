#!/usr/bin/env python3
"""
Convert the raw CPSC2018 (China Physiological Signal Challenge 2018, PhysioNet
"challenge-2020" project's training/cpsc_2018 subset) WFDB records into a
Lance dataset, matching MIMIC-IV-ECG's normalization convention so the
pretrained encoder sees the same input distribution.

Preprocessing mirrors scripts/build_lance_csn.py / demo/preprocess_waveforms0.py:
  - Read physical-unit signal via wfdb.rdrecord(...).p_signal (mV)
  - Per-lead z-score normalization (robust_z_score): NaN->0 first, then
    (x - mean) / std per lead, flat leads (std ~ 0) left at 0
  - Cast to float16

Unlike CSN (records ~uniformly 5000 samples / 10s), a pre-flight header scan
(demo/cpsc_preflight_scan.py) found CPSC2018 records vary 3,000-72,000
samples (6-144s), with 64.7% exceeding 5000 samples and a mean 37% (up to
93%) of signal discarded under a plain first-10s-crop policy. Instead, each
record is split into ceil(n_samples / 5000) non-overlapping 5000-sample
windows (only the final partial window is zero-padded), and each window
becomes its own Lance row carrying the parent record's dx/dx_raw labels.
Windows are z-scored *independently* (not the full record then sliced) so
each row individually matches the statistical scale (per-lead mean 0, std 1)
the pretrained encoder was trained on.

Folds are assigned per RECORD (not per window) via a seeded permutation, then
propagated to every window of that record — this is essential: assigning
folds per-window would leak windows from the same recording across
train/val/test.

Diagnoses are SNOMED-CT codes via the header's Dx comment line (multi-label).
The pre-flight scan found exactly 9 distinct codes -- matching CPSC2018's
well-known official 9-class structure -- built from whatever is actually
observed in the data, not an assumed subset.

Outputs (written to --out-dir, defaulting to the CPSC data dir):
  cpsc2018.lance           — main dataset (one row per window)
  dx_vocabulary_cpsc.json  — list[str] of distinct SNOMED-CT Dx codes (index -> code)

Usage:
  python scripts/build_lance_cpsc.py
  python scripts/build_lance_cpsc.py --raw-dir /path/to/raw --out-dir /path/to/out
"""

import argparse
import json
import math
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
BATCH_SIZE = 2000  # records per write batch (each record contributes 1+ window rows)
N_FOLDS = 20  # train: fold<=17, val: fold==18, test: fold==19 (matches MIMICLanceDataset)

DEFAULT_RAW_DIR = "/snfs2/josefernandes/datasets/lesaude/cpsc-monolith/raw"
DEFAULT_OUT_DIR = "/snfs2/josefernandes/datasets/lesaude/cpsc-monolith"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=0, help="Fold-assignment RNG seed.")
    return p.parse_args()


def robust_z_score(signal: np.ndarray) -> np.ndarray:
    """Per-lead z-score. Matches demo/preprocess_waveforms0.py / build_lance_csn.py
    exactly, applied independently to each 5000-sample window (see module
    docstring) so every row matches the pretrained encoder's expected scale."""
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
    """Pass 1 (fast, header-only): collect record ids + Dx codes + sample
    lengths (for window-count accounting), skip malformed."""
    record_ids, dx_raw_list, sig_lens = [], [], []
    skipped = []
    for hea in tqdm(hea_files, desc="Scanning headers"):
        try:
            header = wfdb.rdheader(str(hea.with_suffix("")))
            if set(LEADS) - set(header.sig_name):
                raise ValueError(f"missing leads: {set(LEADS) - set(header.sig_name)}")
            dx_codes = parse_dx(header.comments)
        except Exception as e:
            skipped.append((hea.name, str(e)))
            continue
        record_ids.append(hea.stem)
        dx_raw_list.append(dx_codes)
        sig_lens.append(header.sig_len)
    return record_ids, dx_raw_list, sig_lens, skipped


def n_windows_for(sig_len: int) -> int:
    return max(1, math.ceil(sig_len / TARGET_LEN))


def read_waveform_windows(raw_dir: Path, record_id: str) -> list[np.ndarray]:
    """Read a record once, split into non-overlapping TARGET_LEN windows
    (zero-padding only the final partial window), z-score each window
    independently. Returns a list of (5000, 12) float16 arrays."""
    record = wfdb.rdrecord(str(_record_path(raw_dir, record_id)))
    signal = record.p_signal  # (length, n_leads), physical units (mV)

    if record.sig_name != LEADS:
        order = [record.sig_name.index(lead) for lead in LEADS]
        signal = signal[:, order]

    length = signal.shape[0]
    windows = []
    for start in range(0, max(length, TARGET_LEN), TARGET_LEN):
        chunk = signal[start : start + TARGET_LEN]
        if chunk.shape[0] < TARGET_LEN:
            chunk = np.pad(chunk, ((0, TARGET_LEN - chunk.shape[0]), (0, 0)), "constant")
        windows.append(robust_z_score(chunk).astype(np.float16))
    return windows


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
            pa.field("window_idx", pa.int16()),
            pa.field("fold", pa.int8()),
            pa.field("dx_raw", pa.list_(pa.string())),
            pa.field("dx", pa.list_(pa.int8(), n_classes)),
            pa.field("waveform", pa.list_(pa.float16(), TARGET_LEN * CHANNELS)),
        ]
    )


def build_lance(raw_dir: Path, record_ids, dx_raw_list, folds, vocabulary, out_path: Path) -> int:
    """Iterates records in batches of BATCH_SIZE; each record's windows are
    all generated from a single wfdb.rdrecord() call and flattened into row
    data before writing. Returns the total number of window-rows written."""
    n_classes = len(vocabulary)
    code_to_idx = {c: i for i, c in enumerate(vocabulary)}
    schema = _make_schema(n_classes)
    n = len(record_ids)
    first_write = True
    total_rows = 0

    for start in tqdm(range(0, n, BATCH_SIZE), desc="Writing Lance dataset"):
        end = min(start + BATCH_SIZE, n)
        batch_ids = record_ids[start:end]
        batch_dx_raw = dx_raw_list[start:end]
        batch_folds = folds[start:end]

        row_ids, row_window_idx, row_folds, row_dx_raw, row_waveforms = [], [], [], [], []
        for rid, dx_raw, fold in zip(batch_ids, batch_dx_raw, batch_folds):
            for w, waveform in enumerate(read_waveform_windows(raw_dir, rid)):
                row_ids.append(rid)
                row_window_idx.append(w)
                row_folds.append(fold)
                row_dx_raw.append(dx_raw)
                row_waveforms.append(waveform)

        b = len(row_ids)
        batch_waveforms = np.stack(row_waveforms)  # (b, 5000, 12)
        batch_dx = np.zeros((b, n_classes), dtype=np.int8)
        for i, codes in enumerate(row_dx_raw):
            for c in codes:
                batch_dx[i, code_to_idx[c]] = 1

        table = pa.table(
            {
                "record_id": pa.array(row_ids, type=pa.string()),
                "window_idx": pa.array(row_window_idx, type=pa.int16()),
                "fold": pa.array(row_folds, type=pa.int8()),
                "dx_raw": pa.array(row_dx_raw, type=pa.list_(pa.string())),
                "dx": to_fixed_list_int8(batch_dx, n_classes),
                "waveform": to_fixed_list_float16(
                    batch_waveforms.reshape(b, -1), TARGET_LEN * CHANNELS
                ),
            },
            schema=schema,
        )

        mode = "create" if first_write else "append"
        lance.write_dataset(table, str(out_path), mode=mode)
        first_write = False
        total_rows += b

    return total_rows


def verify(out_path: Path, raw_dir: Path, expected_n: int):
    print("\nRunning verification checks...")
    ds = lance.dataset(str(out_path))
    assert ds.count_rows() == expected_n, (
        f"Row count mismatch: Lance={ds.count_rows()}, expected={expected_n}"
    )
    print(f"  [OK] Row count: {ds.count_rows():,}")

    rng = np.random.default_rng(42)
    sample_rows = rng.integers(0, expected_n, size=5).tolist()
    lance_rows = ds.take(sample_rows, columns=["record_id", "window_idx", "waveform"]).to_pydict()
    for i, row in enumerate(sample_rows):
        rid = lance_rows["record_id"][i]
        widx = lance_rows["window_idx"][i]
        expected = read_waveform_windows(raw_dir, rid)[widx]
        got = np.array(lance_rows["waveform"][i], dtype=np.float16).reshape(TARGET_LEN, CHANNELS)
        assert np.array_equal(expected, got), f"Waveform mismatch at row {row} ({rid}, window {widx})"
    print("  [OK] Waveform integrity (5 random samples, recomputed from source)")
    print("Verification passed.\n")


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hea_files = sorted(raw_dir.rglob("*.hea"))
    print(f"Found {len(hea_files):,} header files under {raw_dir}")

    record_ids, dx_raw_list, sig_lens, skipped = scan_headers(hea_files)
    print(f"Valid records: {len(record_ids):,}  (skipped {len(skipped)})")
    for name, err in skipped:
        print(f"  SKIP {name}: {err}")

    expected_windows = sum(n_windows_for(s) for s in sig_lens)
    print(f"Expected window rows: {expected_windows:,} "
          f"(mean {expected_windows / len(record_ids):.2f} windows/record)")

    vocabulary = sorted({c for codes in dx_raw_list for c in codes})
    vocab_path = out_dir / "dx_vocabulary_cpsc.json"
    with open(vocab_path, "w") as f:
        json.dump(vocabulary, f)
    print(f"Vocabulary: {len(vocabulary)} SNOMED-CT Dx codes -> {vocab_path}")

    n = len(record_ids)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    folds = np.empty(n, dtype=np.int8)
    folds[perm] = (np.arange(n) % N_FOLDS).astype(np.int8)  # assigned per RECORD, propagated to windows

    lance_path = out_dir / "cpsc2018.lance"
    total_rows = build_lance(raw_dir, record_ids, dx_raw_list, folds, vocabulary, lance_path)

    verify(lance_path, raw_dir, total_rows)

    lance_size_gb = sum(f.stat().st_size for f in lance_path.rglob("*") if f.is_file()) / 1e9

    ds = lance.dataset(str(lance_path))
    fold_col = np.array(ds.to_table(columns=["fold"])["fold"])
    n_train = int((fold_col <= 17).sum())
    n_val = int((fold_col == 18).sum())
    n_test = int((fold_col == 19).sum())
    print(f"Lance dataset size on disk: {lance_size_gb:.2f} GB")
    print(f"Splits (window-rows): train={n_train:,}  val={n_val:,}  test={n_test:,}")
    print(f"Done. {total_rows:,} window-rows from {n:,} records, {len(vocabulary)} Dx classes.")


if __name__ == "__main__":
    main()

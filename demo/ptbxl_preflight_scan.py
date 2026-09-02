#!/usr/bin/env python3
"""
Pre-flight scan for PTB-XL, before writing build_lance_ptbxl.py.

Unlike CSN/CPSC, PTB-XL's labels don't live in the WFDB header comments --
they're in ptbxl_database.csv's scp_codes column (a dict-literal string of
code:likelihood pairs), joined against scp_statements.csv for the code
hierarchy. This scan reads .hea text only (no .dat signal loading) across
the full records500/ corpus to check sample-length uniformity and lead
order against every row of ptbxl_database.csv, cross-references the CSV's
own record count/strat_fold distribution, and tallies the observed SCP code
vocabulary directly from the CSV. Mirrors the equivalent ad-hoc scans done
for CSN/CPSC (demo/csn_experiment_report.md section 3, plan.md step 2).

Usage:
  python demo/ptbxl_preflight_scan.py --raw-dir /snfs2/josefernandes/datasets/lesaude/ptbxl-monolith/raw
"""

import argparse
import ast
import csv
from collections import Counter
from pathlib import Path

import numpy as np

# PTB-XL's headers use "AVR"/"AVL"/"AVF" (uppercase, no lowercase "a"), unlike
# MIMIC/CSN/CPSC's "aVR"/"aVL"/"aVF" -- compare case-insensitively.
CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CANONICAL_LEADS_NORM = [name.upper() for name in CANONICAL_LEADS]
TARGET_LEN = 5000  # 10s @ 500Hz, matching MIMIC/CSN's convention


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw-dir", required=True,
        help="Directory containing (or containing a subdir with) records500/, "
             "ptbxl_database.csv, scp_statements.csv -- the data root printed "
             "by download_ptbxl.py.",
    )
    return p.parse_args()


def find_data_root(raw_dir: Path) -> Path:
    for p in raw_dir.rglob("records500"):
        if p.is_dir():
            return p.parent
    return raw_dir


def parse_header(path: Path):
    lines = path.read_text().splitlines()
    first = lines[0].split()
    record_id, n_sig, fs, n_samples = first[0], int(first[1]), float(first[2]), int(first[3])
    lead_names = [line.split()[-1] for line in lines[1:1 + n_sig]]
    return record_id, n_sig, fs, n_samples, lead_names


def load_database(db_csv: Path) -> list[dict]:
    with open(db_csv, newline="") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    data_root = find_data_root(Path(args.raw_dir))
    db_csv = data_root / "ptbxl_database.csv"
    scp_csv = data_root / "scp_statements.csv"
    print(f"Data root: {data_root}")

    rows = load_database(db_csv)
    print(f"\nptbxl_database.csv: {len(rows):,} rows")
    print(f"columns: {list(rows[0].keys())}")

    if scp_csv.exists():
        n_documented = sum(1 for _ in csv.DictReader(open(scp_csv)))
        print(f"\nscp_statements.csv: {n_documented} documented codes")
    else:
        print("\nscp_statements.csv: NOT FOUND")

    fold_counts = Counter(row["strat_fold"] for row in rows)
    print(f"\nstrat_fold distribution: "
          f"{dict(sorted(fold_counts.items(), key=lambda kv: int(kv[0])))}")

    sample_lengths = []
    fs_values = Counter()
    lead_order_exact = 0
    lead_order_permuted = 0
    lead_order_missing = 0
    skipped = []
    all_scp_codes = set()
    scp_code_counts = Counter()
    malformed_scp = []

    for row in rows:
        ecg_id = row["ecg_id"]
        rel_path = row["filename_hr"].strip()
        hea_path = data_root / f"{rel_path}.hea"

        try:
            _, n_sig, fs, n_samples, lead_names = parse_header(hea_path)
        except Exception as e:
            skipped.append((rel_path, str(e)))
            continue

        sample_lengths.append(n_samples)
        fs_values[fs] += 1

        lead_names_norm = [name.upper() for name in lead_names]
        if lead_names_norm == CANONICAL_LEADS_NORM:
            lead_order_exact += 1
        elif set(lead_names_norm) == set(CANONICAL_LEADS_NORM):
            lead_order_permuted += 1
        else:
            lead_order_missing += 1

        try:
            codes = ast.literal_eval(row["scp_codes"])
        except Exception as e:
            malformed_scp.append((ecg_id, str(e)))
            continue
        for code in codes:
            all_scp_codes.add(code)
            scp_code_counts[code] += 1

    sample_lengths = np.array(sample_lengths)
    n_valid = len(sample_lengths)

    print(f"\nValid headers parsed: {n_valid:,} / {len(rows):,}  (skipped {len(skipped)})")
    for name, err in skipped[:20]:
        print(f"  SKIP {name}: {err}")

    print(f"\nSampling frequency: {dict(fs_values)}")

    print(f"\nSample-length distribution (target = {TARGET_LEN}):")
    print(f"  min={sample_lengths.min()}  max={sample_lengths.max()}  "
          f"mean={sample_lengths.mean():.1f}  median={np.median(sample_lengths):.1f}")
    n_exact = int((sample_lengths == TARGET_LEN).sum())
    n_off = n_valid - n_exact
    print(f"  Records exactly {TARGET_LEN}: {n_exact:,} ({100 * n_exact / n_valid:.1f}%)")
    if n_off:
        off_lengths = sample_lengths[sample_lengths != TARGET_LEN]
        print(f"  Records NOT exactly {TARGET_LEN}: {n_off:,} -- "
              f"distinct off-target lengths: {sorted(set(off_lengths.tolist()))[:20]}")

    print(f"\nLead order: exact canonical match={lead_order_exact:,}, "
          f"same set but reordered={lead_order_permuted:,}, "
          f"missing/extra leads={lead_order_missing:,}")

    print(f"\nDistinct SCP codes observed (scp_codes column): {len(all_scp_codes)}")
    print("Top 20 most frequent codes:")
    for code, count in scp_code_counts.most_common(20):
        print(f"  {code}: {count}")

    if malformed_scp:
        print(f"\nMalformed scp_codes entries: {len(malformed_scp)}")
        for ecg_id, err in malformed_scp[:20]:
            print(f"  ecg_id={ecg_id}: {err}")

    print(f"\nMalformed/missing headers: {len(skipped)}")


if __name__ == "__main__":
    main()

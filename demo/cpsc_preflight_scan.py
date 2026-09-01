#!/usr/bin/env python3
"""
Pre-flight header scan for CPSC2018, before writing build_lance_cpsc.py.

Reads only .hea text (no .mat signal loading) across the full corpus to
check the assumptions the converter will otherwise silently get wrong:
sample-length distribution (CPSC's records vary 6-60s, unlike CSN's near-
uniform 10s — this is the key structural risk area), lead order, Dx code
vocabulary size, and malformed-record count. Mirrors the equivalent ad-hoc
scan done for CSN (see demo/csn_experiment_report.md, section 3).

Usage:
  python demo/cpsc_preflight_scan.py --raw-dir /snfs2/josefernandes/datasets/lesaude/cpsc-monolith/raw
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TARGET_LEN = 5000  # 10s @ 500Hz, matching MIMIC/CSN's crop-to policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    return p.parse_args()


def parse_header(path: Path):
    lines = path.read_text().splitlines()
    first = lines[0].split()
    record_id, n_sig, fs, n_samples = first[0], int(first[1]), float(first[2]), int(first[3])

    lead_names = []
    for line in lines[1:1 + n_sig]:
        lead_names.append(line.split()[-1])

    dx_codes = []
    for line in lines:
        stripped = line.strip()
        # PhysioNet's own doc page shows "#Dx:" (no space) but this mirror's
        # actual files use "# Dx:" (space after #) — match either.
        if stripped.startswith("#") and "Dx:" in stripped:
            after = stripped.split("Dx:", 1)[1]
            dx_codes = [c.strip() for c in after.split(",") if c.strip()]
            break

    return record_id, n_sig, fs, n_samples, lead_names, dx_codes


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    hea_files = sorted(raw_dir.rglob("*.hea"))
    print(f"Found {len(hea_files):,} .hea files under {raw_dir}")

    sample_lengths = []
    fs_values = Counter()
    lead_order_exact = 0
    lead_order_permuted = 0
    lead_order_missing = 0
    all_dx_codes = set()
    dx_code_counts = Counter()
    skipped = []

    for f in hea_files:
        try:
            record_id, n_sig, fs, n_samples, lead_names, dx_codes = parse_header(f)
        except Exception as e:
            skipped.append((f.name, str(e)))
            continue

        sample_lengths.append(n_samples)
        fs_values[fs] += 1

        if lead_names == CANONICAL_LEADS:
            lead_order_exact += 1
        elif set(lead_names) == set(CANONICAL_LEADS):
            lead_order_permuted += 1
        else:
            lead_order_missing += 1

        for c in dx_codes:
            all_dx_codes.add(c)
            dx_code_counts[c] += 1

    sample_lengths = np.array(sample_lengths)
    n_valid = len(sample_lengths)

    print(f"\nValid headers parsed: {n_valid:,}  (skipped {len(skipped)})")
    for name, err in skipped:
        print(f"  SKIP {name}: {err}")

    print(f"\nSampling frequency: {dict(fs_values)}")

    print("\nSample-length distribution (target crop/pad length = 5000):")
    print(f"  min={sample_lengths.min()}  max={sample_lengths.max()}  "
          f"mean={sample_lengths.mean():.1f}  median={np.median(sample_lengths):.1f}")
    for pct in (5, 25, 50, 75, 90, 95, 99):
        print(f"  p{pct}: {np.percentile(sample_lengths, pct):.0f}")
    n_truncated = int((sample_lengths > TARGET_LEN).sum())
    n_padded = int((sample_lengths < TARGET_LEN).sum())
    n_exact = int((sample_lengths == TARGET_LEN).sum())
    print(f"  Records requiring truncation (>{TARGET_LEN}): {n_truncated:,} "
          f"({100 * n_truncated / n_valid:.1f}%)")
    print(f"  Records requiring zero-padding (<{TARGET_LEN}): {n_padded:,} "
          f"({100 * n_padded / n_valid:.1f}%)")
    print(f"  Records exactly {TARGET_LEN}: {n_exact:,} ({100 * n_exact / n_valid:.1f}%)")
    if n_truncated:
        frac_discarded = 1 - (TARGET_LEN / sample_lengths[sample_lengths > TARGET_LEN])
        print(f"  Among truncated records, mean fraction of signal discarded: "
              f"{frac_discarded.mean():.1%} (max {frac_discarded.max():.1%})")

    print(f"\nLead order: exact canonical match={lead_order_exact:,}, "
          f"same set but reordered={lead_order_permuted:,}, "
          f"missing/extra leads={lead_order_missing:,}")

    print(f"\nDistinct SNOMED Dx codes: {len(all_dx_codes)}")
    print("Top 15 most frequent codes:")
    for code, count in dx_code_counts.most_common(15):
        print(f"  {code}: {count}")

    print(f"\nMalformed records: {len(skipped)}")


if __name__ == "__main__":
    main()

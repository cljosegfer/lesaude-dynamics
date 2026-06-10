#!/usr/bin/env python3
"""
Show all ECGs and pairs for one patient to verify ecg_no_within_stay
and pair_type assignment match expectations.

Example expected layout for a patient with 4 stays:
  stay1 {x1, x2, x3}  -> x1 enws=0, x2 enws=1, x3 enws=2 (or any increasing value)
  stay2 {x4, x5}       -> x4 enws=0, x5 enws=1
  stay3 {x6}           -> x6 enws=0
  stay4 {x7, x8}       -> x7 enws=0, x8 enws=1

Expected pairs (only consecutive-in-time):
  x1->x2  within_stay   (same stay, enws increases)
  x2->x3  within_stay
  x3->x4  cross_stay    (enws resets)
  x4->x5  within_stay
  x5->x6  cross_stay
  x6->x7  cross_stay
  x7->x8  within_stay

Note: x1->x4, x4->x6 etc. do NOT exist — only consecutive pairs are built.

Usage:
    python demo/check_pairs.py                        # auto-pick a patient with >=3 stays
    python demo/check_pairs.py --subject 10000032
    python demo/check_pairs.py --min-stays 4 --min-ecgs-per-stay 2
"""

import argparse
import lance
import pandas as pd

LANCE_PATH = "/snfs2/josefernandes/datasets/lesaude/mimic-iv-ecg-monolith/mimic_iv_ecg.lance"
PAIRS_PATH = "/snfs2/josefernandes/datasets/lesaude/mimic-iv-ecg-monolith/pairs.lance"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=None)
    p.add_argument("--min-stays", type=int, default=3,
                   help="Auto-select: minimum number of distinct stays (default 3)")
    p.add_argument("--max-stays", type=int, default=5,
                   help="Auto-select: maximum number of distinct stays (default 5)")
    p.add_argument("--min-ecgs-per-stay", type=int, default=2,
                   help="Auto-select: at least one stay must have this many ECGs (default 2)")
    return p.parse_args()


def load_meta():
    return (
        lance.dataset(LANCE_PATH)
        .to_table(columns=["subject_id", "ecg_no_within_stay", "ecg_time"])
        .to_pandas()
        .rename_axis("lance_idx")
        .reset_index()
    )


def load_pairs():
    return lance.dataset(PAIRS_PATH).to_pandas()[
        ["idx_t", "idx_t1", "subject_id", "pair_type"]
    ]


def infer_stay(group: pd.DataFrame) -> list[int]:
    """Assign stay index (0, 1, 2, …) each time ecg_no_within_stay resets to 0."""
    stay_ids, stay = [], -1
    for enws in group["ecg_no_within_stay"]:
        if enws == 0:
            stay += 1
        stay_ids.append(stay)
    return stay_ids


def auto_pick(meta: pd.DataFrame, min_stays: int, max_stays: int, min_ecgs_per_stay: int) -> int:
    candidates = []
    for subj, grp in meta.groupby("subject_id"):
        grp = grp.sort_values("ecg_time")
        stays = pd.Series(infer_stay(grp), index=grp.index)
        n_stays = stays.nunique()
        max_stay_size = stays.value_counts().max()
        if min_stays <= n_stays <= max_stays and max_stay_size >= min_ecgs_per_stay:
            candidates.append((subj, n_stays, max_stay_size, len(grp)))
    if not candidates:
        raise RuntimeError("No patient meets the criteria — lower --min-stays or --min-ecgs-per-stay")
    candidates.sort(key=lambda x: (x[1], x[3]), reverse=True)
    chosen = candidates[0]
    print(f"Auto-selected subject_id={chosen[0]}  "
          f"(stays={chosen[1]}, max_ecgs_per_stay={chosen[2]}, total_ecgs={chosen[3]})\n")
    return chosen[0]


def main():
    args = parse_args()

    print("Loading data...")
    meta  = load_meta()
    pairs = load_pairs()

    subject_id = args.subject if args.subject is not None else \
        auto_pick(meta, args.min_stays, args.max_stays, args.min_ecgs_per_stay)

    # --- ECGs for this patient, sorted by time ---
    ecgs = meta[meta["subject_id"] == subject_id].sort_values("ecg_time").copy()
    if ecgs.empty:
        raise SystemExit(f"subject_id {subject_id} not found in dataset")

    ecgs = ecgs.reset_index(drop=True)
    ecgs["stay"]  = infer_stay(ecgs)
    ecgs["label"] = ["x" + str(i + 1) for i in ecgs.index]

    print(f"=== ECGs for subject {subject_id} ===")
    print(ecgs[["label", "lance_idx", "stay", "ecg_no_within_stay", "ecg_time"]].to_string(index=False))
    print()

    # --- Pairs for this patient ---
    subj_pairs = pairs[pairs["subject_id"] == subject_id].copy()
    if subj_pairs.empty:
        print("No pairs found for this patient.")
        return

    label_map = dict(zip(ecgs["lance_idx"], ecgs["label"]))
    stay_map  = dict(zip(ecgs["lance_idx"], ecgs["stay"]))

    subj_pairs = subj_pairs.copy()
    subj_pairs["ecg_t"]   = subj_pairs["idx_t"].map(label_map)
    subj_pairs["ecg_t1"]  = subj_pairs["idx_t1"].map(label_map)
    subj_pairs["stay_t"]  = subj_pairs["idx_t"].map(stay_map)
    subj_pairs["stay_t1"] = subj_pairs["idx_t1"].map(stay_map)
    subj_pairs["pair"]    = subj_pairs["ecg_t"] + " -> " + subj_pairs["ecg_t1"]

    # same_stay_check: do both ECGs belong to the same inferred stay?
    subj_pairs["same_stay"] = subj_pairs["stay_t"] == subj_pairs["stay_t1"]

    # mismatch: pair_type disagrees with inferred stay membership
    subj_pairs["mismatch"] = (
        ((subj_pairs["pair_type"] == "within_stay") & ~subj_pairs["same_stay"]) |
        ((subj_pairs["pair_type"] == "cross_stay")  &  subj_pairs["same_stay"])
    )

    print(f"=== Pairs for subject {subject_id} ===")
    cols = ["pair", "stay_t", "stay_t1", "pair_type", "same_stay", "mismatch"]
    print(subj_pairs[cols].to_string(index=False))
    print()

    # --- summary ---
    n_bad = subj_pairs["mismatch"].sum()
    if n_bad:
        print(f"  MISMATCHES: {n_bad} pair(s) where pair_type disagrees with inferred stay.")
        print(subj_pairs[subj_pairs["mismatch"]][cols].to_string(index=False))
    else:
        print("  [OK] All pair_type labels agree with inferred stay membership.")

    # Skip-stay check: cross_stay pairs that jump more than one stay
    cross = subj_pairs[subj_pairs["pair_type"] == "cross_stay"]
    skip  = cross[abs(cross["stay_t1"] - cross["stay_t"]) > 1]
    if not skip.empty:
        print(f"\n  Note: {len(skip)} cross_stay pair(s) skip a stay (stay gap > 1):")
        print(skip[cols].to_string(index=False))
    else:
        print("  [OK] All cross_stay pairs are between adjacent stays (no skips).")


if __name__ == "__main__":
    main()

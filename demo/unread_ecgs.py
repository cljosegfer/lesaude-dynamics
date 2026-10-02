#!/usr/bin/env python3
"""
How many training ECGs does dynamics pretraining never read at all?

MIMICLanceDataset(mode="pair") only ever samples rows referenced as idx_t or
idx_t1 in the pairs file -- unlike mode="monitoring", which reads every row
in the split. Any train ECG that never appears in either pair-index column
is invisible to dynamics pretraining, no matter how many epochs it runs.

Checks this for the CURRENT default dynamics config (configs/pretrain.yaml):
  pairs_path = pairs_multistep_path (pairs_multistep_k12.lance)
  pair_types = [cross_stay]  (this file only ever has this one pair_type)

Breaks the unread set down by why each row is unread:
  - ecg_no_within_stay < 0  : per the source ECG-MIMIC preprocessing code
                              (AI4HealthUOL/ECG-MIMIC, full_preprocessing.py
                              L146-151), every row starts at -1 and is only
                              overwritten if it falls inside an ED stay or a
                              hospital admission window. So enws<0 means this
                              ECG isn't linked to any stay at all (mostly
                              outpatient ECGs), and therefore has NO ICD code
                              linked to it whatsoever -- not "same label as
                              its stay", but no label at all. Dropped before
                              any pairing in both build_pair_index and
                              build_pairs_multistep; confirmed empirically
                              below via the icd label vector itself.
  - ecg_no_within_stay > 0  : not a stay's first ECG -- cross-stay pairs only
                              ever reference enws==0 rows, so these can never
                              be a pair endpoint under a cross-stay-only scheme
  - ecg_no_within_stay == 0, patient has only one recorded stay: no stay N-1
                              or N+1 exists to pair with, at any hop k. (Any
                              enws==0 row for a patient with >=2 stays is
                              already covered by the k=1 hop alone, so this
                              is the only way an enws==0 row ends up unread.)

Usage:
  python demo/unread_ecgs.py --config configs/data.yaml
  python demo/unread_ecgs.py --pairs-path-key pairs_path --pair-types within_stay cross_stay
"""

import argparse

import lance
import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument(
        "--pairs-path-key",
        default="pairs_multistep_path",
        help="configs/data.yaml key naming the pairs file to check (default: the "
             "multistep file, matching configs/pretrain.yaml's current default).",
    )
    p.add_argument(
        "--pair-types", nargs="+", default=["cross_stay"],
        help="Pair types to include, matching MIMICLanceDataset's pair_types (default: cross_stay only).",
    )
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    ds = lance.dataset(config["lance_path"])
    meta = ds.to_table(columns=["study_id", "subject_id", "ecg_no_within_stay", "fold"]).to_pandas()
    meta["lance_idx"] = meta.index

    # icd, decoded via the raw Arrow buffer (avoids to_pandas() materializing
    # 800k separate small array objects for a FixedSizeList column).
    icd = (
        ds.to_table(columns=["icd"]).column("icd").combine_chunks().flatten()
        .to_numpy(zero_copy_only=False)
        .reshape(len(meta), 76)
    )
    meta["icd_empty"] = icd.sum(axis=1) == 0

    train = meta[meta["fold"] <= 17]
    n_train = len(train)

    pairs_path = config[args.pairs_path_key]
    type_filter = " OR ".join(f"pair_type = '{t}'" for t in args.pair_types)
    pairs_df = (
        lance.dataset(pairs_path)
        .to_table(filter=f"(fold <= 17) AND ({type_filter})")
        .to_pandas()[["idx_t", "idx_t1"]]
    )
    referenced = set(
        np.unique(np.concatenate([pairs_df["idx_t"].values, pairs_df["idx_t1"].values])).tolist()
    )

    unread = train[~train["lance_idx"].isin(referenced)]
    n_anomalous = int((unread["ecg_no_within_stay"] < 0).sum())
    n_not_first = int((unread["ecg_no_within_stay"] > 0).sum())
    n_single_stay = int((unread["ecg_no_within_stay"] == 0).sum())
    assert n_anomalous + n_not_first + n_single_stay == len(unread)

    print(f"Pairs file : {pairs_path}")
    print(f"Pair types : {args.pair_types}")
    print(f"Train exams total : {n_train:,}")
    print(f"  read   : {len(referenced):,}  ({100*len(referenced)/n_train:.1f}%)")
    print(f"  unread : {len(unread):,}  ({100*len(unread)/n_train:.1f}%)")
    print()
    print("Unread breakdown:")
    print(f"  ecg_no_within_stay < 0  (anomalous, dropped pre-pairing)      : {n_anomalous:,}")
    print(f"  ecg_no_within_stay > 0  (not a stay's first ECG)             : {n_not_first:,}")
    print(f"  ecg_no_within_stay == 0, single-stay patient (no partner)    : {n_single_stay:,}")

    # --- subject_id / study_id checks, over the FULL metadata (all folds) ---
    n_all = len(meta)
    print(f"\n--- subject_id / study_id checks (all {n_all:,} rows, all folds) ---")

    n_unique_study = meta["study_id"].nunique()
    print(
        f"study_id: {n_unique_study:,} unique / {n_all:,} rows "
        f"({'all unique' if n_unique_study == n_all else f'{n_all - n_unique_study:,} duplicated'})"
    )

    subj_counts = meta["subject_id"].value_counts()
    singleton_subjects = set(subj_counts[subj_counts == 1].index)
    n_patients = meta["subject_id"].nunique()
    print(f"\nsubject_id: {n_patients:,} unique patients")
    print(
        f"  patients with exactly 1 ECG total: {len(singleton_subjects):,} "
        f"({100*len(singleton_subjects)/n_patients:.1f}% of patients, "
        f"{100*len(singleton_subjects)/n_all:.1f}% of rows)"
    )

    all_anomalous = meta[meta["ecg_no_within_stay"] < 0]
    anomalous_is_singleton = all_anomalous["subject_id"].isin(singleton_subjects)
    print(f"\necg_no_within_stay < 0 (all folds): {len(all_anomalous):,} rows")
    print(
        f"  from singleton-subject_id patients:      "
        f"{int(anomalous_is_singleton.sum()):,} ({100*anomalous_is_singleton.mean():.1f}%)"
    )
    print(
        f"  from patients with other (>=0) rows too: "
        f"{int((~anomalous_is_singleton).sum()):,} ({100*(~anomalous_is_singleton).mean():.1f}%)"
    )

    singleton_rows = meta[meta["subject_id"].isin(singleton_subjects)]
    singleton_anomalous_frac = (singleton_rows["ecg_no_within_stay"] < 0).mean()
    overall_anomalous_frac = (meta["ecg_no_within_stay"] < 0).mean()
    print(
        f"\nOf the {len(singleton_rows):,} singleton-patient rows, "
        f"{100*singleton_anomalous_frac:.1f}% have ecg_no_within_stay < 0 "
        f"(vs. {100*overall_anomalous_frac:.1f}% overall)."
    )

    # --- label-emptiness check: do anomalous (enws < 0) exams carry any ICD code at all? ---
    print(f"\n--- label-emptiness check (all {n_all:,} rows, all folds) ---")

    print("Distinct ecg_no_within_stay values < 0:")
    print(meta.loc[meta["ecg_no_within_stay"] < 0, "ecg_no_within_stay"].value_counts().sort_index().to_string())

    anomalous_mask = meta["ecg_no_within_stay"] < 0
    n_anom_all = int(anomalous_mask.sum())
    n_valid_all = int((~anomalous_mask).sum())
    anom_empty = int(meta.loc[anomalous_mask, "icd_empty"].sum())
    valid_empty = int(meta.loc[~anomalous_mask, "icd_empty"].sum())
    print("\nAll-zero ICD label vector:")
    print(f"  anomalous (enws < 0) rows : {anom_empty:,} / {n_anom_all:,}  ({100*anom_empty/n_anom_all:.2f}% empty)")
    print(f"  valid (enws >= 0) rows    : {valid_empty:,} / {n_valid_all:,}  ({100*valid_empty/n_valid_all:.2f}% empty)")

    n_empty = int(meta["icd_empty"].sum())
    print(f"\nOf all {n_empty:,} empty-label rows in the dataset:")
    print(f"  {anom_empty:,} ({100*anom_empty/n_empty:.1f}%) are enws < 0 (anomalous, no stay linked)")
    print(f"  {n_empty - anom_empty:,} ({100*(n_empty-anom_empty)/n_empty:.1f}%) are enws >= 0 (valid stay, but still no ICD code linked)")


if __name__ == "__main__":
    main()

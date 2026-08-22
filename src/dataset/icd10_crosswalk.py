"""
SNOMED-CT (CSN) -> ICD-10 (MIMIC) label crosswalk.

Supports the "Distribution Shift" experiment (papel/experiments.tex): a
MIMIC-finetuned classifier (76-dim sigmoid output, one per 3-digit ICD-10
code) is applied as-is, with no further finetuning, to CSN waveforms. CSN's
diagnoses are SNOMED-CT codes, not ICD-10, so scoring is restricted to the
MIMIC output columns that have a genuine CSN counterpart, using CSN's own
labels (projected through the crosswalk) as ground truth for those columns.

The mapping (snomed_to_icd10_csn.csv, alongside this file) is a
hand-curated crosswalk, not an automated one: CSN is an arrhythmia/
conduction-focused dataset, so the achievable overlap with MIMIC's full
circulatory label space is concentrated in a handful of conduction/
arrhythmia ICD-10 clusters (I21, I44, I45, I47, I48, I49, I51) — MIMIC
diagnoses covering hypertension, ischemic-disease chronicity, heart
failure, etc. have no CSN counterpart and are correctly excluded. Many CSN
labels are pure ECG-report descriptors (ST/T changes, axis deviation,
voltage criteria, interval measurements) with no ICD-10 Chapter IX disease
code at all (they'd fall under R94.31 "abnormal ECG", not a circulatory
diagnosis) — those rows have an empty mimic_icd10 and are never matched.

Each row also carries a confidence tag:
  high   — standard, unambiguous ICD-10-CM convention (e.g. AFib -> I48)
  medium — plausible but somewhat debatable (e.g. generic "myocardial
           infarction" -> I21 acute vs. I25 old, which isn't in MIMIC's 76)
  low    — genuinely uncertain catch-all placements (e.g. several rare
           ectopic-rhythm labels dumped into I49 "other arrhythmias")
  none   — no ICD-10 Chapter IX counterpart at all (excluded)

Multiple SNOMED codes can map to the same 3-digit ICD-10 cluster (e.g. both
"atrial fibrillation" and "atrial flutter" -> I48); build_intersection()
groups by ICD-10 code and project_csn_labels() OR-reduces CSN's multi-hot
labels across each group, so the MIMIC classifier's I48 column is scored
against "is this ECG AFib or flutter" rather than needing an exact
one-to-one code match.
"""

import csv
from pathlib import Path

import numpy as np

_CROSSWALK_PATH = Path(__file__).parent / "snomed_to_icd10_csn.csv"
_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1, "none": 0}


def load_crosswalk(path: Path | str = _CROSSWALK_PATH) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def build_intersection(
    mimic_vocab: list[str],
    csn_vocab: list[str],
    min_confidence: str = "medium",
    crosswalk_path: Path | str = _CROSSWALK_PATH,
) -> dict:
    """
    Returns:
      {
        "icd10_codes":   list[str]        — matched 3-digit ICD-10 codes, sorted
        "mimic_indices": np.ndarray (K,)  — column to select from the MIMIC
                                             classifier's 76-dim output for
                                             icd10_codes[i]
        "csn_groups":    list[np.ndarray] — for icd10_codes[i], the csn_vocab
                                             column indices to OR together to
                                             get its CSN-derived ground truth
      }
    K = number of matched ICD-10 clusters at or above min_confidence.
    """
    if min_confidence not in _CONFIDENCE_RANK:
        raise ValueError(f"min_confidence must be one of {list(_CONFIDENCE_RANK)}, got {min_confidence!r}")
    rank_floor = _CONFIDENCE_RANK[min_confidence]
    rows = load_crosswalk(crosswalk_path)

    csn_idx = {c: i for i, c in enumerate(csn_vocab)}
    mimic_idx = {c: i for i, c in enumerate(mimic_vocab)}

    groups: dict[str, list[int]] = {}
    for row in rows:
        code = row["mimic_icd10"]
        if not code or _CONFIDENCE_RANK[row["confidence"]] < rank_floor:
            continue
        if code not in mimic_idx or row["snomed_code"] not in csn_idx:
            continue  # stay defensive if either vocab drifts from the crosswalk
        groups.setdefault(code, []).append(csn_idx[row["snomed_code"]])

    icd10_codes = sorted(groups)
    return {
        "icd10_codes": icd10_codes,
        "mimic_indices": np.array([mimic_idx[c] for c in icd10_codes], dtype=np.int64),
        "csn_groups": [np.array(groups[c], dtype=np.int64) for c in icd10_codes],
    }


def project_csn_labels(labels: np.ndarray, csn_groups: list[np.ndarray]) -> np.ndarray:
    """
    labels: (N, 94) multi-hot CSN dx array.
    Returns (N, K): OR-reduced onto the matched ICD-10 clusters, in the same
    order as build_intersection()'s icd10_codes.
    """
    return np.stack([labels[:, g].max(axis=1) for g in csn_groups], axis=1)

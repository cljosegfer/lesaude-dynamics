"""
Part 1 Pair Assembly
====================
The pairs, labels and median-beat features that every Part 1 analysis reads (gonogo/go_no_go_plan.md, 2.1).

Pairs: the cross_stay pairs of pairs.lance only (first ECG of stay N -> first ECG of stay N+1 of the same patient,
both inside a stay, i.e. ecg_no_within_stay == 0 on both sides). Never all consecutive ECGs of a patient: ECGs outside
a stay (ecg_no_within_stay == -1) have no codes, stored as zeros, and would create artefact transitions. Within-stay
pairs never change codes (labels are per stay) and are not used here.

Per pair:
  gap_days       (ecg_time[t+1] - ecg_time[t]) / 86400; ecg_time is float64 seconds (MIMIC's shifted dates), never
                 passed to pd.to_datetime, which would read it as nanoseconds
  n_changes      |a_t|_1 over the 76 codes, a_t = y_{t+1} - y_t
  stable         n_changes == 0
  single_code    the changed code's vocabulary index when n_changes == 1, else -1; single_dir +1 onset / -1 resolution
  clean          stable or n_changes == 1 (the plan's clean subset)
  has_b          both ECGs have usable median beats
  fold           the 20 patient-stratified folds of the Lance table (constant per patient)

Median-beat features b: the 39 scalars of anatomie/beat_action_ceiling.py (hr, rr_irr, qrs_width, per-lead qrs, st and
t), from anatomie/median_beats.py's cache (node-local, next to the Lance table), clipped to their 0.1-99.9 percentiles
over the ECGs of these pairs (label-free; detection failures would otherwise dominate).

Event counts: onsets and resolutions of each code over all cross-stay pairs and over single-code transitions, with
the plan's threshold (>= 250 onsets in total over the folds) applied to both, in code_counts.csv. The clean count is
the one that applies (decision of 2026-10-22): 14 codes pass, not 47.

Same-episode pairs: cross-stay pairs under 12 h apart are 99.7% stable and those at 12-24 h 97.9% (28-31% from a
week on), as expected if an ED stay and the admission it led to share their codes. Like within-stay pairs, they
cannot show a transition; once the linkage identifies them, they leave the gain analysis and join the noise floor.

Outputs (outputs/gonogo_part1/, shared NFS, so later steps run on any node): pairs.parquet, labels.npz (y_t, y_t1 per
pair), b.npz (Lance rows, features, names, clip limits), code_counts.csv. Rebuilt only with --rebuild.

Example
-------
python gonogo/part1/pairs.py          # on the node that holds the median-beat cache (gorgona10)
"""

import argparse
import json
import sys
from pathlib import Path

import lance
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "anatomie"))
from beat_action_ceiling import SCALARS, features  # noqa: E402
from encoders import data_paths  # noqa: E402
from median_beats import load_cache  # noqa: E402

OUT = REPO / "outputs" / "gonogo_part1"
POSITIVE = ["I44", "I45", "I47", "I48", "I49"]
AMBIGUOUS = ["I10", "I25", "I50", "I63"]
MIN_ONSETS = 250


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--workers", type=int, default=16, help="only used if the beat cache must be built")
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()


def beat_features(lance_path, rows, args):
    """b (len(rows), 39) float32 for the Lance rows `rows`; NaN where an ECG has no usable beats."""
    cache = load_cache(lance_path, num_workers=args.workers)
    b = np.full((len(rows), len(SCALARS)), np.nan, np.float32)
    ok = np.where(cache["n_beats"][rows] > 0)[0]
    for s in range(0, len(ok), args.chunk):
        i = ok[s:s + args.chunk]
        _, f = features(np.asarray(cache["beats"][rows[i]], dtype=np.float32))
        b[i, 0:2] = cache["rhythm"][rows[i]]
        b[i, 2:] = f
    b[~np.isfinite(b).all(1)] = np.nan
    return b


def build(args):
    paths = data_paths(args.config)
    vocab = json.load(open(paths["vocabulary_path"]))
    meta = lance.dataset(paths["lance_path"]).to_table(
        columns=["subject_id", "ecg_time", "ecg_no_within_stay", "fold", "icd"])
    labels = (meta.column("icd").combine_chunks().flatten().to_numpy(zero_copy_only=False)
              .reshape(meta.num_rows, 76).astype(np.int8))
    ecg_time = meta.column("ecg_time").to_numpy().astype(np.float64)        # seconds, not nanoseconds
    enws = meta.column("ecg_no_within_stay").to_numpy()
    subj = meta.column("subject_id").to_numpy()
    fold = meta.column("fold").to_numpy()

    pairs = lance.dataset(paths["pairs_path"]).to_table(
        columns=["idx_t", "idx_t1", "subject_id", "fold", "pair_type"]).to_pandas()
    pairs = pairs[pairs["pair_type"] == "cross_stay"].drop(columns="pair_type").reset_index(drop=True)
    t, t1 = pairs["idx_t"].values, pairs["idx_t1"].values
    assert (enws[t] == 0).all() and (enws[t1] == 0).all(), "cross-stay ECGs must open a stay"
    assert (subj[t] == pairs["subject_id"].values).all() and (subj[t1] == subj[t]).all()
    assert (fold[t] == pairs["fold"].values).all()
    pairs["gap_days"] = (ecg_time[t1] - ecg_time[t]) / 86400.0
    assert (pairs["gap_days"] > 0).all(), "pairs must be ordered in time"

    yt, yt1 = labels[t], labels[t1]
    a = yt1.astype(np.int8) - yt
    n_changes = np.abs(a).sum(1)
    pairs["n_changes"] = n_changes
    pairs["stable"] = n_changes == 0
    single = n_changes == 1
    pairs["single_code"] = np.where(single, np.abs(a).argmax(1), -1)
    pairs["single_dir"] = np.where(single, a[np.arange(len(a)), np.abs(a).argmax(1)], 0)
    pairs["clean"] = pairs["stable"] | single

    rows = np.unique(np.concatenate([t, t1]))
    b = beat_features(paths["lance_path"], rows, args)
    usable = np.isfinite(b).all(1)
    lo, hi = np.percentile(b[usable], [0.1, 99.9], axis=0)
    b = np.clip(b, lo, hi)
    pos = np.full(len(labels), -1)
    pos[rows] = np.arange(len(rows))
    pairs["has_b"] = usable[pos[t]] & usable[pos[t1]]

    on = (a == 1)
    counts = pd.DataFrame({
        "code": vocab,
        "onsets_all": on.sum(0), "onsets_single": (on & single[:, None]).sum(0),
        "resolutions_all": (a == -1).sum(0), "resolutions_single": ((a == -1) & single[:, None]).sum(0),
        "at_risk_stable": ((yt == 0) & pairs["stable"].values[:, None]).sum(0),
    })
    counts["pass_all"] = counts["onsets_all"] >= MIN_ONSETS
    counts["pass_single"] = counts["onsets_single"] >= MIN_ONSETS

    OUT.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(OUT / "pairs.parquet")
    np.savez_compressed(OUT / "labels.npz", y_t=yt, y_t1=yt1, vocab=np.array(vocab))
    np.savez(OUT / "b.npz", rows=rows, b=b, names=np.array(SCALARS), clip_lo=lo, clip_hi=hi)
    counts.to_csv(OUT / "code_counts.csv", index=False)
    return pairs, counts


def load():
    """(pairs DataFrame, y_t, y_t1, vocab list, b-row lookup: Lance row -> row of b, b array, b names)."""
    pairs = pd.read_parquet(OUT / "pairs.parquet")
    lab = np.load(OUT / "labels.npz")
    bz = np.load(OUT / "b.npz")
    lookup = pd.Series(np.arange(len(bz["rows"])), index=bz["rows"])
    return pairs, lab["y_t"], lab["y_t1"], [str(v) for v in lab["vocab"]], lookup, bz["b"], [str(n) for n in bz["names"]]


def main():
    args = parse_args()
    if args.rebuild or not (OUT / "code_counts.csv").exists():
        pairs, counts = build(args)
    else:
        pairs, counts = pd.read_parquet(OUT / "pairs.parquet"), pd.read_csv(OUT / "code_counts.csv")
    n = len(pairs)
    print(f"cross-stay pairs {n:,} ({pairs['subject_id'].nunique():,} patients); stable {pairs['stable'].mean():.1%}; "
          f"changed {(~pairs['stable']).sum():,}, single-code {(pairs['n_changes'] == 1).sum():,}; "
          f"clean {pairs['clean'].sum():,}; both ECGs with usable beats {pairs['has_b'].mean():.2%}", flush=True)
    q = np.percentile(pairs["gap_days"], [1, 10, 50, 90, 99])
    print("gap days, percentiles 1/10/50/90/99: " + " / ".join(f"{v:.2f}" for v in q), flush=True)
    for col in ("all", "single"):
        p = counts[counts[f"pass_{col}"]]
        print(f"codes with >= {MIN_ONSETS} onsets over {col} transitions: {len(p)} "
              f"(median {p[f'onsets_{col}'].median():.0f})", flush=True)
    show = counts.set_index("code").loc[POSITIVE + AMBIGUOUS]
    print(show[["onsets_all", "onsets_single", "resolutions_single", "at_risk_stable"]].to_string(), flush=True)


if __name__ == "__main__":   # load_cache's (re)build uses spawned DataLoader workers
    main()

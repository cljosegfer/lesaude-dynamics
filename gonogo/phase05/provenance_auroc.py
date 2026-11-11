"""
Phase 0.5b, Statistic 1: Provenance AUROC (label-only)
======================================================
How well the transitions of a label set predict the care setting of the later stay (ED-only vs admission), the
label-provenance part of 0.5b (gonogo/go_no_go_plan.md, revision 6). Nothing here reads an ECG or a representation.

Pairs: usable between-episode cross-stay pairs (b and m at both ECGs, not same-episode), all 20 folds.
Target: setting(t+1) == ed_only. Features: onset and resolution indicators (absent at t and present at t+1, and the
reverse) of each code of the set, from the full Ext-ICD labels (codes_full.npz). Model: L2 logistic regression
(C = 1), cross-fitted over the 20 patient folds (fit on 19, predict the held-out fold). CI: patient-level bootstrap
(1,000 resamples) of the out-of-fold AUROC.

Sets (frozen file outputs/phase05/freeze/label_sets.json, SHA-256 checked):
  A           C0 without the validation subset (the A of every Phase 0.5 rerun; primary)
  C0          the full frozen no-route set, validation subset included (secondary)
  anchors11   the 11 anchors of the superseded 25 Oct approval (comparison)

Output: outputs/phase05/3a_provenance_auroc/{results.json, oof.npz, run.log}

Example
-------
python -I gonogo/phase05/provenance_auroc.py
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "outputs" / "gonogo_part1"
FREEZE = ROOT / "outputs" / "phase05" / "freeze"
OUTD = ROOT / "outputs" / "phase05" / "3a_provenance_auroc"
SHA = "c1feffa3508d9070af4994544c08d62efcc78d0a1e48732e1716b3816cf4bce4"
N_BOOT, SEED = 1000, 0


def load():
    raw = (FREEZE / "label_sets.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA, "frozen label_sets.json changed"
    fz = json.loads(raw)
    pairs, z = pd.read_parquet(P1 / "pairs.parquet"), pd.read_parquet(P1 / "z.parquet")
    assert (z["idx_t"].values == pairs["idx_t"].values).all()
    mz = np.load(P1 / "m.npz")
    lk = pd.Series(np.arange(len(mz["rows"])), index=mz["rows"])
    has_m = mz["has_m"][lk.loc[pairs["idx_t"].values].values] & mz["has_m"][lk.loc[pairs["idx_t1"].values].values]
    use = pairs["has_b"].values & has_m & ~z["same_episode"].values
    C = sp.load_npz(P1 / "codes_full.npz").tocsr()
    names = list(np.load(P1 / "codes_full_names.npy").astype(str))
    return fz, pairs[use].reset_index(drop=True), z[use].reset_index(drop=True), C, names


def features(C, names, pairs, codes):
    j = [names.index(c) for c in codes]
    Ct = C[pairs["idx_t"].values][:, j].astype(bool)
    Ct1 = C[pairs["idx_t1"].values][:, j].astype(bool)
    on = (Ct1 > Ct).astype(np.float32)          # absent at t, present at t+1
    res = (Ct > Ct1).astype(np.float32)         # present at t, absent at t+1
    return sp.hstack([on, res]).tocsr()


def crossfit(X, y, fold):
    p = np.full(len(y), np.nan)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        m = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def boot_auroc(y, p, subj, rng):
    """Patient-level bootstrap: resample subjects with replacement, weight pairs by draw counts."""
    u, inv = np.unique(subj, return_inverse=True)
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv]
        out[b] = roc_auc_score(y, p, sample_weight=w)
    return out


def main():
    t0 = time.time()
    fz, pairs, z, C, names = load()
    val = set(fz["validation_c1"])
    sets = {"A": sorted(c for c in fz["C0"] if c not in val), "C0": sorted(fz["C0"]),
            "anchors11": sorted(fz["superseded_2026_10_25"]["anchor"])}
    y = (z["setting_t1"].values == "ed_only").astype(int)
    fold, subj = pairs["fold"].values, pairs["subject_id"].values
    print(f"usable between-episode pairs {len(y):,}; ED-only at t+1 {y.mean():.3%}; patients {len(np.unique(subj)):,}",
          flush=True)
    res, oof = {}, {}
    for name, codes in sets.items():
        X = features(C, names, pairs, codes)
        p = crossfit(X, y, fold)
        auc = roc_auc_score(y, p)
        bs = boot_auroc(y, p, subj, np.random.default_rng(SEED))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        nnz = np.asarray((X > 0).sum(1)).ravel()
        res[name] = {"n_codes": len(codes), "n_features": X.shape[1], "auroc": round(float(auc), 4),
                     "ci95": [round(float(lo), 4), round(float(hi), 4)],
                     "pairs_with_any_transition": round(float((nnz > 0).mean()), 4),
                     "auroc_by_fold_min_max": [round(float(min(roc_auc_score(y[fold == f], p[fold == f])
                                                         for f in np.unique(fold))), 4),
                                               round(float(max(roc_auc_score(y[fold == f], p[fold == f])
                                                         for f in np.unique(fold))), 4)]}
        oof[name] = p
        print(f"{name:10s} codes {len(codes):3d}  AUROC {auc:.4f} [{lo:.4f}, {hi:.4f}]  "
              f"pairs with any transition {res[name]['pairs_with_any_transition']:.3f}  ({time.time() - t0:.0f}s)",
              flush=True)
    rule = {"threshold": 0.9, "A_meets_auroc_0.9": res["A"]["auroc"] >= 0.9,
            "note": "AUROC >= 0.9 is one of the two conditions of 'provenance only' (0.5b); the other needs "
                    "statistic 2 on the chosen representation"}
    out = {"frozen_sha256": SHA, "pairs": int(len(y)), "ed_only_rate": round(float(y.mean()), 4),
           "model": "L2 logistic C=1, cross-fitted over 20 folds", "bootstrap": f"{N_BOOT} patient resamples",
           "results": res, "rule": rule}
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "results.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(OUTD / "oof.npz", y=y, fold=fold, subject_id=subj, **oof)
    print(json.dumps(rule), f"\nsaved {OUTD / 'results.json'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())

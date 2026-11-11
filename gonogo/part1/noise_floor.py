"""
Part 1 Noise Floor and Variance Decomposition
=============================================
Inputs to Part 2's latent generator (gonogo/go_no_go_plan.md, 2.9; Phase 0, Task 5). Nothing here uses a gain.

Noise floor: the mean squared change per dimension, in units of the between-episode change's SD per dimension, for
  same episode   an ED visit and the admission it led to (identical labels by construction; b, m, h)
  within stay    consecutive ECGs of one stay (identical labels; b and m, computed here for the ECGs the Part 1
                 caches lack; h was embedded for cross-stay ECGs only, so within-stay pairs have no h)
  between        cross-stay pairs between episodes, stable (no cardiac code change) and changed
by gap bin. 1.0 is the between-episode average; a floor near 1 means most change happens without any code change.

Variance decomposition of Δh (and Δb, Δm) on between-episode pairs, as cross-fitted R^2 (20 folds, ridge with the
strength chosen on the next fold, pooled over dimensions):
  gap      a piecewise-linear basis of log10 gap (knots at the gap deciles)
  codes    + the 152 onset / resolution indicators of the 76 cardiac codes (beyond the gap)
  state    + the current value of the target (regression to the mean; beyond the gap)
  patient  between-patient share of Δ (one-way random-effects ANOVA over patients with >= 2 pairs, trace over
           dimensions)
and the patient share of h itself (same ANOVA over the cross-stay ECGs of patients with >= 2 of them), which sets
sigma_p of the generator.

Output: outputs/gonogo_part1/noise_floor.json.

Example
-------
python gonogo/part1/noise_floor.py           # on gorgona10 (the median-beat cache is node-local)
"""

import argparse
import json
import sys
from pathlib import Path

import lance
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from linkage import load_z  # noqa: E402
from pairs import OUT, beat_features, load as load_pairs  # noqa: E402
from measurements import CSV, NAMES, derive  # noqa: E402

GAP_BINS = [0, 1, 7, 30, 1e5]
LAMBDAS = [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lance", default="/scratch/josefernandes/data/mimic_iv_ecg.lance")
    p.add_argument("--pairs", default="/scratch/josefernandes/data/pairs.lance")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunk", type=int, default=8192)
    return p.parse_args()


def anova_share(X, groups):
    """Between-group share of the variance of X (n, d), trace over d, from a one-way random-effects ANOVA."""
    _, g = np.unique(groups, return_inverse=True)
    k, n = g.max() + 1, len(g)
    cnt = np.bincount(g, minlength=k).astype(float)
    means = np.stack([np.bincount(g, weights=X[:, j], minlength=k) for j in range(X.shape[1])], 1) / cnt[:, None]
    grand = X.mean(0)
    ss_b = (cnt[:, None] * (means - grand) ** 2).sum()
    ss_w = ((X - means[g]) ** 2).sum()
    ms_b, ms_w = ss_b / (k - 1), ss_w / (n - k)
    n0 = (n - (cnt ** 2).sum() / n) / (k - 1)
    s2_b = max((ms_b - ms_w) / n0, 0.0)
    return s2_b / (s2_b + ms_w)


def gap_basis(log_gap):
    knots = np.quantile(log_gap, np.linspace(0.1, 0.9, 9))
    return np.column_stack([log_gap] + [np.maximum(log_gap - k, 0) for k in knots])


def ridge_r2(F, Y, folds):
    """Cross-fitted R^2 pooled over the columns of Y: fold k is predicted by a ridge fitted on the folds other than
    k and k+1, with the strength chosen on k+1."""
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    F = np.column_stack([F, np.ones(len(F))])
    pred = np.zeros_like(Y)
    for k in range(20):
        tr, va, te = (folds != k) & (folds != (k + 1) % 20), folds == (k + 1) % 20, folds == k
        G, b = F[tr].T @ F[tr], F[tr].T @ Y[tr]
        best = None
        for lam in LAMBDAS:
            W = np.linalg.solve(G + lam * tr.sum() / 1e4 * np.eye(len(G)), b)
            sse = ((F[va] @ W - Y[va]) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, W)
        pred[te] = F[te] @ best[1]
    return 1 - ((pred - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


def main():
    args = parse_args()
    pairs, yt, yt1, vocab, b_lookup, b, b_names = load_pairs()
    z = load_z()
    hz, mz = np.load(OUT / "h_zeroactions.npz"), np.load(OUT / "m.npz")
    feats = {"b": (b, b_lookup), "h": (hz["h"], pd.Series(np.arange(len(hz["rows"])), index=hz["rows"])),
             "m": (mz["m"], pd.Series(np.arange(len(mz["rows"])), index=mz["rows"]))}
    ok_m = mz["has_m"]
    lk = feats["m"][1]
    usable = (pairs["has_b"].values & ok_m[lk.loc[pairs["idx_t"].values].values]
              & ok_m[lk.loc[pairs["idx_t1"].values].values])
    se = z["same_episode"].values
    between, same = usable & ~se, usable & se

    # ---- within-stay pairs: b and m for the ECGs the caches lack
    w = lance.dataset(args.pairs).to_table(columns=["idx_t", "idx_t1", "pair_type"]).to_pandas()
    w = w[w["pair_type"] == "within_stay"].reset_index(drop=True)
    meta = lance.dataset(args.lance).to_table(columns=["ecg_time"])
    t_sec = meta.column("ecg_time").to_numpy().astype(np.float64)
    w["gap_days"] = (t_sec[w["idx_t1"].values] - t_sec[w["idx_t"].values]) / 86400.0
    rows_w = np.unique(np.concatenate([w["idx_t"].values, w["idx_t1"].values]))
    bw = beat_features(args.lance, rows_w, args)
    bw = np.clip(bw, np.load(OUT / "b.npz")["clip_lo"], np.load(OUT / "b.npz")["clip_hi"])
    mm = pd.read_csv(CSV, usecols=["study_id", "rr_interval", "p_onset", "qrs_onset", "qrs_end", "t_end", "p_axis",
                                   "qrs_axis", "t_axis"]).set_index("study_id")
    sid = lance.dataset(args.lance).to_table(columns=["study_id"]).column("study_id").to_numpy()
    mw, has_mw = derive(mm.loc[sid[rows_w]].reset_index(drop=True))
    for k in ("pr", "p_axis_cos", "p_axis_sin"):
        mw[k] = mw[k].fillna(mw[k].median() if k == "pr" else mw[k].mean())
    mw = mw[NAMES].to_numpy(np.float32)
    lw = pd.Series(np.arange(len(rows_w)), index=rows_w)
    it, it1 = lw.loc[w["idx_t"].values].values, lw.loc[w["idx_t1"].values].values
    w_ok = np.isfinite(bw[it]).all(1) & np.isfinite(bw[it1]).all(1) & has_mw[it] & has_mw[it1]
    within = {"b": (bw[it1] - bw[it])[w_ok], "m": (mw[it1] - mw[it])[w_ok]}
    w_gap = w["gap_days"].values[w_ok]

    out = {"noise_floor": {}, "decomposition": {}}
    gap = pairs["gap_days"].values
    stable = pairs["stable"].values
    for name, (F, look) in feats.items():
        d = F[look.loc[pairs["idx_t1"].values].values] - F[look.loc[pairs["idx_t"].values].values]
        sd = d[between].std(0) + 1e-9
        msq = lambda x: float(((x / sd) ** 2).mean())
        floor = {"between, all": msq(d[between]), "between, stable": msq(d[between & stable]),
                 "between, changed": msq(d[between & ~stable])}
        for lo, hi in zip(GAP_BINS[:-1], GAP_BINS[1:]):
            g = (gap >= lo) & (gap < hi)
            floor[f"between, gap {lo}-{hi} d"] = msq(d[between & g])
            if (same & g).sum() > 100:
                floor[f"same episode, gap {lo}-{hi} d"] = msq(d[same & g])
            if name in within:
                gw = (w_gap >= lo) & (w_gap < hi)
                if gw.sum() > 100:
                    floor[f"within stay, gap {lo}-{hi} d"] = msq(within[name][gw])
        floor["same episode, all"] = msq(d[same])
        if name in within:
            floor["within stay, all"] = msq(within[name])
        floor["n"] = {"between": int(between.sum()), "same episode": int(same.sum()),
                      "within stay": int(w_ok.sum()) if name in within else 0}
        out["noise_floor"][name] = floor

        # ---- decomposition on between-episode pairs
        P = np.where(between)[0]
        Y = d[P] / sd
        folds = pairs["fold"].values[P]
        basis = gap_basis(np.log10(gap[P]))
        A = np.concatenate([(yt1[P].astype(int) - yt[P] == 1), (yt1[P].astype(int) - yt[P] == -1)], 1).astype(float)
        cur = F[look.loc[pairs["idx_t"].values[P]].values]
        r2_gap = ridge_r2(basis, Y, folds)
        r2_codes = ridge_r2(np.column_stack([basis, A]), Y, folds)
        r2_state = ridge_r2(np.column_stack([basis, cur]), Y, folds)
        r2_all = ridge_r2(np.column_stack([basis, A, cur]), Y, folds)
        subj = pairs["subject_id"].values[P]
        multi = pd.Series(subj).map(pd.Series(subj).value_counts()).values >= 2
        out["decomposition"][name] = {"r2_gap": r2_gap, "r2_codes_beyond_gap": r2_codes - r2_gap,
                                      "r2_state_beyond_gap": r2_state - r2_gap, "r2_all": r2_all,
                                      "patient_share_of_delta": anova_share(Y[multi], subj[multi]),
                                      "n_pairs": int(len(P)), "n_pairs_multi": int(multi.sum())}
        print(f"Δ{name}: R2 gap {r2_gap:.4f}; + codes {r2_codes - r2_gap:+.4f}; + state {r2_state - r2_gap:+.4f}; "
              f"all {r2_all:.4f}; patient share of Δ {out['decomposition'][name]['patient_share_of_delta']:.3f}",
              flush=True)
        print("  noise floor: " + "; ".join(f"{k} {v:.2f}" for k, v in floor.items() if k != "n"), flush=True)

    # ---- patient share of h itself (cross-stay ECGs of patients with >= 2 of them)
    h, look = feats["h"]
    ecg = pd.DataFrame({"row": np.concatenate([pairs["idx_t"].values, pairs["idx_t1"].values]),
                        "subj": np.concatenate([pairs["subject_id"].values] * 2)}).drop_duplicates("row")
    ecg = ecg[ecg["subj"].map(ecg["subj"].value_counts()) >= 2]
    H = h[look.loc[ecg["row"].values].values]
    H = (H - H.mean(0)) / (H.std(0) + 1e-9)
    out["patient_share_of_h"] = anova_share(H, ecg["subj"].values)
    print(f"patient share of h (ICC, trace): {out['patient_share_of_h']:.3f} over {len(ecg):,} ECGs", flush=True)
    (OUT / "noise_floor.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"saved {OUT / 'noise_floor.json'}")


if __name__ == "__main__":   # load_cache's (re)build uses spawned DataLoader workers
    main()

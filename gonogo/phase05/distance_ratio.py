"""
Phase 0.5a: Changed-to-Stable Distance Ratio on the Diagnosis Items
===================================================================
For h, intra, g and b, the mean squared distance between the two ECGs of a pair, on pairs whose cardiac codes change
over pairs whose codes do not, on the same 1,000 fold-19 items as the diagnosis of g (one cross-stay pair per patient;
outputs/phase05/6_intra_report/shortcut.json, identical items checked there) A ratio above 1 means the representation moves more when the labels change.
Every representation is standardised per dimension over all exported ECGs, so b's features (bpm, ms, mV) weigh
equally. Reported over all items and over between-episode items only (same-episode pairs are stable by construction
and close in time). CIs: bootstrap over items (one item per patient), 1,000 resamples. Items without b at both ECGs
are dropped for every representation, so all four use the same pairs.

Output: outputs/phase05/10_0.5a_b/distance_ratio.{json,md}

Example
-------
python -I gonogo/phase05/distance_ratio.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gonogo" / "part1"))
from linkage import load_z  # noqa: E402
from pairs import OUT as P1, load as load_pairs  # noqa: E402

OUT = ROOT / "outputs" / "phase05" / "10_0.5a_b"
TAGS = {"h": "zeroactions", "intra": "intra_last", "g": "single_last"}


def main():
    items = json.loads((ROOT / "outputs/phase05/6_intra_report/shortcut.json").read_text())["items"]
    ra, rb = np.array(items["rows_a"]), np.array(items["rows_b"])
    pairs, _, _, _, b_lookup, b, _ = load_pairs()
    z = load_z()
    key = pd.Series(np.arange(len(pairs)), index=pd.MultiIndex.from_arrays([pairs["idx_t"], pairs["idx_t1"]]))
    pi = key.loc[list(zip(ra, rb))].values
    stable = pairs["stable"].values[pi]
    same_ep = z["same_episode"].values[pi]
    has_b = pairs["has_b"].values[pi]
    reps = {}
    for name, tag in TAGS.items():
        e = np.load(P1 / f"h_{tag}.npz")
        lk = pd.Series(np.arange(len(e["rows"])), index=e["rows"])
        X = e["h"].astype(np.float64)
        X = X / X.std(0)
        reps[name] = (X[lk.loc[ra].values], X[lk.loc[rb].values])
    sd = b.std(0)
    keep = has_b
    Bs = b / np.where(sd > 0, sd, 1)
    ia = np.full(len(ra), -1); ib = np.full(len(rb), -1)
    ia[keep] = b_lookup.loc[ra[keep]].values; ib[keep] = b_lookup.loc[rb[keep]].values
    reps["b"] = (np.where(keep[:, None], Bs[ia], np.nan), np.where(keep[:, None], Bs[ib], np.nan))
    rng = np.random.default_rng(0)
    res, lines = {}, []
    for scope, m in (("all items", keep), ("between-episode items", keep & ~same_ep)):
        n_st, n_ch = int((m & stable).sum()), int((m & ~stable).sum())
        res[scope] = {"n_stable": n_st, "n_changed": n_ch}
        idx = np.where(m)[0]
        boots = rng.integers(0, len(idx), size=(1000, len(idx)))
        for name, (xa, xb) in reps.items():
            d = ((xb - xa) ** 2).sum(1)[idx]
            st = stable[idx]
            ratio = d[~st].mean() / d[st].mean()
            bs = [d[bi][~st[bi]].mean() / d[bi][st[bi]].mean() for bi in boots]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            med = np.median(d[~st]) / np.median(d[st])            # robust to the heavy tails of b's features
            bm = [np.median(d[bi][~st[bi]]) / np.median(d[bi][st[bi]]) for bi in boots]
            mlo, mhi = np.percentile(bm, [2.5, 97.5])
            res[scope][name] = {"ratio_changed_over_stable": round(float(ratio), 3), "ci95": [round(lo, 3), round(hi, 3)],
                                "median_ratio": round(float(med), 3), "median_ratio_ci95": [round(mlo, 3), round(mhi, 3)],
                                "mean_sq_dist_stable": round(float(d[st].mean()), 2),
                                "mean_sq_dist_changed": round(float(d[~st].mean()), 2)}
    for scope, r in res.items():
        lines.append(f"\n**{scope}** (stable {r['n_stable']}, changed {r['n_changed']})\n")
        lines.append("| Representation | Changed / stable, means [95% CI] | Changed / stable, medians [95% CI] "
                     "| Mean squared distance, stable / changed |\n|---|---|---|---|")
        for name in ("h", "intra", "g", "b"):
            e = r[name]
            lines.append(f"| {name} | {e['ratio_changed_over_stable']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}] "
                         f"| {e['median_ratio']:.3f} [{e['median_ratio_ci95'][0]:.3f}, {e['median_ratio_ci95'][1]:.3f}] "
                         f"| {e['mean_sq_dist_stable']:.1f} / {e['mean_sq_dist_changed']:.1f} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "distance_ratio.json").write_text(json.dumps(res, indent=1))
    (OUT / "distance_ratio.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

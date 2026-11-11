"""
Phase 0.5 Step 3d on b: Task 1-2 Reruns and the 0.5b Statistic 2
=================================================================
Tables of the short Task 1 and Task 2 reruns on the chosen representation b (go_no_go_plan.md, 0.5a "Rerun", 0.5b),
from gonogo/phase05/run_b.py's results in outputs/phase05/8_3d_b/gain (clean subset, state [b, m], target Δb, frozen
label sets of 0.5d). Gains are R-learner excess gains over the CRT null mean, x1e5, with paired patient-level bootstrap
CIs (gonogo/part1/ci.py: same resamples for every condition of a code). Stars: CRT after BH within the condition and
family (positive / GO), codes with >= 250 onsets (CRT p < 0.05 below that).

Task 1   E under z, z+R, z+A; the fraction of the post-z gain each set removes, 1 - E[z+S] / E[z]
Task 2   the removal by the setting at t+1, D = E[z - setting] - E[z] (pre-registered primary), and the fraction A
         recovers when the setting is withheld, (E[z - s] - E[z - s + A]) / D; the same for all of z from dt (secondary)
0.5b     D_A = E[dt] - E[dt + A] over all pairs and within pairs of constant setting (both stays admissions; both ED-only
         where >= 20 onsets), from the per-pair gains; within a subset, the full-sample null means are subtracted
         (their difference between dt and dt+A is reported). Rule of 0.5b (revision 6, unchanged), per setting-driven
         code: "beyond provenance" if D_A within constant setting has a CI excluding 0 and is >= 1/4 of D_A over all
         pairs; the ambitious framing needs it for a majority. "Provenance only" needs AUROC >= 0.9, which 0.5b
         statistic 1 already excluded (0.817).

Output: outputs/phase05/8_3d_b/{task1.json, task2.json, da.json, tables.md}

Example
-------
python -I gonogo/phase05/tasks_b.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gonogo" / "part1"))
from ci import N_BOOT, fmt, load, ratio, recovered, resample, significance, star  # noqa: E402
from gain import group_of  # noqa: E402
from pairs import OUT as P1  # noqa: E402

D = ROOT / "outputs" / "phase05" / "8_3d_b"
T1 = ["z", "z+R", "z+A"]
T2 = ["dt", "dt+A", "z-setting_t1", "z-setting_t1+A", "z"]
PHASE0_SETTING_CODES = ["K21", "K59", "I50", "I95", "K22"]
AUROC_A = 0.817


def ci_excl0(draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return bool(lo > 0 or hi < 0)


def main():
    runs = load(D / "gain", subset="clean", state="bm")
    sig = significance(runs, sorted({k for e in runs.values() for k in e}))
    L, t1, t2 = [], {}, {}
    L += ["## Task 1 on b (Δb, clean, state [b, m])", "",
          "| Code | Group | Onsets | E after z | E z+R | E z+A | Removed by R | Removed by A |", "|---|---|---|---|---|---|---|---|"]
    for (t, c), e in sorted(runs.items(), key=lambda kv: (group_of(kv[0][1]) == "positive", kv[0][1])):
        if not all(k in e for k in T1):
            continue
        boot, point = resample(e, T1)
        rec = {k: {"E": point[k], "ci": np.percentile(boot[k], [2.5, 97.5]).tolist(), "p": sig[(t, c, k)][0],
                   "q": sig[(t, c, k)][1]} for k in T1}
        cells = []
        for k in ("z+R", "z+A"):
            s, rec[f"removed by {k[2:]}"] = ratio(point["z"] - point[k], boot["z"] - boot[k], point["z"], boot["z"])
            cells.append(s.replace(" unstable", " u"))
        t1[c] = rec
        L.append(f"| {c} | {group_of(c)} | {e['z']['onsets']} | " + " | ".join(fmt(point[k], boot[k]) + star(sig, (t, c, k))
                                                                         for k in T1) + " | " + " | ".join(cells) + " |")
    L += ["", "## Task 2 on b (setting-only withdrawal primary)", "",
          "| Code | Removal by setting(t+1) | A recovers (setting withheld) | Removal by all of z | A recovers (all of z) |",
          "|---|---|---|---|---|"]
    setting_driven = []
    for (t, c), e in sorted(runs.items()):
        if not all(k in e for k in T2):
            continue
        boot, point = resample(e, T2)
        rec = {k: {"E": point[k], "ci": np.percentile(boot[k], [2.5, 97.5]).tolist()} for k in T2}
        cells = []
        for name, w in (("setting(t+1)", "z-setting_t1"), ("all of z", "dt")):
            s, rec[f"removal by {name}"] = ratio(point[w] - point["z"], boot[w] - boot["z"], 1.0, np.ones(N_BOOT))
            r2, rec[f"recovered by A | {name} withheld"] = recovered(point, boot, w, "z", f"{w}+A")
            cells += [s, r2.replace(" unstable", " u")]
        if ci_excl0(boot["z-setting_t1"] - boot["z"]):
            setting_driven.append(c)
        t2[c] = rec
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    src = "rerun on b (setting-only removal with a CI excluding 0)"
    if not setting_driven:
        setting_driven, src = [c for c in PHASE0_SETTING_CODES if ("b", c) in runs], "Phase 0 rows (none on b)"
    # 0.5b statistic 2
    z = pd.read_parquet(P1 / "z.parquet", columns=["setting_t", "setting_t1"])
    da, L2 = {}, []
    not_run = [c for c in setting_driven if not all(k in runs.get(("b", c), {}) for k in ("dt", "dt+A"))]
    for c in setting_driven:
        if c in not_run:
            continue
        e = runs[("b", c)]
        rows, subj = e["dt"]["rows"], e["dt"]["subj"]
        assert (e["dt+A"]["rows"] == rows).all()
        d = (e["dt"]["g"].astype(np.float64) - e["dt+A"]["g"].astype(np.float64))
        dmu = e["dt"]["mu"] - e["dt+A"]["mu"]
        pp = np.load(D / "gain" / f"clean_bm_b_dt_{c}.pairs.npz")
        a = pp["a"].astype(bool)
        st, st1 = z["setting_t"].values[rows], z["setting_t1"].values[rows]
        subsets = {"all pairs": np.ones(len(rows), bool), "both admissions": (st != "ed_only") & (st1 != "ed_only"),
                   "both ED-only": (st == "ed_only") & (st1 == "ed_only")}
        _, inv = np.unique(subj, return_inverse=True)
        P = inv.max() + 1
        W = np.random.default_rng(0).multinomial(P, np.full(P, 1 / P), size=N_BOOT).astype(np.float64)
        res = {}
        for name, k in subsets.items():
            n_on = int((a & k).sum())
            if n_on < 20:
                res[name] = {"onsets": n_on, "pairs": int(k.sum()), "note": "fewer than 20 onsets"}
                continue
            s = np.bincount(inv, weights=np.where(k, d, 0.0), minlength=P)
            n = np.bincount(inv, weights=k.astype(np.float64), minlength=P)
            point = (d[k].mean() - dmu) * 1e5
            boot = ((W @ s) / (W @ n) - dmu) * 1e5
            res[name] = {"onsets": n_on, "pairs": int(k.sum()), "D_A": round(float(point), 3),
                         "ci95": [round(float(x), 3) for x in np.percentile(boot, [2.5, 97.5])],
                         "ci_excludes_0": ci_excl0(boot)}
        allp = res["all pairs"]["D_A"]
        verdicts = []
        for name in ("both admissions", "both ED-only"):
            r = res[name]
            if "D_A" in r:
                r["beyond_provenance"] = bool(r["ci_excludes_0"] and r["D_A"] >= 0.25 * allp)
                verdicts.append(r["beyond_provenance"])
        da[c] = {"subsets": res, "null_mean_difference_x1e5": round(dmu * 1e5, 4),
                 "beyond_provenance": bool(any(verdicts))}
        cell = lambda r: (f"{r['D_A']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] ({r['onsets']} onsets)"
                          if "D_A" in r else f"n/a ({r['onsets']} onsets)")
        L2.append(f"| {c} | {cell(res['all pairs'])} | {cell(res['both admissions'])} | {cell(res['both ED-only'])} "
                  f"| {'yes' if da[c]['beyond_provenance'] else 'no'} |")
    n_bp = sum(v["beyond_provenance"] for v in da.values())
    framing = ("not computed yet" if not da else "ambitious (beyond provenance for a majority)" if n_bp > len(da) / 2
               else "mixed: modest framing (provenance only is excluded by AUROC 0.817 < 0.9)")
    L += ["", f"## 0.5b statistic 2 on b: D_A = E[dt] - E[dt + A], x1e5 (setting-driven codes: {src})", "",
          "| Code | All pairs | Both admissions | Both ED-only | Beyond provenance |", "|---|---|---|---|---|"] + L2
    L += ["", f"Beyond provenance for {n_bp} of {len(da)} setting-driven codes; A's AUROC for setting(t+1) {AUROC_A} < 0.9. "
              f"**0.5b framing: {framing}.**" + (f" Not yet run (dt, dt+A missing): {not_run}." if not_run else "")]
    (D / "task1.json").write_text(json.dumps(t1, indent=1, default=float))
    (D / "task2.json").write_text(json.dumps(t2, indent=1, default=float))
    (D / "da.json").write_text(json.dumps({"setting_driven": setting_driven, "source": src, "codes": da,
                                           "n_beyond_provenance": n_bp, "framing": framing}, indent=1))
    (D / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

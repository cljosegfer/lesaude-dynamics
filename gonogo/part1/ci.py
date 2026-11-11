"""
Part 1 Bootstrap Tables
=======================
Patient-level bootstrap 95% CIs for every gain and gain ratio of a results folder of gonogo/part1/gain.py (Phase 0).

For one code and target, all conditions are computed on the same pairs (the analysis set depends on the code and the
subset only), so every bootstrap resample draws patients once (multinomial counts, 1,000 resamples) and reweights the
per-pair R-learner gains of every condition (<result>.pairs.npz) with the same weights. A gain is reported as its
excess over the CRT null mean, E = G_R - mean_null (null fixed at its full-sample value), so that 0 means "as
uninformative as a fake onset drawn from the propensity". Ratios and differences are computed per resample, and
their CIs are percentile intervals. A ratio whose denominator's CI includes 0 is flagged "unstable".

Tables (each also saved as <folder>/<task>_summary.json):
  task1   E under z, z+R, z+A, z+R+A, and the fraction of the post-z gain each set removes, 1 - E[z+S] / E[z]
  task2   E under dt, z without setting(t+1), z without (age, sex), z, each also + anchors (anc, the 11 approved
          anchors) and + A; the removal a variable achieves, D = E[z - v] - E[z], and the fraction of it that a set
          recovers when the variable is withheld, (E[z - v] - E[z - v + S]) / D (likewise for all of z from dt)
  task3   E under z, z+diag and z+anc+diag (gonogo/part1/k59_diag.py's covariates), the fraction of the post-z gain
          each removes, and the propensity's AUROC
  task4   E under z+anc with the logistic and the gradient-boosted propensity, and their AUROC
  drivers age/sex and setting(t+1) shares of what z removes, (E[z - v] - E[z]) / (E[dt] - E[z]) (2.9; drivers.json)

Example
-------
python gonogo/part1/ci.py task1 outputs/gonogo_part1/task1_labelsets
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gain import bh, group_of  # noqa: E402

N_BOOT = 1000


def load(folder, subset="clean", state="bhm"):
    """{(target, code): {cond: entry}} with the per-pair gains of every result in `folder`."""
    runs = {}
    for f in sorted(Path(folder).glob(f"{subset}_{state}_*.json")):
        r = json.loads(f.read_text())
        pp = f.with_name(f.stem + ".pairs.npz")
        if not pp.exists():
            continue
        z = np.load(pp)
        runs.setdefault((r["target"], r["code"]), {})[r["cond"]] = {
            "G": r["rlearner"]["G"], "mu": r["rlearner"]["crt"]["mean"], "p": r["rlearner"]["crt"]["p"],
            "onsets": r["n_onsets"], "auroc": r["propensity_auroc"], "rows": z["rows"], "subj": z["subj"], "g": z["g"]}
    return runs


def resample(entries, conds, seed=0):
    """Bootstrapped excess gains x1e5: {cond: (n_boot,) array}, plus point estimates {cond: float}."""
    rows = entries[conds[0]]["rows"]
    for c in conds[1:]:
        assert (entries[c]["rows"] == rows).all(), "conditions on different pairs"
    _, inv = np.unique(entries[conds[0]]["subj"], return_inverse=True)
    P = inv.max() + 1
    W = np.random.default_rng(seed).multinomial(P, np.full(P, 1 / P), size=N_BOOT).astype(np.float32)
    n = np.bincount(inv, minlength=P).astype(np.float32)
    boot, point = {}, {}
    for c in conds:
        e = entries[c]
        s = np.bincount(inv, weights=e["g"], minlength=P).astype(np.float32)
        boot[c] = ((W @ s) / (W @ n) - e["mu"]) * 1e5
        point[c] = (e["G"] - e["mu"]) * 1e5
    return boot, point


def fmt(point, draws, digits=1):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return f"{point:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def ratio(num_point, num, den_point, den, digits=2):
    lo_d, hi_d = np.percentile(den, [2.5, 97.5])
    flag = " unstable" if lo_d <= 0 <= hi_d else ""
    return fmt(num_point / den_point, num / den, digits) + flag, {
        "point": num_point / den_point, "ci": np.percentile(num / den, [2.5, 97.5]).tolist(), "unstable": bool(flag)}


def significance(runs, conds):
    """p_crt and BH q (within target x condition x family, codes above 250 onsets) per (target, code, cond)."""
    rows = [{"target": t, "code": c, "cond": k, "p": e[k]["p"], "onsets": e[k]["onsets"],
             "fam": "positive" if group_of(c) == "positive" else "go"}
            for (t, c), e in runs.items() for k in conds if k in e]
    df = pd.DataFrame(rows)
    df["q"] = np.nan
    ok = df["onsets"] >= 250
    for _, g in df[ok].groupby(["target", "cond", "fam"]):
        df.loc[g.index, "q"] = bh(g["p"].values)
    return {(r.target, r.code, r.cond): (r.p, r.q) for r in df.itertuples()}


def star(sig, key):
    p, q = sig[key]
    return "*" if (q < 0.05 if not np.isnan(q) else p < 0.05) else ""


def task1(folder):
    runs, conds = load(folder), ["z", "z+R", "z+A", "z+R+A"]
    sig = significance(runs, conds)
    table, out = [], {}
    for (t, c), e in sorted(runs.items(), key=lambda kv: (kv[0][0], group_of(kv[0][1]) == "positive", kv[0][1])):
        if not all(k in e for k in conds):
            continue
        boot, point = resample(e, conds)
        row = {"target": t, "group": group_of(c), "code": c, "onsets": e["z"]["onsets"]}
        rec = {}
        for k in conds:
            row[f"E {k}"] = fmt(point[k], boot[k]) + star(sig, (t, c, k))
            rec[k] = {"E": point[k], "ci": np.percentile(boot[k], [2.5, 97.5]).tolist(), "p": sig[(t, c, k)][0]}
        for k in conds[1:]:
            s, rec[f"removed by {k[2:]}"] = ratio(point["z"] - point[k], boot["z"] - boot[k], point["z"], boot["z"])
            row[f"removed {k[2:]}"] = s
        table.append(row)
        out[f"{t} {c}"] = rec
    return table, out


def recovered(point, boot, withheld, full, with_set):
    """Fraction of the removal by a variable (withheld -> full) that a set recovers: (E[w] - E[w+S]) / (E[w] - E[f])."""
    return ratio(point[withheld] - point[with_set], boot[withheld] - boot[with_set],
                 point[withheld] - point[full], boot[withheld] - boot[full])


def task2(folder):
    runs = load(folder)
    conds = ["dt", "dt+anc", "dt+A", "z-setting_t1", "z-setting_t1+anc", "z-setting_t1+A", "z-agesex",
             "z-agesex+anc", "z-agesex+A", "z"]
    sig = significance(runs, conds)
    gains, fracs, out = [], [], {}
    for (t, c), e in sorted(runs.items()):
        if not all(k in e for k in conds):
            continue
        boot, point = resample(e, conds)
        g = {"target": t, "group": group_of(c), "code": c, "onsets": e["z"]["onsets"]}
        for k in conds:
            g[k] = fmt(point[k], boot[k]) + star(sig, (t, c, k))
        gains.append(g)
        f = {"target": t, "code": c}
        rec = {k: {"E": point[k], "ci": np.percentile(boot[k], [2.5, 97.5]).tolist()} for k in conds}
        for name, w in (("setting(t+1)", "z-setting_t1"), ("age/sex", "z-agesex"), ("all of z", "dt")):
            s, rec[f"removal by {name}"] = ratio(point[w] - point["z"], boot[w] - boot["z"], 1.0, np.ones(N_BOOT))
            f[f"removal {name}"] = s
            for S in ("anc", "A"):
                s, rec[f"recovered by {S} | {name} withheld"] = recovered(point, boot, w, "z", f"{w}+{S}")
                f[f"{S} recovers {name}"] = s
        fracs.append(f)
        out[f"{t} {c}"] = rec
    return (gains, fracs), out


def task3(folder):
    """E under z, z+diag and z+anc+diag (Task 3's diagnostic covariates), the fraction of the post-z gain each removes,
    and the propensity's AUROC (a jump means the covariates nearly determine the code)."""
    runs, conds = load(folder), ["z", "z+diag", "z+anc+diag"]
    table, out = [], {}
    for (t, c), e in sorted(runs.items()):
        if not all(k in e for k in conds):
            continue
        boot, point = resample(e, conds)
        row = {"target": t, "group": group_of(c), "code": c, "onsets": e["z"]["onsets"]}
        rec = {}
        for k in conds:
            row[f"E {k}"] = fmt(point[k], boot[k]) + ("*" if e[k]["p"] < 0.05 else "")
            row[f"AUROC {k}"] = round(e[k]["auroc"], 3)
            rec[k] = {"E": point[k], "ci": np.percentile(boot[k], [2.5, 97.5]).tolist(), "p": e[k]["p"],
                      "auroc": e[k]["auroc"]}
        for k in conds[1:]:
            s, rec[f"removed by {k[2:]}"] = ratio(point["z"] - point[k], boot["z"] - boot[k], point["z"], boot["z"])
            row[f"removed {k[2:]}"] = s
        table.append(row)
        out[f"{t} {c}"] = rec
    return table, out


def task4(folder_gbm, folder_logit):
    """GBM vs logistic propensity under z+anc for the codes in folder_gbm."""
    gbm, logit = load(folder_gbm), load(folder_logit)
    table, out = [], {}
    for (t, c), e in sorted(gbm.items()):
        for k in e:
            if (t, c) not in logit or k not in logit[(t, c)]:
                continue
            both = {"gbm": e[k], "logistic": logit[(t, c)][k]}
            boot, point = resample(both, ["gbm", "logistic"])
            row = {"target": t, "code": c, "cond": k}
            for name in ("logistic", "gbm"):
                row[f"AUROC {name}"] = round(both[name]["auroc"], 3)
                row[f"E {name}"] = fmt(point[name], boot[name]) + ("*" if both[name]["p"] < 0.05 else "")
                row[f"p_crt {name}"] = round(both[name]["p"], 4)
            table.append(row)
            out[f"{t} {c} {k}"] = {n: {"E": point[n], "ci": np.percentile(boot[n], [2.5, 97.5]).tolist(),
                                       "p": both[n]["p"], "auroc": both[n]["auroc"]} for n in both}
    return table, out


def drivers(folder):
    """Age/sex and setting(t+1) shares of what z removes, with CIs, from a folder with dt, z, z-agesex, z-setting_t1."""
    runs = load(folder)
    conds = ["dt", "z", "z-agesex", "z-setting_t1"]
    table, out = [], {}
    for (t, c), e in sorted(runs.items()):
        if not all(k in e for k in conds):
            continue
        boot, point = resample(e, conds)
        rec = {"E_dt": point["dt"], "E_z": point["z"],
               "z_effect": {"point": point["dt"] - point["z"],
                            "ci": np.percentile(boot["dt"] - boot["z"], [2.5, 97.5]).tolist()}}
        row = {"target": t, "code": c, "z effect": fmt(point["dt"] - point["z"], boot["dt"] - boot["z"])}
        for name, w in (("age/sex", "z-agesex"), ("setting(t+1)", "z-setting_t1")):
            s, rec[f"share {name}"] = ratio(point[w] - point["z"], boot[w] - boot["z"],
                                            point["dt"] - point["z"], boot["dt"] - boot["z"])
            row[f"share {name}"] = s
        lo = np.percentile(boot["dt"] - boot["z"], 2.5)
        shares = {n: rec[f"share {n}"]["point"] for n in ("age/sex", "setting(t+1)")}
        rec["driver"] = max(shares, key=shares.get) if lo > 0 else None
        row["driver"] = rec["driver"] or "(z effect not significant)"
        table.append(row)
        out.setdefault(t, {})[c] = rec
    return table, out


def show(table):
    with pd.option_context("display.width", 320, "display.max_colwidth", 60, "display.max_rows", 500):
        print(pd.DataFrame(table).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["task1", "task2", "task3", "task4", "drivers"])
    p.add_argument("folder")
    p.add_argument("--reference", help="task4: the folder with the logistic-propensity results")
    p.add_argument("--save", help="drivers: where to write the JSON (default: <folder>/drivers_summary.json)")
    args = p.parse_args()
    if args.task == "task1":
        table, out = task1(args.folder)
        show(table)
    elif args.task == "task2":
        (gains, fracs), out = task2(args.folder)
        show(gains)
        print()
        show(fracs)
    elif args.task == "task3":
        table, out = task3(args.folder)
        show(table)
    elif args.task == "task4":
        table, out = task4(args.folder, args.reference)
        show(table)
    else:
        table, out = drivers(args.folder)
        show(table)
    path = Path(args.save) if args.save else Path(args.folder) / f"{args.task}_summary.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"saved {path}")


if __name__ == "__main__":
    main()

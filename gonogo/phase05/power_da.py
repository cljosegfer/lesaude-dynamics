"""
Phase 0.5b, Statistic 2: Power Pre-Check (information only)
===========================================================
How precisely D_A = E[dt] - E[dt + A] (the absolute gain reduction by the no-route set A; go_no_go_plan.md 0.5b,
revision 6) can be estimated within pairs of constant setting, at the real onset counts, before the statistic is run
on the chosen representation. The 0.5b decision rule is not changed by anything here.

Input: the Phase 0 per-pair R-learner gains of Task 2 (outputs/gonogo_part1/task2_metadata/clean_bhm_<t>_{dt,dt+A}_
<code>.pairs.npz; state s_t = [b, h, m] with the current h, clean subset, Phase 0's A of 208 codes). Per pair,
d = g(dt) - g(dt + A); the mean of d over a subset is D_A there, up to the difference of the two CRT null means, a
constant that moves the point estimate but not its spread. Subsets: all pairs of the code's analysis set; both stays
admissions (setting at t and t+1 not ED-only); both ED-only. Uncertainty: patient-level bootstrap (1,000 multinomial
resamples, as gonogo/part1/ci.py), half-width = (97.5th - 2.5th percentile) / 2, in excess-gain units x1e5.

Only spreads and onset counts within the constant-setting subsets are computed, never their point estimates, so the
pre-specified statistic is not looked at early. Attainability uses the Phase 0 all-pairs D_A (exploratory, already in
the Task 2 runs) as the reference effect: if A's reduction persisted in full within constant setting, the power to
see it with a CI excluding 0 is Phi(|D_all| / SE_const - 1.96).

Output: outputs/phase05/4a_power_da/{results.json, table.md}

Example
-------
python -I gonogo/phase05/power_da.py
"""

import json
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "outputs" / "gonogo_part1"
T2 = P1 / "task2_metadata"
OUTD = ROOT / "outputs" / "phase05" / "4a_power_da"
ROWS = [("h", "K21"), ("h", "K59"), ("b", "I50"), ("b", "I95"), ("b", "K22")]   # setting-driven rows of Table 5
N_BOOT, SEED, MIN_ONSETS = 1000, 0, 20


def load(target, code, cond):
    stem = T2 / f"clean_bhm_{target}_{cond}_{code}"
    meta = json.loads(stem.with_suffix(".json").read_text())
    z = np.load(stem.with_suffix(".pairs.npz"))
    return meta, z


def phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def main():
    z = pd.read_parquet(P1 / "z.parquet", columns=["setting_t", "setting_t1"])
    out, lines = {}, []
    for target, code in ROWS:
        m0, g0 = load(target, code, "dt")
        m1, g1 = load(target, code, "dt+A")
        assert (g0["rows"] == g1["rows"]).all(), "conditions on different pairs"
        rows, subj, a = g0["rows"], g0["subj"], g0["a"].astype(bool)
        d = g0["g"].astype(np.float64) - g1["g"].astype(np.float64)
        mu0, mu1 = m0["rlearner"]["crt"]["mean"], m1["rlearner"]["crt"]["mean"]
        d_all_point = ((m0["rlearner"]["G"] - mu0) - (m1["rlearner"]["G"] - mu1)) * 1e5   # Phase 0, exploratory
        st, st1 = z["setting_t"].values[rows], z["setting_t1"].values[rows]
        subsets = {"all pairs": np.ones(len(rows), bool),
                   "both admissions": (st != "ed_only") & (st1 != "ed_only"),
                   "both ED-only": (st == "ed_only") & (st1 == "ed_only")}
        _, inv = np.unique(subj, return_inverse=True)
        P = inv.max() + 1
        W = np.random.default_rng(SEED).multinomial(P, np.full(P, 1 / P), size=N_BOOT).astype(np.float64)
        res = {}
        for name, k in subsets.items():
            n_on = int((a & k).sum())
            entry = {"pairs": int(k.sum()), "onsets": n_on}
            if n_on >= MIN_ONSETS:
                s = np.bincount(inv, weights=np.where(k, d, 0.0), minlength=P)
                n = np.bincount(inv, weights=k.astype(np.float64), minlength=P)
                boot = (W @ s) / (W @ n) * 1e5
                lo, hi = np.percentile(boot, [2.5, 97.5])
                entry["half_width"] = round(float((hi - lo) / 2), 3)
                entry["se"] = round(float(boot.std()), 3)
            else:
                entry["half_width"] = None
                entry["note"] = f"fewer than {MIN_ONSETS} onsets: not estimable"
            res[name] = entry
        # The mean per-pair R-learner gain scales with the onset density, about p(1 - p), so A's reduction persisting
        # in full within a subset means D_all scaled by that density ratio there, not D_all itself.
        p_all = a.mean()
        for s in ("both admissions", "both ED-only"):
            e = res[s]
            if e.get("se"):
                p_s = e["onsets"] / e["pairs"]
                e["expected_D_full_persistence"] = round(float(d_all_point * p_s * (1 - p_s) / (p_all * (1 - p_all))), 3)
                e["power_full_persistence"] = round(phi(abs(e["expected_D_full_persistence"]) / e["se"] - 1.96), 3)
                e["power_half_persistence"] = round(phi(abs(e["expected_D_full_persistence"]) / 2 / e["se"] - 1.96), 3)
        power = max([res[s].get("power_full_persistence", 0.0) for s in ("both admissions", "both ED-only")])
        power_half = max([res[s].get("power_half_persistence", 0.0) for s in ("both admissions", "both ED-only")])
        out[f"{code} (d{target})"] = {"target": target, "code": code, "n_pairs": int(len(rows)),
                                     "n_onsets": int(a.sum()), "D_all_phase0_x1e5": round(float(d_all_point), 3),
                                     "subsets": res, "power_if_full_persistence": round(power, 3),
                                     "power_if_half_persistence": round(power_half, 3),
                                     "attainable": bool(power >= 0.5)}
        cell = lambda s: (f"{res[s]['onsets']} / " + (f"±{res[s]['half_width']:.2f}" if res[s]["half_width"]
                                                     is not None else "n/a"))
        pw = lambda s: (f"{res[s]['expected_D_full_persistence']:.2f} / {res[s]['power_full_persistence']:.2f}"
                        if "power_full_persistence" in res[s] else "n/a")
        lines.append(f"| {code} (Δ{target}) | {cell('all pairs')} | {d_all_point:.2f} | {cell('both admissions')} | "
                     f"{pw('both admissions')} | {cell('both ED-only')} | {pw('both ED-only')} |")
        print(lines[-1], flush=True)
    n_att = sum(v["attainable"] for v in out.values())

    def at_least(k, ps):   # P(at least k successes) for independent successes with probabilities ps
        dist = np.zeros(len(ps) + 1)
        dist[0] = 1.0
        for p in ps:
            dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
            dist[0] *= 1 - p
        return float(dist[k:].sum())

    p3_full = at_least(3, [v["power_if_full_persistence"] for v in out.values()])
    p3_half = at_least(3, [v["power_if_half_persistence"] for v in out.values()])
    verdict = (f"'Beyond provenance' needs a CI excluding 0 within constant setting for 3 of 5 codes (a majority). If A's "
               f"reduction persisted in full within constant setting, {n_att} of 5 codes would reach power >= 0.5 "
               f"(chance of at least 3 of 5: {p3_full:.2f}); if half of it persisted, that chance is {p3_half:.2f}. "
               "Power comes from the both-admissions subset; the both-ED-only subsets are too sparse. So the result is "
               + ("attainable" if n_att >= 3 else "not attainable") + " at these counts only if the reduction largely "
               "persists; a partial persistence would most likely be reported as mixed.")
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "results.json").write_text(json.dumps({"rows": out, "codes_attainable": n_att, "verdict": verdict,
                                                   "p_at_least_3_full": round(p3_full, 3),
                                                   "p_at_least_3_half": round(p3_half, 3),
                                                   "units": "excess gain x1e5", "n_boot": N_BOOT}, indent=1))
    head = ("| Code (target) | All pairs: onsets / ±half-width | Phase 0 D_A, all pairs | Both admissions: onsets / "
            "±half-width | Both admissions: expected D_A / power, full persistence | Both ED-only: onsets / ±half-width "
            "| Both ED-only: expected D_A / power |\n|---|---|---|---|---|---|---|")
    (OUTD / "table.md").write_text(head + "\n" + "\n".join(lines) + "\n\n" + verdict + "\n")
    print(verdict)


if __name__ == "__main__":
    main()

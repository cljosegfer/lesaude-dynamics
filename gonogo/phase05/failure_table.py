"""
The Real-Data Failure Table on b (plan revision 10, step 2b; reported only)
===========================================================================
With the same ψ̂² test as Phase 0.5 (Ledoit-Wolf Σ of b's stable-pair changes, R-learner θ, debiased split product,
patient bootstrap), on folds 10-19 (quarters 10-14 vs 15-19, Σ from the stable pairs of folds 10-19), the number of
codes whose ψ̂² lower bound clears 0, per group, under four conditioning sets of gain.py (clean, state [b, m], Δb):
  naive        none: no z and no anchors
  z            the observed encounter covariates
  z - setting  z without the care setting at t+1 (encounter metadata missing)
  z - s + A    z without the setting, plus the frozen no-route set A; also counted against the margin Γ̂² (3e)
Groups: the 11 C1 controls (no route to the ECG; a cleared bound is a false grounding), the 9 ambiguous codes and the
5 positive controls (a cleared bound is a detection). Inputs: outputs/phase05/8_3d_b/gain/clean_bm_b_<cond>_<code>.

Output: outputs/phase05/11_failure_table/{table.json, table.md}

Example
-------
python -I gonogo/phase05/failure_table.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gonogo" / "part1"))
sys.path.insert(0, str(ROOT / "gonogo" / "phase05"))
from gamma_b import AMBIGUOUS, psi2_code  # noqa: E402
from pairs import POSITIVE  # noqa: E402
from psi2 import metrics, stable_changes  # noqa: E402

G = ROOT / "outputs" / "phase05" / "8_3d_b" / "gain"
OUTD = ROOT / "outputs" / "phase05" / "11_failure_table"
CONDS = [("naive", "none"), ("z", "z"), ("z - setting(t+1)", "z-setting_t1"), ("z - setting(t+1) + A", "z-setting_t1+A")]


def main():
    fz = json.loads((ROOT / "outputs/phase05/freeze/label_sets.json").read_text())
    gamma2 = json.loads((ROOT / "outputs/phase05/9_3e_b/gamma.json").read_text())["pooled"]["Gamma2"]
    groups = {"C1 control": list(fz["validation_c1"]), "ambiguous": AMBIGUOUS, "positive": list(POSITIVE)}
    d_hi, _, _ = stable_changes(list(range(10, 20)), [])
    M = {"lw": metrics(d_hi["b"])[0]["lw"]}
    res, missing = {}, []
    for name, cond in CONDS:
        for grp, codes in groups.items():
            for c in codes:
                f = G / f"clean_bm_b_{cond}_{c}.resid.npz"
                if not f.exists():
                    missing.append(f"{cond}/{c}")
                    continue
                r = psi2_code(f, list(range(10, 15)), list(range(15, 20)), M)
                if r is None:
                    missing.append(f"{cond}/{c} (no onset in a quarter)")
                    continue
                res.setdefault(name, {}).setdefault(grp, {})[c] = {"psi2": r["lw"]["psi2"], "ci95": r["lw"]["ci95"],
                                                                   "onsets": r["onsets"]}
    L = [f"ψ̂² lower bound > 0 (and > Γ̂² = {gamma2:.4f} in the last row), folds 10-19, b, clean; count / codes", "",
         "| Condition | C1 controls (false grounding) | Ambiguous | Positive controls (detection) |", "|---|---|---|---|"]
    table = {}
    for name, _ in CONDS:
        row = {}
        for grp in groups:
            r = res.get(name, {}).get(grp, {})
            n0 = sorted(c for c, v in r.items() if v["ci95"][0] > 0)
            row[grp] = {"n": len(r), "clear_0": n0}
            if name.endswith("+ A"):
                row[grp]["clear_Gamma2"] = sorted(c for c, v in r.items() if v["ci95"][0] > gamma2)
        table[name] = row
        cell = lambda g: (f"{len(row[g]['clear_0'])} / {row[g]['n']}" + (f" ({', '.join(row[g]['clear_0'])})" if row[g]['clear_0'] else "")
                          + (f"; > Γ̂²: {len(row[g]['clear_Gamma2'])}" + (f" ({', '.join(row[g]['clear_Gamma2'])})" if row[g]['clear_Gamma2'] else "")
                             if "clear_Gamma2" in row[g] else ""))
        L.append(f"| {name} | {cell('C1 control')} | {cell('ambiguous')} | {cell('positive')} |")
    L += ["", f"Missing: {missing}" if missing else "No missing runs."]
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "table.json").write_text(json.dumps({"table": table, "psi2": res, "Gamma2": gamma2, "missing": missing}, indent=1))
    (OUTD / "table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

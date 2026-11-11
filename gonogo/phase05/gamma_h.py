"""
Γ̂ by Rotation on h (Plan Revision 13, Step 2)
=============================================
The 3e margin with gamma_b.py's estimator, applied to the reference h (zeroactions export) for the semi-synthetic test
on h. Withheld codes: every rotation code of group g under z + rot<g>, state [b, h, m], target Δh, clean subset, frozen
sets (outputs/phase05/12_h_rotation/gain, run by run_h_gpu.sh). Statistic: debiased ψ̂² (folds 0-9 vs 10-19), Ledoit-Wolf
on the standardised stable-pair changes of h over all folds; v = variance of 1,000 patient-bootstrap draws; pooled
DerSimonian-Laird, Γ̂² = μ̂ + 1.645 τ̂ (or μ̂ + 1.645 SE when τ̂ = 0). Tested codes are not part of this step.

Output: outputs/phase05/12_h_rotation/{gamma_h.json, gamma_h.md}

Example
-------
python -I gonogo/phase05/gamma_h.py             # final, after all 182 codes
python -I gonogo/phase05/gamma_h.py --preview   # code test on the finished codes -> gamma_h_preview.*
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gonogo" / "part1"))
sys.path.insert(0, str(ROOT / "gonogo" / "phase05"))
from gamma_b import dersimonian_laird, psi2_code  # noqa: E402
from psi2 import metrics, stable_changes  # noqa: E402

P05 = ROOT / "outputs" / "phase05"
FROZEN = P05 / "freeze" / "label_sets.json"
OUTD = P05 / "12_h_rotation"


def main():
    preview = "--preview" in sys.argv          # code test on the finished codes only; never read by semisynth.py
    name = "gamma_h_preview" if preview else "gamma_h"
    fz = json.loads(FROZEN.read_text())
    d_all, _, n_all = stable_changes(list(range(20)), ["zeroactions"])
    M, diag = metrics(d_all["h:zeroactions"])
    withheld, dropped = {}, []
    for g, codes in fz["rotation_groups"].items():
        for c in codes:
            f = OUTD / "gain" / f"clean_bhm_h_z+rot{g}_{c}.resid.npz"
            if not f.exists():
                dropped.append((c, "no result"))
                continue
            r = psi2_code(f, list(range(10)), list(range(10, 20)), {"lw": M["lw"]})
            if r is None or not np.isfinite(r["lw"]["psi2"]) or not np.isfinite(r["lw"]["var"]) or r["lw"]["var"] <= 0:
                dropped.append((c, "no onset in a half or degenerate bootstrap"))
                continue
            withheld[c] = {"group": int(g), "profile": fz["C0"][c]["profile"], "onsets": r["onsets"],
                           "psi2": r["lw"]["psi2"], "ci95": r["lw"]["ci95"], "var": r["lw"]["var"]}
    y = np.array([w["psi2"] for w in withheld.values()])
    v = np.array([w["var"] for w in withheld.values()])
    pooled = dersimonian_laird(y, v)
    out = {"metric": "Ledoit-Wolf (primary), h zeroactions", "noise_floor": [n_all, diag], "withheld": withheld,
           "dropped": dropped, "pooled": pooled}
    (OUTD / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
    L = [f"Γ̂ by rotation on h (Ledoit-Wolf; {pooled['N']} withheld codes, {len(dropped)} dropped): μ̂ {pooled['mu']:.4f} "
         f"[SE {pooled['se_mu']:.4f}], τ̂ {pooled['tau']:.4f}, **Γ̂² = {pooled['Gamma2']:.4f}** ({pooled['Gamma2_rule']})."]
    (OUTD / f"{name}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

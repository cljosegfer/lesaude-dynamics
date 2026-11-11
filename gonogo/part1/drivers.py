"""
Part 1 Confounding Drivers
==========================
Which observed context variable accounts for the gain that z removes? (gonogo/go_no_go_plan.md, 2.9.) Reads the
results of gonogo/part1/gain.py with the drop-one conditions (--conds z-gap,z-setting_t,z-setting_t1,z-agesex,
z-coding): z without one group of variables at a time.

Per code and target, with E the excess of the R-learner gain over its CRT null mean (x1e5):
  z effect    E[dt] - E[z], what z removes
  share g     (E[z - g] - E[z]) / (E[dt] - E[z]), the part of that reduction that dropping g alone gives back.
              Shares need not sum to 1 (the groups overlap), and they are unstable when the z effect is small,
              so a driver is reported only where the z effect is at least MIN_EFFECT[target]
  driver      the group with the largest share
Output: outputs/gonogo_part1/drivers.json.

Example
-------
python gonogo/part1/drivers.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pairs import OUT  # noqa: E402

GROUPS = ["gap", "setting_t", "setting_t1", "agesex", "coding"]
MIN_EFFECT = {"h": 0.8, "b": 2.0, "m": 2.0}       # about one null SD of E (0.7 on Δh, 1.2-1.4 on Δb and Δm)


def main():
    rows = []
    for f in (OUT / "gain").glob("clean_bhm_*.json"):
        r = json.loads(f.read_text())
        rows.append({"code": r["code"], "target": r["target"], "Z": r["cond"],
                     "E": (r["rlearner"]["G"] - r["rlearner"]["crt"]["mean"]) * 1e5})
    need = ["dt", "z"] + [f"z-{g}" for g in GROUPS]
    df = pd.DataFrame(rows).pivot_table(index=["target", "code"], columns="Z", values="E").dropna(subset=need)[need]
    out = {}
    for (target, code), e in df.iterrows():
        effect = e["dt"] - e["z"]
        shares = {g: (e[f"z-{g}"] - e["z"]) / effect for g in GROUPS}
        stable = effect >= MIN_EFFECT[target]
        out.setdefault(target, {})[code] = {
            "E_dt": round(e["dt"], 2), "E_z": round(e["z"], 2), "z_effect": round(effect, 2),
            "shares": {g: round(s, 2) for g, s in shares.items()},
            "driver": max(shares, key=shares.get) if stable else None}
        if stable:
            print(f"Δ{target} {code}: z removes {effect:.1f} of {e['dt']:.1f}; driver {out[target][code]['driver']} "
                  + "  ".join(f"{g} {s:+.2f}" for g, s in shares.items()))
    (OUT / "drivers.json").write_text(json.dumps(out, indent=1))
    print(f"saved {OUT / 'drivers.json'}")


if __name__ == "__main__":
    main()

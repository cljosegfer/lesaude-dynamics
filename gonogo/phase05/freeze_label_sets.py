"""
Phase 0.5d: Frozen Anchor Rule
==============================
Freezes the no-route (anchor) set C0, its validation subset, its rotation groups and the route set R before any
Phase 0.5 statistic exists (gonogo/go_no_go_plan.md, 0.5d). Inputs are label statistics only, all computed on
25 Oct before any gain: gonogo/part1/label_sets.py's label_sets.json (frequency and route rule) and
gonogo/part1/controls.py's control_pool.csv (clean onsets, ED-only onsets, separation odds ratio). Nothing here reads
an ECG, a representation or a gain.

Rule
  frequency   non-cardiac 3-character codes with >= 200 transitions (onsets plus resolutions) on the usable
              between-episode pairs of folds <= 17 (label_sets.py's criterion, unchanged)
  route       label_sets.assign: a plausible route to the ECG, cardiac content or a link to grounded events -> R;
              every other frequent code -> candidate no-route code
  separation  Mantel-Haenszel odds ratio of an onset of a grounded cardiac code (I44 I45 I47 I48 I49 I21 I26) on pairs
              with an onset of the candidate, stratified by setting(t+1) x age tertile x sex, over all at-risk
              between-episode pairs; a candidate with OR > 1.5 fails and moves to R (co-onset with grounded events)
  profile     "both" if >= 25 clean onsets go into ED-only stays, else "admission" (roles.py's rule)
  roles       the 11 C1 controls of the 25 Oct approval are the validation subset (never an input); every other
              code of C0 is rotated. The 25 Oct anchor and held-out-anchor roles are superseded
  rotation    five groups, stratified by clean onsets: codes sorted by clean onsets (descending, then code), cut into
              consecutive blocks of five, a seeded permutation (seed 0) of the five group labels within each block

Output: outputs/phase05/freeze/label_sets.json, plus its SHA-256 on stdout (recorded in the plan, 0.5d).

Example
-------
python -I gonogo/phase05/freeze_label_sets.py
"""

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gonogo" / "part1"))
from label_sets import assign  # noqa: E402

P1 = ROOT / "outputs" / "gonogo_part1"
OUT = ROOT / "outputs" / "phase05" / "freeze"
ROLES = ROOT / "gonogo" / "part1" / "control_roles.json"
MAX_OR, MIN_ED_ONSETS, N_GROUPS, SEED = 1.5, 25, 5, 0
GROUNDED = ["I44", "I45", "I47", "I48", "I49", "I21", "I26"]
WHY_FREQUENCY = (
    "Chosen over '>= 250 clean onsets' (which leaves 38 no-route codes, about 26 to rotate). (1) The anchor set works as "
    "many weak proxies jointly; Phase 0 found breadth to matter (the 11 curated anchors recovered nothing on dh where the "
    "208-code set recovered a substantial share), and under the clean-onset criterion each rotation would deconfound "
    "with about 21 codes, close to the failed 11-anchor regime. (2) The context estimator receives onset and resolution "
    "indicators, so transitions in both directions are the relevant count for a proxy; clean onsets are the relevant "
    "count for a tested code. (3) The confirmatory reruns then test the set whose exploratory Phase 0 results motivated "
    "them (Phase 0's A minus screen failures and the validation subset), not a different, smaller set. (4) The low "
    "precision of rare withheld codes is absorbed by the random-effects margin of 0.5c (weights 1/(v_n + tau^2)), which "
    "does not let their estimation noise inflate Gamma-hat. Cost: most rotated codes have < 100 clean onsets, so their "
    "psi^2 are imprecise, and the odds ratio of the separation screen is noisy for rare codes, so some fail by chance "
    "(15 of 208 here, against 1 of the 38 frequent ones); failing only removes candidates.")


def main():
    ls = json.loads((P1 / "label_sets.json").read_text())
    pool = pd.read_csv(P1 / "control_pool.csv").set_index("code")
    roles = json.loads(ROLES.read_text())
    c1 = sorted(roles["roles"]["c1_control"])
    cand = sorted(ls["A"])
    assert all(assign(c)[0] == "A" for c in cand) and set(cand) <= set(pool.index)
    assert set(c1) <= set(cand)

    rec, failed = {}, {}
    for c in cand:
        r = pool.loc[c]
        orv = float(r["co_onset_or"])
        e = {"reason": ls["A"][c], "title": None if pd.isna(r["title"]) else str(r["title"]),
             "transitions_folds_le17": int(ls["transitions"][c]), "onsets_clean": int(r["onsets"]),
             "onsets_clean_ed_only": int(r["onsets_ed_only"]), "onsets_adjusted": int(r["onsets_adjusted"]),
             "co_onset_or": round(orv, 4), "screen": "fail" if orv > MAX_OR else "pass",
             "profile": "both" if r["onsets_ed_only"] >= MIN_ED_ONSETS else "admission"}
        (failed if orv > MAX_OR else rec)[c] = e
    assert not set(c1) & set(failed), "a C1 control failed the screen"

    rot = sorted((c for c in rec if c not in c1), key=lambda c: (-rec[c]["onsets_clean"], c))
    rng = np.random.default_rng(SEED)
    for i in range(0, len(rot), N_GROUPS):
        labels = rng.permutation(N_GROUPS)
        for c, g in zip(rot[i:i + N_GROUPS], labels):
            rec[c]["group"] = int(g)
    for c in rec:
        rec[c]["role"] = "validation_c1" if c in c1 else "rotation"
        rec[c].setdefault("group", None)

    R = dict(ls["R"])
    for c, e in failed.items():
        R[c] = f"co-onset with grounded cardiac codes: separation screen failed (MH OR {e['co_onset_or']:.2f} > {MAX_OR})"
        e["moved_to"] = "R"
    groups = {g: sorted(c for c in rec if rec[c]["group"] == g) for g in range(N_GROUPS)}
    out = {
        "status": "frozen", "frozen": date.today().isoformat(), "script": "gonogo/phase05/freeze_label_sets.py",
        "plan": "gonogo/go_no_go_plan.md revision 6, 0.5d",
        "rule": {
            "frequency": {"criterion": ">= 200 transitions (onsets + resolutions)", "min_transitions": 200,
                          "pairs": "usable between-episode pairs (b and m at both ECGs, not same-episode), folds <= 17",
                          "why": WHY_FREQUENCY},
            "route": "gonogo/part1/label_sets.py assign(): R_REASONS, R_CHAPTERS (C F O V), route exclusions of "
                     "gonogo/part1/roles.py; every other frequent non-cardiac code is a no-route candidate",
            "separation": {"statistic": "Mantel-Haenszel OR of an onset of any grounded cardiac code on pairs with an "
                                        "onset of the candidate vs without; strata setting(t+1) x age tertile x sex; "
                                        "all at-risk between-episode pairs, all folds",
                           "grounded_codes": GROUNDED, "max_or": MAX_OR,
                           "source": "outputs/gonogo_part1/control_pool.csv (gonogo/part1/controls.py, 2026-10-25)",
                           "on_fail": "removed from C0 and added to R"},
            "profile": {"both": f">= {MIN_ED_ONSETS} clean onsets into ED-only stays", "admission": "otherwise"},
            "roles": {"validation_c1": "the 11 C1 controls of the 2026-10-25 approval; never an input to "
                                       "deconfounding; examined in Phase 0, so validation is not blind",
                      "rotation": "every other code of C0"},
            "rotation": {"groups": N_GROUPS, "seed": SEED,
                         "stratified_by": "clean onsets (descending, ties by code), blocks of 5, seeded permutation "
                                          "of the group labels within each block"},
            "supersedes": "anchor and heldout_anchor roles of gonogo/part1/control_roles.json (approved 2026-10-25)"},
        "counts": {"candidates": len(cand), "screen_failed": len(failed), "C0": len(rec),
                   "validation_c1": len(c1), "rotation": len(rot), "R": len(R),
                   "C0_profile": {p: sum(e["profile"] == p for e in rec.values()) for p in ("both", "admission")},
                   "group_sizes": {g: len(v) for g, v in groups.items()}},
        "validation_c1": c1, "rotation_groups": groups, "C0": rec, "screen_failed": failed,
        "R": dict(sorted(R.items())),
        "superseded_2026_10_25": {"anchor": roles["roles"]["anchor"], "heldout_anchor": roles["roles"]["heldout_anchor"]},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "label_sets.json"
    if path.exists():
        raise SystemExit(f"{path} exists: the rule is frozen; do not overwrite it")
    path.write_text(json.dumps(out, indent=1))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "label_sets.sha256").write_text(f"{sha}  label_sets.json\n")
    print(json.dumps(out["counts"], indent=1))
    print("screen failed:", " ".join(f"{c}({e['co_onset_or']:.2f})" for c, e in failed.items()))
    print("groups:", {g: " ".join(v) for g, v in groups.items()})
    print(f"saved {path}\nsha256 {sha}")


if __name__ == "__main__":
    main()

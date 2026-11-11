"""
Part 1 Label Sets R and A
=========================
The code sets of the upper bound for label-based deconfounding (Phase 0, Task 1). Nothing here reads an ECG.

Frequent codes: non-cardiac (outside I00-I99) 3-character codes of the full Ext-ICD labels (gonogo/part1/linkage.py's
codes_full.npz) with >= 200 transitions (onsets plus resolutions) on the usable between-episode pairs of the training
folds (<= 17). Each frequent code goes to exactly one set:
  R   a plausible route to the ECG, cardiac content, or a link to grounded cardiac events (syncope, co-onset): the
      reasons of R_REASONS, plus whole chapters where nearly every code qualifies (R_CHAPTERS). It contains the
      requested E87, E03, E05, N17, N18, A41, R65, J44 and D50-D64, and the 117 codes excluded from the
      negative-control pool (gonogo/part1/roles.py) whenever they are frequent
  A   every other frequent code: no plausible route; includes the 33 codes of the approved negative-control pool
The code under test is always removed from a set before it is used (gonogo/part1/gain.py).

Output: outputs/gonogo_part1/label_sets.json ({"R": {code: reason}, "A": {code: reason}, "transitions": {code: n}}).

Example
-------
python gonogo/part1/label_sets.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from linkage import load_z  # noqa: E402
from pairs import OUT  # noqa: E402
from roles import EXCLUDED, INCLUDED  # noqa: E402

MIN_TRANSITIONS = 200
REQUESTED_R = "E87 E03 E05 N17 N18 A41 R65 J44 D50 D51 D52 D53 D54 D55 D56 D57 D58 D59 D60 D61 D62 D63 D64".split()
R_CHAPTERS = {"C": "malignancy (chemotherapy, radiation, cachexia)",
              "F": "psychiatric or substance use (psychotropic drugs, stimulants, alcohol)",
              "O": "pregnancy (heart rate, axis)",
              "V": "ICD-9 V-code kept by Ext-ICD (cardiac devices, drugs, history; ambiguous with ICD-10)"}
_R_REASONS = {
    "electrolytes, fluids, nutrition or kidney (QT, U and T waves, voltage)":
        "A04 A08 A09 E16 E21 E43 E44 E46 E83 E86 E87 E88 E89 K52 K56 K90 M62 N00 N03 N05 N08 N13 N14 N17 N18 N25 "
        "N28 Q61 R19 R34 R62 R63 R64 R79 R80",
    "endocrine or metabolic with ECG effects, or cardiovascular risk factor":
        "E03 E05 E07 E09 E10 E11 E13 E22 E23 E27 E66 E78 E85 G62 L97 M10 M1A R35 R73 T87 Z68 Z72 Z83 Z89",
    "heart rate through fever, sepsis, anaemia, bleeding or hypoxia":
        "A40 A41 D50 D51 D52 D53 D54 D55 D56 D57 D58 D59 D60 D61 D62 D63 D64 D65 D68 D69 D70 D72 D75 J10 J11 J12 "
        "J15 J18 J69 J80 J81 J86 J93 J95 J96 J98 K92 R04 R06 R09 R23 R50 R57 R65 R68 R71",
    "pulmonary disease or sleep apnea (right heart, hypoxia, AF)": "G47 J43 J44 J45 J47 J84 J90 J91 J94 R91",
    "drugs with ECG effects, chemotherapy or transplant":
        "B20 D35 D46 D47 G30 G31 G40 H40 R78 T38 T40 T42 T43 T44 T45 T46 T50 T81 T86 T88 Z21 Z51 Z79 Z85 Z92 Z94 Z99",
    "cardiac content (symptoms, findings, devices, history, procedures) or syncope link":
        "D86 H35 H81 M94 Q21 Q23 R00 R01 R03 R05 R07 R16 R31 R40 R42 R44 R45 R53 R55 R60 R93 R94 S00 S01 S02 S06 "
        "S09 S27 T82 W01 W06 W10 W18 W19 Y83 Y84 Z45 Z66 Z82 Z86 Z87 Z91 Z95 Z98",
    "brain, stroke, seizure or neuromuscular (brain-heart effects, tremor artefacts)":
        "G06 G20 G25 G45 G72 G81 G82 G83 G90 G91 G92 G93 G95 G97 G99 R13 R20 R25 R26 R27 R29 R41 R47 R48 R56 R90",
    "liver disease (QT, congestion)": "B18 B19 K65 K70 K72 K74 K75 K76 K85 K86 R17 R18 R74",
    "gastroparesis or intestinal ischaemia (diabetes, embolism)": "K31 K55",
    "systemic inflammatory disease (pericarditis, myocarditis, drugs)": "M05 M06 M30 M31 M32 M33 M34 M35",
}
R_REASONS = {c: r for r, codes in _R_REASONS.items() for c in codes.split()}
# Pool exclusions made for ambiguity rather than for a route to the ECG
NOT_A_ROUTE = {"nonspecific or supplementary code", "external cause chapter V-Y (ICD-9 V-code ambiguity)"}


def assign(code):
    """('R', reason) or ('A', reason) for a non-cardiac 3-character code."""
    if code in R_REASONS:
        return "R", R_REASONS[code]
    if code[0] in R_CHAPTERS:
        return "R", R_CHAPTERS[code[0]]
    if code in EXCLUDED and EXCLUDED[code] not in NOT_A_ROUTE:   # pool exclusions for a route stay out of A
        return "R", f"excluded from the negative-control pool: {EXCLUDED[code]}"
    if code in EXCLUDED:
        return "A", f"no plausible route to the ECG ({EXCLUDED[code]})"
    return "A", "no plausible route to the ECG"


def main():
    pairs, z = pd.read_parquet(OUT / "pairs.parquet"), load_z()
    mz = np.load(OUT / "m.npz")
    lk = pd.Series(np.arange(len(mz["rows"])), index=mz["rows"])
    has_m = mz["has_m"][lk.loc[pairs["idx_t"].values].values] & mz["has_m"][lk.loc[pairs["idx_t1"].values].values]
    use = pairs["has_b"].values & has_m & ~z["same_episode"].values & (pairs["fold"].values <= 17)
    C = sp.load_npz(OUT / "codes_full.npz").tocsr()
    names = np.load(OUT / "codes_full_names.npy").astype(str)
    trans = np.asarray((C[pairs["idx_t"].values[use]].astype(bool) != C[pairs["idx_t1"].values[use]].astype(bool))
                       .sum(0)).ravel()
    freq = [(n, int(k)) for n, k in zip(names, trans) if not n.startswith("I") and k >= MIN_TRANSITIONS]
    sets = {"R": {}, "A": {}}
    for code, _ in freq:
        s, reason = assign(code)
        sets[s][code] = reason
    pool = set(INCLUDED) - {"Z66"}               # Z66 was screened out on separation
    assert pool & set(dict(freq)) <= set(sets["A"]), "a pool code fell into R"
    assert {c for c in REQUESTED_R if c in dict(freq)} <= set(sets["R"])
    out = {"min_transitions": MIN_TRANSITIONS, "pairs": "usable between-episode, folds <= 17",
           "R": dict(sorted(sets["R"].items())), "A": dict(sorted(sets["A"].items())),
           "transitions": dict(freq)}
    (OUT / "label_sets.json").write_text(json.dumps(out, indent=1))
    print(f"{len(freq)} frequent non-cardiac codes (>= {MIN_TRANSITIONS} transitions in folds <= 17): "
          f"R {len(sets['R'])}, A {len(sets['A'])}")
    by = pd.Series(sets["R"]).groupby(lambda c: sets["R"][c]).apply(lambda s: " ".join(sorted(s.index)))
    print("R by reason:\n" + "\n".join(f"  {r}: {codes}" for r, codes in by.items()))
    print("A: " + " ".join(sorted(sets["A"])))
    print(f"saved {OUT / 'label_sets.json'}")


if __name__ == "__main__":
    main()

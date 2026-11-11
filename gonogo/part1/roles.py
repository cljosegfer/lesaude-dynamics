"""
Part 1 Negative-Control Roles
=============================
Fixes the negative-control pool and its three disjoint roles before any result on them is seen
(gonogo/go_no_go_plan.md, 2.2; method_section.tex, anchors). Inputs are label statistics only
(gonogo/part1/controls.py's control_pool.csv); nothing here reads an ECG or a gain.

Pool: non-cardiac 3-character codes with >= 250 onsets on the clean between-episode set, minus
  EXCLUDED     a plausible route to the ECG (electrolytes, drugs, devices, heart rate through fever, anemia or
               hypoxia, ...), cardiac content (symptoms that prompt the ECG, cardiac history), or an ambiguous code
               (external causes V-Y: ICD-9 V-codes that are also valid ICD-10 strings survive in Ext-ICD, e.g. V10.3
               "history of breast cancer" read as "pedal cycle rider"; supplementary or nonspecific codes)
  separation   co_onset_or > 1.5 (onsets of grounded cardiac codes on the same pairs, beyond setting, age and sex)
Categories, from the data:
  both         >= 25 onsets into ED-only stays: recorded among ED discharge codes too, so overlap holds across care
               settings (mostly acute complaints, routed to the ECG only through the acute context, e.g. pain)
  admission    < 25: recorded essentially only at admissions (prevalence at ED-only stays <= 0.2%); onsets happen
               only when the later stay is an admission, so these are testable only within such pairs, and they
               are the codes through which the recording route acts (coding intensity)
Roles: within each category, a seeded random permutation (seed 0, codes sorted first) split in thirds, in the order
C1 controls, anchors, held-out anchors. The venous I80/I82/I87 stay the plan's secondary C1 set (cardiac
vocabulary, analysed with the 76 codes).

Output: gonogo/part1/control_roles.json, "status": "approved" (2026-10-25, the proposed split unchanged).

Example
-------
python gonogo/part1/roles.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pairs import OUT  # noqa: E402

ROLES_FILE = Path(__file__).resolve().parent / "control_roles.json"
MIN_ONSETS, MIN_ED_ONSETS, MAX_CO_ONSET_OR, SEED = 250, 25, 1.5, 0
APPROVED = "2026-10-25"   # the user approved the proposed split unchanged; None writes it back as "proposed"
_REASONS = {
    "cardiac symptom or ECG finding (reason the ECG is taken)": "R00 R05 R06 R07 R09 R42 R53 R55 R60 R94",
    "electrolytes, fluids or nutrition (QT, U waves, T waves, voltage)":
        "E43 E46 E83 E86 E87 E88 K52 K56 N17 R19 R79 A04 K85",
    "heart rate through fever, infection severity, anemia, bleeding or hypoxia":
        "A41 R65 R50 D50 D62 D63 D64 K92 J15 J18 J69 J96 J98 R45 F43 D69 D72",
    "endocrine or cardiovascular risk factor (direct effect or separation)": "E03 E11 E13 E66 E78 F17 Z68 M10 G62",
    "pulmonary disease or sleep apnea (right heart, hypoxia, AF)": "J44 J45 G47 J91 R91",
    "drugs with ECG effects (QT, rate) or chemotherapy":
        "Z79 Z92 T45 T40 R78 F11 F14 F12 F19 F10 F31 F32 F41 F60 F03 F05 G40 H40 Z51 Z85 C78 C79 D61 D70 D68",
    "cardiac device, procedure or history": "Z95 T82 T81 Z98 Z86 Z87 Z82 Z99 Z91",
    "brain, stroke or seizure (brain-heart effects, syncope differential)": "G92 G93 R13 R20 R26 R41 R56",
    "liver disease (QT, congestion) or hepatitis": "K74 K76 K86 B18 B19 R18 R74",
    "nonspecific or supplementary code": "B95 B96 R62 R63 Z78 K31 M94 R31",
    "external cause chapter V-Y (ICD-9 V-code ambiguity)": "V15 V17 W18 X58 Y83 Y84 Y92",
}
EXCLUDED = {c: r for r, codes in _REASONS.items() for c in codes.split()}
# Reviewed and kept: no plausible route to the ECG beyond the acute context they mark (pain, mild infection) or the
# coding intensity of the encounter (chronic, non-cardiac). Every code above the threshold must be in one of the two.
INCLUDED = ("R10 R11 M54 N39 M79 R51 M25 J06 L03 K21 K59 K29 G43 G89 M19 M81 M48 M47 N40 K57 K22 K64 K80 K83 Z96 "
            "Z90 Z80 E55 L89 R32 R33 Z23 B37 Z66").split()


def main():
    pool = pd.read_csv(OUT / "control_pool.csv").set_index("code")
    ok = pool[pool["onsets"] >= MIN_ONSETS]
    unreviewed = sorted(set(ok.index) - set(EXCLUDED) - set(INCLUDED))
    assert not unreviewed, f"codes above the threshold with no decision: {unreviewed}"
    assert not set(EXCLUDED) & set(INCLUDED)
    cand = ok.loc[[c for c in ok.index if c in INCLUDED]]
    screened = cand[cand["co_onset_or"] > MAX_CO_ONSET_OR]
    cand = cand[cand["co_onset_or"] <= MAX_CO_ONSET_OR].copy()
    cand["category"] = np.where(cand["onsets_ed_only"] >= MIN_ED_ONSETS, "both", "admission")
    rng = np.random.default_rng(SEED)
    roles = {}
    for cat in ("both", "admission"):
        codes = sorted(cand.index[cand["category"] == cat])
        codes = [codes[i] for i in rng.permutation(len(codes))]
        k = len(codes)
        cut1, cut2 = round(k / 3), round(2 * k / 3)
        for i, c in enumerate(codes):
            roles[c] = "c1_control" if i < cut1 else ("anchor" if i < cut2 else "heldout_anchor")
    cand["role"] = pd.Series(roles)
    out = {"status": "approved" if APPROVED else "proposed", "approved": APPROVED, "seed": SEED,
           "rules": {"min_onsets": MIN_ONSETS, "min_ed_only_onsets": MIN_ED_ONSETS,
                     "max_co_onset_or": MAX_CO_ONSET_OR},
           "roles": {r: sorted(cand.index[cand["role"] == r]) for r in ("c1_control", "anchor", "heldout_anchor")},
           "secondary_c1_venous": ["I80", "I82", "I87"],
           "codes": {c: {"title": str(cand.at[c, "title"]), "category": cand.at[c, "category"],
                         "role": cand.at[c, "role"], "onsets": int(cand.at[c, "onsets"]),
                         "onsets_ed_only": int(cand.at[c, "onsets_ed_only"]),
                         "co_onset_or": round(float(cand.at[c, "co_onset_or"]), 3)} for c in cand.index},
           "excluded": {c: EXCLUDED[c] for c in ok.index if c in EXCLUDED},
           "screened_separation": {c: round(float(screened.at[c, "co_onset_or"]), 3) for c in screened.index}}
    ROLES_FILE.write_text(json.dumps(out, indent=1))
    print(f"{len(ok)} codes >= {MIN_ONSETS} onsets: {len(out['excluded'])} excluded, "
          f"{len(screened)} screened on separation ({', '.join(screened.index)}), {len(cand)} in the pool")
    show = cand.sort_values(["role", "category", "onsets"], ascending=[True, True, False])
    with pd.option_context("display.width", 200, "display.max_colwidth", 48):
        print(show[["role", "category", "onsets", "onsets_ed_only", "prev_ed", "prev_adm", "co_onset_or", "title"]]
              .round(4).to_string())
    print(f"saved {ROLES_FILE}")


if __name__ == "__main__":
    main()

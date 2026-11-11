"""
Part 1 Negative-Control Pool
============================
Label-only statistics of every non-cardiac 3-character code of the full Ext-ICD labels (gonogo/part1/linkage.py's
codes_full.npz), to fix the negative-control pool and its three disjoint roles (C1 controls, anchors, held-out
anchors; gonogo/go_no_go_plan.md, 2.2) before any result on them is seen. Nothing here reads an ECG.

Analysis set of a non-cardiac code c (the clean subset's analogue): cross-stay pairs between episodes whose two ECGs
have usable b and m, with no change in the 76 cardiac codes, and c absent at t. An onset is c present at t+1.
Columns per code:
  onsets            onsets in that set (the plan's threshold: >= 250)
  onsets_ed_only    of which the stay at t+1 is ED-only, i.e. c was recorded among ED discharge codes (overlap: a
                    code recorded only during admissions has its onsets determined by the care setting)
  prev_ed, prev_adm prevalence among cross-stay ECGs of ED-only stays (ED discharge codes) and of admissions
                    (hospital discharge codes): relevance across settings, and the size of the recording route
  onsets_adjusted   onsets over all at-risk between-episode pairs (cardiac codes may change; --subset adjusted)
  co_onset_or       separation: Mantel-Haenszel odds ratio, stratified by setting at t+1 x age tertile x sex (so
                    that neither the encounter nor age explains it), of an onset of any grounded
                    cardiac code (positive controls, I21 myocardial infarction, I26 pulmonary embolism) on pairs with
                    an onset of c vs without, over the adjusted set. Values well above the other candidates' flag a
                    code that travels with grounded events beyond the setting (venous thrombosis with I26)
Output: outputs/gonogo_part1/control_pool.csv (every code with >= 50 onsets), names from hosp/d_icd_diagnoses.

Example
-------
python gonogo/part1/controls.py
"""

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from linkage import PHYSIONET, load_z  # noqa: E402
from pairs import OUT, POSITIVE, load as load_pairs  # noqa: E402

GROUNDED = POSITIVE + ["I21", "I26"]


def names3():
    """3-character ICD-10 category -> title (the category's own row, else its shortest subcode's)."""
    d = pd.read_csv(PHYSIONET / "mimiciv/2.2/hosp/d_icd_diagnoses.csv.gz")
    d = d[d["icd_version"] == 10].assign(cat=lambda x: x["icd_code"].str[:3], n=lambda x: x["icd_code"].str.len())
    return d.sort_values("n").drop_duplicates("cat").set_index("cat")["long_title"]


def mh_or(event, exposed, strata):
    """Mantel-Haenszel odds ratio of event (bool) for exposed vs not, over strata."""
    num = den = 0.0
    for s in np.unique(strata):
        k = strata == s
        n = k.sum()
        a = np.sum(event & exposed & k); b = np.sum(~event & exposed & k)
        c = np.sum(event & ~exposed & k); d = np.sum(~event & ~exposed & k)
        num += a * d / n
        den += b * c / n
    return num / den if den > 0 else np.nan


def main():
    pairs, yt, yt1, vocab, _, _, _ = load_pairs()
    z = load_z()
    mz = np.load(OUT / "m.npz")
    lk = pd.Series(np.arange(len(mz["rows"])), index=mz["rows"])
    has_m = mz["has_m"][lk.loc[pairs["idx_t"].values].values] & mz["has_m"][lk.loc[pairs["idx_t1"].values].values]
    usable = pairs["has_b"].values & has_m & ~z["same_episode"].values
    C = sp.load_npz(OUT / "codes_full.npz").tocsr()
    names = np.load(OUT / "codes_full_names.npy", allow_pickle=False)
    noncard = np.array([not n.startswith("I") for n in names])

    P = np.where(usable)[0]
    Ct = C[pairs["idx_t"].values[P]].astype(bool).tocsc()
    Ct1 = C[pairs["idx_t1"].values[P]].astype(bool).tocsc()
    stable = pairs["stable"].values[P]
    ed1 = z["setting_t1"].values[P] == "ed_only"
    age = z["age"].values[P]
    strata = (pd.Categorical(z["setting_t1"].values[P]).codes * 3
              + np.searchsorted(np.quantile(age, [1 / 3, 2 / 3]), age)) * 2 + z["male"].values[P]
    grounded_on = ((yt[P][:, [vocab.index(c) for c in GROUNDED]] == 0)
                   & (yt1[P][:, [vocab.index(c) for c in GROUNDED]] == 1)).any(1)

    # prevalence by setting over the cross-stay ECGs at t
    ecg = np.unique(pairs["idx_t"].values[P])
    ctx = pd.read_parquet(OUT / "ecg_context.parquet", columns=["setting"])["setting"].values
    Ce = C[ecg].astype(bool)
    is_ed = ctx[ecg] == "ed_only"
    prev_ed = np.asarray(Ce[is_ed].mean(0)).ravel()
    prev_adm = np.asarray(Ce[~is_ed].mean(0)).ravel()

    rows = []
    for j in np.where(noncard)[0]:
        t_has = Ct[:, j].toarray().ravel()
        t1_has = Ct1[:, j].toarray().ravel()
        risk = ~t_has
        onset = risk & t1_has
        clean = risk & stable
        n_on = int((onset & clean).sum())
        if (onset & risk).sum() < 50:
            continue
        rows.append({"code": names[j], "onsets": n_on, "onsets_ed_only": int((onset & clean & ed1).sum()),
                     "onsets_adjusted": int(onset.sum()), "prev_ed": prev_ed[j], "prev_adm": prev_adm[j],
                     "co_onset_or": mh_or(grounded_on[risk], onset[risk], strata[risk])})
    df = pd.DataFrame(rows)
    df["title"] = df["code"].map(names3())
    df = df.sort_values("onsets", ascending=False)
    df.to_csv(OUT / "control_pool.csv", index=False)
    ok = df["onsets"] >= 250
    print(f"usable between-episode pairs {usable.sum():,}; cardiac-stable {stable.sum():,}; "
          f"non-cardiac codes with >= 250 clean onsets: {ok.sum()} (of {len(df)} with >= 50 adjusted onsets)")
    with pd.option_context("display.width", 250, "display.max_colwidth", 60, "display.max_rows", 400):
        print(df[ok].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

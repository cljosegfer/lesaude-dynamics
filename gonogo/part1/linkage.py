"""
Part 1 Linkage and Observed Controls z
======================================
Links every ECG of the Lance table to its encounter and builds the observed controls z of the cross-stay pairs
(gonogo/go_no_go_plan.md, 2.1).

Source: MIMIC-IV-ECG-Ext-ICD (records_w_diag_icd10.csv) already carries, per study_id, the ED stay, the admission the
ED stay led to (ed_hadm_id), the admission the ECG was taken in (hosp_hadm_id), age, sex, fold and
ecg_no_within_stay, so the linkage is a join on study_id, checked at startup: the same 800,035 ECGs, subject, fold,
stay numbering and ecg_time (to the second) as the Lance table, and all_diag_all truncated to the 76 three-character
I codes reproduces the Lance labels on every ECG. all_diag_all is the full-chapter version of the labels: the hospital
discharge codes when the encounter includes an admission, else the ED discharge codes.

A stay (the unit of ecg_no_within_stay, hence of the pair index) is the admission when the ECG was taken in hospital,
else the ED stay (reproduces the stay numbering for 99.87% of ECGs). Per ECG:
  setting    ed_only      taken in the ED, the ED stay led to no admission; labels = ED discharge codes
             ed_admitted  taken in the ED, the ED stay led to admission ed_hadm_id; labels = that admission's codes
             inpatient    taken during an admission (incl. the 14,850 taken in the ED while already admitted)
             outside      in no stay; no labels (missing, stored as zeros); never in a cross-stay pair
  episode    the admission (hosp_hadm_id, else ed_hadm_id), else the ED stay: an ED visit and the admission it led to
             are one episode but two stays
Per cross-stay pair (aligned with pairs.parquet):
  same_episode  the earlier stay is the ED visit that led to the later admission. Both ECGs are then labelled with
                the same admission's discharge codes, so the pair is stable by construction: it cannot show a
                transition and leaves the gain analysis (decision of 2026-10-22; noise floor, with within-stay pairs)
  z (the plan's observed controls): log10 gap in days, setting at t and t+1, age at t, sex, coding system at t and
  t+1. HR at t is in b_t (and m_t).
  coding     icd9 / icd10 / unknown: the system the stay was coded in. Ext-ICD converts ICD-9 codes to ICD-10, so a
             pair whose stays straddle the switch (October 2015, real time) changes coding process, a recording-route
             change. Admissions: hosp/diagnoses_icd's icd_version (10 of 430,852 admissions mixed: majority). ED-only
             stays (no version on disk): ICD-9 if one of the patient's ICD-9 admissions was discharged later, ICD-10
             if one of the patient's ICD-10 admissions was discharged earlier; otherwise from the real-year bounds
             that anchor_year_group gives (before 2015.75 -> ICD-9, after -> ICD-10, a rule checked on admissions at
             startup); else unknown.
  context (candidate drivers for 2.9, not in z: partly consequences of the transition, e.g. an arrhythmia leading to
  admission or ICU): admission type and length of stay, any ICU transfer (hosp/transfers), ED disposition, triage
  acuity and temperature (ed/triage; for admissions through the ED, of that ED stay), number of codes.

Full-chapter labels: 3-character codes of all_diag_all for every in-stay ECG, as a sparse matrix (codes_full.npz).

Outputs (outputs/gonogo_part1/): ecg_context.parquet (per Lance row), z.parquet (per cross-stay pair, same order as
pairs.parquet), codes_full.npz. Rebuilt only with --rebuild.

Example
-------
python gonogo/part1/linkage.py        # any node; reads /snfs2 and the Lance table on /scratch
"""

import argparse
import json
import sys
import time
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pairs import OUT  # noqa: E402

PHYSIONET = Path("/snfs2/josefernandes/datasets/physionet")
SETTINGS = ["ed_only", "ed_admitted", "inpatient"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--physionet", default=str(PHYSIONET))
    p.add_argument("--lance", default="/scratch/josefernandes/data/mimic_iv_ecg.lance")
    p.add_argument("--vocab", default="/scratch/josefernandes/data/icd_vocabulary_76.json")
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()


def codes3(series):
    """Ext-ICD list strings "['I4891', 'E871']" -> list of unique 3-character codes per row."""
    return series.fillna("[]").str.findall(r"'([A-Z0-9]+)'").map(lambda l: sorted({c[:3] for c in l}))


def coding_system(root, ctx, ecg_time, in_stay):
    """icd9 / icd10 / unknown per ECG (see the module docstring); prints the check of the anchor-year rule."""
    dx = pd.read_csv(root / "mimiciv/2.2/hosp/diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_version"])
    ver = dx.groupby("hadm_id")["icd_version"].agg(lambda v: v.value_counts().idxmax())
    adm = pd.read_csv(root / "mimiciv/2.2/hosp/admissions.csv.gz", usecols=["subject_id", "hadm_id", "dischtime"])
    adm["disch"] = pd.to_datetime(adm["dischtime"]).values.astype("datetime64[s]").astype(np.int64)
    adm["ver"] = adm["hadm_id"].map(ver)
    late9 = adm[adm["ver"] == 9].groupby("subject_id")["disch"].max()      # an ICD-9 discharge after s: s is ICD-9
    early10 = adm[adm["ver"] == 10].groupby("subject_id")["disch"].min()   # an ICD-10 discharge before s: ICD-10
    pat = pd.read_csv(root / "mimiciv/2.2/hosp/patients.csv.gz", usecols=["subject_id", "anchor_year", "anchor_year_group"])
    pat = pat.set_index("subject_id")
    subj = ctx["subject_id"].values
    year = pd.to_datetime(ecg_time, unit="s")
    yfrac = (year.year + (year.dayofyear - 1) / 365.25).values
    g0 = pat["anchor_year_group"].str[:4].astype(float).reindex(subj).values
    off = yfrac - pat["anchor_year"].reindex(subj).values
    lo, hi = g0 + off, g0 + 3 + off                     # real calendar year of the ECG lies in [lo, hi)
    by_year = np.select([hi <= 2015.75, lo >= 2015.75], ["icd9", "icd10"], "unknown")

    coding = np.full(len(ctx), "unknown", dtype=object)
    hosp = ctx["setting"].isin(["ed_admitted", "inpatient"]).values
    v = ctx["hadm_id"].map(ver).values
    coding[hosp & (v == 9)], coding[hosp & (v == 10)] = "icd9", "icd10"
    known = hosp & np.isin(coding, ["icd9", "icd10"])
    decided = known & (by_year != "unknown")
    print(f"anchor-year rule on admissions: decides {decided.sum() / known.sum():.3f} of them, agrees with "
          f"diagnoses_icd on {(by_year[decided] == coding[decided]).mean():.4f}", flush=True)
    ed = (ctx["setting"] == "ed_only").values
    t9 = pd.Series(late9).reindex(subj).values
    t10 = pd.Series(early10).reindex(subj).values
    coding[ed & (ecg_time < t9)] = "icd9"
    coding[ed & (ecg_time > t10)] = "icd10"
    rest = ed & (coding == "unknown")
    coding[rest] = by_year[rest]
    coding[~in_stay] = "unknown"
    print("coding system of in-stay ECGs: " + ", ".join(
        f"{s} {pd.Series(coding[in_stay & (ctx['setting'] == s).values]).value_counts(normalize=True).round(3).to_dict()}"
        for s in SETTINGS), flush=True)
    return coding


def build(args):
    root = Path(args.physionet)
    t0 = time.time()
    ext = pd.read_csv(root / "mimic-iv-ecg-ext-icd-labels/1.0.1/records_w_diag_icd10.csv",
                      usecols=["study_id", "subject_id", "ecg_time", "ed_stay_id", "ed_hadm_id", "hosp_hadm_id",
                               "all_diag_all", "gender", "age", "ecg_no_within_stay", "ecg_taken_in_ed",
                               "ecg_taken_in_hosp", "fold"])
    lt = lance.dataset(args.lance).to_table(columns=["study_id", "subject_id", "ecg_time", "ecg_no_within_stay",
                                                     "fold", "icd"])
    L = {k: lt.column(k).to_numpy() for k in ("study_id", "subject_id", "ecg_time", "ecg_no_within_stay", "fold")}
    Y = lt.column("icd").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(lt.num_rows, 76)
    vocab = json.load(open(args.vocab))

    # ---- the join, and the checks that make it the same data as the Lance table
    assert ext["study_id"].is_unique and len(ext) == len(L["study_id"])
    e = ext.set_index("study_id").loc[L["study_id"]].reset_index()
    for k in ("subject_id", "ecg_no_within_stay", "fold"):
        assert (e[k].values == L[k]).all(), k
    secs = pd.to_datetime(e["ecg_time"]).values.astype("datetime64[s]").astype(np.int64)
    assert (secs == L["ecg_time"]).all(), "ecg_time"                           # Lance stores float64 seconds
    full = codes3(e["all_diag_all"])
    pos = {c: i for i, c in enumerate(vocab)}
    rows = np.repeat(np.arange(len(full)), full.map(len).values)
    flat = np.concatenate([np.asarray(l, dtype=object) for l in full.values if len(l)])
    keep = np.array([c in pos for c in flat])
    I76 = np.zeros_like(Y)
    I76[rows[keep], [pos[c] for c in flat[keep]]] = 1
    assert (I76 == Y).all(), "all_diag_all does not reproduce the Lance labels"
    print(f"Ext-ICD joined and checked against the Lance table ({len(e):,} ECGs, {time.time() - t0:.0f} s)",
          flush=True)

    # ---- per ECG: setting, episode, stay
    ed, hosp = e["ecg_taken_in_ed"].values, e["ecg_taken_in_hosp"].values
    in_stay = e["ecg_no_within_stay"].values >= 0
    assert (in_stay == (ed | hosp)).all()
    ed_hadm, hosp_hadm, ed_stay = (e[k].astype("Int64") for k in ("ed_hadm_id", "hosp_hadm_id", "ed_stay_id"))
    setting = np.select([hosp, ed & ed_hadm.notna().values, ed], ["inpatient", "ed_admitted", "ed_only"], "outside")
    hadm = hosp_hadm.where(hosp_hadm.notna(), ed_hadm)
    ctx = pd.DataFrame({"study_id": e["study_id"].values, "subject_id": e["subject_id"].values,
                        "setting": setting, "hadm_id": hadm.values, "ed_stay_id": ed_stay.values,
                        "age": e["age"].values, "male": (e["gender"] == "M").values,
                        "n_codes": full.map(len).values})
    ctx["episode"] = ctx["hadm_id"].fillna(ctx["ed_stay_id"]).where(in_stay)   # hadm 2e7.., ED stays 3e7..
    assert ctx.loc[in_stay, "episode"].notna().all()

    # ---- encounter context (candidate drivers, not z)
    adm = pd.read_csv(root / "mimiciv/2.2/hosp/admissions.csv.gz",
                      usecols=["hadm_id", "admittime", "dischtime", "admission_type", "hospital_expire_flag"])
    adm["los_days"] = (pd.to_datetime(adm["dischtime"]) - pd.to_datetime(adm["admittime"])).dt.total_seconds() / 86400
    tr = pd.read_csv(root / "mimiciv/2.2/hosp/transfers.csv.gz", usecols=["hadm_id", "careunit"])
    icu = tr.loc[tr["careunit"].fillna("").str.contains("Intensive Care|Coronary Care"), "hadm_id"].dropna().unique()
    adm["icu"] = adm["hadm_id"].isin(icu)
    eds = pd.read_csv(root / "mimic-iv-ed/2.2/ed/edstays.csv.gz", usecols=["stay_id", "hadm_id", "disposition"])
    tri = pd.read_csv(root / "mimic-iv-ed/2.2/ed/triage.csv.gz", usecols=["stay_id", "acuity", "temperature"])
    eds = eds.merge(tri, on="stay_id", how="left")
    ctx = ctx.merge(adm[["hadm_id", "admission_type", "los_days", "icu", "hospital_expire_flag"]].astype(
        {"hadm_id": "Int64"}), on="hadm_id", how="left")
    # The ED stay of the encounter: the ECG's own, else the ED stay that led to its admission
    ed_of_hadm = eds.dropna(subset=["hadm_id"]).drop_duplicates("hadm_id").set_index(
        eds.dropna(subset=["hadm_id"]).drop_duplicates("hadm_id")["hadm_id"].astype("Int64"))["stay_id"]
    enc_ed = ctx["ed_stay_id"].fillna(ctx["hadm_id"].map(ed_of_hadm).astype("Int64"))
    ctx = pd.concat([ctx, eds.set_index("stay_id").reindex(enc_ed.values)[["disposition", "acuity", "temperature"]]
                     .reset_index(drop=True)], axis=1)
    assert len(ctx) == len(e)
    ctx["coding"] = coding_system(root, ctx, L["ecg_time"], in_stay)

    # ---- per cross-stay pair
    pairs = pd.read_parquet(OUT / "pairs.parquet")
    t, t1 = pairs["idx_t"].values, pairs["idx_t1"].values
    a, b = ctx.iloc[t].reset_index(drop=True), ctx.iloc[t1].reset_index(drop=True)
    assert (a["setting"] != "outside").all() and (b["setting"] != "outside").all()
    z = pd.DataFrame({"idx_t": t, "idx_t1": t1,
                      "same_episode": (a["episode"].values == b["episode"].values),
                      "log_gap": np.log10(pairs["gap_days"].values),
                      "setting_t": a["setting"].values, "setting_t1": b["setting"].values,
                      "age": a["age"].values, "male": a["male"].values,
                      "coding_t": a["coding"].values, "coding_t1": b["coding"].values})
    for k in ("admission_type", "los_days", "icu", "disposition", "acuity", "temperature", "n_codes"):
        z[f"{k}_t1"] = b[k].values
    z["n_codes_t"] = a["n_codes"].values

    # ---- full-chapter 3-character labels of in-stay ECGs
    names, inv = np.unique(flat, return_inverse=True)
    M = sp.csr_matrix((np.ones(len(flat), np.int8), (rows, inv)), shape=(len(full), len(names)))
    OUT.mkdir(parents=True, exist_ok=True)
    ctx.to_parquet(OUT / "ecg_context.parquet")
    z.to_parquet(OUT / "z.parquet")
    sp.save_npz(OUT / "codes_full.npz", M)
    np.save(OUT / "codes_full_names.npy", names.astype(str))
    print(f"saved ecg_context.parquet, z.parquet, codes_full.npz ({len(names):,} 3-character codes)", flush=True)
    return ctx, z, pairs


def load_z():
    """z.parquet, checked to be aligned with pairs.parquet."""
    pairs, z = pd.read_parquet(OUT / "pairs.parquet"), pd.read_parquet(OUT / "z.parquet")
    assert (z["idx_t"].values == pairs["idx_t"].values).all() and (z["idx_t1"].values == pairs["idx_t1"].values).all()
    return z


def report(z, pairs):
    s = pairs["stable"].values
    se = z["same_episode"].values
    print(f"\ncross-stay pairs {len(z):,}: same episode {se.sum():,} ({se.mean():.1%}); stable {s[se].mean():.4f} "
          f"within same-episode pairs vs {s[~se].mean():.4f} between episodes ({(~se).sum():,} pairs)", flush=True)
    bins = pd.cut(pairs["gap_days"], [0, 0.5, 1, 7, 30, 1e5], right=False)
    g = pd.DataFrame({"bin": bins, "se": se, "stable": s}).groupby("bin", observed=True)
    print(pd.DataFrame({"n": g.size(), "same_episode": g["se"].mean(),
                        "stable | same episode": g.apply(lambda d: d.loc[d.se, "stable"].mean()),
                        "stable | between": g.apply(lambda d: d.loc[~d.se, "stable"].mean())}).round(4).to_string())
    bt = z[~se]
    print("\nsetting t (rows) -> t+1 (columns), between-episode pairs:")
    tab = pd.crosstab(bt["setting_t"], bt["setting_t1"]).reindex(index=SETTINGS, columns=SETTINGS)
    print(tab.to_string())
    stab = pd.crosstab(bt["setting_t"], bt["setting_t1"], values=pairs.loc[~se, "stable"], aggfunc="mean")
    print("stable fraction:\n" + stab.reindex(index=SETTINGS, columns=SETTINGS).round(3).to_string())
    print(f"\nage missing {z['age'].isna().mean():.4f}; male {z['male'].mean():.3f}; z-context missing at t+1: "
          + ", ".join(f"{k} {z[k + '_t1'].isna().mean():.2f}" for k in ("admission_type", "acuity", "temperature")),
          flush=True)


def main():
    args = parse_args()
    if args.rebuild or not (OUT / "z.parquet").exists():
        _, z, pairs = build(args)
    else:
        z, pairs = load_z(), pd.read_parquet(OUT / "pairs.parquet")
    report(z, pairs)


if __name__ == "__main__":
    main()

"""
Part 1 Constipation Diagnostic Covariates
=========================================
Diagnostic covariates for the K59 (constipation) residual after z (Phase 0, Task 3). They are not method inputs.
They may also be post-event variables: an arrhythmia, or its treatment, can change potassium or bring opioids, so
conditioning on them can remove a real effect or open a collider path. A shrinking residual is evidence of a
co-occurring cause (electrolytes, opioid treatment), not proof of confounding.

Per ECG of the cross-stay pairs, within +-24 h of ecg_time (Lance float64 seconds):
  k, mg, tsh     the blood value closest in time (potassium 50971, 52610, 50822, 52452; magnesium 50960; TSH 50993),
                 with a missing indicator (missing values are set to the median)
  opioid, lax    any administration in the window: an eMAR "Administered" event or an ED Pyxis dispensing at a time
                 inside it, or a prescription active at any time inside it
Per pair: these at t and at t+1 (16 columns), z-scored over the pairs.

Inputs, whichever exist (--src):
  bq    BigQuery extracts of physionet-data.mimiciv_2_2_hosp (the unversioned mimiciv_hosp is 3.x) in
        /snfs2/josefernandes/datasets/physionet/task3/: labs.csv.gz (subject_id, hadm_id, charttime, itemid,
        valuenum) and meds.csv.gz (src, subject_id, hadm_id, stay_id, t0, t1, name; eMAR and prescriptions), plus
        mimic-iv-ed/2.2/ed/pyxis.csv.gz downloaded from PhysioNet (MIMIC-IV-ED needs a separate BigQuery grant)
  raw   the PhysioNet tables: mimiciv/2.2/hosp/{labevents, emar, prescriptions}.csv.gz, mimic-iv-ed/2.2/ed/pyxis.csv.gz
Output: outputs/gonogo_part1/task3_diag/pair_covariates.npz (X (n_pairs, 16) float32 aligned with pairs.parquet,
names, coverage), read by gonogo/part1/gain.py's "+diag" set.

Example
-------
python gonogo/part1/k59_diag.py --src bq
"""

import argparse
import re
import sys
from pathlib import Path

import lance
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pairs import OUT  # noqa: E402

ROOT = Path("/snfs2/josefernandes/datasets/physionet")
LABS = {"k": [50971, 52610, 50822, 52452], "mg": [50960], "tsh": [50993]}
OPIOID = re.compile(r"morphine|hydromorphone|dilaudid|oxycodone|percocet|fentanyl|hydrocodone|methadone|tramadol|"
                    r"codeine|oxymorphone|meperidine|buprenorphine|tapentadol")
LAXATIVE = re.compile(r"senna|docusate|colace|bisacodyl|dulcolax|polyethylene glycol|miralax|golytely|lactulose|"
                      r"magnesium citrate|milk of magnesia|magnesium hydroxide|methylnaltrexone|lubiprostone|"
                      r"linaclotide|fleet|glycerin|naloxegol")
WINDOW = 24 * 3600.0
DEST = OUT / "task3_diag"


def secs(series):
    return pd.to_datetime(series).values.astype("datetime64[s]").astype(np.int64).astype(np.float64)


def read_labs(src):
    if src == "bq":
        d = pd.read_csv(ROOT / "task3/labs.csv.gz", usecols=["subject_id", "charttime", "itemid", "valuenum"])
    else:
        keep = [i for ids in LABS.values() for i in ids]
        d = pd.concat(c[c["itemid"].isin(keep)] for c in pd.read_csv(
            ROOT / "mimiciv/2.2/hosp/labevents.csv.gz", usecols=["subject_id", "charttime", "itemid", "valuenum"],
            chunksize=5_000_000))
    d = d.dropna(subset=["valuenum"])
    d["t"] = secs(d["charttime"])
    d["lab"] = d["itemid"].map({i: k for k, ids in LABS.items() for i in ids})
    return d[["subject_id", "t", "lab", "valuenum"]]


def read_meds(src):
    if src == "bq":     # eMAR + prescriptions from BigQuery (mimiciv_2_2_hosp); ED dispensing from the PhysioNet file
        d = pd.read_csv(ROOT / "task3/meds.csv.gz", usecols=["subject_id", "t0", "t1", "name"])
        px = ROOT / "mimic-iv-ed/2.2/ed/pyxis.csv.gz"
        if px.exists():
            p = pd.read_csv(px, usecols=["subject_id", "charttime", "name"]).rename(columns={"charttime": "t0"})
            p["t1"] = p["t0"]
            d = pd.concat([d, p[["subject_id", "t0", "t1", "name"]]])
        else:
            print(f"WARNING: {px} missing; ED-only stays get no medication data", flush=True)
    else:
        e = pd.read_csv(ROOT / "mimiciv/2.2/hosp/emar.csv.gz", usecols=["subject_id", "charttime", "medication",
                                                                         "event_txt"])
        e = e[e["event_txt"] == "Administered"].rename(columns={"medication": "name", "charttime": "t0"})
        e["t1"] = e["t0"]
        r = pd.read_csv(ROOT / "mimiciv/2.2/hosp/prescriptions.csv.gz", usecols=["subject_id", "starttime",
                                                                                  "stoptime", "drug"])
        r = r.rename(columns={"drug": "name", "starttime": "t0", "stoptime": "t1"})
        p = pd.read_csv(ROOT / "mimic-iv-ed/2.2/ed/pyxis.csv.gz", usecols=["subject_id", "charttime", "name"])
        p = p.rename(columns={"charttime": "t0"})
        p["t1"] = p["t0"]
        d = pd.concat([e[["subject_id", "t0", "t1", "name"]], r[["subject_id", "t0", "t1", "name"]],
                       p[["subject_id", "t0", "t1", "name"]]])
    name = d["name"].fillna("").str.lower()
    d["cls"] = np.select([name.str.contains(OPIOID), name.str.contains(LAXATIVE)], ["opioid", "lax"], "")
    d = d[d["cls"] != ""].copy()
    d["t0"], d["t1"] = secs(d["t0"]), secs(d["t1"].fillna(d["t0"]))
    return d[["subject_id", "t0", "t1", "cls"]]


def per_ecg(ecg, labs, meds):
    """ecg: DataFrame (row, subject_id, t). Returns {feature: (len(ecg),) array}."""
    out = {}
    ecg = ecg.sort_values("t")
    for lab in LABS:
        L = labs[labs["lab"] == lab].sort_values("t")
        m = pd.merge_asof(ecg, L[["subject_id", "t", "valuenum"]].rename(columns={"t": "t_lab"}),
                          left_on="t", right_on="t_lab", by="subject_id", direction="nearest", tolerance=WINDOW)
        out[lab] = m.set_index("row")["valuenum"]
    for cls in ("opioid", "lax"):
        M = meds[meds["cls"] == cls]
        j = ecg.merge(M, on="subject_id", how="inner")
        hit = j[(j["t1"] >= j["t"] - WINDOW) & (j["t0"] <= j["t"] + WINDOW)]["row"].unique()
        out[cls] = pd.Series(ecg["row"].isin(hit).values.astype(float), index=ecg["row"].values)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", choices=["bq", "raw"], required=True)
    p.add_argument("--lance", default="/scratch/josefernandes/data/mimic_iv_ecg.lance")
    args = p.parse_args()
    pairs = pd.read_parquet(OUT / "pairs.parquet")
    t_sec = lance.dataset(args.lance).to_table(columns=["ecg_time"]).column("ecg_time").to_numpy().astype(np.float64)
    rows = np.unique(np.concatenate([pairs["idx_t"].values, pairs["idx_t1"].values]))
    subj = pd.Series(np.concatenate([pairs["subject_id"].values] * 2),
                     index=np.concatenate([pairs["idx_t"].values, pairs["idx_t1"].values]))
    subj = subj[~subj.index.duplicated()]
    ecg = pd.DataFrame({"row": rows, "subject_id": subj.loc[rows].values.astype(np.int64), "t": t_sec[rows]})
    feats = per_ecg(ecg, read_labs(args.src), read_meds(args.src))
    cols, names, coverage = [], [], {}
    for side, idx in (("t", pairs["idx_t"].values), ("t1", pairs["idx_t1"].values)):
        for f, s in feats.items():
            v = s.reindex(idx).values.astype(float)
            if f in LABS:
                miss = np.isnan(v)
                coverage[f"{f}_{side}"] = float(1 - miss.mean())
                cols += [np.where(miss, np.nanmedian(v), v), miss.astype(float)]
                names += [f"{f}_{side}", f"{f}_{side}_missing"]
            else:
                coverage[f"{f}_{side}"] = float(v.mean())
                cols.append(v)
                names.append(f"{f}_{side}")
    X = np.column_stack(cols)
    X = ((X - X.mean(0)) / (X.std(0) + 1e-9)).astype(np.float32)
    DEST.mkdir(parents=True, exist_ok=True)
    np.savez(DEST / "pair_covariates.npz", X=X, names=np.array(names), coverage=np.array(list(coverage.items())))
    print("coverage (labs: share with a value within 24 h; drugs: share exposed): "
          + ", ".join(f"{k} {v:.2f}" for k, v in coverage.items()))
    print("CAVEAT: these may be post-event variables (diagnostic only, not method inputs)")
    print(f"saved {DEST / 'pair_covariates.npz'}")


if __name__ == "__main__":
    main()

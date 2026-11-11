"""
Part 1 Machine Measurements m
=============================
The interpretable, ECG-observable representation m of gonogo/go_no_go_plan.md, 2.1: the ECG cart's own measurements
from MIMIC-IV-ECG's machine_measurements.csv, per Lance row.

Checked at startup: one row per study_id, and the same 800,035 ECGs and subjects as the Lance table. Not the date:
the file's ecg_time disagrees with the Lance / Ext-ICD time for 17,850 ECGs (2.2%), by whole days up to decades, with
the same time of day. The Lance date is the right one (99.3% of those ECGs within 12 years of the patient's
anchor_year, none after death; the file's date: 27%, and 44% after the year of death), and the measurements are those
of the right waveform (machine vs median-beat HR r = 0.986 on those ECGs, 0.982 on the others), so only the file's
date field is wrong; it is never used. The content check is that HR agreement (r > 0.95 required).
Missing values are sentinels (29999 = not measured, mostly P waves; a few 32767, 65535, ...): raw values
with |v| >= 29999 become missing, and so do derived values outside physiological ranges:
  hr         60000 / rr_interval, 24-250 bpm
  pr         qrs_onset - p_onset, 60-400 ms
  qrs        qrs_end - qrs_onset, 40-250 ms
  qt         t_end - qrs_onset, 200-700 ms
  qtc        Bazett, qt / sqrt(rr in s), 250-700 ms
  axes       p_axis, qrs_axis, t_axis in -180..270 degrees, as (cos, sin), so that differences do not wrap
  p_missing  P onset or P axis not measured (P absence is itself informative for I48)
PR and the P axis are imputed with their median (cos, sin: their mean) where the P wave is missing, so Δpr is 0 when
P is missing at both ECGs; p_missing carries that information. has_m: hr, qrs, qt, qtc and the QRS and T axes valid.

Output: outputs/gonogo_part1/m.npz (Lance rows of the cross-stay ECGs, m (N, 12) float32, names, has_m). Rebuilt only
with --rebuild.

Example
-------
python gonogo/part1/measurements.py
"""

import argparse
import sys
from pathlib import Path

import lance
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pairs import OUT  # noqa: E402

CSV = "/snfs2/josefernandes/datasets/physionet/mimic-iv-ecg/1.0/machine_measurements.csv"
NAMES = ["hr", "pr", "qrs", "qt", "qtc", "p_axis_cos", "p_axis_sin", "qrs_axis_cos", "qrs_axis_sin",
         "t_axis_cos", "t_axis_sin", "p_missing"]
RANGES = {"hr": (24, 250), "pr": (60, 400), "qrs": (40, 250), "qt": (200, 700), "qtc": (250, 700)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=CSV)
    p.add_argument("--lance", default="/scratch/josefernandes/data/mimic_iv_ecg.lance")
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()


def derive(mm):
    """Raw machine fields -> DataFrame of NAMES (NaN where invalid) plus has_m."""
    raw = mm[["rr_interval", "p_onset", "qrs_onset", "qrs_end", "t_end", "p_axis", "qrs_axis", "t_axis"]].astype(float)
    raw = raw.where(raw.abs() < 29999)
    out = pd.DataFrame(index=mm.index)
    rr = raw["rr_interval"].where(raw["rr_interval"] > 0)
    out["hr"] = 60000.0 / rr
    out["pr"] = raw["qrs_onset"] - raw["p_onset"]
    out["qrs"] = raw["qrs_end"] - raw["qrs_onset"]
    out["qt"] = raw["t_end"] - raw["qrs_onset"]
    out["qtc"] = out["qt"] / np.sqrt(rr / 1000.0)
    for k, (lo, hi) in RANGES.items():
        out[k] = out[k].where(out[k].between(lo, hi))
    for k in ("p_axis", "qrs_axis", "t_axis"):
        ang = np.deg2rad(raw[k].where(raw[k].between(-180, 270)))
        out[f"{k}_cos"], out[f"{k}_sin"] = np.cos(ang), np.sin(ang)
    out["p_missing"] = (raw["p_onset"].isna() | raw["p_axis"].isna()).astype(np.float32)
    has_m = out[["hr", "qrs", "qt", "qtc", "qrs_axis_cos", "t_axis_cos"]].notna().all(1).values
    return out, has_m


def main():
    args = parse_args()
    path = OUT / "m.npz"
    if args.rebuild or not path.exists():
        mm = pd.read_csv(args.csv, usecols=["subject_id", "study_id", "ecg_time", "rr_interval", "p_onset", "qrs_onset",
                                            "qrs_end", "t_end", "p_axis", "qrs_axis", "t_axis"])
        lt = lance.dataset(args.lance).to_table(columns=["study_id", "subject_id", "ecg_time"])
        L = {k: lt.column(k).to_numpy() for k in ("study_id", "subject_id", "ecg_time")}
        assert mm["study_id"].is_unique and len(mm) == len(L["study_id"])
        mm = mm.set_index("study_id").loc[L["study_id"]].reset_index()
        assert (mm["subject_id"].values == L["subject_id"]).all(), "subject_id"
        secs = pd.to_datetime(mm["ecg_time"]).values.astype("datetime64[s]").astype(np.int64)
        date_off = secs != L["ecg_time"]                                         # Lance stores float64 seconds
        print(f"machine_measurements joined on study_id; its ecg_time differs from the Lance time for "
              f"{date_off.sum():,} ECGs (not used)", flush=True)

        pairs = pd.read_parquet(OUT / "pairs.parquet")
        rows = np.unique(np.concatenate([pairs["idx_t"].values, pairs["idx_t1"].values]))
        m, has_m = derive(mm.iloc[rows].reset_index(drop=True))
        bz = np.load(OUT / "b.npz")
        assert (bz["rows"] == rows).all()
        hr_b = bz["b"][:, list(bz["names"]).index("hr")]
        ok = np.isfinite(hr_b) & m["hr"].notna().values
        r_all = np.corrcoef(m["hr"].values[ok], hr_b[ok])[0, 1]
        sel = ok & date_off[rows]
        r_off = np.corrcoef(m["hr"].values[sel], hr_b[sel])[0, 1]
        print(f"machine HR vs median-beat HR: r = {r_all:.4f} ({ok.sum():,} ECGs), {r_off:.4f} on the {sel.sum():,} "
              f"with a differing date", flush=True)
        assert r_all > 0.95 and r_off > 0.95, "machine measurements do not match the waveforms"
        miss = m.isna().mean()
        print("missing (cross-stay ECGs): " + ", ".join(f"{k} {v:.3f}" for k, v in miss.items() if v > 0), flush=True)
        for k in ("pr", "p_axis_cos", "p_axis_sin"):
            m[k] = m[k].fillna(m[k].median() if k == "pr" else m[k].mean())
        np.savez(path, rows=rows, m=m[NAMES].to_numpy(np.float32), names=np.array(NAMES), has_m=has_m)
        print(f"saved {path}: {len(rows):,} ECGs, has_m {has_m.mean():.4f}", flush=True)
    z = np.load(path)
    m, names, has_m = z["m"], list(z["names"]), z["has_m"]
    q = np.nanpercentile(m[has_m], [1, 50, 99], axis=0)
    print("valid ECGs, percentiles 1/50/99: " + "; ".join(f"{n} {q[0, i]:.2f}/{q[1, i]:.2f}/{q[2, i]:.2f}"
                                                         for i, n in enumerate(names)), flush=True)


if __name__ == "__main__":
    main()

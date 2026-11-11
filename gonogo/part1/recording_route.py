"""
Part 1 Recording Route
======================
Does the care setting change what gets coded? (gonogo/go_no_go_plan.md, 2.4.) Labels only; no ECG.

Pairs: cross-stay pairs between episodes (gonogo/part1/linkage.py; same-episode pairs share their codes by
construction). For each code c, two logistic regressions, adjusted for the setting at t, log gap, age and sex:
  resolution  P(c absent at t+1 | c present at t)
  onset       P(c present at t+1 | c absent at t)
as functions of the care setting at t+1 (ED only, ED then admitted; reference: inpatient) and of a switch of coding
system between the stays (ICD-9 at t, ICD-10 at t+1; reference: same system; pairs with an unknown system get their
own indicator). Odds ratios with 95%
confidence intervals from patient-clustered sandwich standard errors (Newton-Raphson fit), and raw rates by setting.
In Ext-ICD, an ED-only stay is labelled with its ED discharge codes and an admission with its hospital discharge codes,
so the setting at t+1 decides which coding process produced the later label.

Codes: --codes, 3-character codes of the full labels (gonogo/part1/linkage.py's codes_full.npz); default: the plan's
chronic cardiac codes. The chronic negative controls are added once their roles are approved.
Output: outputs/gonogo_part1/recording_route.json (one entry per code; existing entries are replaced).

Example
-------
python gonogo/part1/recording_route.py
python gonogo/part1/recording_route.py --codes K21,M19,M81
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from linkage import SETTINGS, load_z  # noqa: E402
from pairs import OUT  # noqa: E402

RESULT = OUT / "recording_route.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--codes", default="I10,I25,I50,I48")
    return p.parse_args()


def logit_cluster(X, y, groups, iters=100, ridge=1.0):
    """Logistic regression by Newton-Raphson with a weak ridge on the slopes, which keeps separated terms finite (codes
    never recorded at ED-only stays have no onsets there; their rates say so directly); returns coefficients and
    patient-clustered sandwich SEs."""
    R = ridge * np.eye(X.shape[1])
    R[0, 0] = 0.0
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        H = X.T @ (X * (p * (1 - p))[:, None]) + R
        step = np.linalg.solve(H, X.T @ (y - p) - R @ beta)
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    _, g = np.unique(groups, return_inverse=True)
    S = np.zeros((g.max() + 1, X.shape[1]))
    np.add.at(S, g, X * (y - p)[:, None] - R @ beta / len(S))
    Hinv = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + R)
    G = len(S)
    V = Hinv @ (S.T @ S) @ Hinv * G / (G - 1)
    return beta, np.sqrt(np.diag(V))


def main():
    args = parse_args()
    pairs, z = pd.read_parquet(OUT / "pairs.parquet"), load_z()
    C = sp.load_npz(OUT / "codes_full.npz").tocsc()
    names = list(np.load(OUT / "codes_full_names.npy"))
    keep = ~z["same_episode"].values
    t, t1 = pairs["idx_t"].values[keep], pairs["idx_t1"].values[keep]
    Z = z[keep].reset_index(drop=True)
    subj = pairs["subject_id"].values[keep]
    d = lambda col, level: (Z[col].values == level).astype(float)
    switch = ((Z["coding_t"] == "icd9") & (Z["coding_t1"] == "icd10")).values.astype(float)
    unknown = ((Z["coding_t"] == "unknown") | (Z["coding_t1"] == "unknown")).values.astype(float)
    X = np.column_stack([np.ones(len(Z)), d("setting_t1", "ed_only"), d("setting_t1", "ed_admitted"), switch,
                         d("setting_t", "ed_only"), d("setting_t", "ed_admitted"), unknown,
                         Z["log_gap"].values, Z["age"].values / 10, Z["male"].values.astype(float)])
    terms = ["t+1 ED only", "t+1 ED then admitted", "ICD-9 -> ICD-10"]
    res = json.loads(RESULT.read_text()) if RESULT.exists() else {}
    for code in args.codes.split(","):
        col = C[:, names.index(code)]
        yt, yt1 = col[t].toarray().ravel() > 0, col[t1].toarray().ravel() > 0
        out = {"n_pairs": int(len(Z))}
        for kind, risk, event in (("resolution", yt, ~yt1), ("onset", ~yt, yt1)):
            by = pd.Series(event[risk]).groupby(Z["setting_t1"].values[risk])
            rates, counts = by.mean().reindex(SETTINGS), by.sum().reindex(SETTINGS)
            beta, se = logit_cluster(X[risk], event[risk].astype(float), subj[risk])
            out[kind] = {"n": int(risk.sum()), "events": int(event[risk].sum()),
                         "rate_by_setting_t1": {k: round(float(v), 4) for k, v in rates.items()},
                         "events_by_setting_t1": {k: int(v) for k, v in counts.items()},
                         "odds_ratio": {terms[i]: [float(np.exp(beta[1 + i])), float(np.exp(beta[1 + i] - 1.96 * se[1 + i])),
                                                   float(np.exp(beta[1 + i] + 1.96 * se[1 + i]))]
                                        for i in range(len(terms))}}
            o = out[kind]["odds_ratio"]
            print(f"{code} {kind:<10} n={risk.sum():>7,} events {event[risk].sum():>6,}  rate at t+1 "
                  + " / ".join(f"{k} {v:.3f}" for k, v in rates.items())
                  + "  OR " + "; ".join(f"{k} {v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" for k, v in o.items()), flush=True)
        res[code] = out
        RESULT.write_text(json.dumps(res, indent=1))       # after every code, so a failure keeps the finished ones
    print(f"saved {RESULT}")


if __name__ == "__main__":
    main()

"""
Part 1 Positive-Control Effect Vectors in h
===========================================
The effect vectors Part 2 injects as grounded effects (gonogo/go_no_go_plan.md, 2.9; Phase 0, Task 5).

Input: gonogo/part1/gain.py results on Δh under z with --save-residuals (<result>.resid.npz: a - pi and the base
model's residuals per pair and dimension). Per code, the partially linear effect of the onset per dimension,
theta = sum (a - pi) r / sum (a - pi)^2, the R-learner's theta (in z-units of Δh; raw h units = theta x sd).
Uncertainty: patient-level bootstrap (1,000 resamples of patients). The plug-in |theta|^2 is biased upward by the
estimation noise (about dims / onsets), so the squared norm is estimated from independent halves, folds 0-9 vs
10-19 (disjoint patients): <theta_A, theta_B>, unbiased, in z-units and in raw units. Debiased cosines between codes
use the same halves, cos(c, d) = (<theta_c,A, theta_d,B> + <theta_c,B, theta_d,A>) / 2 / sqrt(|c|^2 |d|^2).

Output: outputs/gonogo_part1/effects_h.npz (codes, theta_z, theta_raw, se_z, n_onsets, norm2_z and norm2_raw as
[point, lo, hi], cos as a matrix) and effects_h.json (the same summaries, readable).

Example
-------
python gonogo/part1/effects.py outputs/gonogo_part1/task5_effects
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pairs import OUT, POSITIVE  # noqa: E402

N_BOOT = 1000


def per_patient(z):
    """Patient sums of (a - pi) r (P, K) and (a - pi)^2 (P,), each patient's half (folds < 10), and onsets."""
    _, inv = np.unique(z["subj"], return_inverse=True)
    P = inv.max() + 1
    ra, ry = z["ra"].astype(np.float64), z["ry"].astype(np.float64)
    num = np.zeros((P, ry.shape[1]))
    np.add.at(num, inv, ra[:, None] * ry)
    den = np.bincount(inv, weights=ra ** 2, minlength=P)
    half = np.zeros(P, bool)
    half[inv] = z["folds"] < 10
    return num, den, half


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    args = p.parse_args()
    rng = np.random.default_rng(0)
    res, halves, summ = {}, {}, {}
    for code in POSITIVE:
        f = Path(args.folder) / f"clean_bhm_h_z_{code}.resid.npz"
        if not f.exists():
            continue
        z = np.load(f)
        num, den, half = per_patient(z)
        P = len(den)
        W = rng.multinomial(P, np.full(P, 1 / P), size=N_BOOT).astype(np.float64)
        theta = num.sum(0) / den.sum()
        boot = (W @ num) / (W @ den)[:, None]
        tA, tB = num[half].sum(0) / den[half].sum(), num[~half].sum(0) / den[~half].sum()
        bA = (W[:, half] @ num[half]) / (W[:, half] @ den[half])[:, None]
        bB = (W[:, ~half] @ num[~half]) / (W[:, ~half] @ den[~half])[:, None]
        sd2 = z["sd"].astype(np.float64) ** 2
        n2z, n2r = float(tA @ tB), float((tA * tB * sd2).sum())
        b2z, b2r = (bA * bB).sum(1), (bA * bB * sd2).sum(1)
        meta = json.loads((Path(args.folder) / f"clean_bhm_h_z_{code}.json").read_text())
        res[code] = {"theta_z": theta, "theta_raw": theta * np.sqrt(sd2), "se_z": boot.std(0),
                     "norm2_z": [n2z, *np.percentile(b2z, [2.5, 97.5])],
                     "norm2_raw": [n2r, *np.percentile(b2r, [2.5, 97.5])],
                     "plugin_norm2_z": float(theta @ theta), "n_onsets": meta["n_onsets"]}
        halves[code] = (tA, tB, bA, bB)
        summ[code] = {k: (v if not isinstance(v, np.ndarray) else None) for k, v in res[code].items()}
        print(f"{code}: onsets {meta['n_onsets']}; |theta|^2 z-units debiased {n2z:.4f} "
              f"[{res[code]['norm2_z'][1]:.4f}, {res[code]['norm2_z'][2]:.4f}] (plug-in {theta @ theta:.4f}); "
              f"raw units {n2r:.4f} [{res[code]['norm2_raw'][1]:.4f}, {res[code]['norm2_raw'][2]:.4f}]", flush=True)
    codes = list(res)
    cos = np.full((len(codes), len(codes)), np.nan)
    cos_ci = {}
    for i, c in enumerate(codes):
        for j, d in enumerate(codes):
            if i >= j:
                continue
            (cA, cB, bcA, bcB), (dA, dB, bdA, bdB) = halves[c], halves[d]
            nc, nd = res[c]["norm2_z"][0], res[d]["norm2_z"][0]
            if nc <= 0 or nd <= 0:
                continue
            point = (cA @ dB + cB @ dA) / 2 / np.sqrt(nc * nd)
            # Bootstrap with the codes' own resamples (patients differ between codes' analysis sets)
            bc = (bcA * bcB).sum(1)
            bd = (bdA * bdB).sum(1)
            bx = ((bcA * bdB).sum(1) + (bcB * bdA).sum(1)) / 2
            ok = (bc > 0) & (bd > 0)
            cos[i, j] = cos[j, i] = point
            cos_ci[f"{c}-{d}"] = [float(point), *np.percentile(bx[ok] / np.sqrt(bc[ok] * bd[ok]), [2.5, 97.5]).tolist()]
            print(f"  cos({c}, {d}) debiased {point:+.2f} [{cos_ci[f'{c}-{d}'][1]:+.2f}, {cos_ci[f'{c}-{d}'][2]:+.2f}]")
    np.savez(OUT / "effects_h.npz", codes=np.array(codes), theta_z=np.stack([res[c]["theta_z"] for c in codes]),
             theta_raw=np.stack([res[c]["theta_raw"] for c in codes]), se_z=np.stack([res[c]["se_z"] for c in codes]),
             norm2_z=np.array([res[c]["norm2_z"] for c in codes]), norm2_raw=np.array([res[c]["norm2_raw"] for c in codes]),
             n_onsets=np.array([res[c]["n_onsets"] for c in codes]), cos=cos)
    (OUT / "effects_h.json").write_text(json.dumps({"codes": summ, "cos": cos_ci}, indent=1, default=float))
    print(f"saved {OUT / 'effects_h.npz'}")


if __name__ == "__main__":
    main()

"""
Phase 0.5c on b: Sensitivity of the Margin (reported only; plan revision 10, step 2a)
====================================================================================
On the 182 withheld codes of the rotation (outputs/phase05/9_3e_b/gamma.json: ψ̂²_n and bootstrap variances v_n, Ledoit-
Wolf Σ, halves 0-9 vs 10-19), the between-code variance τ² and the margin Γ̂² under three estimators:
  DL    DerSimonian-Laird (the pre-specified primary, gamma_b.py)
  PM    Paule-Mandel: τ² solving Q(τ²) = Σ (y - μ(τ²))² / (v + τ²) = N - 1
  REML  restricted maximum likelihood (Fisher scoring)
Γ̂² = μ̂ + 1.645 τ̂, or μ̂ + 1.645 SE(μ̂) when τ̂ = 0.
SE calibration check: for every withheld code, the residualised labels (a - π̂) of gain.py's residual file are permuted
within folds, which makes every true residual zero; ψ̂²_n and its bootstrap variance are recomputed exactly as for the
real data; Cochran's Q/df = Σ (y - μ_FE)² / v / (N - 1) should then be near 1 if the per-code variances are calibrated
(< 1: overstated, > 1: understated). Several permutation replicates. Also: the 0.5c outcome against a zero margin.

Output: outputs/phase05/9_3e_b/margin_sensitivity.{json,md}

Example
-------
python -I gonogo/phase05/margin_sensitivity.py --perms 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gonogo" / "part1"))
sys.path.insert(0, str(ROOT / "gonogo" / "phase05"))
from psi2 import metrics, stable_changes  # noqa: E402

D = ROOT / "outputs" / "phase05" / "9_3e_b"


def margin(y, v, tau2):
    w = 1 / (v + tau2)
    mu = (w * y).sum() / w.sum()
    se = w.sum() ** -0.5
    g2 = mu + 1.645 * np.sqrt(tau2) if tau2 > 0 else mu + 1.645 * se
    return {"tau2": float(tau2), "tau": float(np.sqrt(tau2)), "mu": float(mu), "se_mu": float(se), "Gamma2": float(g2)}


def q_stat(y, v, tau2=0.0):
    w = 1 / (v + tau2)
    mu = (w * y).sum() / w.sum()
    return float((w * (y - mu) ** 2).sum())


def dl(y, v):
    w = 1 / v
    Q = q_stat(y, v)
    return max(0.0, (Q - (len(y) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))


def pm(y, v):
    f = lambda t: q_stat(y, v, t) - (len(y) - 1)
    if f(0.0) <= 0:
        return 0.0
    hi = max(1e-6, np.var(y))
    while f(hi) > 0:
        hi *= 2
    return float(brentq(f, 0.0, hi))


def reml(y, v, iters=500):
    t = max(0.0, dl(y, v))
    for _ in range(iters):
        w = 1 / (v + t)
        mu = (w * y).sum() / w.sum()
        new = max(0.0, ((w ** 2) * ((y - mu) ** 2 - v)).sum() / (w ** 2).sum() + 1 / w.sum())
        if abs(new - t) < 1e-12:
            break
        t = new
    return float(t)


def psi2_perm(path, M, rng, n_boot=300):
    z = np.load(path)
    ra, ry, folds, subj, sd = (z["ra"].astype(np.float64), z["ry"].astype(np.float64), z["folds"].astype(int),
                               z["subj"], z["sd"].astype(np.float64))
    for f in np.unique(folds):                        # permute the residualised labels within each fold
        k = np.where(folds == f)[0]
        ra[k] = ra[rng.permutation(k)]
    u, inv = np.unique(subj, return_inverse=True)
    num = np.zeros((len(u), ry.shape[1]))
    np.add.at(num, inv, ra[:, None] * ry)
    den = np.bincount(inv, weights=ra ** 2, minlength=len(u))
    pf = np.zeros(len(u), int)
    pf[inv] = folds
    A, B = pf < 10, pf >= 10
    W = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=n_boot).astype(np.float64)
    tA, tB = num[A].sum(0) / den[A].sum() * sd, num[B].sum(0) / den[B].sum() * sd
    bA = (W[:, A] @ num[A]) / (W[:, A] @ den[A])[:, None] * sd
    bB = (W[:, B] @ num[B]) / (W[:, B] @ den[B])[:, None] * sd
    boot = np.einsum("bi,ij,bj->b", bA, M, bB)
    return float(tA @ M @ tB), float(boot[np.isfinite(boot)].var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=3)
    args = ap.parse_args()
    g = json.loads((D / "gamma.json").read_text())
    W = g["withheld"]
    codes = list(W)
    y = np.array([W[c]["psi2"] for c in codes])
    v = np.array([W[c]["var"] for c in codes])
    est = {"DL": margin(y, v, dl(y, v)), "PM": margin(y, v, pm(y, v)), "REML": margin(y, v, reml(y, v))}
    q_real = q_stat(y, v) / (len(y) - 1)
    deltas, _, _ = stable_changes(list(range(20)), [])
    M = metrics(deltas["b"])[0]["lw"]
    rng = np.random.default_rng(0)
    perm = []
    for r in range(args.perms):
        yp, vp = [], []
        for c in codes:
            f = D / "gain" / f"clean_bm_b_z+rot{W[c]['group']}_{c}.resid.npz"
            a, b = psi2_perm(f, M, rng)
            yp.append(a)
            vp.append(b)
        yp, vp = np.array(yp), np.array(vp)
        ok = np.isfinite(yp) & np.isfinite(vp) & (vp > 0)
        perm.append({"replicate": r, "N": int(ok.sum()), "Q_over_df": q_stat(yp[ok], vp[ok]) / (ok.sum() - 1),
                     "mean_psi2": float(yp[ok].mean()), "mean_z": float((yp[ok] / np.sqrt(vp[ok])).mean()),
                     "sd_z": float((yp[ok] / np.sqrt(vp[ok])).std())})
        print(f"permutation {r}: Q/df {perm[-1]['Q_over_df']:.3f}, mean ψ̂² {perm[-1]['mean_psi2']:+.4f}, "
              f"z mean {perm[-1]['mean_z']:+.3f} sd {perm[-1]['sd_z']:.3f}", flush=True)
    pos = {c: t for c, t in g["tested"].items() if t["role"] == "positive"}
    zero_margin = {c: {"lower": t["lw"]["ci95"][0], "clears_0": bool(t["lw"]["ci95"][0] > 0)} for c, t in pos.items()}
    out = {"estimators": est, "Q_over_df_real": q_real, "permutation_check": perm, "positive_vs_zero_margin": zero_margin,
           "n_positive_clearing_0": int(sum(x["clears_0"] for x in zero_margin.values())),
           "note": "reported only; the 0.5c decision uses DL (pre-specified)"}
    (D / "margin_sensitivity.json").write_text(json.dumps(out, indent=1))
    L = ["| Estimator | τ̂² | τ̂ | μ̂ [SE] | Γ̂² |", "|---|---|---|---|---|"]
    L += [f"| {k} | {e['tau2']:.5f} | {e['tau']:.4f} | {e['mu']:.4f} [{e['se_mu']:.4f}] | {e['Gamma2']:.4f} |"
          for k, e in est.items()]
    L += ["", f"Cochran's Q/df on the real withheld codes: {q_real:.3f}.",
          "Permuted labels (true residuals zero): " + "; ".join(f"Q/df {p['Q_over_df']:.3f} (mean ψ̂² {p['mean_psi2']:+.4f}, "
                                                                f"z sd {p['sd_z']:.2f})" for p in perm),
          "", "Positive controls against a zero margin (folds 10-19): " + ", ".join(
              f"{c} lower {x['lower']:+.3f} ({'clears' if x['clears_0'] else 'fails'})" for c, x in zero_margin.items())]
    (D / "margin_sensitivity.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

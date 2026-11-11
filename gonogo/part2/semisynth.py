"""
Part 2: Semi-Synthetic Test on b at γ_mix = 0 and the Scale Choice (plan revision 11, rule (c))
================================================================================================
Real pairs, real noise, real covariates; only the injected codes are artificial (go_no_go_plan.md 3.2, revisions 10-11).

Base data   the 35,009 real usable stable between-episode pairs of b (b and m at both ECGs, not same-episode, no
            cardiac code change), with real b_t, b_t+1, z, A transitions, patients and folds.
Injection   per code, onset pairs are sampled from the pool with weights exp(0.75 * score), the score being the
            standardised mean of a random projection of b_t and age (the 3.2 risk profile). G codes add scale x Δ_c
            (shrunk to the debiased norm, revision 11; I47, I48, I49, one-feature, two-feature, diffuse) to b_t+1 on
            event pairs; events are recorded with κ = 0.9 at an admission and 0.9 x 0.13 at an ED-only next stay, and an
            unrecorded event keeps its effect and label 0 (revision 10 e). U codes: the same risk profile, no effect.
            γ_mix = 0: no code follows the driver. Onset counts: I47 302, I48 777, I49 129, one-feature 185,
            two-feature 204, diffuse 351 (recorded); each U code has its G partner's count.
Estimators  Part 1 estimator: cross-fitted R-learner on the real patient folds, base model ridge on [b_t, z without
            setting(t+1), A transitions], propensity logistic on [b_t, z without setting(t+1)]. Oracle: the same plus
            setting(t+1), the withheld driver.
Inference   revision 11: T = θ̂_A' M θ̂_B (halves 0-9 / 10-19, M = Ledoit-Wolf on the real stable-pair changes); bias-
            corrected plug-in variance from half-specific patient bootstraps; stratified permutation p-value (strata
            fold x gap quintile x age tertile, x setting(t+1) for the oracle); percentile bootstrap secondary.
Rule (c)    C2 is read at the smallest scale in {0.5, 1, 2, 4} at which the ORACLE's mean G-vs-U AUROC of T over the
            draws reaches 0.9.
Realism     report-only: G detection at 1x with the real folds 10-19 protocol (quarters 10-14 vs 15-19, lower bound
            of the percentile bootstrap > 0), against the real result (1 of 5 positive controls).

Output: outputs/part2/semisynth/{gamma0_rows.csv, gamma0.json, gamma0.md}

Revision 12 (commands crt-validate, gamma0, sweep; the revision-11 run is the command rev11)
-------------------------------------------------------------------------------------------
CRT         the stratified permutation test is replaced by a conditional randomization test: 500 null label vectors
            ã ~ Bernoulli(π̂(x)) from the cross-fitted propensity; T recomputed with the residuals fixed and per-draw
            denominators; p = (1 + #{T_k >= T}) / 501.
Noise       patient-bootstrapped Part 1 base-model residuals of the real stable pairs (a random patient, then one of its
            pairs) are added to every injected event pair, scaled by c so that the Part 1 residual-variance ratio (event
            pairs vs other pairs, mean squared residual summed over b's dimensions in units of the stable-pair change sd)
            reaches the target: G codes their source control's ratio (I47 2.36, I48 1.92, I49 1.90; synthetic patterns
            1.92), U/UC/GC codes the C1 median 1.111. c is solved per scale (the effect adds to the ratio). The same noise
            enters both estimators.
UC/GC       each onset follows the driver (weight 10 at an ED-only next stay, 1 otherwise) with probability γ_mix, else
            the risk profile. GC codes carry the I47/I48/I49 effects with κ < 1; the number of events is set so that the
            expected number of recorded onsets matches the count at every γ_mix.

python -I gonogo/part2/semisynth.py crt-validate --draws 16
python -I gonogo/part2/semisynth.py gamma0 --draws 8
python -I gonogo/part2/semisynth.py sweep --draws 8 --scale 1
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import beta as beta_dist, kstest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gonogo" / "part1"))
sys.path.insert(0, str(ROOT / "gonogo" / "phase05"))
sys.path.insert(0, str(ROOT / "gonogo" / "part2"))
from latent import ALPHA, COUNTS, KAPPA_ADM, KAPPA_ED, SCALES, logistic_cf, new_inference, strata_perms  # noqa: E402
from linkage import load_z  # noqa: E402
from pairs import OUT as P1, load as load_pairs  # noqa: E402
from psi2 import metrics  # noqa: E402

OUTD = ROOT / "outputs" / "part2" / "semisynth"
CAL = ROOT / "outputs" / "part2" / "latent"
FROZEN = ROOT / "outputs" / "phase05" / "freeze" / "label_sets.json"
GAMMA2 = 0.0135
N_FOLDS = 20


def load_pool(rep="b"):
    pairs, _, _, _, b_lookup, b, names = load_pairs()
    if rep == "h":                     # plan revision 13: the reference h (zeroactions export, 256 dims)
        hz = np.load(P1 / "h_zeroactions.npz")
        b, names = hz["h"], [f"h{i}" for i in range(hz["h"].shape[1])]
        b_lookup = pd.Series(np.arange(len(hz["rows"])), index=hz["rows"])
    z = load_z()
    mz = np.load(P1 / "m.npz")
    lk = pd.Series(np.arange(len(mz["rows"])), index=mz["rows"])
    has_m = mz["has_m"][lk.loc[pairs["idx_t"].values].values] & mz["has_m"][lk.loc[pairs["idx_t1"].values].values]
    use = pairs["has_b"].values & has_m & ~z["same_episode"].values
    rows_all = np.unique(np.concatenate([b_lookup.loc[pairs["idx_t"].values[use]].values,
                                         b_lookup.loc[pairs["idx_t1"].values[use]].values]))
    sd_x = b[rows_all].std(0)
    keep = use & pairs["stable"].values
    P, Z = pairs[keep].reset_index(drop=True), z[keep].reset_index(drop=True)
    bt = b[b_lookup.loc[P["idx_t"].values].values].astype(np.float64)
    bt1 = b[b_lookup.loc[P["idx_t1"].values].values].astype(np.float64)
    onehot = lambda col: pd.get_dummies(Z[col].astype(str)).to_numpy(np.float64)
    zf = np.c_[Z["log_gap"].values, onehot("setting_t"), Z["age"].values, Z["male"].values.astype(float),
               onehot("coding_t"), onehot("coding_t1")]
    zs = onehot("setting_t1")
    # A transitions (frozen A = C0 without the validation subset), onset and resolution indicators
    import scipy.sparse as sp
    fz = json.loads(FROZEN.read_text())
    A = sorted(c for c in fz["C0"] if c not in set(fz["validation_c1"]))
    C = sp.load_npz(P1 / "codes_full.npz").tocsr()
    nm = list(np.load(P1 / "codes_full_names.npy").astype(str))
    j = [nm.index(c) for c in A]
    Ct = C[P["idx_t"].values][:, j].astype(bool).toarray()
    Ct1 = C[P["idx_t1"].values][:, j].astype(bool).toarray()
    At = np.c_[(Ct1 & ~Ct), (Ct & ~Ct1)].astype(np.float64)
    std = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-9)
    return {"bt": bt, "bt1": bt1, "sd_x": sd_x, "names": names, "zf": std(zf), "zs": zs, "A": At,
            "age": Z["age"].values.astype(float), "log_gap": Z["log_gap"].values,
            "ed1": (Z["setting_t1"].values == "ed_only"), "subj": P["subject_id"].values, "fold": P["fold"].values}


class Ridge:
    """Cross-fitted ridge with the per-fold factorisations computed once (residuals of any Y on the same X)."""

    def __init__(self, X, folds, lam=1e-3):
        self.Xa, self.folds = np.c_[X, np.ones(len(X))], folds
        G = self.Xa.T @ self.Xa
        self.fac = {}
        for f in range(N_FOLDS):
            te = folds == f
            Gf = G - self.Xa[te].T @ self.Xa[te]
            self.fac[f] = cho_factor(Gf + lam * (~te).sum() * np.eye(len(G)))

    def resid(self, Y):
        Y = Y.reshape(len(Y), -1)
        XtY = self.Xa.T @ Y
        R = np.empty_like(Y)
        for f in range(N_FOLDS):
            te = self.folds == f
            W = cho_solve(self.fac[f], XtY - self.Xa[te].T @ Y[te])
            R[te] = Y[te] - self.Xa[te] @ W
        return R


def rev11(args):
    d = load_pool()
    n, dim = d["bt"].shape
    D = d["bt1"] - d["bt"]
    M = metrics(D)[0]["lw"]                                        # Ledoit-Wolf on the real stable-pair changes
    cz = np.load(CAL / "calibration_b.npz")
    vec = {k[5:]: cz[k] * d["sd_x"] for k in cz.files if k.startswith("vecs_")}      # raw b units
    Lsb = np.linalg.cholesky(cz["Sigma_b"] + 1e-9 * np.eye(dim))
    med = float(cz["diffuse_target"])
    bts = (d["bt"] - d["bt"].mean(0)) / d["bt"].std(0)
    feats = {"part1": np.c_[bts, d["zf"]], "oracle": np.c_[bts, d["zf"], d["zs"]]}
    ridge = {e: Ridge(np.c_[F, d["A"]], d["fold"]) for e, F in feats.items()}
    Rb = {e: r.resid(D) for e, r in ridge.items()}
    _, inv = np.unique(d["subj"], return_inverse=True)
    gap5 = np.searchsorted(np.quantile(d["log_gap"], [0.2, 0.4, 0.6, 0.8]), d["log_gap"])
    age3 = np.searchsorted(np.quantile(d["age"], [1 / 3, 2 / 3]), d["age"])
    strata = {"part1": (d["fold"] * 5 + gap5) * 3 + age3}
    strata["oracle"] = strata["part1"] * 2 + d["ed1"]
    age_z = (d["age"] - d["age"].mean()) / d["age"].std()
    kbar = float(np.mean(np.where(d["ed1"], KAPPA_ED, KAPPA_ADM)))
    rng = np.random.default_rng(1110)
    rows = []
    for dr in range(args.draws):
        ud = rng.standard_normal(dim)
        vecs = {**vec, "diffuse": (Lsb @ (ud / np.linalg.norm(ud)) * np.sqrt(med)) * d["sd_x"]}
        for kind in ("G", "U"):
            for c, n_rec in COUNTS.items():
                w = rng.standard_normal(dim)
                proj = bts @ (w / np.linalg.norm(w))
                score = (proj / proj.std() + age_z) / np.sqrt(2)
                wt = np.exp(ALPHA * score / score.std())
                n_ev = int(round(n_rec / kbar)) if kind == "G" else n_rec
                ev = rng.choice(n, size=n_ev, replace=False, p=wt / wt.sum())
                e = np.zeros(n)
                e[ev] = 1
                a = e.copy()
                if kind == "G":
                    a[ev] = (rng.random(n_ev) < np.where(d["ed1"][ev], KAPPA_ED, KAPPA_ADM)).astype(float)
                for est, F in feats.items():
                    pi = logistic_cf(F, a, d["fold"])
                    ra = a - pi
                    Re = ridge[est].resid(e)[:, :1] * vecs[c][None] if kind == "G" else np.zeros_like(D)
                    res = new_inference(ra, Rb[est], Re, inv, d["fold"], M, rng, SCALES if kind == "G" else [0.0],
                                        strata=strata[est])
                    q1 = new_inference(ra, Rb[est], Re, inv, d["fold"], M, rng, [1.0] if kind == "G" else [0.0],
                                       strata=strata[est], split=(tuple(range(10, 15)), tuple(range(15, 20))))
                    auc_p = roc_auc_score(a, pi)
                    for sc, r in res.items():
                        rq = q1.get(1.0 if kind == "G" else 0.0)
                        rows.append({"draw": dr, "kind": kind, "code": c, "estimator": est, "scale": sc,
                                     "onsets": int(a.sum()), "events": int(e.sum()), "propensity_auroc": float(auc_p),
                                     **r, "q_lo": rq["lo"], "q_perc_lo": rq["perc_lo"]})
        print(f"draw {dr + 1}/{args.draws} done", flush=True)
    df = pd.DataFrame(rows)
    OUTD.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTD / "gamma0_rows.csv", index=False)
    U = df[df.kind == "U"]
    summ = {}
    for est in ("oracle", "part1"):
        u = U[U.estimator == est]
        for sc in SCALES:
            g = df[(df.kind == "G") & (df.estimator == est) & (df.scale == sc)]
            aucs = [roc_auc_score(np.r_[np.ones(6), np.zeros(6)], np.r_[g[g.draw == k]["T"].values, u[u.draw == k]["T"].values])
                    for k in range(args.draws)]
            summ[f"{est}|{sc}"] = {"auroc_mean": float(np.mean(aucs)), "auroc_min": float(np.min(aucs)),
                                   "G_lo_gt_0": float((g.lo > 0).mean()), "G_lo_gt_Gamma": float((g.lo > GAMMA2).mean()),
                                   "G_perm_p_lt_05": float((g.p_perm < 0.05).mean()),
                                   "by_vector_lo_gt_Gamma": g.groupby("code").apply(lambda x: float((x.lo > GAMMA2).mean())).to_dict()}
        summ[f"{est}|U"] = {"U_lo_gt_0": float((u.lo > 0).mean()), "U_lo_gt_Gamma": float((u.lo > GAMMA2).mean()),
                            "U_perm_p_lt_05": float((u.p_perm < 0.05).mean())}
    chosen = next((sc for sc in SCALES if summ[f"oracle|{sc}"]["auroc_mean"] >= 0.9), None)
    g1 = df[(df.kind == "G") & (df.scale == 1.0)]
    realism = {est: {"quarters_percentile_lo_gt_0": float((g1[g1.estimator == est].q_perc_lo > 0).mean()),
                     "quarters_new_lo_gt_0": float((g1[g1.estimator == est].q_lo > 0).mean()),
                     "by_vector_quarters_percentile": g1[g1.estimator == est].groupby("code").apply(
                         lambda x: float((x.q_perc_lo > 0).mean())).to_dict()} for est in ("oracle", "part1")}
    out = {"draws": args.draws, "n_pool": int(n), "kappa_bar": kbar, "propensity_auroc_median": float(df.propensity_auroc.median()),
           "summary": summ, "chosen_scale": chosen,
           "rule_c": (f"C2 is read at scale {chosen}x" if chosen else "no scale reaches oracle AUROC 0.9: C2 cannot be read"),
           "realism_1x": realism, "real_folds_10_19": "1 of 5 positive controls (quarters, percentile lower bound > 0)"}
    (OUTD / "gamma0.json").write_text(json.dumps(out, indent=1))
    L = [f"Semi-synthetic test on b at γ_mix = 0: {n:,} real stable pairs, {args.draws} draws x (6 G + 6 U), κ̄ {kbar:.3f}, "
         f"median propensity AUROC {out['propensity_auroc_median']:.3f}.", "",
         "| Estimator | Scale | G-vs-U AUROC, mean (min) over draws | G: lower bound > 0 | > Γ̂² | permutation p < 0.05 |",
         "|---|---|---|---|---|---|"]
    for est in ("oracle", "part1"):
        for sc in SCALES:
            s_ = summ[f"{est}|{sc}"]
            L.append(f"| {est} | {sc}x | {s_['auroc_mean']:.3f} ({s_['auroc_min']:.3f}) | {s_['G_lo_gt_0']:.2f} | "
                     f"{s_['G_lo_gt_Gamma']:.2f} | {s_['G_perm_p_lt_05']:.2f} |")
        su = summ[f"{est}|U"]
        L.append(f"| {est} | U codes (θ = 0) | — | {su['U_lo_gt_0']:.2f} | {su['U_lo_gt_Gamma']:.2f} | {su['U_perm_p_lt_05']:.2f} |")
    L += ["", f"**Rule (c): {out['rule_c']}.**", "",
          "Realism check at 1x (quarters 10-14 vs 15-19, percentile lower bound > 0; real: 1 of 5): " + "; ".join(
              f"{est} {r['quarters_percentile_lo_gt_0']:.2f} (new interval {r['quarters_new_lo_gt_0']:.2f}; by vector "
              + ", ".join(f"{k} {v:.2f}" for k, v in r["by_vector_quarters_percentile"].items()) + ")"
              for est, r in realism.items())]
    (OUTD / "gamma0.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ------------------------------------------------------------------------------------------------ revision 12
HALVES = (tuple(range(10)), tuple(range(10, 20)))
GPU_LOGISTIC = [False]
QUARTERS = (tuple(range(10, 15)), tuple(range(15, 20)))
TARGET_G = {"I47": 2.36, "I48": 1.92, "I49": 1.90, "one-feature": 1.92, "two-feature": 1.92, "diffuse": 1.92}
TARGET_U = 1.111                       # median ratio on real onset pairs of the 11 C1 controls (measured 11 Nov 2026)
GC_CODES = ("I47", "I48", "I49")
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DRIVER_OR = 10.0
S = {}                                 # shared state, filled before the workers fork


OUTD_H = ROOT / "outputs" / "part2" / "semisynth_h"
ROT_H = ROOT / "outputs" / "phase05" / "12_h_rotation"


def setup(rep="b", variants=("part1", "oracle")):
    """Shared state. Input variants (plan revision 13): naive (b_t only), z-s ([b_t, z without setting(t+1)]), part1
    (z-s plus A in the base model; the Part 1 estimator), oracle (part1 plus setting(t+1) in both models)."""
    d = load_pool(rep)
    n, dim = d["bt"].shape
    D = d["bt1"] - d["bt"]
    M = metrics(D)[0]["lw"]
    bts = (d["bt"] - d["bt"].mean(0)) / d["bt"].std(0)
    spec = {"naive": (bts, bts), "z-s": (np.c_[bts, d["zf"]], np.c_[bts, d["zf"]]),
            "part1": (np.c_[bts, d["zf"], d["A"]], np.c_[bts, d["zf"]]),
            "oracle": (np.c_[bts, d["zf"], d["zs"], d["A"]], np.c_[bts, d["zf"], d["zs"]])}
    feats = {v: spec[v][1] for v in variants}
    ridge = {v: Ridge(spec[v][0], d["fold"]) for v in variants}
    if rep == "b":
        cz = np.load(CAL / "calibration_b.npz")
        S.update(vec={k[5:]: cz[k] * d["sd_x"] for k in cz.files if k.startswith("vecs_")},
                 Lsb=np.linalg.cholesky(cz["Sigma_b"] + 1e-9 * np.eye(dim)), med=float(cz["diffuse_target"]),
                 target_g=TARGET_G, target_u=TARGET_U, gamma2=GAMMA2, outd=OUTD)
    else:
        ch = json.loads((OUTD_H / "calibration_h.json").read_text())
        v48 = np.load(OUTD_H / "calibration_h.npz")["vec_I48_raw"]
        gj = ROT_H / "gamma_h.json"
        S.update(vec={c: v48 for c in COUNTS}, target_g={c: ch["target_g_I48"] for c in COUNTS},
                 target_u=ch["target_u_c1_median"],
                 gamma2=json.loads(gj.read_text())["pooled"]["Gamma2"] if gj.exists() else np.nan, outd=OUTD_H)
    S["rep"] = rep
    S["gpu_logistic"] = GPU_LOGISTIC[0]
    _, inv = np.unique(d["subj"], return_inverse=True)
    gap5 = np.searchsorted(np.quantile(d["log_gap"], [0.2, 0.4, 0.6, 0.8]), d["log_gap"])
    age3 = np.searchsorted(np.quantile(d["age"], [1 / 3, 2 / 3]), d["age"])
    strata = {"part1": (d["fold"] * 5 + gap5) * 3 + age3}
    strata["oracle"] = strata["part1"] * 2 + d["ed1"]
    kap = np.where(d["ed1"], KAPPA_ED, KAPPA_ADM)
    order = np.argsort(inv, kind="stable")
    cnt = np.bincount(inv)
    S.update(d=d, n=n, dim=dim, D=D, M=M, sdD=D.std(0), feats=feats, ridge=ridge, inv=inv, strata=strata,
             Rb={e: r.resid(D) for e, r in ridge.items()}, bts=bts, age_z=(d["age"] - d["age"].mean()) / d["age"].std(),
             kap=kap, kbar=float(kap.mean()), wd=np.where(d["ed1"], DRIVER_OR, 1.0), order=order, start=np.r_[0, np.cumsum(cnt)[:-1]], cnt=cnt,
             Smat=sp.csr_matrix((np.ones(n), (inv, np.arange(n))), shape=(inv.max() + 1, n)))
    S["kbar_d"] = float((S["wd"] * kap).sum() / S["wd"].sum())


def logistic_cf_gpu(X, a, folds, iters=25, l2=1e-3):
    """latent.logistic_cf (same IRLS, ridge, start and stopping rule) in float64 on the GPU, for h's 256-dim state."""
    import torch
    dev = torch.device("cuda")
    Xa = torch.as_tensor(np.c_[X, np.ones(len(X))], device=dev, dtype=torch.float64)
    at = torch.as_tensor(a, device=dev, dtype=torch.float64)
    p = Xa.shape[1]
    R = torch.diag(torch.cat([torch.ones(p - 1), torch.zeros(1)])).to(dev, torch.float64)
    pi = np.empty(len(a))
    for f in range(N_FOLDS):
        trn = folds != f
        tr = torch.as_tensor(trn, device=dev)
        Xt, yt = Xa[tr], at[tr]
        ntr = Xt.shape[0]
        w = torch.zeros(p, device=dev, dtype=torch.float64)
        m = float(yt.mean())
        w[-1] = np.log(m / (1 - m))
        for _ in range(iters):
            mu = torch.sigmoid(Xt @ w)
            reg = w.clone()
            reg[-1] = 0
            g = Xt.T @ (mu - yt) + l2 * ntr * reg
            H = (Xt * (mu * (1 - mu))[:, None]).T @ Xt + l2 * ntr * R
            step = torch.linalg.solve(H, g)
            w -= step
            if float(step.abs().max()) < 1e-8:
                break
        pi[~trn] = torch.sigmoid(Xa[~tr] @ w).cpu().numpy()
    return np.clip(pi, 1e-6, 1 - 1e-6)


def fit_propensity(F, a):
    return (logistic_cf_gpu if S.get("gpu_logistic") else logistic_cf)(F, a, S["d"]["fold"])


def infer(ra, comps, coefs, rng, pi=None, n_crt=500, n_boot=400, n_perm=0, strata=None, split=HALVES):
    """Revision-11 inference with the revision-12 CRT. The residual matrix is R = Σ_j coef_j comps_j, with one tuple of
    coefficients per key of `coefs`. Returns {key: {T, var, lo, hi, perc_lo, perc_hi, p_crt[, p_perm]}}."""
    inv, pf_row, M, Sm = S["inv"], S["d"]["fold"], S["M"], S["Smat"]
    P = Sm.shape[0]
    pf = np.zeros(P, int)
    pf[inv] = pf_row
    nums = [Sm @ (ra[:, None] * C) for C in comps]
    den = Sm @ (ra ** 2)
    halves = [np.flatnonzero(np.isin(pf, s)) for s in split]
    rows_h = [np.isin(pf_row, s) for s in split]
    Ws = [rng.multinomial(len(h), np.full(len(h), 1 / len(h)), size=n_boot).astype(np.float64) for h in halves]
    crt = None
    if pi is not None:
        Ak = (rng.random((n_crt, len(ra))) < pi).astype(np.float64) - pi[None]
        crt = [([Ak[:, rh] @ C[rh] for C in comps], (Ak[:, rh] ** 2).sum(1)) for rh in rows_h]
        del Ak
    perm = None
    if n_perm:
        idx = strata_perms(rng, strata, n_perm)
        perm = [([ra[idx][:, rh] @ C[rh] for C in comps], np.full(n_perm, (ra[rh] ** 2).sum())) for rh in rows_h]

    def null_T(parts, cf):
        th = [sum(c * x for c, x in zip(cf, pp)) / dk[:, None] for pp, dk in parts]
        return np.einsum("ki,ij,kj->k", th[0], M, th[1])

    out = {}
    for key, cf in coefs.items():
        nb = sum(c * x for c, x in zip(cf, nums))
        th = [nb[h].sum(0) / den[h].sum() for h in halves]
        draws = [(W @ nb[h]) / (W @ den[h])[:, None] for W, h in zip(Ws, halves)]
        V = [np.cov(d_, rowvar=False) for d_ in draws]
        tr = float(np.trace(M @ V[0] @ M @ V[1]))
        var = max(float(th[1] @ M @ V[0] @ M @ th[1]) - tr, 0.0) + max(float(th[0] @ M @ V[1] @ M @ th[0]) - tr, 0.0) + tr
        T = float(th[0] @ M @ th[1])
        Tb = np.einsum("bi,ij,bj->b", draws[0], M, draws[1])
        sd_ = np.sqrt(var)
        rec = {"T": T, "var": var, "lo": T - 1.96 * sd_, "hi": T + 1.96 * sd_,
               "perc_lo": float(np.percentile(Tb, 2.5)), "perc_hi": float(np.percentile(Tb, 97.5))}
        if crt is not None:
            rec["p_crt"] = float((1 + (null_T(crt, cf) >= T).sum()) / (1 + n_crt))
        if perm is not None:
            rec["p_perm"] = float((1 + (null_T(perm, cf) >= T).sum()) / (1 + n_perm))
        out[key] = rec
    return out


def risk_weights(rng):
    w = rng.standard_normal(S["dim"])
    proj = S["bts"] @ (w / np.linalg.norm(w))
    score = (proj / proj.std() + S["age_z"]) / np.sqrt(2)
    return np.exp(ALPHA * score / score.std())


def onsets(rng, n_target, recorded, gamma=0.0):
    """Event pairs and recorded labels. With prob. gamma an event follows the driver, else the risk profile. With
    `recorded` (G, GC) events are recorded with κ by setting(t+1), and the number of events is n_target / E[κ]."""
    n = S["n"]
    kb = gamma * S["kbar_d"] + (1 - gamma) * S["kbar"]
    n_ev = int(round(n_target / kb)) if recorded else n_target
    n_d = rng.binomial(n_ev, gamma) if gamma > 0 else 0
    wt = risk_weights(rng)
    ev_d = rng.choice(n, size=n_d, replace=False, p=S["wd"] / S["wd"].sum()) if n_d else np.array([], int)
    wt[ev_d] = 0
    ev = np.r_[ev_d, rng.choice(n, size=n_ev - n_d, replace=False, p=wt / wt.sum())].astype(int)
    e = np.zeros(n)
    e[ev] = 1
    a = e.copy()
    if recorded:
        a[ev] = (rng.random(n_ev) < S["kap"][ev]).astype(float)
    return e, a, n_d


def solve_c(Rf, Rn, ev, target):
    """Smallest c >= 0 with ratio(Rf + c Rn) = target (ratio of event-pair to other-pair mean squared residual, in
    stable-pair change sd units); 0 if the ratio is already at or above the target."""
    s = S["sdD"]
    Rf, Rn = Rf / s, Rn / s
    m = np.zeros(len(Rf), bool)
    m[ev] = True
    q = lambda X, Y, mk: float((X[mk] * Y[mk]).sum(1).mean())
    N = (q(Rn, Rn, m), 2 * q(Rf, Rn, m), q(Rf, Rf, m))
    Dn = (q(Rn, Rn, ~m), 2 * q(Rf, Rn, ~m), q(Rf, Rf, ~m))
    r0 = N[2] / Dn[2]
    if r0 >= target:
        return 0.0, r0
    roots = np.roots([N[0] - target * Dn[0], N[1] - target * Dn[1], N[2] - target * Dn[2]])
    roots = np.real(roots[np.isreal(roots) & (np.real(roots) > 0)])
    return float(roots.min()), r0


def ratio(R, ev):
    s = S["sdD"]
    m = np.zeros(len(R), bool)
    m[ev] = True
    q = ((R / s) ** 2).sum(1)
    return float(q[m].mean() / q[~m].mean())


def eval_code(rng, e, a, vec, scales, target, perm=False, quarters=False, estimators=("part1", "oracle")):
    """One injected code: noise, then both estimators. Returns a list of row dicts (one per estimator x scale)."""
    ev = np.flatnonzero(e)
    ridge, d = S["ridge"], S["d"]
    Re = {est: ridge[est].resid(e)[:, :1] * vec[None] for est in estimators} if vec is not None else None
    cs, r0s, r1s, Rn = {}, {}, {}, None
    if target is not None:
        pick = rng.integers(S["cnt"].size, size=len(ev))
        donor = S["order"][S["start"][pick] + (rng.random(len(ev)) * S["cnt"][pick]).astype(int)]
        eta = np.zeros_like(S["D"])
        eta[ev] = S["Rb"]["part1"][donor]
        Rn = {est: ridge[est].resid(eta) for est in estimators}
        for sc in scales:
            Rf = S["Rb"]["part1"] + (sc * Re["part1"] if vec is not None else 0)
            cs[sc], r0s[sc] = solve_c(Rf, Rn["part1"], ev, target)
            r1s[sc] = ratio(Rf + cs[sc] * Rn["part1"], ev)
    rows = []
    for est in estimators:
        F = S["feats"][est]
        pi = fit_propensity(F, a)
        ra = a - pi
        comps = [S["Rb"][est]] + ([Re[est]] if vec is not None else []) + ([Rn[est]] if Rn is not None else [])
        cf = lambda sc: (1.0,) + ((sc,) if vec is not None else ()) + ((cs[sc],) if Rn is not None else ())
        res = infer(ra, comps, {sc: cf(sc) for sc in scales}, rng, pi=pi, n_perm=200 if perm else 0,
                    strata=S["strata"].get(est))
        q = infer(ra, comps, {1.0: cf(1.0)}, rng, split=QUARTERS) if quarters and 1.0 in scales else {}
        auc_p = roc_auc_score(a, pi)
        for sc, r in res.items():
            truth = float((sc * vec) @ S["M"] @ (sc * vec)) if vec is not None else 0.0
            rows.append({"estimator": est, "scale": sc, "onsets": int(a.sum()), "events": int(e.sum()),
                         "propensity_auroc": float(auc_p), "psi2_true": truth, **r,
                         "noise_c": cs.get(sc, 0.0), "ratio_before": r0s.get(sc, np.nan), "ratio_after": r1s.get(sc, np.nan),
                         "q_perc_lo": q[1.0]["perc_lo"] if sc == 1.0 and q else np.nan,
                         "q_lo": q[1.0]["lo"] if sc == 1.0 and q else np.nan})
    return rows


def draw_vecs(rng):
    if S["rep"] == "h":
        return dict(S["vec"])
    ud = rng.standard_normal(S["dim"])
    return {**S["vec"], "diffuse": (S["Lsb"] @ (ud / np.linalg.norm(ud)) * np.sqrt(S["med"])) * S["d"]["sd_x"]}


def work_validate(dr):
    rng = np.random.default_rng([1210, dr])
    rows, labs = [], {}
    for c, n_rec in COUNTS.items():
        e, a, _ = onsets(rng, n_rec, recorded=False)
        labs[f"U|{c}|{dr}"] = np.flatnonzero(a)
        for r in eval_code(rng, e, a, None, [0.0], None, perm=True):
            rows.append({"draw": dr, "kind": "U", "code": c, **r})
    return rows, labs


def work_gamma0(dr):
    rng = np.random.default_rng([1211, dr])
    vecs = draw_vecs(rng)
    rows, labs = [], {}
    for kind in ("G", "U"):
        for c, n_rec in COUNTS.items():
            e, a, _ = onsets(rng, n_rec, recorded=(kind == "G"))
            labs[f"{kind}|{c}|{dr}|ev"], labs[f"{kind}|{c}|{dr}|rec"] = np.flatnonzero(e), np.flatnonzero(a)
            out = (eval_code(rng, e, a, vecs[c], SCALES, S["target_g"][c], quarters=True) if kind == "G"
                   else eval_code(rng, e, a, None, [0.0], S["target_u"]))
            rows += [{"draw": dr, "kind": kind, "code": c, **r} for r in out]
    return rows, labs


def work_sweep(args):
    dr, scale = args
    rng = np.random.default_rng([1212, dr])
    vecs = draw_vecs(rng)
    rows, labs = [], {}
    jobs = [("G", c, n_rec, None) for c, n_rec in COUNTS.items()] + [("U", c, n_rec, None) for c, n_rec in COUNTS.items()]
    jobs += [(k, c, COUNTS[c], g) for g in GAMMAS for k, cs in (("UC", COUNTS), ("GC", GC_CODES)) for c in cs]
    for kind, c, n_rec, g in jobs:
        eff = kind in ("G", "GC")
        e, a, n_d = onsets(rng, n_rec, recorded=eff, gamma=g or 0.0)
        tag = f"{kind}|{c}|{dr}|{g}"
        labs[tag + "|ev"], labs[tag + "|rec"] = np.flatnonzero(e), np.flatnonzero(a)
        target = S["target_g"][c] if kind == "G" else S["target_u"]
        out = eval_code(rng, e, a, vecs[c] if eff else None, [scale] if eff else [0.0], target)
        rows += [{"draw": dr, "kind": kind, "code": c, "gamma_mix": g, "driver_events": int(n_d), **r} for r in out]
    print(f"sweep draw {dr} done", flush=True)
    return rows, labs


def run_pool(fn, items, workers):
    ctx = mp.get_context("fork")
    with ctx.Pool(min(workers, len(items))) as pool:
        res = pool.map(fn, items, chunksize=1)
    rows = [r for rr, _ in res for r in rr]
    labs = {k: v for _, ll in res for k, v in ll.items()}
    return pd.DataFrame(rows), labs


def cp(k, n):
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return lo, hi


def cmd_validate(args):
    setup(args.rep)
    df, labs = run_pool(work_validate, list(range(args.draws)), args.workers)
    S["outd"].mkdir(parents=True, exist_ok=True)
    df.to_csv(S["outd"] / "crt_validation_rows.csv", index=False)
    np.savez_compressed(S["outd"] / "crt_validation_labels.npz", **{k.replace("|", "__"): v for k, v in labs.items()})
    summ, stop = {}, False
    for est in ("part1", "oracle"):
        u = df[df.estimator == est]
        n_ = len(u)
        rec = {"n": n_, "propensity_auroc_median": float(u.propensity_auroc.median())}
        for name, col in (("crt", "p_crt"), ("permutation", "p_perm")):
            k = int((u[col] < 0.05).sum())
            rec[name] = {"rejections": k, "rate": k / n_, "cp95": cp(k, n_)}
        rec["new_lo_gt_0"] = float((u.lo > 0).mean())
        rec["crt_p_uniform_ks"] = float(kstest(u.p_crt, "uniform").pvalue)
        summ[est] = rec
        stop |= rec["crt"]["cp95"][0] > 0.05
    out = {"draws": args.draws, "codes_per_estimator": int(len(df) / 2), "summary": summ, "stop": bool(stop),
           "decision": ("STOP: the CRT's binomial CI excludes 0.05 from above" if stop else
                        "pass: no estimator's CRT CI excludes 0.05 from above")}
    (S["outd"] / "crt_validation.json").write_text(json.dumps(out, indent=1))
    L = [f"CRT validation on U codes of the γ_mix = 0 generator (no added noise), {args.draws} draws x 6 = "
         f"{out['codes_per_estimator']} per estimator, rejection at p < 0.05, Clopper-Pearson 95% CI.", "",
         "| Estimator | CRT rejections | CRT rate [95% CI] | Permutation rejections | Permutation rate [95% CI] | new lower bound > 0 | KS p (CRT p ~ U(0,1)) |",
         "|---|---|---|---|---|---|---|"]
    for est, r in summ.items():
        L.append(f"| {est} | {r['crt']['rejections']}/{r['n']} | {r['crt']['rate']:.3f} [{r['crt']['cp95'][0]:.3f}, "
                 f"{r['crt']['cp95'][1]:.3f}] | {r['permutation']['rejections']}/{r['n']} | {r['permutation']['rate']:.3f} "
                 f"[{r['permutation']['cp95'][0]:.3f}, {r['permutation']['cp95'][1]:.3f}] | {r['new_lo_gt_0']:.3f} | "
                 f"{r['crt_p_uniform_ks']:.3f} |")
    L += ["", f"**{out['decision']}.**"]
    (S["outd"] / "crt_validation.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def g_vs_u_auc(g, u, draws):
    out = []
    for k in draws:
        tg, tu = g[g.draw == k]["T"].values, u[u.draw == k]["T"].values
        out.append(roc_auc_score(np.r_[np.ones(len(tg)), np.zeros(len(tu))], np.r_[tg, tu]))
    return out


def cmd_gamma0(args):
    setup(args.rep)
    if args.summary_only:
        df = pd.read_csv(S["outd"] / "gamma0_rev12_rows.csv")
    else:
        df, labs = run_pool(work_gamma0, list(range(args.draws)), args.workers)
        S["outd"].mkdir(parents=True, exist_ok=True)
        df.to_csv(S["outd"] / "gamma0_rev12_rows.csv", index=False)
        np.savez_compressed(S["outd"] / "gamma0_rev12_labels.npz", **{k.replace("|", "__"): v for k, v in labs.items()})
    U = df[df.kind == "U"]
    summ = {}
    for est in ("oracle", "part1"):
        u = U[U.estimator == est]
        for sc in SCALES:
            g = df[(df.kind == "G") & (df.estimator == est) & (df.scale == sc)]
            aucs = g_vs_u_auc(g, u, range(args.draws))
            summ[f"{est}|{sc}"] = {"auroc_mean": float(np.mean(aucs)), "auroc_min": float(np.min(aucs)),
                                   "G_lo_gt_0": float((g.lo > 0).mean()), "G_lo_gt_Gamma": float((g.lo > S["gamma2"]).mean()),
                                   "G_crt_p_lt_05": float((g.p_crt < 0.05).mean()),
                                   "ratio_before_median": float(g.ratio_before.median()),
                                   "ratio_after_median": float(g.ratio_after.median()),
                                   "by_vector_lo_gt_Gamma": g.groupby("code").apply(lambda x: float((x.lo > S["gamma2"]).mean())).to_dict()}
        k = int((u.p_crt < 0.05).sum())
        summ[f"{est}|U"] = {"U_lo_gt_0": float((u.lo > 0).mean()), "U_lo_gt_Gamma": float((u.lo > S["gamma2"]).mean()),
                            "U_crt_rejections": k, "n": len(u), "U_crt_cp95": cp(k, len(u)),
                            "ratio_before_median": float(u.ratio_before.median()),
                            "ratio_after_median": float(u.ratio_after.median())}
    chosen = next((sc for sc in SCALES if summ[f"oracle|{sc}"]["auroc_mean"] >= 0.9), None)
    g1 = df[(df.kind == "G") & (df.scale == 1.0)]
    realism = {est: {"quarters_percentile_lo_gt_0": float((g1[g1.estimator == est].q_perc_lo > 0).mean()),
                     "quarters_new_lo_gt_0": float((g1[g1.estimator == est].q_lo > 0).mean()),
                     "by_vector_quarters_percentile": g1[g1.estimator == est].groupby("code").apply(
                         lambda x: float((x.q_perc_lo > 0).mean())).to_dict()} for est in ("oracle", "part1")}
    noise = (df[df.estimator == "part1"].groupby(["kind", "code", "scale"])[["ratio_before", "ratio_after", "noise_c"]]
             .median().round(3).reset_index())
    out = {"draws": args.draws, "n_pool": S["n"], "kappa_bar": S["kbar"],
           "propensity_auroc_median": float(df.propensity_auroc.median()), "summary": summ, "chosen_scale": chosen,
           "rule_c": (f"C2 is read at scale {chosen}x" if chosen else "no scale reaches oracle AUROC 0.9: C2 cannot be read"),
           "realism_1x": realism, "real_folds_10_19": ("1" if S["rep"] == "b" else "0") + " of 5 positive controls (quarters, percentile lower bound > 0)",
           "noise": noise.to_dict(orient="records")}
    (S["outd"] / "gamma0_rev12.json").write_text(json.dumps(out, indent=1))
    L = [f"Semi-synthetic test on {S['rep']} at γ_mix = 0, revision 12 (κ < 1 on G codes, realistic noise, CRT): {S['n']:,} real "
         f"stable pairs, {args.draws} draws x (6 G + 6 U), κ̄ {S['kbar']:.3f}, median propensity AUROC "
         f"{out['propensity_auroc_median']:.3f}.", "",
         "| Estimator | Scale | G-vs-U AUROC, mean (min) over draws | G: lower bound > 0 | > Γ̂² | CRT p < 0.05 | Part 1 ratio before → after noise (median) |",
         "|---|---|---|---|---|---|---|"]
    for est in ("oracle", "part1"):
        for sc in SCALES:
            s_ = summ[f"{est}|{sc}"]
            L.append(f"| {est} | {sc}x | {s_['auroc_mean']:.3f} ({s_['auroc_min']:.3f}) | {s_['G_lo_gt_0']:.2f} | "
                     f"{s_['G_lo_gt_Gamma']:.2f} | {s_['G_crt_p_lt_05']:.2f} | {s_['ratio_before_median']:.2f} → "
                     f"{s_['ratio_after_median']:.2f} |")
        su = summ[f"{est}|U"]
        L.append(f"| {est} | U codes (θ = 0) | — | {su['U_lo_gt_0']:.2f} | {su['U_lo_gt_Gamma']:.2f} | "
                 f"{su['U_crt_rejections']}/{su['n']} [{su['U_crt_cp95'][0]:.3f}, {su['U_crt_cp95'][1]:.3f}] | "
                 f"{su['ratio_before_median']:.2f} → {su['ratio_after_median']:.2f} |")
    L += ["", f"**Rule (c): {out['rule_c']}.**", "",
          "Realism check at 1x, report-only (quarters 10-14 vs 15-19, percentile lower bound > 0; real: " + out["real_folds_10_19"] + "): " + "; ".join(
              f"{est} {r['quarters_percentile_lo_gt_0']:.2f} (new interval {r['quarters_new_lo_gt_0']:.2f}; by vector "
              + ", ".join(f"{k} {v:.2f}" for k, v in r["by_vector_quarters_percentile"].items()) + ")"
              for est, r in realism.items()), "", "Noise (Part 1, median over draws):", "",
          noise.to_markdown(index=False)]
    (S["outd"] / "gamma0_rev12.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def f1_threshold(tg, tu):
    """Threshold on T maximising the G-vs-U F1 (predict grounded when T >= t)."""
    best, bt = -1.0, np.inf
    for t in np.unique(np.r_[tg, tu]):
        tp, fp = (tg >= t).sum(), (tu >= t).sum()
        f1 = 2 * tp / (2 * tp + fp + (len(tg) - tp))
        if f1 > best:
            best, bt = f1, t
    return bt


def cmd_sweep(args):
    setup(args.rep)
    if args.summary_only:
        df = pd.read_csv(S["outd"] / "sweep_rows.csv")
    else:
        df, labs = run_pool(work_sweep, [(k, args.scale) for k in range(args.draws)], args.workers)
        S["outd"].mkdir(parents=True, exist_ok=True)
        df.to_csv(S["outd"] / "sweep_rows.csv", index=False)
        np.savez_compressed(S["outd"] / "sweep_labels.npz", **{k.replace("|", "__"): v for k, v in labs.items()})
    thr = {(est, k): f1_threshold(df[(df.kind == "G") & (df.estimator == est) & (df.draw == k)]["T"].values,
                                  df[(df.kind == "U") & (df.estimator == est) & (df.draw == k)]["T"].values)
           for est in ("part1", "oracle") for k in range(args.draws)}
    df["above_f1_thr"] = [r.T >= thr[(r.estimator, r.draw)] for r in df.itertuples()]
    df["bias_ratio"] = np.where(df.psi2_true > 0, np.sqrt(np.clip(df["T"], 0, None)) / np.sqrt(df.psi2_true.where(df.psi2_true > 0)), np.nan)
    rows = []
    for est in ("part1", "oracle"):
        for kind in ("G", "U"):
            x = df[(df.kind == kind) & (df.estimator == est)]
            rows.append({"estimator": est, "kind": kind, "gamma_mix": "—", "n": len(x),
                         "above_F1_threshold": x.above_f1_thr.mean(), "lo_gt_Gamma": (x.lo > S["gamma2"]).mean(),
                         "crt_p_lt_05": (x.p_crt < 0.05).mean(),
                         "bias_ratio_median": x.bias_ratio.median() if kind == "G" else np.nan,
                         "T_over_truth_mean": (x["T"] / x.psi2_true).mean() if kind == "G" else np.nan})
        for g in GAMMAS:
            for kind in ("UC", "GC"):
                x = df[(df.kind == kind) & (df.estimator == est) & (df.gamma_mix == g)]
                rows.append({"estimator": est, "kind": kind, "gamma_mix": g, "n": len(x),
                             "above_F1_threshold": x.above_f1_thr.mean(), "lo_gt_Gamma": (x.lo > S["gamma2"]).mean(),
                             "crt_p_lt_05": (x.p_crt < 0.05).mean(),
                             "bias_ratio_median": x.bias_ratio.median() if kind == "GC" else np.nan,
                             "T_over_truth_mean": (x["T"] / x.psi2_true).mean() if kind == "GC" else np.nan,
                             "recorded_onsets_mean": x.onsets.mean(), "driver_events_mean": x.driver_events.mean()})
    tab = pd.DataFrame(rows)
    k_uc = {(est, g): int((df[(df.kind == "UC") & (df.estimator == est) & (df.gamma_mix == g)].lo > S["gamma2"]).sum())
            for est in ("part1", "oracle") for g in GAMMAS}
    out = {"draws": args.draws, "scale": args.scale, "gammas": GAMMAS, "driver_odds_ratio": DRIVER_OR,
           "kbar_driver": S["kbar_d"], "kbar_risk": S["kbar"], "table": tab.to_dict(orient="records"),
           "uc_false_grounding_lo_gt_Gamma_cp95": {f"{e}|{g}": cp(k, args.draws * 6) for (e, g), k in k_uc.items()}}
    (S["outd"] / "sweep.json").write_text(json.dumps(out, indent=1, default=float))
    L = [f"γ_mix sweep at {args.scale}x, revision 12 (Part 1 estimator and oracle only): {args.draws} draws x (6 G + 6 U "
         f"+ 5 γ_mix x (6 UC + 3 GC)); driver odds ratio {DRIVER_OR:g} for an ED-only next stay; E[κ] driver "
         f"{S['kbar_d']:.3f}, risk {S['kbar']:.3f}. UC false grounding: (i) T above the draw's G-vs-U F1 threshold, (ii) "
         f"lower bound > Γ̂² = {S['gamma2']}, (iii) CRT p < 0.05. GC effect bias: median √max(T, 0)/√ψ²_true.", "",
         tab.round(3).to_markdown(index=False)]
    (S["outd"] / "sweep.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ------------------------------------------------------------------------------------------------ revision 13
VARIANTS = ("naive", "z-s", "part1", "oracle")
H_I48 = ROOT / "outputs" / "phase05" / "4b_references" / "gain_clean" / "clean_bhm_h_z_I48.resid.npz"
C1 = ("G43", "J06", "K21", "K22", "K29", "K57", "K59", "K64", "K80", "N40", "Z96")


def onset_ratio(path):
    """Residual-variance ratio of a real gain.py run: mean squared standardised residual summed over dimensions, on
    recorded-onset pairs vs the other pairs (the rev-12 measurement)."""
    z = np.load(path)
    a = np.load(str(path).replace(".resid.npz", ".pairs.npz"))["a"] > 0.5
    q = (z["ry"].astype(np.float64) ** 2).sum(1)
    return float(q[a].mean() / q[~a].mean()), int(a.sum())


def cmd_calibrate_h(args):
    """Plan revision 13: h's I48 vector at its debiased norm (Ledoit-Wolf metric of h) and the noise targets on h."""
    from latent import halves_theta, theta_all_folds
    d = load_pool("h")
    M = metrics(d["bt1"] - d["bt"])[0]["lw"]
    th = theta_all_folds(H_I48)
    A, B = halves_theta(H_I48, 1.0)
    deb, plug = float(A @ M @ B), float(th @ M @ th)
    OUTD_H.mkdir(parents=True, exist_ok=True)
    c1 = {c: onset_ratio(ROT_H / "gain_c1" / f"clean_bhm_h_z_{c}.resid.npz") for c in C1}
    r48 = onset_ratio(H_I48)
    out = {"source": str(H_I48.relative_to(ROOT)), "metric": "Ledoit-Wolf on standardised stable-pair changes of h",
           "psi2_plug_in": plug, "psi2_debiased_halves": deb, "factor": float(np.sqrt(max(deb, 0) / plug)),
           "target_g_I48": r48[0], "I48_onsets": r48[1], "c1_ratios": {c: r[0] for c, r in c1.items()},
           "target_u_c1_median": float(np.median([r[0] for r in c1.values()])), "n_pool": int(len(d["bt"]))}
    if deb <= 0:
        out["stop"] = "debiased norm of I48 on h <= 0: stop and report (plan revision 13)"
    (OUTD_H / "calibration_h.json").write_text(json.dumps(out, indent=1))
    np.savez(OUTD_H / "calibration_h.npz", theta_raw=th, vec_I48_raw=th * out["factor"])
    print(json.dumps(out, indent=1))
    if deb <= 0:
        raise SystemExit(out["stop"])


def work_decompose(args):
    dr, gammas, scale = args
    # one task per draw (b) or per (draw, gamma_mix) when split (h, --split-gamma): own stream per task
    rng = np.random.default_rng([1213, dr] if len(gammas) > 1 else [1213, dr, int(round(gammas[0] * 100))])
    rows = []
    for g in gammas:
        for kind, codes in (("UC", list(COUNTS)), ("GC", list(GC_CODES))):
            for c in codes:
                tag = f"{kind}__{c}__{dr}__{g}"
                e, a = np.zeros(S["n"]), np.zeros(S["n"])
                e[S["labels"][tag + "__ev"]], a[S["labels"][tag + "__rec"]] = 1, 1
                eff = kind == "GC"
                out = eval_code(rng, e, a, S["vec"][c] if eff else None, [scale] if eff else [0.0], S["target_u"],
                                estimators=VARIANTS)
                rows += [{"draw": dr, "kind": kind, "code": c, "gamma_mix": g, **r} for r in out]
    print(f"decompose draw {dr} done", flush=True)
    return rows, {}


def cmd_decompose(args):
    setup(args.rep, variants=VARIANTS)
    gammas = [float(x) for x in args.gammas.split(",")]
    if args.summary_only:
        df = pd.read_csv(S["outd"] / "decompose_rows.csv")
    else:
        S["labels"] = dict(np.load(S["outd"] / "sweep_labels.npz"))
        tasks = ([(k, [g], args.scale) for k in range(args.draws) for g in gammas] if args.split_gamma
                 else [(k, gammas, args.scale) for k in range(args.draws)])
        df, _ = run_pool(work_decompose, tasks, args.workers)
        df.to_csv(S["outd"] / "decompose_rows.csv", index=False)
    G2 = S["gamma2"]
    df["bias_ratio"] = np.sqrt(df["T"].clip(lower=0)) / np.sqrt(df.psi2_true.where(df.psi2_true > 0))
    rows = []
    for g in gammas:
        for v in VARIANTS:
            u = df[(df.kind == "UC") & (df.estimator == v) & (df.gamma_mix == g)]
            x = df[(df.kind == "GC") & (df.estimator == v) & (df.gamma_mix == g)]
            k = int((u.p_crt < 0.05).sum())
            rows.append({"gamma_mix": g, "variant": v, "UC_n": len(u), "UC_crt_p_lt_05": k / len(u),
                         "UC_crt_cp95": "[{:.3f}, {:.3f}]".format(*cp(k, len(u))),
                         "UC_lo_gt_0": (u.lo > 0).mean(), "UC_lo_gt_Gamma": (u.lo > G2).mean(),
                         "UC_T_median": u["T"].median(), "UC_T_mean": u["T"].mean(),
                         "UC_prop_auroc": u.propensity_auroc.median(),
                         "GC_bias_median": x.bias_ratio.median(), "GC_bias_q10": x.bias_ratio.quantile(0.1),
                         "GC_bias_q90": x.bias_ratio.quantile(0.9), "GC_lo_gt_Gamma": (x.lo > G2).mean()})
    tab = pd.DataFrame(rows)
    out = {"rep": S["rep"], "draws": args.draws, "scale": args.scale, "gammas": gammas, "Gamma2": G2,
           "variants": {"naive": "b_t only (base and propensity)", "z-s": "[state, z without setting(t+1)] in both",
                        "part1": "z-s plus A in the base model (the Part 1 estimator)",
                        "oracle": "part1 plus setting(t+1) in both"}, "table": tab.to_dict(orient="records")}
    (S["outd"] / "decompose.json").write_text(json.dumps(out, indent=1, default=float))
    L = [f"Input-variant decomposition (plan revision 13, pre-declared secondary) on {S['rep']}: the rev-12 sweep's "
         f"γ_mix = {', '.join(map(str, gammas))} draws ({args.draws} draws x (6 UC + 3 GC)), scale {args.scale}x, noise "
         f"redrawn (seed 1213) and shared by the variants; Γ̂² = {G2:.4f}. GC bias = √max(T, 0)/√ψ²_true.", "",
         tab.round(3).to_markdown(index=False)]
    (S["outd"] / "decompose.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rev11", "crt-validate", "gamma0", "sweep", "calibrate-h", "decompose"])
    ap.add_argument("--rep", default="b", choices=["b", "h"])
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--gammas", default="1.0,0.5")
    ap.add_argument("--summary-only", action="store_true", help="recompute the summary from the saved rows")
    ap.add_argument("--split-gamma", action="store_true", help="decompose: one task per (draw, gamma_mix)")
    ap.add_argument("--gpu-logistic", action="store_true", help="propensity IRLS on the GPU (same algorithm)")
    a_ = ap.parse_args()
    GPU_LOGISTIC[0] = a_.gpu_logistic
    {"rev11": rev11, "crt-validate": cmd_validate, "gamma0": cmd_gamma0, "sweep": cmd_sweep,
     "calibrate-h": cmd_calibrate_h, "decompose": cmd_decompose}[a_.cmd](a_)

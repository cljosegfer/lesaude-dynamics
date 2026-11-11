"""
Part 2: Latent Environment Calibrated to b, and the 3.6 Power Check
==================================================================
go_no_go_plan.md Section 3, revision 10 (d), (e), (c), (g). Two commands:

calibrate   Measures on b (usable between-episode pairs: b and m at both ECGs, not same-episode), in units of b's
            cross-sectional SD per feature: the cross-sectional covariance V; the stable-pair change covariance Σ_b (the
            noise floor); the patient share π_p of b (one-way random-effects ANOVA over patients with >= 2 ECGs, as
            gonogo/part1/noise_floor.py); the linear R² of the stable-pair change from the current state (5-fold
            cross-fitted ridge); the stable fraction; the frequency q of an ED-only stay at t+1 on stable pairs and the
            mean shift β of the change with it; the per-patient counts of stable pairs (the clean pool); and the effect
            vectors to inject: I47, I48, I49 (R-learner θ under z on all 20 folds, 8_3d_b), the one-feature vector
            (I44's QRS-width component) and the two-feature vector (I48's HR and RR-irregularity components).
            -> outputs/part2/latent/calibration_b.npz, calibration_b.json

power       The b-calibrated generator (dim 39):
              x_t = p + s_t,  p ~ N(0, π_p V),  s_t ~ N(0, (1 - π_p) V)
              Δx  = -(1 - ρ) s_t + u β + ε + scale · e Δ_c,   u ~ Bern(q) the hidden context (ED-only stay at t+1),
              ε ~ N(0, Σ_ε), Σ_ε = Σ_b - (1 - ρ)² (1 - π_p) V - q (1 - q) β β' (negative eigenvalues clipped),
              1 - ρ chosen so that the linear R² of the change from x_t equals b's.
            The stable pool reproduces b's stable pairs per patient. Each code has its own analysis set (the clean
            subset): the pool plus its event pairs, on new patients whose p is tilted along a risk direction (the
            "risk profile" shortcut, propensity AUROC near the real ~0.7). G codes add scale x their vector; U codes
            add nothing. Weak labels (revision 10 e): a G event is recorded with probability κ = 0.9 at an admission
            and 0.9 x 0.13 at an ED-only stay; an unrecorded event stays in the analysis set labelled 0, effect
            included. The number of true events is set so that the recorded onsets match the real clean counts
            (I47 302, I48 777, I49 129, one-feature 185 as I44, two-feature 204 as I45, diffuse 351, the median
            passing cardiac code); each U code has the count of its G partner. γ = 0: labels do not depend on u.
            Estimators: the Part 1 estimator (cross-fitted ridge base model and logistic propensity on x_t over 20
            patient folds, R-learner θ, debiased ψ̂² with Ledoit-Wolf Σ of the stable-pair changes, patient bootstrap)
            and the oracle (the same, with u added to the base model and the propensity). ψ̂² on halves 0-9 vs 10-19
            (rule c) and on quarters 10-14 vs 15-19 (reporting protocol). Scales 0.5, 1, 2, 4.
            -> outputs/part2/latent/power_b.json, power_b.md

Example
-------
python -I gonogo/part2/latent.py calibrate
python -I gonogo/part2/latent.py power --draws 8
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gonogo" / "part1"))
from linkage import load_z  # noqa: E402
from noise_floor import anova_share  # noqa: E402
from pairs import OUT as P1, load as load_pairs  # noqa: E402

OUTD = ROOT / "outputs" / "part2" / "latent"
G3D = ROOT / "outputs" / "phase05" / "8_3d_b" / "gain"
SCALES = [0.5, 1.0, 2.0, 4.0]
GAMMA2 = 0.014                       # the real margin of 3e (pooled DL, Ledoit-Wolf)
KAPPA_ADM, KAPPA_ED = 0.9, 0.9 * 0.13
COUNTS = {"I47": 302, "I48": 777, "I49": 129, "one-feature": 185, "two-feature": 204, "diffuse": 351}
N_FOLDS, N_BOOT = 20, 500
ALPHA = 0.75          # risk-profile strength on the observed state (propensity AUROC near the real ~0.7)


# ---------------------------------------------------------------- calibration
def theta_all_folds(path):
    z = np.load(path)
    ra, ry, sd = z["ra"].astype(np.float64), z["ry"].astype(np.float64), z["sd"].astype(np.float64)
    return (ra[:, None] * ry).sum(0) / (ra ** 2).sum() * sd          # raw b units


def calibrate():
    pairs, _, _, _, b_lookup, b, names = load_pairs()
    z = load_z()
    mz = np.load(P1 / "m.npz")
    lk = pd.Series(np.arange(len(mz["rows"])), index=mz["rows"])
    has_m = mz["has_m"][lk.loc[pairs["idx_t"].values].values] & mz["has_m"][lk.loc[pairs["idx_t1"].values].values]
    use = pairs["has_b"].values & has_m & ~z["same_episode"].values
    P = pairs[use]
    it, it1 = b_lookup.loc[P["idx_t"].values].values, b_lookup.loc[P["idx_t1"].values].values
    rows = np.unique(np.concatenate([it, it1]))
    sd_x = b[rows].std(0)
    B = b / sd_x                                                       # standardised b
    V = np.cov(B[rows], rowvar=False)
    stable = P["stable"].values
    D = B[it1] - B[it]
    Sigma_b = np.cov(D[stable], rowvar=False)
    # patient share of b over the ECGs of patients with >= 2 of them
    ecg = pd.DataFrame({"row": np.concatenate([it, it1]), "subj": np.concatenate([P["subject_id"].values] * 2)})
    ecg = ecg.drop_duplicates("row")
    multi = ecg["subj"].map(ecg["subj"].value_counts()).values >= 2
    pi_p = anova_share(B[ecg["row"].values[multi]], ecg["subj"].values[multi])
    # linear R² of the stable change from the current state (5-fold cross-fitted ridge)
    Xs, Ds = B[it][stable], D[stable]
    subj = P["subject_id"].values[stable]
    fold = pd.Series(subj).map({s: i % 5 for i, s in enumerate(np.unique(subj))}).values
    pred = np.zeros_like(Ds)
    for f in range(5):
        tr, te = fold != f, fold == f
        Xa = np.c_[Xs[tr], np.ones(tr.sum())]
        W = np.linalg.solve(Xa.T @ Xa + 1e-3 * len(Xa) * np.eye(Xa.shape[1]), Xa.T @ Ds[tr])
        pred[te] = np.c_[Xs[te], np.ones(te.sum())] @ W
    r2 = 1 - ((Ds - pred) ** 2).sum() / ((Ds - Ds.mean(0)) ** 2).sum()
    ed = (z["setting_t1"].values[use] == "ed_only")
    q = float(ed[stable].mean())
    beta = Ds[ed[stable]].mean(0) - Ds[~ed[stable]].mean(0)
    per_patient = pd.Series(subj).value_counts().values
    # effect vectors (standardised units)
    vec = {c: theta_all_folds(G3D / f"clean_bm_b_z_{c}.resid.npz") / sd_x for c in ("I47", "I48", "I49")}
    j = {n: k for k, n in enumerate(names)}
    t44 = theta_all_folds(G3D / "clean_bm_b_z_I44.resid.npz") / sd_x
    one = np.zeros(len(names)); one[j["qrs_width"]] = t44[j["qrs_width"]]
    two = np.zeros(len(names)); two[[j["hr"], j["rr_irr"]]] = vec["I48"][[j["hr"], j["rr_irr"]]]
    vec["one-feature"], vec["two-feature"] = one, two
    lw = LedoitWolf().fit(Ds / Ds.std(0))
    S_inv = np.linalg.inv(lw.covariance_) / np.outer(Ds.std(0), Ds.std(0))
    psi2 = {c: float(v @ S_inv @ v) for c, v in vec.items()}
    cal = {"dim": int(len(names)), "n_usable_pairs": int(use.sum()), "stable_fraction": float(stable.mean()),
           "n_stable_pairs": int(stable.sum()), "n_stable_patients": int(len(per_patient)),
           "patient_share_b": float(pi_p), "r2_change_from_state": float(r2), "q_ed_only_t1": q,
           "beta_norm": float(np.linalg.norm(beta)), "trace_V": float(np.trace(V)), "trace_Sigma_b": float(np.trace(Sigma_b)),
           "true_psi2_of_vectors_at_1x": psi2, "features": names}
    OUTD.mkdir(parents=True, exist_ok=True)
    np.savez(OUTD / "calibration_b.npz", V=V, Sigma_b=Sigma_b, beta=beta, per_patient=per_patient,
             **{f"vec_{k}": v for k, v in vec.items()})
    (OUTD / "calibration_b.json").write_text(json.dumps(cal, indent=1))
    print(json.dumps({k: v for k, v in cal.items() if k != "features"}, indent=1))


# ---------------------------------------------------------------- generator and estimators
def psd_clip(S):
    ev, U = np.linalg.eigh(S)
    return (U * np.clip(ev, 0, None)) @ U.T, float(np.clip(-ev, 0, None).sum() / np.abs(ev).sum())


def logistic_cf(X, a, folds, iters=25, l2=1e-3):
    """Cross-fitted logistic regression by IRLS; returns out-of-fold propensities."""
    Xa = np.c_[X, np.ones(len(X))]
    pi = np.empty(len(a))
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        w = np.zeros(Xa.shape[1])
        w[-1] = np.log(a[tr].mean() / (1 - a[tr].mean()))
        for _ in range(iters):
            mu = 1 / (1 + np.exp(-Xa[tr] @ w))
            g = Xa[tr].T @ (mu - a[tr]) + l2 * tr.sum() * np.r_[w[:-1], 0]
            H = (Xa[tr] * (mu * (1 - mu))[:, None]).T @ Xa[tr] + l2 * tr.sum() * np.diag(np.r_[np.ones(len(w) - 1), 0])
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        pi[te] = 1 / (1 + np.exp(-Xa[te] @ w))
    return np.clip(pi, 1e-6, 1 - 1e-6)


def ridge_cf(X, Y, folds, lam=1e-3):
    """Cross-fitted ridge residuals of Y (n, d) on X."""
    Xa = np.c_[X, np.ones(len(X))]
    R = np.empty_like(Y)
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        W = np.linalg.solve(Xa[tr].T @ Xa[tr] + lam * tr.sum() * np.eye(Xa.shape[1]), Xa[tr].T @ Y[tr])
        R[te] = Y[te] - Xa[te] @ W
    return R


def psi2_halves(ra, R, subj, folds, M, W, inv, split):
    """ψ̂² and its bootstrap CI (shared multinomial weights W over patients)."""
    P = W.shape[1]
    num = np.zeros((P, R.shape[1]))
    np.add.at(num, inv, ra[:, None] * R)
    den = np.bincount(inv, weights=ra ** 2, minlength=P)
    pf = np.zeros(P, int)
    pf[inv] = folds
    A, B = np.isin(pf, split[0]), np.isin(pf, split[1])
    tA, tB = num[A].sum(0) / den[A].sum(), num[B].sum(0) / den[B].sum()
    bA = (W[:, A] @ num[A]) / (W[:, A] @ den[A])[:, None]
    bB = (W[:, B] @ num[B]) / (W[:, B] @ den[B])[:, None]
    boot = np.einsum("bi,ij,bj->b", bA, M, bB)
    lo, hi = np.percentile(boot[np.isfinite(boot)], [2.5, 97.5])
    return float(tA @ M @ tB), float(lo), float(hi)


def power(draws):
    cal = json.loads((OUTD / "calibration_b.json").read_text())
    z = np.load(OUTD / "calibration_b.npz")
    V, Sb, beta, per_patient = z["V"], z["Sigma_b"], z["beta"], z["per_patient"]
    vec = {k[4:]: z[k] for k in z.files if k.startswith("vec_")}
    d, pi_p, q, r2 = cal["dim"], cal["patient_share_b"], cal["q_ed_only_t1"], cal["r2_change_from_state"]
    one_minus_rho = np.sqrt(r2 * np.trace(Sb) / np.trace(V)) / (1 - pi_p)
    S_eps, clipped = psd_clip(Sb - one_minus_rho ** 2 * (1 - pi_p) * V - q * (1 - q) * np.outer(beta, beta))
    Lp, Ls, Le = (np.linalg.cholesky(pi_p * V + 1e-9 * np.eye(d)), np.linalg.cholesky((1 - pi_p) * V + 1e-9 * np.eye(d)),
                  np.linalg.cholesky(S_eps + 1e-9 * np.eye(d)))
    med = float(np.median([cal["true_psi2_of_vectors_at_1x"][c] for c in ("I47", "I48", "I49")]))
    kbar = (1 - q) * KAPPA_ADM + q * KAPPA_ED
    rng = np.random.default_rng(0)
    rows = []
    for dr in range(draws):
        # stable pool: b's stable pairs per patient
        n_pool_pat = len(per_patient)
        pat = np.repeat(np.arange(n_pool_pat), per_patient)
        p = rng.standard_normal((n_pool_pat, d)) @ Lp.T
        s = rng.standard_normal((len(pat), d)) @ Ls.T
        u = (rng.random(len(pat)) < q).astype(float)
        X0 = p[pat] + s
        D0 = -one_minus_rho * s + u[:, None] * beta + rng.standard_normal((len(pat), d)) @ Le.T
        lw = LedoitWolf().fit(D0 / D0.std(0))
        M = np.linalg.inv(lw.covariance_) / np.outer(D0.std(0), D0.std(0))
        u_diffuse = rng.standard_normal(d)
        Lsb = np.linalg.cholesky(Sb + 1e-9 * np.eye(d))
        vdiff = Lsb @ (u_diffuse / np.linalg.norm(u_diffuse)) * np.sqrt(med)
        vecs = {**vec, "diffuse": vdiff}
        fold_pat = rng.integers(0, N_FOLDS, n_pool_pat + 10 ** 6)        # folds of pool and event patients
        for kind in ("G", "U"):
            for c, n_rec in COUNTS.items():
                n_ev = int(round(n_rec / kbar)) if kind == "G" else n_rec
                # Risk profile on the OBSERVED state (plan revision 10 correction): candidates are drawn from the
                # population, and onset pairs are sampled with weights exp(ALPHA * w'x / sd(w'x)), so that, given x_t,
                # the label is independent of the change (U codes are null) and the propensity AUROC is near ~0.7
                w = rng.standard_normal(d)
                w /= np.linalg.norm(w)
                n_cand = 8 * n_ev
                pc = rng.standard_normal((n_cand, d)) @ Lp.T
                sc_ = rng.standard_normal((n_cand, d)) @ Ls.T
                score = (pc + sc_) @ w
                wt = np.exp(ALPHA * (score - score.mean()) / score.std())
                pick = rng.choice(n_cand, size=n_ev, replace=False, p=wt / wt.sum())
                pe, se = pc[pick], sc_[pick]
                ue = (rng.random(n_ev) < q).astype(float)
                Xe = pe + se
                De = -one_minus_rho * se + ue[:, None] * beta + rng.standard_normal((n_ev, d)) @ Le.T
                rec = rng.random(n_ev) < np.where(ue > 0, KAPPA_ED, KAPPA_ADM) if kind == "G" else np.ones(n_ev, bool)
                X = np.r_[X0, Xe]
                Dbase = np.r_[D0, De]
                E = np.r_[np.zeros(len(X0)), np.ones(n_ev)][:, None] * vecs[c][None] if kind == "G" else 0 * Dbase
                a = np.r_[np.zeros(len(X0)), rec.astype(float)]
                uu = np.r_[u, ue]
                subj = np.r_[pat, n_pool_pat + np.arange(n_ev)]
                folds = fold_pat[subj]
                _, inv = np.unique(subj, return_inverse=True)
                Wb = rng.multinomial(inv.max() + 1, np.full(inv.max() + 1, 1 / (inv.max() + 1)), size=N_BOOT).astype(float)
                feats = {"part1": X, "oracle": np.c_[X, uu]}
                for est, F in feats.items():
                    pi = logistic_cf(F, a, folds)
                    ra = a - pi
                    Rb, Re = ridge_cf(F, Dbase, folds), ridge_cf(F, E, folds) if kind == "G" else 0
                    auc_p = roc_auc_score(a, pi)
                    for sc in SCALES:
                        R = Rb + sc * Re if kind == "G" else Rb
                        h = psi2_halves(ra, R, subj, folds, M, Wb, inv, (list(range(10)), list(range(10, 20))))
                        qq = psi2_halves(ra, R, subj, folds, M, Wb, inv, (list(range(10, 15)), list(range(15, 20))))
                        rows.append({"draw": dr, "kind": kind, "code": c, "estimator": est, "scale": sc,
                                     "onsets": int(a.sum()), "true_psi2": (sc ** 2 * float(vecs[c] @ M @ vecs[c])
                                                                          if kind == "G" else 0.0),
                                     "psi2": h[0], "lo": h[1], "hi": h[2], "psi2_q": qq[0], "lo_q": qq[1],
                                     "propensity_auroc": float(auc_p)})
        print(f"draw {dr + 1}/{draws} done", flush=True)
    df = pd.DataFrame(rows)
    out = {"calibration": {k: v for k, v in cal.items() if k != "features"}, "one_minus_rho": float(one_minus_rho),
           "S_eps_clipped_fraction": clipped, "kappa_bar": kbar, "draws": draws, "n_boot": N_BOOT, "Gamma2": GAMMA2,
           "propensity_auroc_median": float(df["propensity_auroc"].median())}
    summ, L = {}, []
    for est in ("oracle", "part1"):
        for sc in SCALES:
            g = df[(df.estimator == est) & (df.scale == sc)]
            aucs = [roc_auc_score(np.r_[np.ones(6), np.zeros(6)],
                                  np.r_[g[(g.draw == dr) & (g.kind == "G")]["psi2"].values,
                                        g[(g.draw == dr) & (g.kind == "U")]["psi2"].values]) for dr in range(draws)]
            G, U = g[g.kind == "G"], g[g.kind == "U"]
            summ[f"{est}|{sc}"] = {
                "auroc_G_vs_U_mean": float(np.mean(aucs)), "auroc_G_vs_U_pooled":
                    float(roc_auc_score(np.r_[np.ones(len(G)), np.zeros(len(U))], np.r_[G.psi2.values, U.psi2.values])),
                "G_detect_lo_gt_0": float((G.lo > 0).mean()), "G_detect_lo_gt_Gamma": float((G.lo > GAMMA2).mean()),
                "G_detect_quarters_lo_gt_0": float((G.lo_q > 0).mean()),
                "U_false_lo_gt_0": float((U.lo > 0).mean()), "U_false_lo_gt_Gamma": float((U.lo > GAMMA2).mean()),
                "G_detect_by_vector_lo_gt_Gamma": G.groupby("code").apply(lambda x: float((x.lo > GAMMA2).mean())).to_dict()}
    chosen = next((sc for sc in SCALES if summ[f"oracle|{sc}"]["auroc_G_vs_U_mean"] >= 0.9), None)
    out["summary"], out["chosen_scale"] = summ, chosen
    out["rule_c"] = ("C2 is read at scale %sx" % chosen) if chosen else "no scale reaches oracle AUROC 0.9: C2 cannot be read"
    df.to_csv(OUTD / "power_b_rows.csv", index=False)
    (OUTD / "power_b.json").write_text(json.dumps(out, indent=1))
    L += [f"b-calibrated latent environment: dim {cal['dim']}, patient share {cal['patient_share_b']:.3f}, R² of the change "
          f"from the state {cal['r2_change_from_state']:.3f}, stable fraction {cal['stable_fraction']:.3f}, q(ED-only at t+1) "
          f"{q:.3f}, 1 - ρ {one_minus_rho:.3f}, κ̄ {kbar:.3f}, median propensity AUROC {out['propensity_auroc_median']:.3f}; "
          f"{draws} draws x 6 G + 6 U codes, {N_BOOT} bootstrap draws.", "",
          "| Estimator | Scale | G-vs-U AUROC (mean over draws) | pooled | G detected: lo > 0 | lo > Γ̂² | quarters lo > 0 | U false: lo > 0 | lo > Γ̂² |",
          "|---|---|---|---|---|---|---|---|---|"]
    for est in ("oracle", "part1"):
        for sc in SCALES:
            s_ = summ[f"{est}|{sc}"]
            L.append(f"| {est} | {sc}x | {s_['auroc_G_vs_U_mean']:.3f} | {s_['auroc_G_vs_U_pooled']:.3f} | "
                     f"{s_['G_detect_lo_gt_0']:.2f} | {s_['G_detect_lo_gt_Gamma']:.2f} | {s_['G_detect_quarters_lo_gt_0']:.2f} | "
                     f"{s_['U_false_lo_gt_0']:.2f} | {s_['U_false_lo_gt_Gamma']:.2f} |")
    L += ["", "G detection (lo > Γ̂²) by vector, Part 1 estimator: " + "; ".join(
        f"{sc}x " + ", ".join(f"{k} {v:.2f}" for k, v in summ[f'part1|{sc}']['G_detect_by_vector_lo_gt_Gamma'].items())
        for sc in SCALES), "", f"**Rule (c): {out['rule_c']}.**"]
    (OUTD / "power_b.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- revision 11: shrunk vectors and the new inference
def halves_theta(path, sd_x):
    z = np.load(path)
    ra, ry, sd, folds = z["ra"].astype(np.float64), z["ry"].astype(np.float64), z["sd"].astype(np.float64), z["folds"]
    out = []
    for m in (folds < 10, folds >= 10):
        out.append((ra[m, None] * ry[m]).sum(0) / (ra[m] ** 2).sum() * sd / sd_x)
    return out


def shrink():
    """Shrinks every injected vector to its debiased norm (plan revision 11): v * sqrt(max(debiased, 0) / plug-in),
    the debiased norm being the split product of the 20-fold halves restricted to the vector's coordinates; the diffuse
    vector takes the median debiased norm of I47, I48, I49. Writes vec_shrunk_* and psi2 tables into calibration_b."""
    z = np.load(OUTD / "calibration_b.npz")
    cal = json.loads((OUTD / "calibration_b.json").read_text())
    names = cal["features"]
    # Debiased and plug-in norms in one metric (the inverse stable-pair change covariance of standardised b; the
    # Ledoit-Wolf intensity on b is ~0.003), so their ratio, the shrink factor, is internally consistent
    Sb = z["Sigma_b"]
    S_inv = np.linalg.inv(Sb + 1e-6 * np.trace(Sb) / len(Sb) * np.eye(len(Sb)))
    j = {n: k for k, n in enumerate(names)}
    th = {c: halves_theta(G3D / f"clean_bm_b_z_{c}.resid.npz", 1.0) for c in ("I44", "I47", "I48", "I49")}
    sd_x = theta_all_folds(G3D / "clean_bm_b_z_I48.resid.npz") / z["vec_I48"]      # vec = theta_raw / sd_x
    supp = {"I47": np.arange(len(names)), "I48": np.arange(len(names)), "I49": np.arange(len(names)),
            "one-feature": np.array([j["qrs_width"]]), "two-feature": np.array([j["hr"], j["rr_irr"]])}
    src = {"I47": "I47", "I48": "I48", "I49": "I49", "one-feature": "I44", "two-feature": "I48"}
    deb, plug, shrunk = {}, {}, {}
    for k, S in supp.items():
        A, B = th[src[k]]
        a, bb = np.zeros(len(names)), np.zeros(len(names))
        a[S], bb[S] = A[S] / sd_x[S], B[S] / sd_x[S]
        v = z[f"vec_{k}"]
        deb[k] = float(a @ S_inv @ bb)
        plug[k] = float(v @ S_inv @ v)
        shrunk[k] = v * np.sqrt(max(deb[k], 0.0) / plug[k])
    med = float(np.median([deb[c] for c in ("I47", "I48", "I49")]))
    cal["rev11_shrink"] = {"plug_in_psi2": plug, "debiased_psi2": deb, "factor": {k: float(np.sqrt(max(deb[k], 0) / plug[k]))
                                                                                 for k in deb},
                           "diffuse_target_psi2": med,
                           "metric": "inverse of the stable-pair change covariance of standardised b (as the plug-in)"}
    data = {k: z[k] for k in z.files}
    data.update({f"vecs_{k}": v for k, v in shrunk.items()})
    data["diffuse_target"] = np.array(med)
    np.savez(OUTD / "calibration_b.npz", **data)
    (OUTD / "calibration_b.json").write_text(json.dumps(cal, indent=1))
    print(json.dumps(cal["rev11_shrink"], indent=1))


def strata_perms(rng, strata, K):
    """K permutation index vectors (K, n), each permuting positions within strata."""
    n = len(strata)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    idx = np.empty((K, n), dtype=np.int64)
    for k in range(K):
        out = np.arange(n)
        for g in groups:
            out[g] = g[rng.permutation(len(g))]
        idx[k] = out
    return idx


def new_inference(ra, Rb, Re, inv, pf_row, M, rng, scales, n_boot=400, n_perm=200, strata=None,
                  split=(tuple(range(10)), tuple(range(10, 20)))):
    """Plan revision 11: T = θ̂_A' M θ̂_B (halves 0-9 / 10-19); V_A, V_B by patient bootstrap within each half; bias-
    corrected plug-in Var(T); interval T ± 1.96 sd; percentile bootstrap (secondary); stratified permutation p-value;
    and one permuted-label replicate (T, Var) for Q/df. Rb, Re: residuals of the base change and of the unit effect
    (R = Rb + scale Re). Returns {scale: dict}."""
    P = inv.max() + 1
    pf = np.zeros(P, int)
    pf[inv] = pf_row
    num_b, num_e = np.zeros((P, Rb.shape[1])), np.zeros((P, Rb.shape[1]))

    def sums(r):
        nb, ne = np.zeros((P, Rb.shape[1])), np.zeros((P, Rb.shape[1]))
        np.add.at(nb, inv, r[:, None] * Rb)
        np.add.at(ne, inv, r[:, None] * Re)
        return nb, ne, np.bincount(inv, weights=r ** 2, minlength=P)

    num_b, num_e, den = sums(ra)
    halves = [np.flatnonzero(np.isin(pf, split[0])), np.flatnonzero(np.isin(pf, split[1]))]
    Ws = [rng.multinomial(len(h), np.full(len(h), 1 / len(h)), size=n_boot).astype(np.float64) for h in halves]
    rows_h = [np.isin(pf_row, split[0]), np.isin(pf_row, split[1])]
    idx = strata_perms(rng, strata, n_perm)
    perm_parts = []
    for rh in rows_h:
        rp = ra[idx][:, rh]                                                  # (K, n_h)
        perm_parts.append((rp @ Rb[rh], rp @ Re[rh], (ra[rh] ** 2).sum()))
    ra_q = ra[idx[0]]
    qb, qe, qden = sums(ra_q)

    def stats(nb, ne, dn, sc):
        th = [(nb[h] + sc * ne[h]).sum(0) / dn[h].sum() for h in halves]
        draws = [(W @ (nb[h] + sc * ne[h])) / (W @ dn[h])[:, None] for W, h in zip(Ws, halves)]
        V = [np.cov(d_, rowvar=False) for d_ in draws]
        tr = float(np.trace(M @ V[0] @ M @ V[1]))
        var = (max(float(th[1] @ M @ V[0] @ M @ th[1]) - tr, 0.0) + max(float(th[0] @ M @ V[1] @ M @ th[0]) - tr, 0.0) + tr)
        T = float(th[0] @ M @ th[1])
        Tb = np.einsum("bi,ij,bj->b", draws[0], M, draws[1])
        return T, var, Tb

    out = {}
    for sc in scales:
        T, var, Tb = stats(num_b, num_e, den, sc)
        thp = [(pb + sc * pe) / dd for pb, pe, dd in perm_parts]
        Tp = np.einsum("ki,ij,kj->k", thp[0], M, thp[1])
        Tq, varq, _ = stats(qb, qe, qden, sc)
        sdv = np.sqrt(var)
        out[sc] = {"T": T, "var": var, "lo": T - 1.96 * sdv, "hi": T + 1.96 * sdv,
                   "perc_lo": float(np.percentile(Tb, 2.5)), "perc_hi": float(np.percentile(Tb, 97.5)),
                   "p_perm": float((1 + (Tp >= T).sum()) / (1 + len(Tp))), "T_permlab": Tq, "var_permlab": varq}
    return out


def validate(draws, n_u=12):
    """Coverage of the revision-11 intervals in the b-calibrated latent environment (κ = 1, risk on observed x_t,
    shrunk vectors), Part 1 estimator; truth ψ² = (s Δ)' M (s Δ), 0 for U codes."""
    cal = json.loads((OUTD / "calibration_b.json").read_text())
    z = np.load(OUTD / "calibration_b.npz")
    V, Sb, beta, per_patient = z["V"], z["Sigma_b"], z["beta"], z["per_patient"]
    vec = {k[5:]: z[k] for k in z.files if k.startswith("vecs_")}
    d, pi_p, q, r2 = cal["dim"], cal["patient_share_b"], cal["q_ed_only_t1"], cal["r2_change_from_state"]
    one_minus_rho = np.sqrt(r2 * np.trace(Sb) / np.trace(V)) / (1 - pi_p)
    S_eps, _ = psd_clip(Sb - one_minus_rho ** 2 * (1 - pi_p) * V - q * (1 - q) * np.outer(beta, beta))
    Lp, Ls, Le = (np.linalg.cholesky(pi_p * V + 1e-9 * np.eye(d)), np.linalg.cholesky((1 - pi_p) * V + 1e-9 * np.eye(d)),
                  np.linalg.cholesky(S_eps + 1e-9 * np.eye(d)))
    Lsb = np.linalg.cholesky(Sb + 1e-9 * np.eye(d))
    med = float(z["diffuse_target"])
    rng = np.random.default_rng(11)
    rows = []
    for dr in range(draws):
        n_pool_pat = len(per_patient)
        pat = np.repeat(np.arange(n_pool_pat), per_patient)
        p = rng.standard_normal((n_pool_pat, d)) @ Lp.T
        s = rng.standard_normal((len(pat), d)) @ Ls.T
        u = (rng.random(len(pat)) < q).astype(float)
        X0 = p[pat] + s
        D0 = -one_minus_rho * s + u[:, None] * beta + rng.standard_normal((len(pat), d)) @ Le.T
        lw = LedoitWolf().fit(D0 / D0.std(0))
        M = np.linalg.inv(lw.covariance_) / np.outer(D0.std(0), D0.std(0))
        ud = rng.standard_normal(d)
        vecs = {**vec, "diffuse": Lsb @ (ud / np.linalg.norm(ud)) * np.sqrt(med)}
        fold_pat = rng.integers(0, N_FOLDS, n_pool_pat + 10 ** 6)
        codes = [("G", c) for c in COUNTS] + [("U", list(COUNTS)[i % 6]) for i in range(n_u)]
        for kind, c in codes:
            n_ev = COUNTS[c]
            w = rng.standard_normal(d)
            w /= np.linalg.norm(w)
            n_cand = 8 * n_ev
            pc = rng.standard_normal((n_cand, d)) @ Lp.T
            sc_ = rng.standard_normal((n_cand, d)) @ Ls.T
            score = (pc + sc_) @ w
            wt = np.exp(ALPHA * (score - score.mean()) / score.std())
            pick = rng.choice(n_cand, size=n_ev, replace=False, p=wt / wt.sum())
            pe, se = pc[pick], sc_[pick]
            ue = (rng.random(n_ev) < q).astype(float)
            Xe = pe + se
            De = -one_minus_rho * se + ue[:, None] * beta + rng.standard_normal((n_ev, d)) @ Le.T
            X, Dbase = np.r_[X0, Xe], np.r_[D0, De]
            E = (np.r_[np.zeros(len(X0)), np.ones(n_ev)][:, None] * vecs[c][None]) if kind == "G" else np.zeros_like(Dbase)
            a = np.r_[np.zeros(len(X0)), np.ones(n_ev)]
            subj = np.r_[pat, n_pool_pat + np.arange(n_ev)]
            folds = fold_pat[subj]
            _, inv = np.unique(subj, return_inverse=True)
            pi = logistic_cf(X, a, folds)
            ra = a - pi
            Rb, Re = ridge_cf(X, Dbase, folds), ridge_cf(X, E, folds)
            quint = np.searchsorted(np.quantile(pi, [0.2, 0.4, 0.6, 0.8]), pi)
            res = new_inference(ra, Rb, Re, inv, folds, M, rng, SCALES if kind == "G" else [0.0],
                                strata=folds * 5 + quint)
            for sc, r in res.items():
                truth = float((sc * vecs[c]) @ M @ (sc * vecs[c])) if kind == "G" else 0.0
                rows.append({"draw": dr, "kind": kind, "code": c, "scale": sc, "truth": truth, **r,
                             "cover": bool(r["lo"] <= truth <= r["hi"]),
                             "perc_cover": bool(r["perc_lo"] <= truth <= r["perc_hi"])})
        print(f"draw {dr + 1}/{draws} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTD / "validate_rev11_rows.csv", index=False)
    summ, L = {}, ["| θ | Codes | Coverage, new interval | Coverage, percentile (secondary) | Permutation test rejects (p < 0.05) | Mean truth ψ² | Mean T |",
                   "|---|---|---|---|---|---|---|"]
    for lab, g in [("0 (U codes)", df[df.kind == "U"])] + [(f"{sc}× (G codes)", df[(df.kind == "G") & (df.scale == sc)])
                                                           for sc in SCALES]:
        s_ = {"n": int(len(g)), "coverage": float(g.cover.mean()), "coverage_percentile": float(g.perc_cover.mean()),
              "perm_reject": float((g.p_perm < 0.05).mean()), "truth_mean": float(g.truth.mean()), "T_mean": float(g["T"].mean())}
        summ[lab] = s_
        L.append(f"| {lab} | {s_['n']} | {s_['coverage']:.3f} | {s_['coverage_percentile']:.3f} | {s_['perm_reject']:.3f} | "
                 f"{s_['truth_mean']:.3f} | {s_['T_mean']:.3f} |")
    # Q/df on permuted labels (one replicate per code), with the new variance
    qd = {}
    for lab, g in [("U", df[df.kind == "U"]), ("G at 1x", df[(df.kind == "G") & (df.scale == 1.0)]), ("all codes", df.drop_duplicates(["draw", "kind", "code", "T_permlab"]))]:
        y, v = g["T_permlab"].values, g["var_permlab"].values
        ok = v > 0
        w = 1 / v[ok]
        mu = (w * y[ok]).sum() / w.sum()
        qd[lab] = {"N": int(ok.sum()), "Q_over_df": float((w * (y[ok] - mu) ** 2).sum() / (ok.sum() - 1)),
                   "z_sd": float((y[ok] / np.sqrt(v[ok])).std())}
    cov0 = summ["0 (U codes)"]["coverage"]
    verdict = ("PASS: coverage at θ = 0 is %.3f ≥ 0.93" % cov0) if cov0 >= 0.93 else \
              ("STOP: coverage at θ = 0 is %.3f < 0.93" % cov0)
    out = {"draws": draws, "n_u_per_draw": n_u, "summary": summ, "Q_over_df_permuted_labels": qd, "verdict": verdict,
           "design": "κ = 1, risk on observed x_t, shrunk vectors, Part 1 estimator, halves 0-9 / 10-19"}
    (OUTD / "validate_rev11.json").write_text(json.dumps(out, indent=1))
    L += ["", "Q/df on permuted labels (new variance): " + "; ".join(f"{k}: {v['Q_over_df']:.3f} (z sd {v['z_sd']:.2f}, N {v['N']})"
                                                                    for k, v in qd.items()), "", f"**{verdict}.**"]
    (OUTD / "validate_rev11.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "power", "shrink", "validate"])
    ap.add_argument("--draws", type=int, default=8)
    a = ap.parse_args()
    {"calibrate": calibrate, "power": lambda: power(a.draws), "shrink": shrink,
     "validate": lambda: validate(a.draws)}[a.cmd]()


if __name__ == "__main__":
    main()

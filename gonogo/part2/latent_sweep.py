"""
Latent γ Sweep with Baseline 1, the Part 1 Estimator and the Oracle (Plan Revision 14, Step 3)
==============================================================================================
b-calibrated latent environment (latent.py: dim 39 in b's cross-sectional SD units, patient share, state-dependent
change, stable-pair noise floor, hidden context u_{t+1} = "next stay ED-only" with its real frequency q and shift β) at
1x with the shrunk vectors, Gaussian noise, ε = ε_u = 0, no delay.

Per draw   one pool of 35,009 pairs with b's per-patient counts, 20 patient folds. 6 G and 6 U codes (shared across γ):
           onset pairs sampled without replacement with weights exp(α score(x_t)); 6 UC and 3 GC codes per γ: weights
           exp(α score(x_t) + γ u_{t+1}). G and GC: effects on every event pair; recorded with κ by u_{t+1}, with the
           number of events n_rec / E[κ | selection]. One multi-label dataset per (draw, γ): the change plus the effects of
           the draw's G codes and of its GC codes at that γ.
Methods    Part 1 estimator: per code, cross-fitted ridge base model and logistic propensity on x_t; ψ̂² on halves, the
           new interval and the CRT (semisynth.infer). Oracle: the same plus u_{t+1}. Baseline 1: baseline1.py's head
           with state x_t and all injected labels as actions, 12 epochs, 3 seeds, flip score, CRT with 500 joint redraws.
Seeds      pool and G/U codes [1400, draw]; UC/GC [1401, draw, 10 γ].

Commands: estimators (CPU, Part 1 + oracle), baseline1 (GPU), summary.
Output: outputs/part2/latent_sweep/
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gonogo" / "part2"))
from latent import ALPHA, COUNTS, KAPPA_ADM, KAPPA_ED, N_FOLDS, logistic_cf, psd_clip  # noqa: E402

CAL = ROOT / "outputs" / "part2" / "latent"
OUTD = ROOT / "outputs" / "part2" / "latent_sweep"
GAMMAS = (0.0, 0.5, 1.0, 2.0, 4.0)
GC_CODES = ("I47", "I48", "I49")
GAMMA2 = 0.0135
SCALE = 1.0


def calib():
    cal = json.loads((CAL / "calibration_b.json").read_text())
    z = np.load(CAL / "calibration_b.npz")
    V, Sb, beta, per_patient = z["V"], z["Sigma_b"], z["beta"], z["per_patient"]
    d, pi_p, q, r2 = cal["dim"], cal["patient_share_b"], cal["q_ed_only_t1"], cal["r2_change_from_state"]
    omr = np.sqrt(r2 * np.trace(Sb) / np.trace(V)) / (1 - pi_p)
    S_eps, _ = psd_clip(Sb - omr ** 2 * (1 - pi_p) * V - q * (1 - q) * np.outer(beta, beta))
    ch = lambda A: np.linalg.cholesky(A + 1e-9 * np.eye(d))
    vecs = {k[5:]: z[k] for k in z.files if k.startswith("vecs_")}
    return {"d": d, "q": q, "omr": omr, "beta": beta, "per_patient": per_patient, "Lp": ch(pi_p * V),
            "Ls": ch((1 - pi_p) * V), "Le": ch(S_eps), "Lsb": ch(Sb), "vecs": vecs, "med": float(z["diffuse_target"])}


def onsets(rng, X, u, n_target, recorded, gamma):
    n, d = X.shape
    w = rng.standard_normal(d)
    sc = X @ (w / np.linalg.norm(w))
    wt = np.exp(ALPHA * (sc - sc.mean()) / sc.std() + gamma * u)
    kap = np.where(u > 0, KAPPA_ED, KAPPA_ADM)
    n_ev = int(round(n_target / float((wt * kap).sum() / wt.sum()))) if recorded else n_target
    ev = rng.choice(n, size=n_ev, replace=False, p=wt / wt.sum())
    e = np.zeros(n)
    e[ev] = 1
    a = e.copy()
    if recorded:
        a[ev] = (rng.random(n_ev) < kap[ev]).astype(float)
    return e, a


def generate(C, dr):
    """Pool, G/U codes (shared) and UC/GC codes per γ of draw dr. Returns a dict."""
    rng = np.random.default_rng([1400, dr])
    d, pp = C["d"], C["per_patient"]
    n_pat = len(pp)
    pat = np.repeat(np.arange(n_pat), pp)
    n = len(pat)
    p = rng.standard_normal((n_pat, d)) @ C["Lp"].T
    s = rng.standard_normal((n, d)) @ C["Ls"].T
    u = (rng.random(n) < C["q"]).astype(float)
    X = p[pat] + s
    D0 = -C["omr"] * s + u[:, None] * C["beta"] + rng.standard_normal((n, d)) @ C["Le"].T
    folds = rng.integers(0, N_FOLDS, n_pat)[pat]
    lw = LedoitWolf().fit(D0 / D0.std(0))
    M = np.linalg.inv(lw.covariance_) / np.outer(D0.std(0), D0.std(0))
    ud = rng.standard_normal(d)
    vecs = {**C["vecs"], "diffuse": C["Lsb"] @ (ud / np.linalg.norm(ud)) * np.sqrt(C["med"])}
    codes = []
    for kind in ("G", "U"):
        for c, n_rec in COUNTS.items():
            e, a = onsets(rng, X, u, n_rec, kind == "G", 0.0)
            codes.append({"kind": kind, "code": c, "gamma": None, "e": e, "a": a, "vec": vecs[c] if kind == "G" else None})
    for g in GAMMAS:
        rg = np.random.default_rng([1401, dr, int(round(g * 10))])
        for kind, cs in (("UC", list(COUNTS)), ("GC", list(GC_CODES))):
            for c in cs:
                e, a = onsets(rg, X, u, COUNTS[c], kind == "GC", g)
                codes.append({"kind": kind, "code": c, "gamma": g, "e": e, "a": a,
                              "vec": vecs[c] if kind == "GC" else None})
    return {"X": X, "u": u, "D0": D0, "pat": pat, "folds": folds, "M": M, "codes": codes, "n": n}


def dataset(G, gamma):
    """Codes and outcome of the multi-label dataset at γ: the draw's G and U codes plus its UC and GC codes at γ."""
    cs = [c for c in G["codes"] if c["gamma"] is None or c["gamma"] == gamma]
    D = G["D0"].copy()
    for c in cs:
        if c["vec"] is not None:
            D += SCALE * c["e"][:, None] * c["vec"][None]
    return cs, D


# ------------------------------------------------------------------------------------------------ Part 1 and oracle
def work_estimators(task):
    dr, gamma = task
    f = OUTD / "estimators" / f"draw{dr}_g{gamma}.csv"
    if f.exists():
        return
    import semisynth as ss
    C = calib()
    G = generate(C, dr)
    cs, D = dataset(G, gamma)
    _, inv = np.unique(G["pat"], return_inverse=True)
    n = G["n"]
    ss.S.update(inv=inv, d={"fold": G["folds"]}, M=G["M"],
                Smat=sp.csr_matrix((np.ones(n), (inv, np.arange(n))), shape=(inv.max() + 1, n)))
    rng = np.random.default_rng([1402, dr, int(round(gamma * 10))])
    feats = {"part1": G["X"], "oracle": np.c_[G["X"], G["u"]]}
    rows = []
    for est, F in feats.items():
        R = ss.Ridge(F, G["folds"]).resid(D)
        for c in cs:
            pi = logistic_cf(F, c["a"], G["folds"])
            r = ss.infer(c["a"] - pi, [R], {0.0: (1.0,)}, rng, pi=pi)[0.0]
            truth = float((SCALE * c["vec"]) @ G["M"] @ (SCALE * c["vec"])) if c["vec"] is not None else 0.0
            rows.append({"draw": dr, "gamma": gamma, "kind": c["kind"], "code": c["code"], "estimator": est,
                         "onsets": int(c["a"].sum()), "events": int(c["e"].sum()), "psi2_true": truth,
                         "propensity_auroc": float(roc_auc_score(c["a"], pi)), **r})
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(f, index=False)
    print(f"  estimators draw {dr} γ {gamma} done", flush=True)


def cmd_estimators(args):
    tasks = [(dr, g) for g in GAMMAS for dr in range(args.draws)]
    with mp.get_context("fork").Pool(args.workers) as pool:
        pool.map(work_estimators, tasks, chunksize=1)


# ------------------------------------------------------------------------------------------------ baseline 1
def cmd_baseline1(args):
    import torch
    import baseline1 as b1
    epochs = json.loads((b1.OUTD / "pilot.json").read_text())["best_epoch"]
    C = calib()
    outd = OUTD / "baseline1"
    outd.mkdir(parents=True, exist_ok=True)
    for gamma in [float(x) for x in args.gammas.split(",")]:
        for dr in range(args.draws):
            f = outd / f"draw{dr}_g{gamma}.npz"
            if f.exists():
                continue
            G = generate(C, dr)
            cs, D = dataset(G, gamma)
            lab = np.stack([c["a"] for c in cs], 1)
            pis = np.stack([logistic_cf(G["X"], c["a"], G["folds"]) for c in cs], 1)
            sd = G["D0"].std(0)
            xs, Y = b1.to_dev(G["X"], (D - G["D0"].mean(0)) / sd)
            aR = torch.zeros((G["n"], 0), device=b1.DEV)
            Mt = torch.as_tensor(G["M"], device=b1.DEV, dtype=torch.float32)
            sdt = torch.as_tensor(sd, device=b1.DEV, dtype=torch.float32)
            rows = torch.as_tensor(np.sort(np.random.default_rng(4096).choice(G["n"], b1.N_FLIP, replace=False)),
                                   device=b1.DEV)
            res = [b1.crt_config(xs, aR, lab, pis, Y, sdt, Mt, epochs, seed=s + 1, order_seed=100 + s, flip_rows=rows,
                                 n_null=args.n_null, gen_seed=20_000 + 1000 * dr + 100 * s + int(gamma * 10))
                   for s in range(3)]
            meta = [{"kind": c["kind"], "code": c["code"], "onsets": int(c["a"].sum()), "events": int(c["e"].sum()),
                     "psi2_true": float((SCALE * c["vec"]) @ G["M"] @ (SCALE * c["vec"])) if c["vec"] is not None else 0.0}
                    for c in cs]
            np.savez(f, delta_obs=np.stack([r[0] for r in res]), S_obs=np.stack([r[1] for r in res]),
                     S_null=np.stack([r[2] for r in res]), meta=json.dumps(meta), epochs=epochs)
            print(f"  latent baseline1 draw {dr} γ {gamma} done", flush=True)


# ------------------------------------------------------------------------------------------------ summary
def f1_threshold(tg, tu):
    best, bt = -1.0, np.inf
    for t in np.unique(np.r_[tg, tu]):
        tp, fp = (tg >= t).sum(), (tu >= t).sum()
        f1 = 2 * tp / (2 * tp + fp + (len(tg) - tp))
        if f1 > best:
            best, bt = f1, t
    return bt


def cmd_summary(args):
    import baseline1 as b1
    rows = []
    est_files = sorted((OUTD / "estimators").glob("*.csv"))
    if est_files:
        E = pd.concat([pd.read_csv(f) for f in est_files])
        for (dr, g, est), x in E.groupby(["draw", "gamma", "estimator"]):
            thr = f1_threshold(x[x.kind == "G"]["T"].values, x[x.kind == "U"]["T"].values)
            auc = roc_auc_score(np.r_[np.ones((x.kind == "G").sum()), np.zeros((x.kind == "U").sum())],
                                np.r_[x[x.kind == "G"]["T"].values, x[x.kind == "U"]["T"].values])
            for r in x.itertuples():
                rows.append({"method": est, "draw": dr, "gamma": g, "seed": 0, "kind": r.kind, "code": r.code,
                             "above_f1": r.T >= thr, "p_crt": r.p_crt, "lo_gt_Gamma": r.lo > GAMMA2, "auc": auc,
                             "bias": np.sqrt(max(r.T, 0) / r.psi2_true) if r.psi2_true > 0 else np.nan})
    b_files = sorted((OUTD / "baseline1").glob("*.npz"))
    if b_files:
        B = b1.score_rows(b_files)
        B["draw"] = B.file.str.extract(r"draw(\d+)_")[0].astype(int)
        B["gamma"] = B.file.str.extract(r"_g([\d.]+)\.npz")[0].astype(float)
        for r in B.itertuples():
            rows.append({"method": "baseline1", "draw": r.draw, "gamma": r.gamma, "seed": r.seed, "kind": r.kind,
                         "code": r.code, "above_f1": r.above_f1, "p_crt": r.p_crt, "lo_gt_Gamma": np.nan,
                         "auc": r.auc_G_vs_U, "bias": r.bias})
    df = pd.DataFrame(rows)
    df.to_csv(OUTD / "summary_rows.csv", index=False)
    tab = []
    for (m, g), x in df.groupby(["method", "gamma"]):
        uc, gg, gc, u = (x[x.kind == k] for k in ("UC", "G", "GC", "U"))
        tab.append({"method": m, "gamma": g, "draws": x.draw.nunique(), "seeds": x.seed.nunique(),
                    "UC_primary": uc.above_f1.mean(), "UC_crt": (uc.p_crt < 0.05).mean(),
                    "UC_lo_gt_Gamma": uc.lo_gt_Gamma.astype(float).mean() if m != "baseline1" else np.nan,
                    "U_primary": u.above_f1.mean(), "U_crt": (u.p_crt < 0.05).mean(),
                    "G_primary": gg.above_f1.mean(), "G_crt": (gg.p_crt < 0.05).mean(),
                    "G_lo_gt_Gamma": gg.lo_gt_Gamma.astype(float).mean() if m != "baseline1" else np.nan,
                    "AUROC_G_vs_U": x.groupby(["draw", "seed"]).auc.first().mean(),
                    "GC_bias_median": gc.bias.median(), "GC_crt": (gc.p_crt < 0.05).mean()})
    T = pd.DataFrame(tab)
    begins = {}
    for m, x in T.groupby("method"):
        x = x.sort_values("gamma")
        for col in ("UC_primary", "UC_crt", "UC_lo_gt_Gamma"):
            if x[col].isna().all():
                continue
            for lvl in (0.2, 0.5):
                hit = x[x[col] > lvl]
                begins[f"{m}|{col}|>{lvl}"] = float(hit.gamma.iloc[0]) if len(hit) else None
    out = {"table": T.to_dict(orient="records"), "begins": begins, "Gamma2": GAMMA2,
           "note": "baseline 1 CRT uses joint redraws of all injected labels: anti-conservative (U codes rejected 0.58-0.69), not interpretable"}
    (OUTD / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    L = ["Latent γ sweep (plan revision 14): b-calibrated environment, 1x, 8 draws x (6 G + 6 U + 6 UC + 3 GC) per γ. "
         "Primary: UC above the draw's G-vs-U F1 threshold. Secondary: CRT p < 0.05 (baseline 1: joint redraw, "
         f"ANTI-conservative here, see plan rev 14 result 3b). Part 1 / oracle also: lower bound > Γ̂² = {GAMMA2}.", "", T.round(3).to_markdown(index=False),
         "", "Where false grounding begins (smallest γ with UC rate above the level; None = never in the grid):", ""]
    L += [f"- {k}: {v}" for k, v in begins.items()]
    (OUTD / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["estimators", "baseline1", "summary"])
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gammas", default="0.0,0.5,1.0,2.0,4.0")
    ap.add_argument("--n-null", type=int, default=500)
    a_ = ap.parse_args()
    {"estimators": cmd_estimators, "baseline1": cmd_baseline1, "summary": cmd_summary}[a_.cmd](a_)

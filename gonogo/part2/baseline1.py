"""
Baseline 1 (Monolithic Action-Conditioned Predictor) with a Conditional Randomization Null (Plan Revision 14)
=============================================================================================================
The workshop dynamics head (src/models/dynamics.py: ActionProjector Proj_ω + DynamicsPredictor Dyn_φ) on a frozen
representation, scored per code by its flip readout, with the secondary C2 metric of revision 14: significance of the
score against 500 null fits in which every injected label is redrawn from its cross-fitted propensity.

Model       Proj_ω: Linear(action -> 64), ReLU, Linear(64 -> 64), ReLU, Linear(64 -> 64).
            Dyn_φ:  Linear(state + 64 -> 256), ReLU, residual block (256), Linear(256 -> dim). Target: the change,
            standardised per dimension (displacement prediction), MSE. Adam lr 1e-3, weight decay 1e-4, batch 1024,
            a fixed number of epochs (pilot), no early stopping. K independent copies are trained in parallel
            (batched weights); every copy has its own initialisation seed; the minibatch order is shared.
Score       flip readout δ_c = mean over a fixed subset of 4,096 pairs of f(x, a with c = 1) - f(x, a with c = 0), in
            raw units; S_c = δ_c' M δ_c (M = Ledoit-Wolf metric of ψ̂²). GC effect bias = sqrt(S_c / ψ²_true).
CRT         per (draw, γ_mix, seed): the observed fit and 500 null fits; in each null fit all injected labels are redrawn
            from their cross-fitted propensities (joint redraw: declared conservative per code in plan rev 14, found
            anti-conservative: the shared action pathway goes unused in the null fits; see plan rev 14 result 3b). p_c = (1 + #{S_c null >= S_c}) / 501.

Commands
--------
pilot      fixed epoch count from the real b data without injected codes (80/20 patient split, patience 4, <= 40)
semisynth  b at 1x, γ_mix in {1, 0}: one multi-label dataset per (draw, γ_mix) from the rev-12 sweep's labels
latent     the latent γ sweep (see latent_sweep.py)
summary    tables for semisynth

Output: outputs/part2/baseline1/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gonogo" / "part2"))
OUTD = ROOT / "outputs" / "part2" / "baseline1"
E_DIM, H_DIM = 64, 256
LR, WD, BATCH = 1e-3, 1e-4, 1024
N_NULL = 500
N_FLIP = 4096
DEV = torch.device("cuda")


# ------------------------------------------------------------------------------------------------ batched head
def _lin(gen, K, fan_in, fan_out):
    b = 1 / np.sqrt(fan_in)
    W = (torch.rand((K, fan_in, fan_out), generator=gen) * 2 - 1) * b
    c = (torch.rand((K, 1, fan_out), generator=gen) * 2 - 1) * b
    return W, c


def init_params(seeds, s_dim, a_real, a_lab, out_dim):
    """Parameters of K = len(seeds) heads; head k is initialised from torch.Generator(seed_k) (same layout as nn.Linear's
    default uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)))."""
    per = []
    for s in seeds:
        g = torch.Generator().manual_seed(int(s))
        a_in = a_real + a_lab
        W1, c1 = _lin(g, 1, a_in, E_DIM)
        W2, c2 = _lin(g, 1, E_DIM, E_DIM)
        W3, c3 = _lin(g, 1, E_DIM, E_DIM)
        V1, d1 = _lin(g, 1, s_dim + E_DIM, H_DIM)
        R1, r1 = _lin(g, 1, H_DIM, H_DIM)
        R2, r2 = _lin(g, 1, H_DIM, H_DIM)
        Vo, do = _lin(g, 1, H_DIM, out_dim)
        per.append([W1[:, :a_real], W1[:, a_real:], c1, W2, c2, W3, c3, V1[:, :s_dim], V1[:, s_dim:], d1,
                    R1, r1, R2, r2, Vo, do])
    return [torch.cat([p[i] for p in per]).to(DEV).requires_grad_(True) for i in range(len(per[0]))]


def forward(P, xs, aR, lab):
    """xs (B, s), aR (B, a_real) shared; lab (K, B, a_lab) per head. Returns (K, B, out)."""
    W1r, W1l, c1, W2, c2, W3, c3, V1s, V1e, d1, R1, r1, R2, r2, Vo, do = P
    e = torch.einsum("bi,kie->kbe", aR, W1r) if aR.shape[1] else 0
    e = torch.relu(e + torch.bmm(lab, W1l) + c1)
    e = torch.relu(torch.bmm(e, W2) + c2)
    e = torch.bmm(e, W3) + c3                                        # Proj_ω(a)
    h = torch.relu(torch.einsum("bs,ksh->kbh", xs, V1s) + torch.bmm(e, V1e) + d1)
    h = torch.relu(h + torch.bmm(torch.relu(torch.bmm(h, R1) + r1), R2) + r2)   # residual block
    return torch.bmm(h, Vo) + do


def train_heads(seeds, xs, aR, labs, Y, epochs, order_seed, chunk=256, verbose=False):
    """Trains len(seeds) heads; labs (K, n, a_lab) float16 on DEV (head k sees labs[k]). Returns params per chunk."""
    n = xs.shape[0]
    out = []
    for c0 in range(0, len(seeds), chunk):
        sl = slice(c0, min(c0 + chunk, len(seeds)))
        P = init_params(seeds[sl], xs.shape[1], aR.shape[1], labs.shape[2], Y.shape[1])
        opt = torch.optim.Adam(P, lr=LR, weight_decay=WD)
        g = torch.Generator(device="cpu").manual_seed(int(order_seed))
        for ep in range(epochs):
            perm = torch.randperm(n, generator=g).to(DEV)
            for i in range(0, n, BATCH):
                idx = perm[i:i + BATCH]
                pred = forward(P, xs[idx], aR[idx], labs[sl][:, idx].float())
                loss = ((pred - Y[idx][None]) ** 2).mean(dim=(1, 2)).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        out.append((sl, [p.detach() for p in P]))
        if verbose:
            print(f"    trained heads {sl.start}-{sl.stop - 1}", flush=True)
    return out


@torch.no_grad()
def flip_scores(trained, xs, aR, labs, rows, sd, M):
    """Flip readout per head and injected code: δ (K, C, dim) in raw units and S = δ' M δ (K, C)."""
    K, C = labs.shape[0], labs.shape[2]
    xs_, aR_ = xs[rows], aR[rows]
    deltas = torch.empty((K, C, sd.shape[0]), device=DEV)
    for sl, P in trained:
        lab = labs[sl][:, rows].float()
        for c in range(C):
            on, off = lab.clone(), lab.clone()
            on[..., c], off[..., c] = 1.0, 0.0
            deltas[sl, c] = (forward(P, xs_, aR_, on) - forward(P, xs_, aR_, off)).mean(1)
    d_raw = deltas * sd[None, None]
    S = torch.einsum("kcd,de,kce->kc", d_raw, M, d_raw)
    return d_raw.cpu().numpy(), S.cpu().numpy()


def crt_config(xs, aR, lab_obs, pis, Y, sd, M, epochs, seed, order_seed, flip_rows, n_null=N_NULL, gen_seed=0):
    """Observed fit + n_null null fits (all injected labels redrawn from pis), one seed. Returns δ_obs, S_obs, S_null."""
    n, C = lab_obs.shape
    g = torch.Generator(device=DEV).manual_seed(int(gen_seed))
    pis_t = torch.as_tensor(pis, device=DEV, dtype=torch.float32)                    # (n, C)
    null = (torch.rand((n_null, n, C), generator=g, device=DEV) < pis_t[None]).half()
    labs = torch.cat([torch.as_tensor(lab_obs, device=DEV).half()[None], null])     # (1 + n_null, n, C)
    seeds = [seed * 1_000_003] + [seed * 1_000_003 + 1 + k for k in range(n_null)]
    trained = train_heads(seeds, xs, aR, labs, Y, epochs, order_seed)
    d_raw, S = flip_scores(trained, xs, aR, labs, flip_rows, sd, M)
    del labs, null
    torch.cuda.empty_cache()
    return d_raw[0], S[0], S[1:]


# ------------------------------------------------------------------------------------------------ b data
def b_setup():
    import semisynth as ss
    ss.setup("b", variants=("part1",))
    S = ss.S
    d = S["d"]
    At = d["A"]
    k = At.shape[1] // 2
    a_ws = At[:, :k] - At[:, k:]                                   # workshop convention: +1 onset, -1 resolution
    state = np.c_[S["bts"], d["zf"]]
    return ss, S, state.astype(np.float32), a_ws.astype(np.float32)


def b_dataset(ss, S, dr, gamma, scale=1.0):
    """Joint multi-label dataset of draw dr at γ_mix = gamma: the rev-12 sweep's labels, all injected at once, noise
    per code by the rev-12 procedure (seed 1214)."""
    L = dict(np.load(S["outd"] / "sweep_labels.npz"))
    vecs = ss.draw_vecs(np.random.default_rng([1212, dr]))         # the sweep's vectors (diffuse included)
    rng = np.random.default_rng([1214, dr, int(round(gamma * 100))])
    codes = ([("G", c, "None") for c in ss.COUNTS] + [("U", c, "None") for c in ss.COUNTS]
             + [("UC", c, str(float(gamma))) for c in ss.COUNTS] + [("GC", c, str(float(gamma))) for c in ss.GC_CODES])
    n = S["n"]
    Dj = S["D"].copy()
    labs, meta = [], []
    for kind, c, g in codes:
        tag = f"{kind}__{c}__{dr}__{g}"
        e, a = np.zeros(n), np.zeros(n)
        e[L[tag + "__ev"]], a[L[tag + "__rec"]] = 1, 1
        ev = np.flatnonzero(e)
        eff = kind in ("G", "GC")
        vec = vecs[c] if eff else None
        target = S["target_g"][c] if kind == "G" else S["target_u"]
        # rev-12 noise: patient-bootstrapped Part 1 residuals on the event pairs, c solved on the code's own residuals
        pick = rng.integers(S["cnt"].size, size=len(ev))
        donor = S["order"][S["start"][pick] + (rng.random(len(ev)) * S["cnt"][pick]).astype(int)]
        eta = np.zeros_like(S["D"])
        eta[ev] = S["Rb"]["part1"][donor]
        Rn = S["ridge"]["part1"].resid(eta)
        Rf = S["Rb"]["part1"] + (scale * S["ridge"]["part1"].resid(e)[:, :1] * vec[None] if eff else 0)
        cc, r0 = ss.solve_c(Rf, Rn, ev, target)
        Dj = Dj + cc * eta + (scale * e[:, None] * vec[None] if eff else 0)
        labs.append(a)
        meta.append({"kind": kind, "code": c, "events": int(e.sum()), "onsets": int(a.sum()), "noise_c": cc,
                     "ratio_before": r0, "psi2_true": float((scale * vec) @ S["M"] @ (scale * vec)) if eff else 0.0})
    return Dj, np.array(labs).T, meta


def to_dev(*arrs):
    return [torch.as_tensor(a, device=DEV, dtype=torch.float32) for a in arrs]


# ------------------------------------------------------------------------------------------------ commands
def cmd_pilot(args):
    ss, S, state, a_ws = b_setup()
    D = S["D"]
    Y = (D - D.mean(0)) / S["sdD"]
    subj = S["d"]["subj"]
    u = np.unique(subj)
    val_pat = np.random.default_rng(14).choice(u, size=len(u) // 5, replace=False)
    va = np.isin(subj, val_pat)
    xs, aR, Yt = to_dev(state, a_ws, Y)
    lab = torch.zeros((1, len(D), 0), device=DEV, dtype=torch.float16)
    tr_idx = torch.as_tensor(np.flatnonzero(~va), device=DEV)
    va_idx = torch.as_tensor(np.flatnonzero(va), device=DEV)
    P = init_params([0], xs.shape[1], aR.shape[1], 0, Yt.shape[1])
    opt = torch.optim.Adam(P, lr=LR, weight_decay=WD)
    g = torch.Generator().manual_seed(0)
    hist, best, bad = [], (np.inf, 0), 0
    for ep in range(1, 41):
        perm = tr_idx[torch.randperm(len(tr_idx), generator=g).to(DEV)]
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            loss = ((forward(P, xs[idx], aR[idx], lab[:, idx].float()) - Yt[idx][None]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = float(((forward(P, xs[va_idx], aR[va_idx], lab[:, va_idx].float()) - Yt[va_idx][None]) ** 2).mean())
        hist.append(vl)
        if vl < best[0]:
            best, bad = (vl, ep), 0
        else:
            bad += 1
            if bad >= 4:
                break
    out = {"best_epoch": best[1], "best_val_mse": best[0], "val_mse_by_epoch": hist,
           "rule": "real b data without injected codes; 80/20 patient split (seed 14); patience 4; <= 40 epochs"}
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "pilot.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def cmd_semisynth(args):
    epochs = json.loads((OUTD / "pilot.json").read_text())["best_epoch"]
    ss, S, state, a_ws = b_setup()
    xs, aR = to_dev(state, a_ws)
    Mt = torch.as_tensor(S["M"], device=DEV, dtype=torch.float32)
    sdt = torch.as_tensor(S["sdD"], device=DEV, dtype=torch.float32)
    flip_rows = torch.as_tensor(np.sort(np.random.default_rng(4096).choice(S["n"], N_FLIP, replace=False)), device=DEV)
    from latent import logistic_cf
    outd = OUTD / "semisynth"
    outd.mkdir(parents=True, exist_ok=True)
    for gamma in [float(x) for x in args.gammas.split(",")]:
        for dr in range(args.draws):
            f = outd / f"draw{dr}_g{gamma}.npz"
            if f.exists():
                continue
            t0 = time.time()
            Dj, lab, meta = b_dataset(ss, S, dr, gamma)
            Y = (Dj - S["D"].mean(0)) / S["sdD"]
            pis = np.stack([logistic_cf(S["feats"]["part1"], lab[:, j], S["d"]["fold"]) for j in range(lab.shape[1])], 1)
            (Yt,) = to_dev(Y)
            res = [crt_config(xs, aR, lab, pis, Yt, sdt, Mt, epochs, seed=s + 1, order_seed=100 + s, flip_rows=flip_rows,
                              n_null=args.n_null, gen_seed=10_000 * dr + 100 * s + int(gamma * 10))
                   for s in range(3)]
            np.savez(f, delta_obs=np.stack([r[0] for r in res]), S_obs=np.stack([r[1] for r in res]),
                     S_null=np.stack([r[2] for r in res]), meta=json.dumps(meta), epochs=epochs)
            print(f"  semisynth draw {dr} γ_mix {gamma}: {time.time() - t0:.0f} s", flush=True)


def f1_threshold(tg, tu):
    best, bt = -1.0, np.inf
    for t in np.unique(np.r_[tg, tu]):
        tp, fp = (tg >= t).sum(), (tu >= t).sum()
        f1 = 2 * tp / (2 * tp + fp + (len(tg) - tp))
        if f1 > best:
            best, bt = f1, t
    return bt


def score_rows(files, method="baseline1"):
    """Per (draw, γ, seed, code) rows from the saved npz files: score, CRT p, primary-threshold reading, bias."""
    rows = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        kinds = np.array([m["kind"] for m in meta])
        for s in range(z["S_obs"].shape[0]):
            So, Sn = z["S_obs"][s], z["S_null"][s]
            p = (1 + (Sn >= So[None]).sum(0)) / (1 + Sn.shape[0])
            thr = f1_threshold(So[kinds == "G"], So[kinds == "U"])
            auc = roc_auc_score(np.r_[np.ones((kinds == "G").sum()), np.zeros((kinds == "U").sum())],
                                np.r_[So[kinds == "G"], So[kinds == "U"]])
            for j, m in enumerate(meta):
                rows.append({"file": f.name, "seed": s, **{k: m[k] for k in ("kind", "code", "onsets", "psi2_true")},
                             "score": float(So[j]), "p_crt": float(p[j]), "above_f1": bool(So[j] >= thr),
                             "auc_G_vs_U": float(auc),
                             "bias": float(np.sqrt(So[j] / m["psi2_true"])) if m["psi2_true"] > 0 else np.nan})
    return pd.DataFrame(rows)


def cp(k, n):
    from scipy.stats import beta
    return (0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)))


def cmd_summary(args):
    outd = OUTD / "semisynth"
    tabs, out = [], {}
    for gamma in [float(x) for x in args.gammas.split(",")]:
        files = sorted(outd.glob(f"draw*_g{gamma}.npz"))
        if not files:
            continue
        df = score_rows(files)
        df.to_csv(outd / f"rows_g{gamma}.csv", index=False)
        r = {"gamma_mix": gamma, "draws": len(files), "seeds": int(df.seed.nunique())}
        for kind in ("G", "U", "UC", "GC"):
            x = df[df.kind == kind]
            k = int((x.p_crt < 0.05).sum())
            r[f"{kind}_above_F1"] = float(x.above_f1.mean())
            r[f"{kind}_crt_p_lt_05"] = k / len(x)
            r[f"{kind}_crt_cp95"] = cp(k, len(x))
        r["AUROC_G_vs_U_mean"] = float(df.groupby(["file", "seed"]).auc_G_vs_U.first().mean())
        gc = df[df.kind == "GC"]
        r["GC_bias_median"], r["GC_bias_q10"], r["GC_bias_q90"] = (float(gc.bias.median()), float(gc.bias.quantile(0.1)),
                                                                   float(gc.bias.quantile(0.9)))
        r["UC_above_F1_by_seed"] = df[df.kind == "UC"].groupby("seed").above_f1.mean().round(3).tolist()
        tabs.append(r)
    out["table"] = tabs
    g1 = next((t for t in tabs if t["gamma_mix"] == 1.0), None)
    if g1 is not None:
        v = g1["UC_above_F1"]
        out["c2_reading"] = (f"baseline 1 UC false grounding (primary) at γ_mix = 1 is {v:.3f} <= 0.2: C2 is NO-GO on b"
                             if v <= 0.2 else
                             f"baseline 1 UC false grounding (primary) at γ_mix = 1 is {v:.3f} > 0.2: C2 stays open "
                             f"(GO needs every non-oracle baseline >= 0.5; Partial depends on the best baseline)")
    (OUTD / "semisynth_summary.json").write_text(json.dumps(out, indent=1, default=float))
    L = ["Baseline 1 (workshop head) on b at 1x, revision 14. Primary: UC above the draw's G-vs-U F1 threshold. "
         "Secondary: CRT p < 0.05 (500 null fits, joint redraw: mildly anti-conservative on b, see plan rev 14 result 3b).", "",
         "| γ_mix | draws x seeds | G-vs-U AUROC | UC false grounding: primary / CRT [95% CI] | U: primary / CRT | G detected: primary / CRT | GC bias, median (10–90%) | GC: CRT |",
         "|---|---|---|---|---|---|---|---|"]
    for t in tabs:
        L.append(f"| {t['gamma_mix']:g} | {t['draws']} x {t['seeds']} | {t['AUROC_G_vs_U_mean']:.3f} | {t['UC_above_F1']:.3f} / "
                 f"{t['UC_crt_p_lt_05']:.3f} [{t['UC_crt_cp95'][0]:.3f}, {t['UC_crt_cp95'][1]:.3f}] | "
                 f"{t['U_above_F1']:.3f} / {t['U_crt_p_lt_05']:.3f} | {t['G_above_F1']:.3f} / {t['G_crt_p_lt_05']:.3f} | "
                 f"{t['GC_bias_median']:.2f} ({t['GC_bias_q10']:.2f}–{t['GC_bias_q90']:.2f}) | {t['GC_crt_p_lt_05']:.3f} |")
    if "c2_reading" in out:
        L += ["", f"**C2 reading: {out['c2_reading']}.**"]
    (OUTD / "semisynth_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_timing(args):
    """Speed test on random data of the semi-synthetic shape (no real labels): K heads, given epochs."""
    n, s, aR_, C, D = 35009, 54, 182, 21, 39
    xs = torch.randn(n, s, device=DEV)
    aR = (torch.rand(n, aR_, device=DEV) < 0.01).float()
    labs = (torch.rand(args.k, n, C, device=DEV) < 0.01).half()
    Y = torch.randn(n, D, device=DEV)
    t0 = time.time()
    tr = train_heads(list(range(args.k)), xs, aR, labs, Y, args.epochs, 0)
    torch.cuda.synchronize()
    t1 = time.time()
    flip_scores(tr, xs, aR, labs, torch.arange(N_FLIP, device=DEV), torch.ones(D, device=DEV), torch.eye(D, device=DEV))
    torch.cuda.synchronize()
    print(f"K {args.k}, epochs {args.epochs}: train {t1 - t0:.1f} s, flip {time.time() - t1:.1f} s; "
          f"peak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "semisynth", "summary", "timing"])
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--gammas", default="1.0,0.0")
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    a_ = ap.parse_args()
    {"pilot": cmd_pilot, "semisynth": cmd_semisynth, "summary": cmd_summary, "timing": cmd_timing}[a_.cmd](a_)

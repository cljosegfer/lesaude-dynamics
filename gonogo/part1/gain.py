"""
Part 1 Cross-Fitted Gain G_c
============================
Does the onset label a_c carry information about the transition beyond the current state? (gonogo/go_no_go_plan.md,
2.3, 2.5, 2.6.) Sanity check 1 runs it on the positive controls I44, I45, I47, I48, I49 with the gap as the only
control, before any other code is interpreted.

Decisions of 2026-10-22, after checking the plan against the data:
  1. The clean subset is primary and the event threshold (>= --min-onsets onsets) counts its onsets: 14 codes pass,
     not the plan's 47 (counted over all cross-stay pairs). Codes below are reported as insufficient data and left out
     of the Benjamini-Hochberg family. --subset adjusted is the sensitivity analysis.
  2. Same-episode pairs (an ED stay and the admission it led to: under a day apart, 98-99.7% stable) will be dropped
     here once the linkage defines them, and join the within-stay pairs as the noise floor (not done yet).
  3. The primary statistic is the R-learner gain with its CRT p-value; the plan's MLP gain is secondary (it can catch
     state-dependent effects, but two separately trained MLPs differ by ~1e-4 on Δb whatever the code).

Pairs (gonogo/part1/pairs.py): cross-stay pairs between episodes (gonogo/part1/linkage.py: an ED visit and the
admission it led to share their codes, so those pairs are stable by construction and are left out), at risk for an
onset of c (y_t,c = 0), whose two ECGs have usable beats and machine measurements.
  --subset clean      the plan's clean subset: stable pairs (a_t = 0) and the single-code onsets of c, so a_c = 0 means
                      that nothing was coded differently and no co-occurring change can confound a_c
  --subset adjusted   every at-risk cross-stay pair, with the other codes' onset and resolution indicators a_{-c}
                      (150 columns) in both predictors, as in the method's conditioning set S_t; uses all onsets of c,
                      but conditioning on a_{-c} can block effects mediated by co-recorded codes
Targets, each dimension z-scored over the analysis pairs:
  b   Δb = b_{t+1} - b_t, the 39 median-beat scalars
  h   Δh = h_{t+1} - h_t, the 256 dims of the frozen no-action ViT (gonogo/part1/embed_h.py)
  m   Δm = m_{t+1} - m_t, the 12 machine measurements (gonogo/part1/measurements.py)
State s_t: --state bhm = [b_t, h_t, m_t] (default: the plan's h_t plus the current value of every target, which holds
the plan's "HR at t"), or a subset of b, h, m.
Conditioning sets Z: none; dt = log10 of the gap in days; z = the plan's observed controls (gonogo/part1/linkage.py):
log gap, care setting at t and t+1 (ED only / ED then admitted / inpatient), age, sex.

Predictors: base f(s_t, Z[, a_{-c}]) and full f(s_t, Z[, a_{-c}], a_c), MLPs with two GELU hidden layers of --hidden
units, the same optimizer, epochs and early-stopping rule; the full one reads one more input column. Common random
numbers: within a fold and seed, base, full, placebo and permutation models start from the same weights (the column's
weight at 0) and see the same batches, so their loss difference reflects the column, not the initialization (with
independent starts, seed-to-seed noise in G was ~1e-4, above the expected signal of the small positive controls). Cross-fitting over
the 20 patient-stratified folds: the model of fold k fits the 18 folds other than k and k+1 (mod 20), stops early on the
MSE of fold k+1 and predicts fold k, so every pair gets one out-of-fold prediction. All fold models, variants and
replicas train at once as one batched network.
Gain: G_c = L_base - L_full, out-of-fold MSE in z-units averaged over pairs and target dimensions, an estimate of the
conditional mutual information I(Δ; a_c | s_t, Z) up to scale; also per onset (x N / n_onsets) and per dimension.
Seeds: --seeds runs (init and batch order); the estimate is the mean over seeds.
Placebo: a cross-fitted L2 logistic propensity P(a_c = 1 | s_t, Z[, a_{-c}]), a fake onset drawn from it with an
  intercept shift that matches the prevalence (one draw per seed), G~_c computed identically against the same base.
Permutation test of the MLP gain: a_c shuffled within the plan's strata (gap quintile x care setting at t+1 x age
  tertile), the full model refitted with one seed per permutation (wider than the 3-seed mean, so conservative);
  p = (1 + #{G_perm >= G_c}) / (1 + perms). Opt-in (~4 min per analysis): conditions listed in --perm-conds.
Uncertainty: patient-level bootstrap of the seed-averaged per-pair loss differences.
Primary statistic (rlearner): the out-of-fold gain of base + (a - pi) theta, theta a constant effect per dimension
  fitted on the other folds from the base residuals (Robinson / R-learner form; silent on a = 0 pairs like the method's
  code channels), with a conditional randomization test (--null-draws fake onsets from the propensity; exact if the
  propensity is right) and the same stratified permutations. Benjamini-Hochberg over codes on the CRT p-values, per
  (subset, state, target, Z), in the summary.
Effect per dimension (the secondary estimator's direction, as a partially linear cross-fitted effect): theta_k =
  sum (a - pi)(Δ_k - f_base) / sum (a - pi)^2 with the out-of-fold propensity pi and base prediction, in SDs of Δ_k
  (and in raw units for b), bootstrap CI over patients.

Resumable: one JSON per (subset, state, target, Z, code) in outputs/gonogo_part1/gain/; existing results are skipped,
and a larger --perms only adds the missing permutations.

Example
-------
python gonogo/part1/gain.py                                  # sanity check 1 (positive controls, Δb and Δh)
python gonogo/part1/gain.py --codes pass --conds dt          # every code with >= 250 clean onsets
python gonogo/part1/gain.py --subset adjusted --conds dt     # sensitivity analysis
python gonogo/part1/gain.py --summary                        # print the tables of every result on disk
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from linkage import SETTINGS, load_z  # noqa: E402
from pairs import OUT, POSITIVE, load as load_pairs  # noqa: E402

RES = OUT / "gain"
ROLES = Path(__file__).resolve().parent / "control_roles.json"
# --label-sets: the frozen sets of go/no-go Phase 0.5d instead of Phase 0's label_sets.json (A = C0 without the
# validation subset, R = frozen R, rot<g> = the rotation codes outside group g); None keeps Phase 0's sets
LABEL_SETS_PATH = None
FROZEN_SHA = "c1feffa3508d9070af4994544c08d62efcc78d0a1e48732e1716b3816cf4bce4"
SET_NAMES = {"A", "R", "rot0", "rot1", "rot2", "rot3", "rot4"}
VENOUS = ["I80", "I82", "I87"]
CODINGS = ["icd9", "icd10", "unknown"]
Z_GROUPS = {"gap": ["log_gap"], "setting_t": ["setting_t"], "setting_t1": ["setting_t1"], "agesex": ["age", "male"],
            "coding": ["coding_t", "coding_t1"]}
Z_VARS = [v for vs in Z_GROUPS.values() for v in vs]
N_FOLDS = 20
KEY_DIMS = {"b": {"I44": ["qrs_width"], "I45": ["qrs_width"], "I47": ["hr"], "I48": ["rr_irr", "hr"],
                  "I49": ["rr_irr"]},
            "m": {"I44": ["qrs"], "I45": ["qrs"], "I47": ["hr"], "I48": ["p_missing", "hr"], "I49": ["hr"]}}
PROPENSITY_L2 = 1e-3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--codes", default="positive",
                   help="'positive' (the five controls), 'pass' (codes with >= --min-onsets onsets in the subset), "
                        "or a comma-separated list")
    p.add_argument("--min-onsets", type=int, default=250, help="the plan's event threshold")
    p.add_argument("--targets", default="b,h,m")
    p.add_argument("--conds", default="none,dt,z", help="conditioning sets Z")
    p.add_argument("--perm-conds", default="", help="conditions that get the (secondary) MLP permutation test")
    p.add_argument("--state", default="bhm", choices=["b", "h", "bh", "bm", "bhm"],
                   help="representations whose current value enters the state s_t")
    p.add_argument("--subset", default="clean", choices=["clean", "adjusted"])
    p.add_argument("--h-tag", default="zeroactions")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--perms", type=int, default=100)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--perms-per-call", type=int, default=10)
    p.add_argument("--null-draws", type=int, default=1000, help="CRT and permutation draws of the R-learner gain")
    p.add_argument("--save-residuals", action="store_true",
                   help="also write <result>.resid.npz (a - pi, base residuals): for effect vectors")
    p.add_argument("--propensity", default="logistic", choices=["logistic", "mlp", "gbm"],
                   help="cross-fitted propensity model (mlp: write to a separate --out-dir)")
    p.add_argument("--summary", action="store_true", help="only print the table of the results on disk")
    p.add_argument("--out-dir", default=str(RES))
    p.add_argument("--keep-going", action="store_true", help="log a failing code and continue with the next")
    p.add_argument("--label-sets", default="",
                   help="frozen sets of Phase 0.5d (outputs/phase05/freeze/label_sets.json, SHA-checked): A, R, rot<g>")
    return p.parse_args()


# ---------------------------------------------------------------- batched cross-fitted MLPs
class BatchedMLP(nn.Module):
    """M independent MLPs on shared inputs x (B, D); model m also reads its own binary column a[m], zero for base
    models (whose column weight then never gets a gradient). W1 is stored (D, M, H), so the first layer is one GEMM.
    Common random numbers: models of the same group (the same test fold) start from the same weights, drawn for the
    groups only (so a given seed gives the same start in every call), and the column weight starts at 0; trained on
    the same batches, a base and a full model then differ only through the column."""

    def __init__(self, d_in, d_out, hidden, group, n_groups=N_FOLDS):
        super().__init__()
        g = group.cpu()
        U = lambda shape, fan_in, dim=0: nn.Parameter(torch.empty(shape).uniform_(-fan_in ** -0.5, fan_in ** -0.5)
                                                      .index_select(dim, g).contiguous())
        self.W1, self.b1 = U((d_in, n_groups, hidden), d_in + 1, dim=1), U((n_groups, 1, hidden), d_in + 1)
        self.wa = nn.Parameter(torch.zeros(len(g), 1, hidden))
        self.W2, self.b2 = U((n_groups, hidden, hidden), hidden), U((n_groups, 1, hidden), hidden)
        self.W3, self.b3 = U((n_groups, hidden, d_out), hidden), U((n_groups, 1, d_out), hidden)

    def forward(self, x, a, idx=None):
        """x (B, D), a (M', B) -> (M', B, d_out) for the models idx (all if None)."""
        sel = (lambda t: t) if idx is None else (lambda t: t[idx])
        W1 = self.W1 if idx is None else self.W1[:, idx]
        D, M, H = W1.shape
        z = (x @ W1.reshape(D, M * H)).view(len(x), M, H).transpose(0, 1)
        z = F.gelu(z + sel(self.b1) + a[..., None] * sel(self.wa))
        z = F.gelu(torch.baddbmm(sel(self.b2), z, sel(self.W2)))
        return torch.baddbmm(sel(self.b3), z, sel(self.W3))


def fit_predict(X, Y, folds, A, test_fold, seed, args, keep_pred, loss="mse"):
    """Trains M = len(test_fold) models at once. Model m fits the pairs outside folds test_fold[m] and
    (test_fold[m] + 1) % 20, stops early on the latter, and predicts the former.
    X (N, D), Y (N, K) float32; folds (N,) long; A (M, N) float, model m's column (zeros for a base model).
    loss: "mse", or "bce" for binary Y (outputs are then logits).
    Returns, per model, its test fold's predictions (n_f, K) if keep_pred, else per-pair loss (n_f,), at the epoch of
    its best validation loss, plus that epoch."""
    crit = ((lambda p, y: ((p - y) ** 2).mean(-1)) if loss == "mse" else
            (lambda p, y: F.binary_cross_entropy_with_logits(p, y.expand_as(p), reduction="none").mean(-1)))
    torch.manual_seed(seed)
    M, (N, D), K = len(test_fold), X.shape, Y.shape[1]
    val_fold = (test_fold + 1) % N_FOLDS
    model = BatchedMLP(D, K, args.hidden, group=test_fold).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    train_ok = ((folds[None] != test_fold[:, None]) & (folds[None] != val_fold[:, None])).float()   # (M, N)
    rows = [torch.where(folds == f)[0] for f in range(N_FOLDS)]
    best = torch.full((M,), float("inf"), device="cuda")
    since = torch.zeros(M, dtype=torch.long, device="cuda")
    best_epoch = torch.zeros(M, dtype=torch.long, device="cuda")
    out = [None] * M
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for epoch in range(args.max_epochs):
        model.train()
        torch.backends.cuda.matmul.allow_tf32 = True          # training only; every reported loss is fp32
        perm = torch.randperm(N, device="cuda", generator=gen)
        for s in range(0, N, args.batch_size):
            i = perm[s:s + args.batch_size]
            w = train_ok[:, i]
            batch_loss = ((crit(model(X[i], A[:, i]), Y[i]) * w).sum(1) / w.sum(1).clamp(min=1)).sum()
            opt.zero_grad(set_to_none=True)
            batch_loss.backward()
            opt.step()
        torch.backends.cuda.matmul.allow_tf32 = False
        model.eval()
        with torch.no_grad():
            val = torch.empty(M, device="cuda")
            for f in range(N_FOLDS):
                ms = torch.where(val_fold == f)[0]
                if len(ms):
                    val[ms] = crit(model(X[rows[f]], A[ms][:, rows[f]], ms), Y[rows[f]]).mean(1)
            improved = val < best
            best, best_epoch = torch.where(improved, val, best), torch.where(improved, epoch, best_epoch)
            since = torch.where(improved, 0, since + 1)
            for f in range(N_FOLDS):
                ms = torch.where(improved & (test_fold == f))[0]
                if len(ms):
                    p = model(X[rows[f]], A[ms][:, rows[f]], ms)
                    res = p if keep_pred else crit(p, Y[rows[f]])
                    for j, m in enumerate(ms.tolist()):
                        out[m] = res[j].clone()
        if (since >= args.patience).all():
            break
    return out, best_epoch.cpu().numpy()


def assemble(out, models, rows, N):
    """Out-of-fold array (N, ...) from the per-model test-fold outputs of `models` (one per fold, in fold order)."""
    full = torch.empty((N,) + out[models[0]].shape[1:], device="cuda")
    for f, m in enumerate(models):
        full[rows[f]] = out[m]
    return full


# ---------------------------------------------------------------- propensity, placebo, permutations
def propensity(X, a, folds):
    """Cross-fitted logit P(a = 1 | X): fold k's model is an L2 logistic regression fitted on the other 19 folds."""
    W = torch.zeros(N_FOLDS, X.shape[1], device="cuda", requires_grad=True)
    b = torch.zeros(N_FOLDS, device="cuda", requires_grad=True)
    w = (folds[:, None] != torch.arange(N_FOLDS, device="cuda")[None]).float()       # (N, 20)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=500, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        nll = F.binary_cross_entropy_with_logits(X @ W.T + b, a[:, None].expand(-1, N_FOLDS), reduction="none")
        loss = ((nll * w).sum(0) / w.sum(0)).sum() + 0.5 * PROPENSITY_L2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return (X @ W.T + b).gather(1, folds[:, None]).squeeze(1)


def propensity_gbm(X, a, folds):
    """Cross-fitted logit P(a = 1 | X) from gradient-boosted trees (sklearn HistGradientBoostingClassifier, CPU): fold
    k's model fits the other 19 folds, with early stopping on a random 10% of them."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xc, ac, fc = X.cpu().numpy(), a.cpu().numpy().astype(int), folds.cpu().numpy()
    logit = np.zeros(len(ac))
    for k in range(N_FOLDS):
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=40,
                                           l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=20, random_state=k)
        m.fit(Xc[fc != k], ac[fc != k])
        p = m.predict_proba(Xc[fc == k])[:, 1].clip(1e-6, 1 - 1e-6)
        logit[fc == k] = np.log(p / (1 - p))
    return torch.tensor(logit, dtype=torch.float32, device="cuda")


def propensity_mlp(X, a, folds, args):
    """Cross-fitted logit P(a = 1 | X) from an MLP classifier (the base models' architecture with one output, binary
    cross-entropy, the same early stopping): captures interactions the logistic model misses."""
    rows = [torch.where(folds == f)[0] for f in range(N_FOLDS)]
    out, _ = fit_predict(X, a[:, None], folds, torch.zeros(N_FOLDS, len(a), device="cuda"),
                         torch.arange(N_FOLDS, device="cuda"), seed=0, args=args, keep_pred=True, loss="bce")
    return assemble(out, list(range(N_FOLDS)), rows, len(a))[:, 0]


def placebo_prob(logit, n_pos):
    """sigmoid(logit + shift), the shift set so that the expected count is n_pos."""
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if torch.sigmoid(logit + mid).sum() > n_pos else (mid, hi)
    return torch.sigmoid(logit + (lo + hi) / 2)


def placebo(logit, n_pos, gen, J=None):
    """Fake onsets drawn from the propensity with the real prevalence: (N,), or (J, N) for J draws."""
    p = placebo_prob(logit, n_pos)
    return torch.bernoulli(p if J is None else p.expand(J, -1).contiguous(), generator=gen)


def auroc(score, y):
    r = torch.empty_like(score)
    r[score.argsort()] = torch.arange(1, len(score) + 1, device=score.device, dtype=score.dtype)
    n1 = y.sum()
    return ((r[y > 0].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))).item()


# ---------------------------------------------------------------- patient bootstrap
class Bootstrap:
    """Patient-level bootstrap of ratios of sums: resampled sum(num) / sum(den) over pairs."""

    def __init__(self, subj, n_boot, seed=0):
        _, self.inv = np.unique(subj, return_inverse=True)
        self.P = self.inv.max() + 1
        rng = np.random.default_rng(seed)
        self.W = torch.tensor(rng.multinomial(self.P, np.full(self.P, 1 / self.P), size=n_boot),
                              dtype=torch.float32, device="cuda")                       # (n_boot, P)
        self.inv_t = torch.tensor(self.inv, device="cuda")

    def sums(self, x):
        x = x if x.dim() > 1 else x[:, None]
        return torch.zeros((self.P, x.shape[1]), device="cuda").index_add_(0, self.inv_t, x.float())

    def ratio(self, num, den):
        """(n_boot, K) resampled ratios; num (N, K) or (N,), den (N,) or (N, K)."""
        return (self.W @ self.sums(num)) / (self.W @ self.sums(den))


def ci(t):
    q = torch.quantile(t, torch.tensor([0.025, 0.975], device=t.device), dim=0)
    return q.squeeze(-1).cpu().numpy().tolist() if q.dim() > 1 and q.shape[-1] == 1 else q.cpu().numpy().tolist()


# ---------------------------------------------------------------- one analysis
def roles():
    """The approved negative-control roles (gonogo/part1/roles.py)."""
    r = json.loads(ROLES.read_text())
    assert r["status"] == "approved", f"{ROLES} is {r['status']}, not approved"
    return r["roles"]


def presence(data, code):
    """(present at t, present at t+1) per pair, for a cardiac code (the 76 labels) or any other 3-character code."""
    if code in data["vocab"]:
        j = data["vocab"].index(code)
        return data["yt"][:, j] == 1, data["yt1"][:, j] == 1
    k = data["full_names"].index(code)
    return data["full_t"][:, k].toarray().ravel() > 0, data["full_t1"][:, k].toarray().ravel() > 0


def at_risk(data, code, subset):
    """The analysis pairs of code c: between episodes, usable b and m, onset admissible (c absent at t), and for
    clean: no change in the 76 cardiac codes except, for a cardiac code, c's own onset."""
    pairs = data["pairs"]
    sel = data["usable"] & ~presence(data, code)[0]
    if subset == "clean":
        if code in data["vocab"]:
            j = data["vocab"].index(code)
            sel &= pairs["stable"].values | ((pairs["single_code"].values == j) & (pairs["single_dir"].values == 1))
        else:
            sel &= pairs["stable"].values
    return sel


def resolve_codes(spec, data, args):
    """Comma-separated codes or presets: positive (the five controls), pass (cardiac codes with >= --min-onsets onsets
    in the subset), ambiguous (pass without the positive controls), c1 / anchor / heldout_anchor (approved roles)."""
    codes = []
    for s in spec.split(","):
        if s == "positive":
            codes += POSITIVE
        elif s in ("pass", "ambiguous"):
            n = {c: int(presence(data, c)[1][at_risk(data, c, args.subset)].sum()) for c in data["vocab"]}
            codes += [c for c in data["vocab"] if n[c] >= args.min_onsets and (s == "pass" or c not in POSITIVE)]
        elif s in ("c1", "anchor", "heldout_anchor"):
            codes += roles()["c1_control" if s == "c1" else s]
        else:
            codes.append(s)
    return list(dict.fromkeys(codes))


def group_of(code):
    if code in POSITIVE:
        return "positive"
    if code in VENOUS:
        return "venous"
    if ROLES.exists() and json.loads(ROLES.read_text())["status"] == "approved":
        for role, codes in roles().items():
            if code in codes:
                return role
    return "ambiguous"


def analysis_set(data, code, target, cond, args):
    pairs, yt, yt1, vocab = data["pairs"], data["yt"], data["yt1"], data["vocab"]
    sel = at_risk(data, code, args.subset)
    P = pairs[sel]
    a = presence(data, code)[1][sel].astype(np.float32)
    get = lambda name, col: data[name][data[f"{name}_lookup"].loc[P[col].values].values]
    state = [get(name, "idx_t") for name in args.state]
    Zp = data["z"][sel]
    log_gap = Zp["log_gap"].values[:, None]
    onehot = lambda col, levels: (Zp[col].values[:, None] == np.array(levels)[None]).astype(np.float32)
    zgroups = {"gap": [log_gap], "setting_t": [onehot("setting_t", SETTINGS)],
               "setting_t1": [onehot("setting_t1", SETTINGS)], "agesex": [Zp[["age", "male"]].to_numpy(np.float32)],
               "coding": [onehot("coding_t", CODINGS), onehot("coding_t1", CODINGS)]}
    z = [c for cs in zgroups.values() for c in cs]
    base, extra = parse_cond(cond)
    zcols = {"none": [], "dt": [log_gap], "z": z}
    if base == "za":                                # legacy: the 11 anchors, standardized with the state
        zcols["za"] = z + [data["anchor_trans"][sel][:, [i for i, n in enumerate(data["anchor_names"])
                                                          if not n.startswith(code + "_")]]]
    if base.startswith("z-"):                       # drop-one driver analysis (2.9): z without one group
        zcols[base] = [c for g, cs in zgroups.items() if g != base[2:] for c in cs]
    cont = np.concatenate(state + zcols[base], 1)
    cont = (cont - cont.mean(0)) / (cont.std(0) + 1e-6)
    cols = [cont]
    for s in extra:                                 # label-transition sets, as 0/1 columns (the tested code removed)
        M, names = data["label_trans"][s]
        cols.append(M[sel][:, [i for i, n in enumerate(names) if not n.startswith(code + "_")]].astype(np.float32))
    if args.subset == "adjusted":
        d = yt1[sel].astype(np.int8) - yt[sel]
        other = np.concatenate([d == 1, d == -1], 1)
        if code in vocab:
            j = vocab.index(code)
            other = np.delete(other, [j, 76 + j], axis=1)
        cols.append(other.astype(np.float32))
    delta = get(target, "idx_t1") - get(target, "idx_t")
    sd = delta.std(0) + 1e-6
    T = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device="cuda")
    # The plan's permutation strata (2.5): gap quintile x care setting at t+1 x age tertile
    cut = lambda x, k: np.searchsorted(np.quantile(x, np.linspace(0, 1, k + 1)[1:-1]), x)
    strata = (cut(Zp["log_gap"].values, 5) * 3 + pd.Categorical(Zp["setting_t1"], categories=SETTINGS).codes) * 3 \
        + cut(Zp["age"].values, 3)
    return {"X": T(np.concatenate(cols, 1)), "Y": T((delta - delta.mean(0)) / sd), "sd": sd,
            "a": T(a), "folds": torch.tensor(P["fold"].values, dtype=torch.long, device="cuda"),
            "subj": P["subject_id"].values, "strata": strata, "rows": np.where(sel)[0]}


def parse_cond(cond):
    """'z-setting_t1+anc+A' -> ('z-setting_t1', ['anc', 'A']). Bases: none, dt, z, z-<group of Z_GROUPS>, and the
    legacy za (z + the 11 anchors standardized with the state). Sets: anc (the 11 approved anchors), A and R
    (gonogo/part1/label_sets.py), each as onset and resolution indicators."""
    base, *extra = cond.split("+")
    assert base in ("none", "dt", "z", "za") or (base.startswith("z-") and base[2:] in Z_GROUPS), cond
    assert set(extra) <= {"anc", "diag"} | SET_NAMES and len(set(extra)) == len(extra), cond
    assert LABEL_SETS_PATH is not None or not {s for s in extra if s.startswith("rot")}, "rot<g> needs --label-sets"
    return base, extra


def label_sets():
    if LABEL_SETS_PATH is None:
        return json.loads((OUT / "label_sets.json").read_text())
    import hashlib
    raw = Path(LABEL_SETS_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FROZEN_SHA, "the frozen label_sets.json has changed"
    fz = json.loads(raw)
    val = set(fz["validation_c1"])
    sets = {"A": sorted(c for c in fz["C0"] if c not in val), "R": sorted(fz["R"])}
    for g in fz["rotation_groups"]:
        sets[f"rot{g}"] = sorted(c for gg, cs in fz["rotation_groups"].items() if gg != g for c in cs)
    return sets


def hparams(args, cond):
    """Everything a result depends on; a cached result with other hparams is recomputed. z's variables, the anchors
    and the label sets are part of it for the conditions that use them."""
    hp = {"init": "shared per fold (CRN)", "episodes": "between", "strata": "gap5 x setting_t1 x age3",
          **{k: getattr(args, k) for k in ("hidden", "batch_size", "lr", "wd", "max_epochs", "patience", "seeds",
                                           "boot", "h_tag", "perms_per_call", "null_draws")}}
    base, extra = parse_cond(cond)
    if base in ("z", "za"):
        hp["z_vars"] = Z_VARS
    if base.startswith("z-"):
        hp["z_vars"] = [v for g, vs in Z_GROUPS.items() if g != base[2:] for v in vs]
    if base == "za" or "anc" in extra:
        hp["anchors"] = roles()["anchor"]
    for s in sorted(SET_NAMES):
        if s in extra:
            hp[f"set_{s}"] = sorted(label_sets()[s])
            if LABEL_SETS_PATH is not None:
                hp["label_sets_sha256"] = FROZEN_SHA
    if "diag" in extra:
        hp["diag"] = "task3_diag/pair_covariates.npz (K, Mg, TSH, opioid, laxative within 24 h, at t and t+1)"
    if args.propensity != "logistic":
        hp["propensity"] = args.propensity
    return hp


def run_one(data, code, target, cond, path, args):
    t0 = time.time()
    res = json.loads(path.read_text()) if path.exists() else None
    if res and res["hparams"] != hparams(args, cond):
        print(f"  {path.name} was computed with other hparams; recomputing it", flush=True)
        res = None
    S = analysis_set(data, code, target, cond, args)
    X, Y, a, folds = S["X"], S["Y"], S["a"], S["folds"]
    N, K = Y.shape
    rows = [torch.where(folds == f)[0] for f in range(N_FOLDS)]
    fold_ids = torch.arange(N_FOLDS, device="cuda")
    n_on = int(a.sum().item())
    boot = Bootstrap(S["subj"], args.boot)
    ones = torch.ones(N, device="cuda")

    logit = {"logistic": lambda: propensity(X, a, folds), "mlp": lambda: propensity_mlp(X, a, folds, args),
             "gbm": lambda: propensity_gbm(X, a, folds)}[args.propensity]()
    pi = torch.sigmoid(logit)
    # ---- base, full and placebo, one batched call per seed
    se_base, pred_base, se_full, se_plac, G, G_plac, epochs = [], [], [], [], [], [], []
    for s in range(args.seeds):
        gen = torch.Generator(device="cuda").manual_seed(1000 + s)
        A = torch.stack([torch.zeros_like(a), a, placebo(logit, n_on, gen)])           # (3, N)
        A = A.repeat_interleave(N_FOLDS, 0)                                            # variant-major, fold-minor
        fit, ep = fit_predict(X, Y, folds, A, fold_ids.repeat(3), seed=s, args=args, keep_pred=True)
        pb, pf, pp = (assemble(fit, list(range(v * N_FOLDS, (v + 1) * N_FOLDS)), rows, N) for v in range(3))
        sb, sf, sp = ((p - Y) ** 2 for p in (pb, pf, pp))                              # (N, K)
        se_base.append(sb), pred_base.append(pb), se_full.append(sf), se_plac.append(sp)
        G.append((sb - sf).mean().item()), G_plac.append((sb - sp).mean().item())
        epochs.append(ep.tolist())
        print(f"  seed {s}: L_base {sb.mean().item():.5f}  G {G[-1]:+.2e}  placebo {G_plac[-1]:+.2e}  "
              f"best epochs median {np.median(ep):.0f}", flush=True)
    sb, sf, sp = (torch.stack(x).mean(0) for x in (se_base, se_full, se_plac))
    d, dp = (sb - sf).mean(1), (sb - sp).mean(1)                                        # per pair, over dims

    out = {"code": code, "target": target, "cond": cond, "state": args.state, "subset": args.subset,
           "n_pairs": N, "n_onsets": n_on, "n_patients": int(len(np.unique(S["subj"]))), "dims": K,
           "L_base": sb.mean().item(), "L_full": sf.mean().item(),
           "G": float(np.mean(G)), "G_seeds": G, "G_per_onset": float(np.mean(G)) * N / n_on,
           "G_ci": ci(boot.ratio(d, ones)), "G_placebo": float(np.mean(G_plac)), "G_placebo_seeds": G_plac,
           "G_placebo_ci": ci(boot.ratio(dp, ones)), "G_minus_placebo_ci": ci(boot.ratio(d - dp, ones)),
           "propensity_auroc": auroc(logit, a), "best_epochs": epochs,
           "hparams": hparams(args, cond)}
    # ---- per-dimension gain and partially linear effect
    ra, ry = a - pi, Y - torch.stack(pred_base).mean(0)
    if args.save_residuals:     # for effect vectors with patient-level CIs (gonogo/part1/effects.py)
        np.savez(path.with_name(path.stem + ".resid.npz"), ra=ra.cpu().numpy().astype(np.float32),
                 ry=ry.cpu().numpy().astype(np.float16), folds=folds.cpu().numpy().astype(np.int8),
                 subj=S["subj"], sd=np.asarray(S["sd"], dtype=np.float32))
    theta = ((ra[:, None] * ry).sum(0) / (ra ** 2).sum()).cpu().numpy()
    theta_bt = boot.ratio(ra[:, None] * ry, ra ** 2)
    G_dim, G_dim_bt = (sb - sf).mean(0).cpu().numpy(), boot.ratio(sb - sf, ones)
    if target in ("b", "m"):
        lo_t, hi_t = (np.array(x) for x in ci(theta_bt))
        lo_g, hi_g = (np.array(x) for x in ci(G_dim_bt))
        out["per_dim"] = {n: {"G": float(G_dim[k]), "G_ci": [float(lo_g[k]), float(hi_g[k])],
                              "theta_sd": float(theta[k]), "theta_sd_ci": [float(lo_t[k]), float(hi_t[k])],
                              "theta_raw": float(theta[k] * S["sd"][k])}
                          for k, n in enumerate(data[f"{target}_names"])}
    else:
        out["theta_h"] = theta.tolist()
        out["theta_h_norm"] = float(np.linalg.norm(theta))
        out["theta_h_norm_ci"] = ci(theta_bt.norm(dim=1, keepdim=True))

    strata = torch.tensor(S["strata"], device="cuda", dtype=torch.float32)
    by_stratum = torch.argsort(strata, stable=True)

    # ---- low-variance variant: cross-fitted R-learner gain. Full = base + (a - pi) theta^(-k), with a constant effect
    # per dimension fitted on the folds other than k from the out-of-fold base residuals; silent on a = 0 pairs up to
    # pi, so the training noise of a second network cannot enter there. Per dimension, from per-fold sums only:
    # G_k = (1/N) sum_f [2 theta_fk num_fk - theta_fk^2 den_f], num_fk = sum_{i in f} (a - pi) ry_ik,
    # den_f = sum_{i in f} (a - pi)^2. Nulls: CRT (fake onsets from the propensity, the null "Δ independent of a given
    # the state") and the stratified permutation of the plan.
    onehot = F.one_hot(folds, N_FOLDS).float()                                          # (N, 20)

    def fold_sums(Amat):
        r = Amat - pi[None]
        return (r[:, None, :] * onehot.T[None]) @ ry, (r ** 2) @ onehot                 # (J, 20, K), (J, 20)

    def loo_theta(num, den):
        return (num.sum(1, keepdim=True) - num) / (den.sum(1, keepdim=True) - den)[..., None]

    def dim_gains(Amat):
        num, den = fold_sums(Amat)
        th = loo_theta(num, den)
        return (2 * th * num - th ** 2 * den[..., None]).sum(1) / N                     # (J, K)

    def null_draws(kind, J, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        if kind == "crt":
            return placebo(logit, n_on, gen, J)
        A = torch.empty((J, N), device="cuda")
        for r in range(J):
            A[r, by_stratum] = a[torch.argsort(strata + torch.rand(N, device="cuda", generator=gen))]
        return A

    ra1 = a - pi
    th1 = loo_theta(*fold_sums(a[None]))[0]                                             # (20, K)
    g_pair = 2 * ra1[:, None] * th1[folds] * ry - (ra1 ** 2)[:, None] * th1[folds] ** 2   # (N, K) per-pair gains
    GR_dim = g_pair.mean(0)
    GR = GR_dim.mean().item()
    rl = {"G": GR, "G_per_onset": GR * N / n_on, "G_ci": ci(boot.ratio(g_pair.mean(1), ones))}
    nulls = {}
    for kind, offset in (("crt", 20_000), ("perm", 30_000)):
        chunk = 25
        nulls[kind] = torch.cat([dim_gains(null_draws(kind, chunk, offset + c)) for c in range(args.null_draws // chunk)])
        tot = nulls[kind].mean(1)
        rl[kind] = {"n": len(tot), "p": float((1 + (tot >= GR).sum().item()) / (1 + len(tot))),
                    "mean": tot.mean().item(), "sd": tot.std().item()}
    out["rlearner"] = rl
    # Split-half effects for unbiased squared norms (folds 0-9 vs 10-19 are disjoint patients), in z-units
    num, den = fold_sums(a[None])
    out["theta_halves"] = [(num[0, h].sum(0) / den[0, h].sum()).cpu().numpy().tolist()
                           for h in (slice(0, N_FOLDS // 2), slice(N_FOLDS // 2, N_FOLDS))]
    out["theta_se"] = theta_bt.std(0).cpu().numpy().tolist()
    out["target_sd"] = np.asarray(S["sd"]).tolist()
    # Per-pair gains, for paired patient-level bootstraps across conditions (gonogo/part1/ci.py)
    np.savez(path.with_name(path.stem + ".pairs.npz"), rows=S["rows"].astype(np.int32), subj=S["subj"],
             g=g_pair.mean(1).cpu().numpy().astype(np.float32), a=a.cpu().numpy().astype(np.int8))
    if target in ("b", "m"):
        lo_r, hi_r = (np.array(x) for x in ci(boot.ratio(g_pair, ones)))
        p_dim = {kind: ((1 + (nulls[kind] >= GR_dim[None]).sum(0)) / (1 + len(nulls[kind]))).cpu().numpy()
                 for kind in nulls}
        for k, n in enumerate(data[f"{target}_names"]):
            out["per_dim"][n].update({"GR": GR_dim[k].item(), "GR_ci": [float(lo_r[k]), float(hi_r[k])],
                                      "p_crt": float(p_dim["crt"][k]), "p_perm": float(p_dim["perm"][k])})
    print(f"  R-learner G {GR:+.2e} CI [{rl['G_ci'][0]:+.1e}, {rl['G_ci'][1]:+.1e}]  p_crt {rl['crt']['p']:.4f} "
          f"(null {rl['crt']['mean']:+.1e} ± {rl['crt']['sd']:.1e})  p_perm {rl['perm']['p']:.4f}", flush=True)
    out["time_s"] = time.time() - t0 + (res or {}).get("time_s", 0.0)
    path.write_text(json.dumps(out, indent=1))

    # ---- stratified permutation test of the MLP gain (resumable: adds only the missing permutations)
    if cond in args.perm_conds.split(","):
        prev = res.get("perm", {}).get("gains", []) if res else []
        gains = list(prev)
        base_seed = [x.mean(1) for x in se_base]
        while len(gains) < args.perms:
            n = min(args.perms_per_call, args.perms - len(gains))
            s_call = (len(gains) // args.perms_per_call) % args.seeds       # shares init and batches with that base
            gen = torch.Generator(device="cuda").manual_seed(10_000 + len(gains))
            A = torch.empty((n, N), device="cuda")
            for r in range(n):
                order = torch.argsort(strata + torch.rand(N, device="cuda", generator=gen))
                A[r, by_stratum] = a[order]
            out_p, _ = fit_predict(X, Y, folds, A.repeat_interleave(N_FOLDS, 0), fold_ids.repeat(n),
                                   seed=s_call, args=args, keep_pred=False)
            for r in range(n):
                se_p = assemble(out_p, list(range(r * N_FOLDS, (r + 1) * N_FOLDS)), rows, N)
                gains.append((base_seed[s_call] - se_p).mean().item())
            g = np.array(gains)
            out["perm"] = {"gains": gains, "n": len(gains), "p": float((1 + (g >= out["G"]).sum()) / (1 + len(g))),
                           "mean": float(g.mean()), "sd": float(g.std()), "strata": "gap5 x setting_t1 x age3"}
            out["time_s"] = time.time() - t0 + (res or {}).get("time_s", 0.0)
            path.write_text(json.dumps(out, indent=1))           # a cut job keeps every finished batch
            print(f"  permutations {len(gains)}/{args.perms}", flush=True)
    out["time_s"] = time.time() - t0 + (res or {}).get("time_s", 0.0)
    path.write_text(json.dumps(out, indent=1))
    return out


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    out = np.empty_like(p)
    out[order] = np.minimum(np.minimum.accumulate(q[::-1])[::-1], 1.0)
    return out


def summary(res_dir, min_onsets):
    """The primary statistic (R-learner gain, CRT; Benjamini-Hochberg over the codes above the event threshold, per
    subset / state / target / Z, separately for the positive controls and for the GO family: C1 and venous controls
    and ambiguous codes), the plan's MLP gain, and the C1 view: per code, the gain under each Z (2.7, 2.8)."""
    recs = [json.loads(p.read_text()) for p in sorted(Path(res_dir).glob("*.json"))]
    if not recs:
        return
    df = pd.DataFrame([{"subset": r["subset"], "state": r["state"], "target": r["target"], "Z": r["cond"],
                        "code": r["code"], "group": group_of(r["code"]), "onsets": r["n_onsets"],
                        "G_R": r["rlearner"]["G"], "p": r["rlearner"]["crt"]["p"]} for r in recs])
    df["family"] = np.where(df["group"] == "positive", "positive", "go")
    df["q"] = np.nan
    ok = df["onsets"] >= min_onsets
    for _, g in df[ok].groupby(["subset", "state", "target", "Z", "family"]):
        df.loc[g.index, "q"] = bh(g["p"].values)
    primary, secondary = [], []
    for (_, row), r in zip(df.iterrows(), recs):
        rl, base = r["rlearner"], {k: row[k] for k in ("subset", "state", "target", "Z", "code", "group", "onsets")}
        key = ""
        if r["target"] in KEY_DIMS and r["code"] in KEY_DIMS[r["target"]]:
            key = "  ".join(f"{n} θ {r['per_dim'][n]['theta_sd']:+.2f} SD [{r['per_dim'][n]['theta_sd_ci'][0]:+.2f}, "
                            f"{r['per_dim'][n]['theta_sd_ci'][1]:+.2f}] p_crt {r['per_dim'][n]['p_crt']:.3f}"
                            for n in KEY_DIMS[r["target"]][r["code"]])
        primary.append({**base, "status": "ok" if row["onsets"] >= min_onsets else f"insufficient (<{min_onsets})",
                        "G_R": f"{rl['G']:+.2e}", "CI": f"[{rl['G_ci'][0]:+.1e}, {rl['G_ci'][1]:+.1e}]",
                        "CRT null": f"{rl['crt']['mean']:+.1e} ± {rl['crt']['sd']:.1e}",
                        "p_crt": f"{rl['crt']['p']:.4f}", "q_BH": "" if np.isnan(row["q"]) else f"{row['q']:.4f}",
                        "p_perm": f"{rl['perm']['p']:.4f}", "prop AUROC": f"{r['propensity_auroc']:.3f}",
                        "pre-specified dims": key})
        secondary.append({**base, "G_MLP": f"{r['G']:+.2e}", "CI": f"[{r['G_ci'][0]:+.1e}, {r['G_ci'][1]:+.1e}]",
                          "seeds": " ".join(f"{g:+.1e}" for g in r["G_seeds"]), "placebo": f"{r['G_placebo']:+.1e}",
                          "p_perm": f"{r['perm']['p']:.3f} ({r['perm']['n']})" if "perm" in r else ""})
    with pd.option_context("display.width", 250, "display.max_colwidth", 140, "display.max_rows", 2000):
        print("PRIMARY: R-learner gain, CRT p-value, BH over codes above the event threshold")
        print(pd.DataFrame(primary).to_string(index=False))
        print("\nSECONDARY: the plan's MLP gain (base vs full)")
        print(pd.DataFrame(secondary).to_string(index=False))
        # C1 view: G_R x 1e5 per target and Z; * q_BH < 0.05 in the family; ~ p_crt < 0.05 below the threshold
        df["cell"] = [f"{g * 1e5:+.1f}" + (("*" if q < 0.05 else "") if not np.isnan(q) else ("~" if p < 0.05 else ""))
                      for g, q, p in zip(df["G_R"], df["q"], df["p"])]
        for subset, d in df.groupby("subset"):
            have = set(zip(d["target"], d["Z"]))
            cols = [(t, z) for t in ("h", "m", "b") for z in ("none", "dt", "z", "za") if (t, z) in have]
            view = d.pivot_table(index=["group", "code", "onsets"], columns=["target", "Z"], values="cell",
                                 aggfunc="first").reindex(columns=cols)
            print(f"\nC1 VIEW ({subset}): G_R x 1e5 by target and Z; * q_BH < 0.05 in the family; ~ p_crt < 0.05 "
                  f"below the event threshold")
            print(view.fillna("").to_string())


def main():
    args = parse_args()
    global LABEL_SETS_PATH
    if args.label_sets:
        LABEL_SETS_PATH = args.label_sets
        label_sets()                                # fails early if the frozen file changed
    if args.summary:
        return summary(args.out_dir, args.min_onsets)
    res_dir = Path(args.out_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    targets = args.targets.split(",")
    pairs, yt, yt1, vocab, b_lookup, b, b_names = load_pairs()
    data = {"pairs": pairs, "yt": yt, "yt1": yt1, "vocab": vocab, "b": b, "b_lookup": b_lookup, "b_names": b_names,
            "z": load_z()}
    if "h" in targets or "h" in args.state:
        hz = np.load(OUT / f"h_{args.h_tag}.npz")
        data["h"], data["h_lookup"] = hz["h"], pd.Series(np.arange(len(hz["rows"])), index=hz["rows"])
    mz = np.load(OUT / "m.npz")
    data["m"], data["m_names"] = mz["m"], [str(n) for n in mz["names"]]
    data["m_lookup"] = pd.Series(np.arange(len(mz["rows"])), index=mz["rows"])
    has_m = mz["has_m"][data["m_lookup"].loc[pairs["idx_t"].values].values] \
        & mz["has_m"][data["m_lookup"].loc[pairs["idx_t1"].values].values]
    # Same-episode pairs are stable by construction (decision of 2026-10-22): noise floor only, never analysed here
    data["usable"] = pairs["has_b"].values & has_m & ~data["z"]["same_episode"].values
    print(f"usable pairs {data['usable'].sum():,} (between episodes, b and m usable at both ECGs)", flush=True)
    full = sp.load_npz(OUT / "codes_full.npz").tocsr()
    data["full_names"] = [str(n) for n in np.load(OUT / "codes_full_names.npy")]
    data["full_t"] = full[pairs["idx_t"].values].tocsc()
    data["full_t1"] = full[pairs["idx_t1"].values].tocsc()
    def transitions(codes):
        """Onset and resolution indicators of `codes` per pair, (n_pairs, 2 len(codes)) uint8, and column names."""
        cols, names = [], []
        for c in codes:
            at, at1 = presence(data, c)
            cols += [at1 & ~at, at & ~at1]
            names += [f"{c}_onset", f"{c}_resolution"]
        return np.stack(cols, 1).astype(np.uint8), names

    parsed = [parse_cond(c) for c in args.conds.split(",")]
    if any(base == "za" for base, _ in parsed):
        # Anchor transitions: onset and resolution indicators of each approved anchor (method_section.tex, a^{C0}_t)
        M, data["anchor_names"] = transitions(roles()["anchor"])
        data["anchor_trans"] = M.astype(np.float32)
    needed = {s for _, extra in parsed for s in extra}
    data["label_trans"] = {s: transitions(roles()["anchor"] if s == "anc" else sorted(label_sets()[s]))
                           for s in needed - {"diag"}}
    if "diag" in needed:        # Task 3's diagnostic covariates (gonogo/part1/k59_diag.py), z-scored, per pair
        dz = np.load(OUT / "task3_diag" / "pair_covariates.npz")
        assert len(dz["X"]) == len(pairs)
        data["label_trans"]["diag"] = (dz["X"], [str(n) for n in dz["names"]])
    codes = resolve_codes(args.codes, data, args)
    print(f"{len(codes)} codes: {','.join(codes)}", flush=True)
    for target in targets:
        for cond in args.conds.split(","):
            for code in codes:
                path = res_dir / f"{args.subset}_{args.state}_{target}_{cond}_{code}.json"
                if path.exists():
                    r = json.loads(path.read_text())
                    if r["hparams"] == hparams(args, cond) and (cond not in args.perm_conds.split(",")
                                                          or r.get("perm", {}).get("n", 0) >= args.perms):
                        continue
                print(f"{args.subset} state={args.state} Δ{target} Z={cond} {code}", flush=True)
                try:
                    r = run_one(data, code, target, cond, path, args)
                except Exception as e:                      # --keep-going: record the failure, go on to the next code
                    if not args.keep_going:
                        raise
                    print(f"  FAILED {code} Δ{target} Z={cond}: {type(e).__name__}: {e}", flush=True)
                    torch.cuda.empty_cache()
                    continue
                print(f"  G {r['G']:+.2e} CI [{r['G_ci'][0]:+.1e}, {r['G_ci'][1]:+.1e}]  placebo {r['G_placebo']:+.1e}"
                      f"  onsets {r['n_onsets']:,}/{r['n_pairs']:,}"
                      + (f"  p_perm {r['perm']['p']:.3f}" if "perm" in r else "") + f"  ({r['time_s']:.0f} s)",
                      flush=True)
    summary(res_dir, args.min_onsets)


if __name__ == "__main__":
    main()

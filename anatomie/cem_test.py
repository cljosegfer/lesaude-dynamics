"""
CEM Planning Test for Forward Checkpoints (CNN vs ViT)
=======================================================
Can a forward checkpoint's own dynamics model read the diagnosis change of a held-out pair (X_t, X_t+1, y_t) out of
its energy
    E(a) = mean_D ||Dyn(h_t, Proj(a)) - h_t+1||^2,
and does it read it better on pairs it trained on? Every readout searches or scores admissible actions only (onset
where y_t,i = 0, resolution where y_t,i = 1) under the train base-rate priors of scripts/cem_planning.py.

Populations: k=1 cross-stay pairs of pairs.lance (the pairs of scripts/cem_planning.py and evaluate_inverse.py)
-----------
  test      fold 19, the paper's test pairs: non-members
  members   fold 17, a training fold built the same way; the share inside each checkpoint's own training pairs is
            checked (multistep checkpoints train on every hop k <= 12, k = 1 included)
  val       fold 18: picks the flip readout's temperature, nothing else

Readouts of the same energy
---------------------------
  cem       scripts/cem_planning.py's OnsetResolutionCEM with plain E and its config defaults: contextualized soft
            (y_t (1 - p_res) + (1 - y_t) p_ons from the final marginals) and hard (clip(y_t + a*)) AUROC; flips per
            pair, E(a*) / E(0), exact matches and search failures [E(a*) > E(a_gt)] on pairs with a change
  decoys    no search: the true action ranked by E against admissible decoys, the families and n_decoys of
            scripts/pretrain.py's _DecoyProbe (same: the true numbers of onsets and resolutions on random labels;
            extra: the truth plus one flip, where < 0.5 means flipping more lowers E, the exploit; zero: doing
            nothing, P(E(a_gt) < E(0)) on pairs with a change; prior: draws from the base-rate prior, CEM's first
            proposal). Within-pair AUROC (ties 1/2), averaged over pairs; chance 0.5
  flip      per-label readout (anatomie/flip_readout.py): delta_i = E(0) - E(e_i) for the single admissible flip e_i
            of label i, p = sigmoid(logit(prior_i) + delta_i / tau), contextualized as above; tau from a grid scaled
            by the checkpoint's own delta spread on val (embedding scales differ between backbones), tau = inf being
            the prior alone, i.e. carry-forward. Also onset / resolution AUROC of delta alone over the live entries:
            prior-free, does E single out the labels that changed?
Every decoy and flip metric is repeated against the pair's own h_t instead of h_t+1 ("nothing changed": same patient,
no change). What survives there is read from h_t (which changes are plausible for this patient), not from the
outcome; the evidence the energy carries is real - nothing changed.

References on the same pairs: carry-forward (y_t) and the inverse head of --inverse (paper Inverse,
Inv(h_t+1 - h_t)), whose fold 17 is in its own training pairs too.

Statistics: point estimates on all pairs; 95% intervals from a patient-level bootstrap (--n-boot, seed 42; a
patient's pairs share ECGs). The resamples are the same for every metric and every checkpoint, so evidence (real -
nothing changed) and --compare differences (A - B, also of the evidence) are paired.

Example
-------
python anatomie/cem_test.py \\
    --ckpt cnn=runs/runs/20260930/084313/e9ebd97377ce/checkpoints/last.ckpt \\
    --ckpt vit=runs/runs/20261006/044610/064d3fbda75f/checkpoints/last.ckpt \\
    --cache /scratch/josefernandes/cache/cem_test_h.pt --fig outputs/cem_test/cem_test.png
python anatomie/cem_test.py --plot-only --fig outputs/cem_test/cem_test.png
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import REPO, data_paths, embed, load_dynamics, load_encoder, load_inverse_head  # noqa: E402

POPS = {"members": 17, "val": 18, "test": 19}       # population -> fold of pairs.lance's cross-stay pairs
SCORED = ("test", "members")
FAMILIES = ("same", "extra", "zero", "prior")
# Metrics with a nothing-changed twin: evidence = real - nothing changed, from the same resamples
EVIDENCE = ("flip", "onset", "resolution", *(f"decoy_{f}" for f in FAMILIES))
# Multiples of the val delta spread. Large tau is carry-forward with E breaking the ties within each label's
# y_t = 0 and y_t = 1 groups, so the grid runs far enough to reach that limit.
TAU_GRID = 10.0 ** np.arange(-1.5, 4.01, 0.5)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", default=[], metavar="NAME=SPEC",
                   help="forward checkpoint to test (repeatable)")
    p.add_argument("--inverse", default="archive/inverse.ckpt", help="reference inverse checkpoint ('' for none)")
    p.add_argument("--compare", nargs="*", default=["cnn:vit"], metavar="A:B", help="paired differences A - B")
    p.add_argument("--n-samples", type=int, default=512)
    p.add_argument("--n-elites", type=int, default=64)
    p.add_argument("--n-iters", type=int, default=16)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--p-min", type=float, default=0.005)
    p.add_argument("--n-decoys", type=int, default=16)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-pairs", type=int, default=None, help="first N pairs of each population (smoke tests)")
    p.add_argument("--batch-pairs", type=int, default=256, help="pairs per CEM / scoring batch")
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--cache", default=None, help="optional .pt file for h, reused per checkpoint file")
    p.add_argument("--out", default="outputs/cem_test/results.json")
    p.add_argument("--fig", default=None)
    p.add_argument("--plot-only", action="store_true", help="redraw --fig from an existing --out")
    return p.parse_args()


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- populations
def populations(paths, max_pairs):
    """Cross-stay pairs of each population, in pairs.lance order (as MIMICLanceDataset(mode="pair") reads them)."""
    t = (lance.dataset(paths["pairs_path"])
         .to_table(filter="pair_type = 'cross_stay'", columns=["idx_t", "idx_t1", "subject_id", "fold"]).to_pandas())
    return {p: t[t["fold"] == f].reset_index(drop=True).iloc[:max_pairs] for p, f in POPS.items()}


def training_info(spec, paths, pops):
    """What the checkpoint trained on, and the share of each population's pairs inside its own training pairs."""
    ck = torch.load(REPO / spec, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters") or {}
    vit = hp.get("vit")
    local = {Path(paths[k]).name: paths[k] for k in ("pairs_path", "pairs_multistep_path")}
    name = Path(str(hp.get("pairs_path", ""))).name
    types = tuple(hp.get("pair_types") or ("within_stay", "cross_stay"))
    t = lance.dataset(local[name]).to_table(columns=["idx_t", "idx_t1", "fold", "pair_type"]).to_pandas()
    t = t[(t["fold"] <= 17) & t["pair_type"].isin(types)]
    trained = pd.MultiIndex.from_arrays([t["idx_t"].values, t["idx_t1"].values])
    inside = lambda df: float(pd.MultiIndex.from_arrays([df["idx_t"].values, df["idx_t1"].values]).isin(trained).mean())
    return {"backbone": f"ViT1d {dict(vit)['dim']}x{dict(vit)['depth']}" if hp.get("backbone") == "vit1d"
            else "ResNet1d", "epochs": int(ck.get("epoch", -1)) + 1, "pairs": name, "pair_types": list(types),
            "train_frac": float(hp.get("train_frac", 1.0)), "lambda_reg": hp.get("lambda_reg"),
            "predictor_hidden_dim": hp.get("predictor_hidden_dim"),
            "in_training_pairs": {p: inside(df) for p, df in pops.items()}}


def lance_labels(lance_path, rows):
    """The 76 labels of the ECGs at sorted Lance rows `rows`, (N, 76) float32."""
    icd = lance.dataset(lance_path).take(rows.tolist(), columns=["icd"]).column("icd").combine_chunks().flatten()
    return torch.from_numpy(icd.to_numpy(zero_copy_only=False).reshape(len(rows), 76).astype(np.float32))


def cached_h(cache, specs, rows, lance_path, batch_size, workers):
    """h of every ECG at `rows` for every spec (float32, on the GPU), reusing --cache entries with the same rows and
    checkpoint file (by modification time); only the missing checkpoints are embedded."""
    key = {n: (s, (REPO / s).stat().st_mtime) for n, s in specs.items()}
    z = {"rows": rows, "key": {}, "h": {}}
    if cache and Path(cache).exists():
        old = torch.load(cache, weights_only=False)      # our own cache file
        if np.array_equal(old["rows"], rows):
            z = old
    todo = {n: s for n, s in specs.items() if z["key"].get(n) != key[n]}
    if todo:
        print(f"embedding {len(rows):,} ECGs with {', '.join(todo)}", flush=True)
        t0 = time.time()
        feats, _ = embed({n: load_encoder(s) for n, s in todo.items()}, rows, lance_path, layers=("h",),
                         batch_size=batch_size, num_workers=workers)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)
        for n in todo:
            z["h"][n], z["key"][n] = feats[n]["h"], key[n]
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            torch.save(z, cache)
            print(f"saved h: {cache}", flush=True)
    elif specs:
        print(f"loaded cached h: {cache}", flush=True)
    return {n: z["h"][n].cuda() for n in specs}


# ---------------------------------------------------------------- statistics
def macro_auroc(P, Y, M=None):
    """Macro AUROC over the classes with both outcomes (scripts/evaluate_inverse.py's macro), all classes at once
    from mid-ranks (ties count 1/2). M (bool, like Y) keeps only its entries per class: the masked onset and
    resolution heads. Masked entries are sent to +inf, above every kept score, so they don't move the kept ranks."""
    P, Y = P.T.double(), Y.T.double()                  # (C, N)
    if M is not None:
        M = M.T
        P, Y = P.masked_fill(~M, float("inf")), Y * M
        neg = (M & (Y == 0)).sum(1)
    else:
        neg = (Y == 0).sum(1)
    S = P.sort(1).values.contiguous()
    rank = (torch.searchsorted(S, P.contiguous()) + torch.searchsorted(S, P.contiguous(), right=True) + 1) / 2
    pos = Y.sum(1)
    auc = ((rank * Y).sum(1) - pos * (pos + 1) / 2) / (pos * neg)
    return auc[(pos > 0) & (neg > 0)].mean()


def cluster_indices(groups, n_boot, seed):
    """n_boot patient-level resamples of the rows (every row of each drawn patient), the same for every checkpoint."""
    _, inv = np.unique(groups, return_inverse=True)
    order = torch.as_tensor(np.argsort(inv, kind="stable"), device="cuda")
    counts = torch.as_tensor(np.bincount(inv), device="cuda")
    starts = torch.cumsum(counts, 0) - counts
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for _ in range(n_boot):
        pick = torch.randint(0, len(counts), (len(counts),), device="cuda", generator=g)
        c = counts[pick]
        within = torch.arange(int(c.sum()), device="cuda") - torch.repeat_interleave(torch.cumsum(c, 0) - c, c)
        out.append(order[torch.repeat_interleave(starts[pick], c) + within])
    return out


def with_reps(stat, idx):
    """(point estimate on all rows, bootstrap replicates) of stat(rows)."""
    return float(stat(slice(None))), np.array([float(stat(i)) for i in idx])


def summ(point, reps):
    return {"auc": point, "lo": float(np.nanpercentile(reps, 2.5)), "hi": float(np.nanpercentile(reps, 97.5))}


# ---------------------------------------------------------------- energies and readouts
def energies(dyn, ht, target, cand, chunk=65536):
    """E for candidate actions cand (B, N, C) of pairs with states ht (B, D) and targets (B, D) -> (B, N), fp32."""
    B, N, C = cand.shape
    flat = cand.reshape(B * N, C)
    rows = torch.arange(B * N, device=cand.device) // N
    E = torch.empty(B * N, device=cand.device)
    for s in range(0, B * N, chunk):
        r = rows[s:s + chunk]
        E[s:s + chunk] = (dyn(ht[r], flat[s:s + chunk]) - target[r]).pow(2).mean(1)
    return E.view(B, N)


def subsets(allowed, counts, k, gen):
    """k random subsets of each row's allowed labels with counts[i] labels each (_DecoyProbe._subsets)."""
    s = torch.rand(allowed.shape[0], k, allowed.shape[1], generator=gen, device=allowed.device)
    rank = s.masked_fill(~allowed.unsqueeze(1), -1.0).argsort(-1, descending=True).argsort(-1)
    return (rank < counts.view(-1, 1, 1)).float()


def decoy_candidates(yt, at, k, prior, gen):
    """(B, 3k + 2, C): [truth, k same-count, k extra-flip, zero, k prior draws] (_DecoyProbe._candidates)."""
    B, C = yt.shape
    same = subsets(yt == 0, (at == 1).sum(1), k, gen) - subsets(yt == 1, (at == -1).sum(1), k, gen)
    one = torch.ones(B, dtype=torch.long, device=yt.device)
    extra = at.unsqueeze(1) + subsets(at == 0, one, k, gen) * (1.0 - 2.0 * yt).unsqueeze(1)
    p_ons, p_res = (p.expand(B, k, C).contiguous() for p in prior)
    pri = (torch.bernoulli(p_ons, generator=gen) * (1.0 - yt).unsqueeze(1)
           - torch.bernoulli(p_res, generator=gen) * yt.unsqueeze(1))
    return torch.cat([at.unsqueeze(1), same, extra, torch.zeros_like(at).unsqueeze(1), pri], 1)


def decoy_rates(E, cand, at, k):
    """Per-pair within-pair AUROC of each family (NaN where the family doesn't apply)."""
    w = (E > E[:, :1]).float() + 0.5 * (E == E[:, :1]).float()
    valid = (cand != at.unsqueeze(1)).any(-1).float()              # a decoy equal to the truth doesn't count
    active = (at != 0).any(1)
    cols = {"same": slice(1, k + 1), "extra": slice(k + 1, 2 * k + 1), "zero": slice(2 * k + 1, 2 * k + 2),
            "prior": slice(2 * k + 2, 3 * k + 2)}
    rows = {"same": active, "extra": torch.ones_like(active), "zero": active, "prior": torch.ones_like(active)}
    out = {}
    for fam, c in cols.items():
        v = valid[:, c]
        rate = (w[:, c] * v).sum(1) / v.sum(1).clamp(min=1)
        out[fam] = torch.where(rows[fam] & (v.sum(1) > 0), rate, torch.nan)
    return out


@torch.no_grad()
def score_population(dyn, H, Y, A, planner, prior, args, run_cem=True):
    """CEM, decoys and flip deltas of one population. H = (ht, ht1), Y = yt, A = at, all on the GPU."""
    ht, ht1 = H
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    eye = torch.eye(Y.shape[1], device="cuda")
    keys = ("p_ons", "p_res", "a_star", "e_star", "e_zero", "e_gt", "delta", "delta_nochange")
    out = {k: [] for k in keys} | {f"decoy_{f}{s}": [] for f in FAMILIES for s in ("", "_nochange")}
    for b in torch.arange(len(Y), device="cuda").split(args.batch_pairs):
        h, h1, y, a = ht[b], ht1[b], Y[b], A[b]
        if run_cem:
            res = planner.plan(lambda c: energies(dyn, h, h1, c), y, *prior, gen)
            for k in ("p_ons", "p_res", "a_star", "e_star", "e_zero"):
                out[k].append(res[k])
        out["e_gt"].append(energies(dyn, h, h1, a.unsqueeze(1)).squeeze(1))
        flips = torch.cat([torch.zeros_like(y).unsqueeze(1), eye * (1 - 2 * y).unsqueeze(1)], 1)   # (B, 1 + C, C)
        cand = decoy_candidates(y, a, args.n_decoys, prior, gen)
        for suffix, target in (("", h1), ("_nochange", h)):
            E = energies(dyn, h, target, flips)
            out[f"delta{suffix}"].append(E[:, :1] - E[:, 1:])
            for fam, r in decoy_rates(energies(dyn, h, target, cand), cand, a, args.n_decoys).items():
                out[f"decoy_{fam}{suffix}"].append(r)
    return {k: torch.cat(v) for k, v in out.items() if v}


def flip_ctx(delta, Y, prior, tau):
    """Contextualized score of the per-label readout; tau None = the prior alone."""
    logit = lambda p: torch.log(p) - torch.log1p(-p)
    shift = 0 if tau is None else delta / tau
    p_ons, p_res = torch.sigmoid(logit(prior[0]) + shift), torch.sigmoid(logit(prior[1]) + shift)
    return Y * (1 - p_res) + (1 - Y) * p_ons


def metrics(s, Y, A, prior, tau, idx):
    """Point estimates and bootstrap replicates of every metric of one scored population."""
    Y1 = (Y + A).clamp(0, 1)
    m_ons, m_res = Y == 0, Y == 1
    on, off = (A == 1).double(), (A == -1).double()
    scores = {}
    if "a_star" in s:
        scores["cem_soft"] = (torch.where(m_res, 1 - s["p_res"].nan_to_num(), s["p_ons"].nan_to_num()), Y1, None)
        scores["cem_hard"] = ((Y + s["a_star"]).clamp(0, 1), Y1, None)
    for suffix in ("", "_nochange"):
        d = s[f"delta{suffix}"]
        scores[f"flip{suffix}"] = (flip_ctx(d, Y, prior, tau), Y1, None)
        scores[f"onset{suffix}"] = (d, on, m_ons)
        scores[f"resolution{suffix}"] = (d, off, m_res)    # either head: delta > 0 means the flip lowers E
    point, reps = {}, {}
    for k, (P, T, M) in scores.items():
        point[k], reps[k] = with_reps(lambda i: macro_auroc(P[i], T[i], None if M is None else M[i]), idx)
    for f in FAMILIES:
        for suffix in ("", "_nochange"):
            r = s[f"decoy_{f}{suffix}"]
            point[f"decoy_{f}{suffix}"], reps[f"decoy_{f}{suffix}"] = with_reps(lambda i: r[i].nanmean(), idx)
    # Plain numbers, no intervals
    active = (A != 0).any(1)
    extra = {"mse_ratio": float(s["e_gt"].mean() / s["do_nothing"].mean()),
             "pairs": len(Y), "active": int(active.sum())}
    if "a_star" in s:
        a = s["a_star"]
        extra |= {"onsets_per_pair": float((a == 1).sum(1).double().mean()),
                  "resolutions_per_pair": float((a == -1).sum(1).double().mean()),
                  "gt_onsets_per_pair": float((A == 1).sum(1).double().mean()),
                  "gt_resolutions_per_pair": float((A == -1).sum(1).double().mean()),
                  "e_star_over_e_zero": float((s["e_star"] / s["e_zero"]).mean()),
                  "e_gt_over_e_zero": float((s["e_gt"] / s["e_zero"]).mean()),
                  "exact_active": float((a == A).all(1)[active].double().mean()),
                  "search_failure_active": float((s["e_star"] > s["e_gt"])[active].double().mean()),
                  "pred_stable": float(((a != 0).sum(1) == 0).double().mean())}
    return {k: summ(point[k], reps[k]) for k in point} | extra, reps


def choose_tau(delta_val, Y, A, prior):
    """tau on val from multiples of the delta spread, against the prior alone (tau = inf)."""
    Y1, scale = (Y + A).clamp(0, 1), float(delta_val.std())
    grid = {float(m): macro_auroc(flip_ctx(delta_val, Y, prior, m * scale), Y1).item() for m in TAU_GRID}
    best = max(grid, key=grid.get)
    prior_only = macro_auroc(flip_ctx(delta_val, Y, prior, None), Y1).item()
    if best in (TAU_GRID[0], TAU_GRID[-1]):
        print(f"  WARNING: tau at the edge of its grid ({best:g} x delta spread)", flush=True)
    return {"tau": best * scale, "multiple": best, "scale": scale, "val": grid[best], "val_prior_only": prior_only,
            "val_grid": grid}


# ---------------------------------------------------------------- report
def fmt(t):
    return f"{t['auc']:.3f} [{t['lo']:.3f}-{t['hi']:.3f}]"


def to_json(x):
    if isinstance(x, dict):
        return {str(k): to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def print_checkpoint(name, r):
    info = r["training"]
    inside = "  ".join(f"{p} {v:.3f}" for p, v in info["in_training_pairs"].items())
    print(f"\n== {name}: {info['backbone']}, {info['pairs']} {'+'.join(info['pair_types'])}, {info['epochs']} epochs; "
          f"pairs inside its training pairs: {inside}")
    t = r["tau"]
    print(f"   flip tau {t['multiple']:g} x {t['scale']:.3g}: val {t['val']:.4f} (prior only {t['val_prior_only']:.4f})")
    for p in SCORED:
        m = r[p]
        print(f"   {p:<8} CEM soft {fmt(m['cem_soft'])}  hard {fmt(m['cem_hard'])}  |  flip {fmt(m['flip'])}  "
              f"nothing changed {fmt(m['flip_nochange'])}")
        print(f"   {'':<8} onset {fmt(m['onset'])} (nc {m['onset_nochange']['auc']:.3f})  resolution "
              f"{fmt(m['resolution'])} (nc {m['resolution_nochange']['auc']:.3f})  E(a_gt)/do-nothing "
              f"{m['mse_ratio']:.3f}")
        print(f"   {'':<8} decoys " + "  ".join(f"{f} {m[f'decoy_{f}']['auc']:.3f}/{m[f'decoy_{f}_nochange']['auc']:.3f}"
                                              for f in FAMILIES) + "   (real/nothing changed)")
        print(f"   {'':<8} CEM flips/pair {m['onsets_per_pair']:.2f} ons {m['resolutions_per_pair']:.2f} res "
              f"(GT {m['gt_onsets_per_pair']:.2f}/{m['gt_resolutions_per_pair']:.2f})  E(a*)/E(0) "
              f"{m['e_star_over_e_zero']:.3f}  E(a_gt)/E(0) {m['e_gt_over_e_zero']:.3f}  exact {m['exact_active']:.3f}"
              f"  search failure {m['search_failure_active']:.3f}  a*=0 {m['pred_stable']:.3f}", flush=True)


def main():
    args = parse_args()
    out_path = Path(args.out)
    if args.plot_only:
        plot(json.loads(out_path.read_text()), args.fig)
        return
    if not args.ckpt:
        raise SystemExit("give at least one --ckpt NAME=SPEC")
    paths = data_paths(args.config)
    cem = load_script("cem_planning")

    pops = populations(paths, args.max_pairs)
    print("pairs: " + ", ".join(f"{p} {len(df):,} ({df['subject_id'].nunique():,} patients)" for p, df in pops.items()),
          flush=True)
    specs = dict(c.split("=", 1) for c in args.ckpt)
    reps_path = out_path.with_name(out_path.stem + "_reps.pt")

    config = {k: v for k, v in vars(args).items() if k not in ("plot_only", "ckpt", "compare", "fig", "workers",
                                                                 "batch_size", "cache", "out")}
    results = {"config": config, "populations": {}, "references": {}, "checkpoints": {}, "compare": {}}
    all_reps = {}
    # Resumable: checkpoints already in --out under the same protocol are kept, and results are saved after each one
    if out_path.exists() and reps_path.exists():
        old = json.loads(out_path.read_text())
        if old["config"] == config:
            results["checkpoints"] = {n: r for n, r in old["checkpoints"].items() if specs.get(n) == r["spec"]}
            all_reps = {n: v for n, v in torch.load(reps_path, weights_only=False).items()
                        if n in results["checkpoints"]}
            if results["checkpoints"]:
                print(f"resuming: {', '.join(results['checkpoints'])} already done", flush=True)
    todo = {n: s for n, s in specs.items() if n not in results["checkpoints"]}
    refs = {"inverse": args.inverse} if args.inverse else {}

    rows = np.unique(np.concatenate([df[c].values for df in pops.values() for c in ("idx_t", "idx_t1")]))
    H = cached_h(args.cache, todo | refs, rows, paths["lance_path"], args.batch_size, args.workers)
    pos = {p: (torch.as_tensor(np.searchsorted(rows, df["idx_t"].values), device="cuda"),
               torch.as_tensor(np.searchsorted(rows, df["idx_t1"].values), device="cuda")) for p, df in pops.items()}
    labels = lance_labels(paths["lance_path"], rows).cuda()
    Y = {p: labels[i] for p, (i, _) in pos.items()}
    A = {p: (labels[j] - labels[i]).clamp(-1, 1) for p, (i, j) in pos.items()}
    idx = {p: cluster_indices(pops[p]["subject_id"].values, args.n_boot, args.seed) for p in SCORED}
    for p in POPS:
        results["populations"][p] = {"fold": POPS[p], "pairs": len(pops[p]), "patients": int(pops[p]["subject_id"].nunique()),
                                     "with_change": int((A[p] != 0).any(1).sum())}

    counts = cem._transition_counts(paths["lance_path"], paths["pairs_path"], ("cross_stay",), 76)
    prior = tuple(torch.from_numpy(p).cuda() for p in cem._priors(counts, "base_rate"))
    planner = cem.OnsetResolutionCEM(args.n_samples, args.n_elites, args.n_iters, args.smoothing, args.p_min)

    # References on the same pairs and resamples
    for p in SCORED:
        Y1 = (Y[p] + A[p]).clamp(0, 1)
        ref = {"carry_forward": with_reps(lambda i: macro_auroc(Y[p][i], Y1[i]), idx[p])}
        if refs:
            head = load_inverse_head(refs["inverse"])
            h = {k: H["inverse"][v] for k, v in zip(("t", "t1"), pos[p])}
            with torch.no_grad():
                ons, res = (torch.sigmoid(z) for z in head(h["t1"] - h["t"]))
            ctx = Y[p] * (1 - res) + (1 - Y[p]) * ons
            ref["inverse"] = with_reps(lambda i: macro_auroc(ctx[i], Y1[i]), idx[p])
        results["references"][p] = {k: summ(*v) for k, v in ref.items()}
        all_reps.setdefault("_references", {})[p] = {k: v[1] for k, v in ref.items()}
    if refs:
        results["references"]["inverse_training"] = training_info(refs["inverse"], paths, pops)
    print("references: " + "   ".join(f"{p}: " + ", ".join(f"{k} {fmt(v)}" for k, v in results["references"][p].items())
                                      for p in SCORED), flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for name, spec in todo.items():
        dyn = load_dynamics(spec)
        if dyn is None:
            print(f"\n== {name}: no forward model in {spec}; skipped", flush=True)
            continue
        t0 = time.time()
        Hp = {p: (H[name][i], H[name][j]) for p, (i, j) in pos.items()}
        val = score_population(dyn, Hp["val"], Y["val"], A["val"], planner, prior, args, run_cem=False)
        tau = choose_tau(val["delta"], Y["val"], A["val"], prior)
        r = {"spec": spec, "training": training_info(spec, paths, pops), "tau": tau}
        all_reps[name] = {}
        for p in SCORED:
            s = score_population(dyn, Hp[p], Y[p], A[p], planner, prior, args)
            s["do_nothing"] = (Hp[p][1] - Hp[p][0]).pow(2).mean(1)
            r[p], all_reps[name][p] = metrics(s, Y[p], A[p], prior, tau["tau"], idx[p])
        r["seconds"] = time.time() - t0
        results["checkpoints"][name] = r
        print_checkpoint(name, r)
        out_path.write_text(json.dumps(to_json(results), indent=2))
        torch.save(all_reps, reps_path)

    results["checkpoints"] = {n: results["checkpoints"][n] for n in specs if n in results["checkpoints"]}
    signed = lambda t: f"{t['auc']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}]"
    # Evidence read from the outcome: each metric minus its nothing-changed twin, as a metric of its own
    for n, r in results["checkpoints"].items():
        for p in SCORED:
            for k in EVIDENCE:
                all_reps[n][p][f"evidence_{k}"] = all_reps[n][p][k] - all_reps[n][p][f"{k}_nochange"]
            r[p]["evidence"] = {k: summ(r[p][k]["auc"] - r[p][f"{k}_nochange"]["auc"], all_reps[n][p][f"evidence_{k}"])
                                for k in EVIDENCE}
    print("\nevidence = real - nothing changed:")
    for n, r in results["checkpoints"].items():
        print(f"   {n:<12} " + "   ".join(f"{p}: " + "  ".join(f"{k} {signed(r[p]['evidence'][k])}"
                                                               for k in ("flip", "onset", "decoy_same"))
                                         for p in SCORED))
    for pair in args.compare:
        a, b = pair.split(":")
        if a in results["checkpoints"] and b in results["checkpoints"]:
            point = lambda n, p, k: (results["checkpoints"][n][p]["evidence"][k.removeprefix("evidence_")]["auc"]
                                     if k.startswith("evidence_") else results["checkpoints"][n][p][k]["auc"])
            results["compare"][f"{a}-{b}"] = {
                p: {k: summ(point(a, p, k) - point(b, p, k), all_reps[a][p][k] - all_reps[b][p][k])
                    for k in all_reps[a][p]} for p in SCORED}
            print(f"\n{a} - {b} (paired, patient-level bootstrap):")
            for p in SCORED:
                d = results["compare"][f"{a}-{b}"][p]
                print(f"   {p:<8} " + "  ".join(f"{k} {signed(d[k])}" for k in (
                    "cem_soft", "cem_hard", "flip", "evidence_flip", "evidence_onset", "decoy_same",
                    "evidence_decoy_same", "decoy_zero")))
    out_path.write_text(json.dumps(to_json(results), indent=2))
    torch.save(all_reps, reps_path)
    print(f"\nSaved {out_path}")
    if args.fig:
        plot(json.loads(out_path.read_text()), args.fig)


# ---------------------------------------------------------------- figure
# The chrome and first two categorical slots of anatomie/memorization.py and anatomie/linear_probe.py; everything
# that is not the matched pair is context, in the muted gray.
INK = {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
COLOR = {"cnn": "#2a78d6", "vit": "#eb6834"}     # color follows the entity, not its position
NAMES = {"cnn": "CNN (matched)", "vit": "ViT (matched)", "cnn_ep5": "CNN, epoch 5", "cnn_lowreg": "CNN, λ 0.005 / wd 0.01",
         "cnn_k1": "CNN, k=1", "vit_k1": "ViT, k=1, 405 ep", "vit_k1_p128": "ViT, k=1, predictor 128",
         "paper_dyn": "paper Dynamics (CNN)"}


def plot(results, path, series=("cnn", "vit")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ck = results["checkpoints"]
    names = list(ck)[::-1]                       # first on top
    series = [s for s in series if s in ck]
    color = lambda n: COLOR.get(n, INK["muted"]) if n in series else INK["muted"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK["axis"], "axes.labelcolor": INK["secondary"],
                         "xtick.color": INK["muted"], "ytick.color": INK["muted"], "axes.titlecolor": INK["primary"]})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 0.42 * len(names) + 2.6), dpi=150, facecolor=INK["surface"],
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})
    lw, ms = 1.25, 5.5

    def frame(ax, title):
        ax.set_facecolor(INK["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=INK["grid"], linewidth=0.5)
        ax.set_title(title, loc="left", fontsize=10)

    def dot(ax, t, y, c, marker="o", filled=True):
        ax.plot([t["lo"], t["hi"]], [y, y], color=c, linewidth=lw, solid_capstyle="round", zorder=2)
        ax.plot(t["auc"], y, marker, color=c, markersize=ms, markerfacecolor=c if filled else INK["surface"],
                markeredgecolor=c if not filled else INK["surface"], markeredgewidth=1.0, zorder=3)

    def references(ax, pop):
        ref = results["references"][pop]
        for k, style, label, side in (("carry_forward", "--", "carry-forward", -1), ("inverse", ":", "paper Inverse", 1)):
            if k in ref:
                ax.axvline(ref[k]["auc"], color=INK["secondary"], linewidth=0.9, linestyle=style, zorder=1)
                ax.annotate(f"{label} {ref[k]['auc']:.3f}", (ref[k]["auc"], len(names) - 0.4), xytext=(3 * side, 0),
                            textcoords="offset points", fontsize=7.5, color=INK["secondary"], va="bottom",
                            ha="left" if side > 0 else "right")

    def rows(ax):
        ax.set_yticks(range(len(names)), [NAMES.get(n, n) for n in names])
        ax.tick_params(axis="y", length=0, labelcolor=INK["secondary"])
        ax.set_ylim(-0.6, len(names) + 0.2)

    # (a) test: the paper's CEM readout against the per-label readout and its nothing-changed control
    ax = axes[0]
    frame(ax, "a  Contextualized AUROC: CEM vs per-label readout")
    for i, n in enumerate(names):
        m, c = ck[n]["test"], color(n)
        dot(ax, m["cem_soft"], i + 0.22, c, marker="s")
        dot(ax, m["flip"], i, c)
        dot(ax, m["flip_nochange"], i - 0.22, c, filled=False)
    references(ax, "test")
    rows(ax)
    ax.set_xlabel("macro AUROC of the next labels; 95% patient-level bootstrap")

    # (b, c) the true action under E without search: on pairs the checkpoint trained on vs held-out pairs
    for ax, fam, title in ((axes[1], "same", "b  Which labels changed? Truth vs same-count decoys"),
                           (axes[2], "zero", "c  Truth vs doing nothing")):
        frame(ax, title)
        for i, n in enumerate(names):
            c = color(n)
            dot(ax, ck[n]["members"][f"decoy_{fam}"], i + 0.22, c, marker="D")
            dot(ax, ck[n]["test"][f"decoy_{fam}"], i, c)
            if fam == "same":
                dot(ax, ck[n]["test"][f"decoy_{fam}_nochange"], i - 0.22, c, filled=False)
        ax.axvline(0.5, color=INK["axis"], linewidth=0.75, zorder=1)
        rows(ax)
        ax.set_yticklabels([])
    axes[1].set_xlabel("P(E(truth) < E(decoy)), within pair, pairs with a change; 0.5 = chance")
    axes[2].set_xlabel("P(E(truth) < E(0)), pairs with a change; 0.5 = chance")

    mk = lambda marker, label, filled=True: Line2D([], [], color=INK["secondary"], linewidth=0, marker=marker,
                                                   markersize=ms, markerfacecolor=INK["secondary"] if filled
                                                   else INK["surface"], label=label)
    handles = [Line2D([], [], color=COLOR[s], linewidth=lw * 1.6, label=NAMES[s]) for s in series]
    handles += [Line2D([], [], color=INK["muted"], linewidth=lw * 1.6, label="other forward checkpoints"),
                mk("s", "CEM, soft readout (test)"), mk("o", "test pairs (fold 19)"),
                mk("o", "test, scored against h_t (nothing changed)", False), mk("D", "training pairs (fold 17)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8.5,
               labelcolor=INK["primary"], bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=INK["surface"])
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":   # spawned DataLoader workers re-import this file
    main()

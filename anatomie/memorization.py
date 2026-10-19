"""
Memorization Test for Forward-Pretrained Encoders (CNN vs ViT)
================================================================
Does a forward checkpoint treat the pairs it trained on differently from the pairs of patients it never saw? A
membership-inference design: members and non-members are drawn identically, so any statistic that separates them
measures memorization, with chance at exactly 0.5. Random-init encoders, which have no members, are the control.

Populations (one random anchor per patient, so observations are independent)
-----------
An anchor is a stay s with a second ECG and a following stay s+1:
  A    first ECG of stay s    trained as the k=1 cross-stay pair A -> B by every cross-stay checkpoint
  A'   second ECG of stay s   in no training pair of a cross-stay checkpoint (verified per checkpoint)
  B    first ECG of stay s+1
  members       training-fold patients (folds 0-17), --n-members of them
  non-members   held-out patients (--holdout-folds; default 18 and 19, never trained on; fold 18 only selected
                best-val checkpoints, never last.ckpt)

Tests: AUROC of members vs non-members with a DeLong 95% CI; 0.5 = no memorization
-----
Encoder, at every readout layer of anatomie/encoders.py. Distances are relative: r(x, B) = |z(x) - z(B)|^2 divided by
the mean squared distance from z(x) to the other B's of the same population. That removes population-level scale
and centring differences (an encoder can shrink unseen ECGs toward the centre).
  pair        score -log r(A, B)                   trained pairs sit closer: recording and/or patient memorization
  patient     score -log r(A', B)                  an unseen recording of a seen patient sits closer: the patient
                                                   is memorized
  recording   score -[log r(A,B) - log r(A',B)]    the trained recording beats an unseen one from the same stay,
                                                   beyond the gap non-members show: the recording is memorized
  spread      score |z(A) - mu|^2                  trained recordings occupy more of the space than unseen ones
                                                   (mu = pooled mean; > 0.5 = members further out)
  dose        the pair test at --dose-layer within buckets of the patient's number of stays, with each bucket's
              training exposure of A (pairs per epoch containing A in the checkpoint's own pair file)
Forward model Dyn(h, Proj(a)) on the A -> B pairs with their true action a, every error relative to the do-nothing
error |h_B - h_A|^2 of the same pair:
  total       score -E(a)                          the forward loss
  zero        score -E(0)                          Dyn's fit with the action removed (shrinkage, patient structure)
  action      score 1 - E(a)/E(0), active pairs    the share of the zero-action error the true action removes (a
                                                   per-pair action R^2); memorized action effects help members only
  total and zero divide by the do-nothing error, which an encoder that collapsed member pairs drives to ~0: read
  them only for encoders whose pair test is ~0.5; action does not depend on it.
  plus per-population aggregates: E(a), E(0), the best one-parameter shrinkage toward the mean, and the action
  effect's R^2 on the residual h_B - Dyn(h_A, 0) and cosine with it (anatomie/dyn_decompose.py's definitions)
Training curves: per-epoch train and val forward loss from the run's local wandb log, when one matches the run dir.

Interpretation
--------------
  random-init encoders ~0.5 everywhere      the populations are matched; deviations elsewhere are memorization
  pair >> 0.5, recording >> 0.5             the encoder memorized specific training recordings
  patient > 0.5                             unseen recordings of training patients also sit closer: identity is
                                            learned for training patients only
  dose rising with stays                    memorization grows with exposure (multistep: up to 24 pairs per ECG
                                            per epoch)
  encoder tests ~0.5 but action >> 0.5      the encoder generalizes; the forward model memorized the transitions

Example
-------
python anatomie/memorization.py \\
    --ckpt cnn_random=random --ckpt vit_random=random-vit \\
    --ckpt cnn=runs/runs/20260930/084313/e9ebd97377ce/checkpoints/last.ckpt \\
    --ckpt vit=runs/runs/20261006/044610/064d3fbda75f/checkpoints/last.ckpt \\
    --out outputs/memorization/results.json --fig outputs/memorization/memorization.png
python anatomie/memorization.py --plot-only --out outputs/memorization/results.json \\
    --fig outputs/memorization/memorization.png     # redraw the figure without recomputing
"""

import argparse
import functools
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import REPO, data_paths, embed, load_dynamics, load_encoder  # noqa: E402

BUCKETS = {"2": (2, 2), "3-4": (3, 4), "5-8": (5, 8), ">=9": (9, 10**6)}   # stays per patient
ENC_TESTS = ("pair", "patient", "recording", "spread")
FWD_TESTS = ("total", "zero", "action")
RANDOM_SPECS = ("random", "random-vit")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", default=[], metavar="NAME=SPEC",
                   help="encoder to test (repeatable): 'random', 'random-vit' or a checkpoint path")
    p.add_argument("--holdout-folds", type=int, nargs="+", default=[18, 19])
    p.add_argument("--n-members", type=int, default=5000, help="member patients, one anchor each")
    p.add_argument("--dose-layer", default="h")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default="outputs/memorization/results.json")
    p.add_argument("--fig", default=None, help="optional PNG with the main panels")
    p.add_argument("--plot-only", action="store_true", help="redraw --fig from an existing --out")
    return p.parse_args()


# ---------------------------------------------------------------- statistics
def auroc(pos, neg):
    """P(member score > non-member score), ties 1/2, with a DeLong 95% CI (independent observations)."""
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    m, n = len(pos), len(neg)
    if m < 10 or n < 10:
        return {"auc": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": [m, n]}
    r = rankdata(np.concatenate([pos, neg]))
    v10 = (r[:m] - rankdata(pos)) / n           # share of non-members each member beats
    v01 = 1.0 - (r[m:] - rankdata(neg)) / m     # share of members each non-member loses to
    auc = v10.mean()
    se = np.sqrt(v10.var(ddof=1) / m + v01.var(ddof=1) / n)
    return {"auc": float(auc), "lo": float(auc - 1.96 * se), "hi": float(auc + 1.96 * se), "n": [m, n]}


def fmt(t):
    return f"{t['auc']:.3f} [{t['lo']:.3f}-{t['hi']:.3f}]" if t and np.isfinite(t["auc"]) else f"{'-':^19}"


# ---------------------------------------------------------------- populations
def anchor_table(lance_path):
    """Every anchor (stay with a second ECG and a following stay), with its patient's fold and number of stays.
    Stays are numbered as scripts/build_pairs_multistep.py does: enws >= 0, time order, a new stay at each enws 0."""
    meta = lance.dataset(lance_path).to_table(
        columns=["subject_id", "ecg_time", "fold", "ecg_no_within_stay"]).to_pandas()
    meta["idx"] = meta.index
    meta = meta[meta["ecg_no_within_stay"] >= 0].sort_values(["subject_id", "ecg_time"])
    meta["stay"] = (meta["ecg_no_within_stay"] == 0).astype(int).groupby(meta["subject_id"]).cumsum()
    meta = meta[meta["stay"] >= 1]
    first = meta[meta["ecg_no_within_stay"] == 0].set_index(["subject_id", "stay"])["idx"]
    second = (meta[meta["ecg_no_within_stay"] == 1].drop_duplicates(["subject_id", "stay"])
              .set_index(["subject_id", "stay"])["idx"])
    k = second.index
    nxt = pd.MultiIndex.from_arrays([k.get_level_values(0), k.get_level_values(1) + 1])
    keep = nxt.isin(first.index)
    k, nxt = k[keep], nxt[keep]
    subject = k.get_level_values(0).values
    fold = meta.groupby("subject_id")["fold"].first()
    n_stays = first.groupby(level=0).size()
    return pd.DataFrame({"subject": subject, "stay": k.get_level_values(1).values, "A": first.loc[k].values,
                         "A2": second.loc[k].values, "B": first.loc[nxt].values,
                         "fold": fold.loc[subject].values, "n_stays": n_stays.loc[subject].values})


def sample_populations(anchors, holdout_folds, n_members, seed):
    one = anchors.sample(frac=1.0, random_state=seed).drop_duplicates("subject")    # one random anchor per patient
    members = one[one["fold"] <= 17]
    members = members.sample(n=min(n_members, len(members)), random_state=seed)
    nonmembers = one[one["fold"].isin(holdout_folds)]
    return members.reset_index(drop=True), nonmembers.reset_index(drop=True)


# ---------------------------------------------------------------- what a checkpoint trained on
@functools.lru_cache(maxsize=None)
def load_pairs(path, types):
    t = lance.dataset(path).to_table(columns=["idx_t", "idx_t1", "fold", "pair_type"]).to_pandas()
    return t[t["pair_type"].isin(types)]


def checkpoint_info(spec, paths, members, nonmembers, holdout_folds):
    """Training pairs, epochs and membership checks of a checkpoint; None for random encoders."""
    if spec in RANDOM_SPECS:
        return None
    ck = torch.load(REPO / spec, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters") or {}
    local = {Path(paths[k]).name: paths[k] for k in ("pairs_path", "pairs_multistep_path")}
    pairs_name = Path(str(hp.get("pairs_path", ""))).name
    types = tuple(hp.get("pair_types") or ("within_stay", "cross_stay"))
    pairs = load_pairs(local[pairs_name], types)
    train, held = pairs[pairs["fold"] <= 17], pairs[pairs["fold"].isin(holdout_folds)]
    trained = pd.MultiIndex.from_arrays([train["idx_t"].values, train["idx_t1"].values])
    ab = lambda df: pd.MultiIndex.from_arrays([df["A"].values, df["B"].values])
    count = lambda p: pd.concat([p["idx_t"], p["idx_t1"]]).value_counts()
    c_train, c_held = count(train), count(held)
    vit = hp.get("vit")
    return {
        "backbone": (f"ViT1d {dict(vit)['dim']}x{dict(vit)['depth']}" if hp.get("backbone") == "vit1d"
                     else "ResNet1d"),
        "pairs": pairs_name, "pair_types": list(types), "epochs": int(ck.get("epoch", -1)) + 1,
        "train_frac": float(hp.get("train_frac", 1.0)), "lambda_reg": hp.get("lambda_reg"),
        "weight_decay": hp.get("weight_decay"), "predictor_hidden_dim": hp.get("predictor_hidden_dim"),
        "check": {"members_AB_trained": float(ab(members).isin(trained).mean()),
                  "members_A2_trained": float(members["A2"].isin(c_train.index).mean()),
                  "nonmembers_AB_trained": float(ab(nonmembers).isin(trained).mean())},
        # pairs per epoch that contain A: the members' exposure, and the non-members' would-be exposure
        "_exposure": (c_train.reindex(members["A"].values).fillna(0).values,
                      c_held.reindex(nonmembers["A"].values).fillna(0).values),
    }


# ---------------------------------------------------------------- tests
def rel_dist(Za, Zb, Zpool):
    """|a - b|^2 and the mean |a - pool_j|^2 (closed form), per row."""
    d = (Za - Zb).pow(2).sum(1)
    ref = Za.pow(2).sum(1) - 2 * Za @ Zpool.mean(0) + Zpool.pow(2).sum(1).mean()
    return d, ref


def encoder_tests(Z, M, N):
    """Z (rows, D) float64; M, N: index tensors A, A2, B of members and non-members."""
    r, desc = {}, {}
    for tag, P in (("m", M), ("n", N)):
        pool = Z[P["B"]]
        for pair, src in (("AB", "A"), ("A2B", "A2")):
            d, ref = rel_dist(Z[P[src]], Z[P["B"]], pool)
            r[tag, pair] = torch.log(d.clamp_min(1e-30)) - torch.log(ref)
            desc[f"{tag}_{pair}"] = float(d.sum() / ref.sum())
    mu = Z[torch.cat([M["A"], N["A"]])].mean(0)
    spread = {tag: (Z[P["A"]] - mu).pow(2).sum(1) for tag, P in (("m", M), ("n", N))}
    desc["spread_ratio"] = float(spread["m"].mean() / spread["n"].mean())
    return {
        "rel": desc,
        "pair": auroc(-r["m", "AB"], -r["n", "AB"]),
        "patient": auroc(-r["m", "A2B"], -r["n", "A2B"]),
        "recording": auroc(-(r["m", "AB"] - r["m", "A2B"]), -(r["n", "AB"] - r["n", "A2B"])),
        "spread": auroc(spread["m"], spread["n"]),
        "_log_r_AB": (r["m", "AB"], r["n", "AB"]),
    }


def dose_tests(log_r, members, nonmembers, info):
    out = {}
    for name, (lo, hi) in BUCKETS.items():
        mm = ((members["n_stays"] >= lo) & (members["n_stays"] <= hi)).values
        nm = ((nonmembers["n_stays"] >= lo) & (nonmembers["n_stays"] <= hi)).values
        t = auroc(-log_r[0][torch.as_tensor(mm)], -log_r[1][torch.as_tensor(nm)])
        if info is not None:
            t["exposure_per_epoch"] = float(info["_exposure"][0][mm].mean()) if mm.any() else float("nan")
        out[name] = t
    return out


@torch.no_grad()
def forward_tests(dyn, H, Y, M, N):
    res, scores = {}, {}
    for tag, P in (("members", M), ("non-members", N)):
        hA, hB = H[P["A"]], H[P["B"]]
        a = (Y[P["B"]] - Y[P["A"]]).clamp(-1, 1)
        p, p0 = dyn(hA, a), dyn(hA, torch.zeros_like(a))
        nothing = (hB - hA).pow(2).sum(1).clamp_min(1e-12)
        Ea, E0 = (p - hB).pow(2).sum(1), (p0 - hB).pow(2).sum(1)
        act = (a != 0).any(1)
        mu = hA.mean(0)
        beta = ((hA - mu) * (hB - mu)).sum() / (hA - mu).pow(2).sum()
        shrink = (beta * (hA - mu) + mu - hB).pow(2).sum(1)
        e, resid = (p - p0)[act], (hB - p0)[act]
        res[tag] = {"dyn": (Ea.sum() / nothing.sum()).item(), "dyn0": (E0.sum() / nothing.sum()).item(),
                    "shrink": (shrink.sum() / nothing.sum()).item(), "beta": beta.item(),
                    "r2_action": (1 - (resid - e).pow(2).sum() / resid.pow(2).sum()).item(),
                    "cos": F.cosine_similarity(e, resid, dim=1).mean().item(), "active": act.float().mean().item()}
        # action: the share of the zero-action error the true action removes, a per-pair action R^2 that, unlike
        # (E0 - Ea) / nothing, stays finite when an encoder has collapsed a member pair (nothing ~ 0)
        scores[tag] = {"total": -(Ea / nothing), "zero": -(E0 / nothing),
                       "action": (1 - Ea / E0.clamp_min(1e-12))[act]}
    for t in FWD_TESTS:
        res[t] = auroc(scores["members"][t].cpu(), scores["non-members"][t].cpu())
    return res


# ---------------------------------------------------------------- training curves
def wandb_curves(spec):
    """Per-epoch train / val forward loss from the local wandb run started within 2 min of the run dir's time."""
    m = re.search(r"runs/runs/(\d{8})/(\d{6})/", spec)
    if not m:
        return None
    t0 = datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S")
    best = None
    for d in (REPO / "wandb").glob(f"run-{m[1]}_*"):
        mm = re.match(r"run-(\d{8})_(\d{6})-", d.name)
        dt = abs((datetime.strptime(mm[1] + mm[2], "%Y%m%d%H%M%S") - t0).total_seconds())
        if dt <= 120 and (best is None or dt < best[0]) and any(d.glob("*.wandb")):
            best = (dt, d)
    if best is None:
        return None
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal import datastore
    ds = datastore.DataStore()
    ds.open_for_scan(str(next(best[1].glob("*.wandb"))))
    epochs = {}
    while (data := ds.scan_data()) is not None:
        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {it.key or "/".join(it.nested_key): json.loads(it.value_json) for it in rec.history.item}
        if "epoch" in row:
            epochs.setdefault(int(row["epoch"]), {}).update(row)
    keep = [e for e in sorted(epochs) if isinstance(epochs[e].get("train/pred_loss_epoch"), (int, float))
            and isinstance(epochs[e].get("val/pred_loss"), (int, float))]
    return {"wandb_run": best[1].name, "epoch": keep,
            "train_pred": [epochs[e]["train/pred_loss_epoch"] for e in keep],
            "val_pred": [epochs[e]["val/pred_loss"] for e in keep]}


# ---------------------------------------------------------------- report
def print_encoder(name, info, enc, dose, dose_layer, fwd, curves):
    head = "random init (no members: the control)" if info is None else (
        f"{info['backbone']}, {info['pairs']} {'+'.join(info['pair_types'])}, {info['epochs']} epochs, "
        f"lambda {info['lambda_reg']}, wd {info['weight_decay']}, predictor {info['predictor_hidden_dim']}")
    print(f"\n== {name}: {head}")
    if info is not None:
        c = info["check"]
        print(f"   membership check: members' A->B trained {c['members_AB_trained']:.3f} (want 1), members' A' "
              f"trained {c['members_A2_trained']:.3f} (want 0), non-members' A->B trained "
              f"{c['nonmembers_AB_trained']:.3f} (want 0)")
    print(f"   {'layer':<8}{'rel A-B m/n':>14}{'rel Aprime-B m/n':>18}{'spread m/n':>11}   "
          + "".join(f"{t:<22}" for t in ENC_TESTS))
    for layer, t in enc.items():
        r = t["rel"]
        print(f"   {layer:<8}{r['m_AB']:>7.3f}/{r['n_AB']:<6.3f}{r['m_A2B']:>10.3f}/{r['n_A2B']:<7.3f}"
              f"{r['spread_ratio']:>9.2f}     " + "".join(f"{fmt(t[k]):<22}" for k in ENC_TESTS))
    print(f"   dose ({dose_layer}):  "
          + "  ".join(f"{b} stays: {fmt(t)}" + (f" exp {t['exposure_per_epoch']:.1f}/ep" if "exposure_per_epoch" in t else "")
                      for b, t in dose.items()))
    if fwd:
        for pop in ("members", "non-members"):
            f = fwd[pop]
            print(f"   forward {pop:<12} E(a) {f['dyn']:.3f}  E(0) {f['dyn0']:.3f}  shrink {f['shrink']:.3f}  "
                  f"action R2 {f['r2_action']:+.3f}  cos {f['cos']:+.3f}   (relative to doing nothing)")
        print("   forward AUROC  " + "  ".join(f"{t} {fmt(fwd[t])}" for t in FWD_TESTS))
    if curves:
        ratio = np.array(curves["train_pred"]) / np.array(curves["val_pred"])
        ep = curves["epoch"]
        show = sorted({i for i in (0, 1, 2, 5, 10, 25, 50, len(ep) - 1) if i < len(ep)})
        print(f"   train/val forward loss ({curves['wandb_run']}): "
              + "  ".join(f"ep{ep[i]} {ratio[i]:.2f}" for i in show))


def to_json(x):
    if isinstance(x, dict):
        return {str(k): to_json(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [to_json(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main():
    args = parse_args()
    out_path = Path(args.out)
    if args.plot_only:
        plot(json.loads(out_path.read_text()), args.fig)
        return
    if not args.ckpt:
        raise SystemExit("give at least one --ckpt NAME=SPEC")
    paths = data_paths(args.config)

    members, nonmembers = sample_populations(anchor_table(paths["lance_path"]), args.holdout_folds,
                                             args.n_members, args.seed)
    print(f"members: {len(members):,} training-fold patients; non-members: {len(nonmembers):,} patients of folds "
          f"{args.holdout_folds}; one anchor each", flush=True)
    specs = dict(c.split("=", 1) for c in args.ckpt)
    infos = {n: checkpoint_info(s, paths, members, nonmembers, args.holdout_folds) for n, s in specs.items()}

    rows = np.unique(np.concatenate([df[c].values for df in (members, nonmembers) for c in ("A", "A2", "B")]))
    encoders = {n: load_encoder(s) for n, s in specs.items()}
    feats, labels = embed(encoders, rows, paths["lance_path"], batch_size=args.batch_size, num_workers=args.workers)
    print(f"embedded {len(rows):,} ECGs with {len(encoders)} encoders", flush=True)
    ix = lambda df: {c: torch.as_tensor(np.searchsorted(rows, df[c].values)) for c in ("A", "A2", "B")}
    M, N = ix(members), ix(nonmembers)

    results = {"config": {k: v for k, v in vars(args).items() if k != "plot_only"},
               "populations": {"members": len(members), "nonmembers": len(nonmembers),
                               "members_by_stays": {b: int(((members["n_stays"] >= lo) & (members["n_stays"] <= hi)).sum())
                                                    for b, (lo, hi) in BUCKETS.items()},
                               "nonmembers_by_stays": {b: int(((nonmembers["n_stays"] >= lo) & (nonmembers["n_stays"] <= hi)).sum())
                                                       for b, (lo, hi) in BUCKETS.items()}},
               "encoders": {}}
    for name, spec in specs.items():
        enc = {layer: encoder_tests(Z.double(), M, N) for layer, Z in feats[name].items()}
        dose = dose_tests(enc[args.dose_layer]["_log_r_AB"], members, nonmembers, infos[name])
        dyn = load_dynamics(spec)
        fwd = forward_tests(dyn, feats[name]["h"].cuda(), labels.cuda().float(),
                            {k: v.cuda() for k, v in M.items()}, {k: v.cuda() for k, v in N.items()}) if dyn else None
        curves = wandb_curves(spec)
        print_encoder(name, infos[name], enc, dose, args.dose_layer, fwd, curves)
        results["encoders"][name] = {"spec": spec, "arch": type(encoders[name].net).__name__,
                                     "encoder_params": sum(p.numel() for p in encoders[name].net.parameters()),
                                     "training": infos[name], "layers": enc, "dose": dose,
                                     "dose_layer": args.dose_layer, "forward": fwd, "curves": curves}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_json(results), indent=2))
    print(f"\nSaved {out_path}")
    if args.fig:
        plot(json.loads(out_path.read_text()), args.fig)


# ---------------------------------------------------------------- figure
# Chrome and the first two categorical slots of the dataviz skill's documented palette (palette.md: slots 1-3 are
# validated all-pairs in both modes); random-init controls are context, in the muted gray.
INK = {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
COLOR = {"cnn": "#2a78d6", "vit": "#eb6834"}     # color follows the entity, not its position
STAGES = ("stem /\npatch", "stage 1", "stage 2", "stage 3", "stage 4", "h")
FWD_LABELS = {"total": "forward loss\nE(a)", "zero": "zero-action fit\nE(0)",
              "action": "action's share of the error\n1 - E(a)/E(0)"}


def plot(results, path, series=("cnn", "vit"), controls=("cnn_random", "vit_random")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    enc = results["encoders"]
    series = [s for s in series if s in enc]
    controls = [c for c in controls if c in enc]
    name = lambda n: (f"{enc[n]['training']['backbone']}, {enc[n]['encoder_params'] / 1e6:.2f}M params"
                      if enc[n].get("training") else f"random init, {enc[n]['arch']}")
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK["axis"], "axes.labelcolor": INK["secondary"],
                         "xtick.color": INK["muted"], "ytick.color": INK["muted"], "axes.titlecolor": INK["primary"]})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150, facecolor=INK["surface"])
    lw, ms = 1.25, 5.5

    def frame(ax, title, ylabel, chance=0.5):
        ax.set_facecolor(INK["surface"])
        ax.grid(axis="y", color=INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axhline(chance, color=INK["axis"], linewidth=0.75, zorder=1)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(ylabel)

    def points(ax, x, tests, color, line=True, z=3, end_label=False):
        y = np.array([t["auc"] for t in tests])
        lo, hi = np.array([t["lo"] for t in tests]), np.array([t["hi"] for t in tests])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=color, elinewidth=lw, capsize=0, zorder=z)
        ax.plot(x, y, "-" if line else "none", color=color, linewidth=lw * 1.6 if line else 0, marker="o",
                markersize=ms, markerfacecolor=color, markeredgecolor=INK["surface"], markeredgewidth=1.0, zorder=z + 1)
        if end_label:
            end_value(ax, x[-1], y[-1])

    def end_value(ax, x, y):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(7, 0), textcoords="offset points", va="center",
                    color=INK["secondary"], fontsize=8.5)

    # (a) pair test at every readout layer
    ax = axes[0, 0]
    frame(ax, "a  Trained pairs separated from new patients' pairs, by layer", "membership AUROC (pair test)")
    x = np.arange(len(STAGES))
    for c in controls:
        ax.plot(x, [t["pair"]["auc"] for t in enc[c]["layers"].values()], color=INK["muted"], linewidth=lw, zorder=2)
    for s, color in ((s, COLOR[s]) for s in series):
        points(ax, x, [t["pair"] for t in enc[s]["layers"].values()], color, end_label=True)
    ax.set_xticks(x, STAGES)
    ax.set_xlim(-0.3, len(STAGES) - 0.55)
    ax.set_xlabel("readout layer (CNN: stem, layer1-4; ViT: patch embedding, block1-4)")
    ax.set_ylim(0.4, 1.02)

    # (b) dose: the pair test at h by the patient's number of stays (exposure of A per epoch in the pair file)
    ax = axes[0, 1]
    frame(ax, "b  Memorization by training exposure (layer h)", "membership AUROC (pair test)")
    buckets = list(enc[series[0]]["dose"]) if series else list(BUCKETS)
    x = np.arange(len(buckets))
    for c in controls:
        ax.plot(x, [enc[c]["dose"][b]["auc"] for b in buckets], color=INK["muted"], linewidth=lw, zorder=2)
    for i, (s, color) in enumerate((s, COLOR[s]) for s in series):
        points(ax, x + (i - 0.5) * 0.08, [enc[s]["dose"][b] for b in buckets], color)
    exposure = enc[series[0]]["dose"] if series else {}
    ax.set_xticks(x, [f"{b} stays\n{exposure[b]['exposure_per_epoch']:.1f} pairs/epoch" if "exposure_per_epoch"
                      in exposure.get(b, {}) else f"{b} stays" for b in buckets])
    ax.set_xlabel("patient's stays (training pairs containing the anchor ECG per epoch, multistep)")
    ax.set_ylim(0.4, 1.02)

    # (c) forward-model tests
    ax = axes[1, 0]
    frame(ax, "c  Forward model: what separates training transitions", "membership AUROC")
    x = np.arange(len(FWD_TESTS))
    for i, (s, color) in enumerate((s, COLOR[s]) for s in series):
        if enc[s].get("forward"):
            points(ax, x + (i - 0.5) * 0.12, [enc[s]["forward"][t] for t in FWD_TESTS], color, line=False)
    ax.set_xticks(x, [FWD_LABELS[t] for t in FWD_TESTS])
    ax.set_xlim(-0.5, len(FWD_TESTS) - 0.5)
    ax.set_ylim(0.4, 1.02)

    # (d) train / val forward loss over training, from the logs
    ax = axes[1, 1]
    frame(ax, "d  Train / val forward loss during training", "train loss / val loss", chance=1.0)
    for s, color in ((s, COLOR[s]) for s in series):
        cv = enc[s].get("curves")
        if cv:
            ep = np.array(cv["epoch"])
            ratio = np.array(cv["train_pred"]) / np.array(cv["val_pred"])
            keep = ep >= 1      # epoch 0 averages over the ViT's collapsed-scale warmup
            ax.plot(ep[keep], ratio[keep], color=color, linewidth=lw * 1.6, solid_capstyle="round", zorder=3)
            end_value(ax, ep[keep][-1], ratio[keep][-1])
    ax.set_xlabel("epoch (from 1; 1.0 = no train/val gap)")
    ax.set_ylim(0, 1.25)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.06)

    handles = [Line2D([], [], color=c, linewidth=lw * 1.6, marker="o", markersize=ms, markerfacecolor=c,
                      markeredgecolor=INK["surface"], label=f"{s.upper() if s in ('cnn', 'vit') else s}: {name(s)}")
               for s, c in ((s, COLOR[s]) for s in series)]
    if controls:
        handles.append(Line2D([], [], color=INK["muted"], linewidth=lw,
                              label="random-init controls (" + ", ".join(enc[c]["arch"] for c in controls) + ")"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=9,
               labelcolor=INK["primary"], bbox_to_anchor=(0.5, 0.995))
    fig.text(0.5, 0.947, "AUROC 0.5 = members and non-members indistinguishable; bars are DeLong 95% CIs",
             ha="center", color=INK["secondary"], fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=INK["surface"])
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":   # spawned DataLoader workers re-import this file
    main()

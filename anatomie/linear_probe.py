"""
Linear-Probe Test for Pretrained Encoders (CNN vs ViT)
=======================================================
How linearly decodable are the 76 diagnosis labels from a frozen encoder, at which layer, and how much does the
probe depend on the encoder having trained on the ECG?

Protocol: the paper's triage evaluation, run offline and identically for every encoder
--------
  probe     logistic regression (76 outputs) on standardized features of one readout layer (anatomie/encoders.py:
            CNN stem, layer1-4, h; ViT patch, block1-4, h), fitted by full-batch L-BFGS with an L2 penalty from --l2
            chosen on val; deterministic. Mini-batch AdamW with early stopping (the first version of this script,
            and layer_readout.py) had not converged after 8,000 steps and reached worse optima on some layers
            (CNN layer3 val 0.676 against 0.699), and the best penalty differs between encoders (the trained ViT's h
            prefers none, CNN layer3 1e-4), so it is chosen per layer
  train     --n-train first-of-stay ECGs of training-fold patients (folds 0-17), excluding the seen patients below
  val       every first-of-stay ECG of fold 18: picks the penalty and the best readout layer
  test      every first-of-stay ECG of fold 19 (scripts/evaluate.py's triage test set): macro AUROC over the classes
            with both outcomes, --n-boot bootstrap resamples (seed 42), mean and 95% interval as evaluate.py reports
  Reported at h (the paper's readout) and at the val-selected layer; per-layer test AUROCs are point estimates.

Seen vs unseen
--------------
The forward objective conditions on a = y(t+1) - y(t), so the labels of training ECGs reach the encoder; a memorizing
encoder also places unseen ECGs in a different region (anatomie/memorization.py, spread test). The probe is scored on
two populations it never trained on, drawn the same way:
  seen      first-of-stay ECGs of --n-seen training-fold patients with >= 2 stays, held out from the probe; for
            pair-trained checkpoints every one is in the checkpoint's own training pairs (checked)
  unseen    first-of-stay ECGs of fold-19 patients with >= 2 stays
  gap = AUROC(seen) - AUROC(unseen), from independent bootstrap resamples. Random-init encoders saw nothing, so
  their gap is the control; a label-trained encoder (supervised) shows what memorized labels look like.

Interpretation
--------------
  best layer >> h        the readout the paper uses is not where the encoder's diagnostic information sits
  gap >> control gap     the probe works better on ECGs the encoder trained on: memorized geometry and/or labels
                         that reached the encoder through the action

Example
-------
python anatomie/linear_probe.py \\
    --ckpt cnn_random=random --ckpt vit_random=random-vit \\
    --ckpt cnn=runs/runs/20260930/084313/e9ebd97377ce/checkpoints/last.ckpt \\
    --ckpt vit=runs/runs/20261006/044610/064d3fbda75f/checkpoints/last.ckpt \\
    --out outputs/linear_probe/results.json --fig outputs/linear_probe/linear_probe.png
python anatomie/linear_probe.py --plot-only --out outputs/linear_probe/results.json \\
    --fig outputs/linear_probe/linear_probe.png
"""

import argparse
import json
import sys
import time
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multilabel_auroc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import REPO, MIMICLanceDataset, data_paths, load_encoder  # noqa: E402

RANDOM_SPECS = ("random", "random-vit")
SETS = ("test", "seen", "unseen")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", default=[], metavar="NAME=SPEC",
                   help="encoder to test (repeatable): 'random', 'random-vit' or a checkpoint path")
    p.add_argument("--n-train", type=int, default=100_000, help="probe-training ECGs")
    p.add_argument("--n-seen", type=int, default=5000, help="held-out training patients for the seen set")
    p.add_argument("--l2", type=float, nargs="+", default=[0.0, 1e-6, 1e-5, 1e-4, 1e-3],
                   help="L2 penalties tried per layer; the best on val is kept")
    p.add_argument("--max-iter", type=int, default=1000, help="L-BFGS iterations (binding only for l2 = 0)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--cache", default=None,
                   help="optional .pt file for the features (float16), reused when the ECGs and checkpoints match")
    p.add_argument("--out", default="outputs/linear_probe/results.json")
    p.add_argument("--fig", default=None)
    p.add_argument("--plot-only", action="store_true", help="redraw --fig from an existing --out")
    return p.parse_args()


# ---------------------------------------------------------------- populations
def ecg_sets(lance_path, n_train, n_seen, seed):
    """Lance rows of the probe's train / val / test ECGs and the seen / unseen sets (all first ECGs of a stay)."""
    meta = lance.dataset(lance_path).to_table(columns=["subject_id", "fold", "ecg_no_within_stay"]).to_pandas()
    meta["idx"] = meta.index
    first = meta[meta["ecg_no_within_stay"] == 0]          # MIMICLanceDataset(mode="triage") rows
    stays = first.groupby("subject_id").size()
    first = first.assign(n_stays=stays.loc[first["subject_id"]].values)
    rng = np.random.default_rng(seed)
    train_fold = first[first["fold"] <= 17]
    multi = train_fold.loc[train_fold["n_stays"] >= 2, "subject_id"].unique()
    seen_patients = rng.choice(multi, size=min(n_seen, len(multi)), replace=False)
    seen = train_fold[train_fold["subject_id"].isin(seen_patients)]
    pool = train_fold[~train_fold["subject_id"].isin(seen_patients)]
    train = pool.iloc[np.sort(rng.choice(len(pool), size=min(n_train, len(pool)), replace=False))]
    test = first[first["fold"] == 19]
    by_idx = lambda df: df.sort_values("idx")
    sets = {"train": by_idx(train), "val": by_idx(first[first["fold"] == 18]), "test": by_idx(test),
            "seen": by_idx(seen), "unseen": by_idx(test[test["n_stays"] >= 2])}
    # rows (sorted) and each row's patient, for the patient-level bootstrap
    return {k: v["idx"].values for k, v in sets.items()}, {k: v["subject_id"].values for k, v in sets.items()}


def training_info(spec, paths, seen_rows):
    """What the checkpoint trained on, and the share of seen ECGs inside its own training pairs."""
    if spec in RANDOM_SPECS:
        return {"trained_on": "nothing (random init)"}
    ck = torch.load(REPO / spec, map_location="cpu", weights_only=False)
    hp, sd = ck.get("hyper_parameters") or {}, ck["state_dict"]
    vit = hp.get("vit")
    info = {"backbone": f"ViT1d {dict(vit)['dim']}x{dict(vit)['depth']}" if hp.get("backbone") == "vit1d"
            else "ResNet1d", "epochs": int(ck.get("epoch", -1)) + 1}
    if not any(k.startswith(("projector.pred.", "projector.proj.", "projector.inv.", "projector.action.")) for k in sd):
        info["trained_on"] = "labels (supervised)"
        return info
    local = {Path(paths[k]).name: paths[k] for k in ("pairs_path", "pairs_multistep_path")}
    name = Path(str(hp.get("pairs_path", ""))).name
    types = tuple(hp.get("pair_types") or ("within_stay", "cross_stay"))
    t = lance.dataset(local[name]).to_table(columns=["idx_t", "idx_t1", "fold", "pair_type"]).to_pandas()
    t = t[(t["fold"] <= 17) & t["pair_type"].isin(types)]
    used = pd.Index(np.unique(np.concatenate([t["idx_t"].values, t["idx_t1"].values])))
    info.update(trained_on=f"pairs: {name} {'+'.join(types)}", seen_in_pairs=float(used.isin(seen_rows).sum()
                                                                                     / len(seen_rows)))
    return info


# ---------------------------------------------------------------- features
@torch.no_grad()
def embed_all(encoders, rows, lance_path, batch_size, workers):
    """Every readout layer of every encoder for the ECGs at `rows`, in one pass; float16 on CPU to bound memory."""
    ds = MIMICLanceDataset(lance_path, split="train", mode="monitoring")
    ds.rows = rows                      # monitoring mode reads any rows through self.rows
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers, multiprocessing_context="spawn",
                        pin_memory=True)
    feats = {n: None for n in encoders}
    labels = torch.empty(len(rows), 76)
    s, t0 = 0, time.time()
    for i, batch in enumerate(loader):
        x, B = batch["waveform"].cuda(non_blocking=True), len(batch["waveform"])
        labels[s:s + B] = batch["label"]
        for n, enc in encoders.items():
            out = enc(x)
            if feats[n] is None:
                feats[n] = {l: torch.empty(len(rows), v.shape[1], dtype=torch.float16) for l, v in out.items()}
            for l, v in out.items():
                feats[n][l][s:s + B] = v.half().cpu()
        s += B
        if i % 100 == 0 and i:
            print(f"  embedded {s:,}/{len(rows):,} ECGs ({s / (time.time() - t0):.0f}/s)", flush=True)
    return feats, labels


def cached_embed(cache, encoders, specs, rows, lance_path, batch_size, workers):
    """embed_all, reusing --cache when it holds the same ECGs and the same checkpoint files (by modification time)."""
    key = {n: (s, 0.0 if s in RANDOM_SPECS else (REPO / s).stat().st_mtime) for n, s in specs.items()}
    if cache and Path(cache).exists():
        z = torch.load(cache, weights_only=False)      # our own cache file
        if np.array_equal(z["rows"], rows) and all(z["key"].get(n) == k for n, k in key.items()):
            print(f"loaded cached features: {cache}", flush=True)
            return {n: z["feats"][n] for n in specs}, z["labels"]
        print("cache does not match the ECGs or checkpoints; re-embedding", flush=True)
    feats, labels = embed_all(encoders, rows, lance_path, batch_size, workers)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"rows": rows, "key": key, "feats": feats, "labels": labels}, cache)
        print(f"saved features: {cache}", flush=True)
    return feats, labels


# ---------------------------------------------------------------- probe and metrics
def macro_auroc(P, Y):
    """Macro AUROC over the classes with both outcomes (scripts/evaluate_inverse.py's macro)."""
    per = multilabel_auroc(P, Y.long(), num_labels=Y.shape[1], average="none")
    n_pos = Y.sum(0)
    return per[(n_pos > 0) & (n_pos < len(Y))].mean()


def bootstrap(P, Y, n_boot, seed=42):
    """Bootstrap replicates of the macro AUROC (resampling ECGs, as scripts/evaluate.py)."""
    g = torch.Generator(device=P.device).manual_seed(seed)
    return torch.stack([macro_auroc(P[idx], Y[idx])
                        for idx in (torch.randint(0, len(Y), (len(Y),), device=P.device, generator=g)
                                    for _ in range(n_boot))]).cpu().numpy()


def cluster_bootstrap(P, Y, groups, n_boot, seed=42):
    """Bootstrap replicates resampling patients, not ECGs: a patient's ECGs are correlated, so an ECG-level
    bootstrap understates the variance of sets with several ECGs per patient."""
    _, inv = np.unique(groups, return_inverse=True)
    order = torch.as_tensor(np.argsort(inv, kind="stable"), device=P.device)
    counts = torch.as_tensor(np.bincount(inv), device=P.device)
    starts = torch.cumsum(counts, 0) - counts
    g = torch.Generator(device=P.device).manual_seed(seed)
    reps = []
    for _ in range(n_boot):
        pick = torch.randint(0, len(counts), (len(counts),), device=P.device, generator=g)
        c = counts[pick]
        within = torch.arange(int(c.sum()), device=P.device) - torch.repeat_interleave(torch.cumsum(c, 0) - c, c)
        idx = order[torch.repeat_interleave(starts[pick], c) + within]
        reps.append(macro_auroc(P[idx], Y[idx]))
    return torch.stack(reps).cpu().numpy()


def summary(reps):
    return {"auc": float(reps.mean()), "lo": float(np.percentile(reps, 2.5)), "hi": float(np.percentile(reps, 97.5))}


def fit_probe(X, Y, Xval, Yval, l2s, max_iter):
    """Full-batch L-BFGS logistic regression, BCE + 0.5 * l2 * |W|^2 on standardized features, for each l2; the l2
    with the best val macro AUROC is kept. Deterministic (zero init); l2 = 0 has no optimum and stops at max_iter.
    Returns (predict function, chosen l2, its val AUROC)."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X, Xval = (X - mu) / sd, (Xval - mu) / sd
    best = None
    for l2 in l2s:
        W = torch.zeros(X.shape[1], Y.shape[1], device=X.device, requires_grad=True)
        b = torch.zeros(Y.shape[1], device=X.device, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(X @ W + b, Y) + 0.5 * l2 * W.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            auc = macro_auroc(torch.sigmoid(Xval @ W + b), Yval).item()
        if best is None or auc > best[0]:
            best = (auc, W.detach(), b.detach(), l2)
    _, W, b, l2 = best

    @torch.no_grad()
    def predict(Z):
        return torch.sigmoid(((Z - mu) / sd) @ W + b)
    return predict, l2, best[0]


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


def main():
    args = parse_args()
    out_path = Path(args.out)
    if args.plot_only:
        plot(json.loads(out_path.read_text()), args.fig)
        return
    if not args.ckpt:
        raise SystemExit("give at least one --ckpt NAME=SPEC")
    paths = data_paths(args.config)

    sets, patients = ecg_sets(paths["lance_path"], args.n_train, args.n_seen, args.seed)
    print("ECGs: " + ", ".join(f"{k} {len(v):,} ({len(np.unique(patients[k])):,} patients)" for k, v in sets.items()),
          flush=True)
    specs = dict(c.split("=", 1) for c in args.ckpt)
    infos = {n: training_info(s, paths, sets["seen"]) for n, s in specs.items()}

    rows = np.unique(np.concatenate(list(sets.values())))
    encoders = {n: load_encoder(s) for n, s in specs.items()}
    feats, labels = cached_embed(args.cache, encoders, specs, rows, paths["lance_path"], args.batch_size, args.workers)
    pos = {k: torch.as_tensor(np.searchsorted(rows, v)) for k, v in sets.items()}
    Y = {k: labels[v].cuda() for k, v in pos.items()}

    results = {"config": {k: v for k, v in vars(args).items() if k != "plot_only"},
               "sets": {k: len(v) for k, v in sets.items()}, "encoders": {}}
    # Resumable: encoders already in --out under the same protocol are kept, and results are saved after each one
    same = ("n_train", "n_seen", "l2", "max_iter", "n_boot", "seed", "config")
    if out_path.exists():
        old = json.loads(out_path.read_text())
        if all(old["config"].get(k) == results["config"][k] for k in same):
            results["encoders"] = {n: r for n, r in old["encoders"].items() if specs.get(n) == r["spec"]}
            if results["encoders"]:
                print(f"resuming: {', '.join(results['encoders'])} already done", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for name, spec in specs.items():
        if name in results["encoders"]:
            continue
        t0 = time.time()
        layers, probes = {}, {}
        for layer, Z in feats[name].items():
            X = {k: Z[v].float().cuda() for k, v in pos.items()}
            predict, l2, val_auc = fit_probe(X["train"], Y["train"], X["val"], Y["val"], args.l2, args.max_iter)
            P = {k: predict(X[k]) for k in SETS}
            layers[layer] = {"val": val_auc, "l2": l2, "test": macro_auroc(P["test"], Y["test"]).item()}
            probes[layer] = P
        best = max(layers, key=lambda l: layers[l]["val"])
        readouts = {}
        for tag, layer in (("h", "h"), ("best", best)):
            P = probes[layer]
            # test: ECG-level, as scripts/evaluate.py; seen / unseen: patient-level, for a valid gap interval
            reps = {"test": bootstrap(P["test"], Y["test"], args.n_boot),
                    **{k: cluster_bootstrap(P[k], Y[k], patients[k], args.n_boot) for k in ("seen", "unseen")}}
            readouts[tag] = {"layer": layer, **{k: summary(reps[k]) for k in SETS},
                             "gap": summary(reps["seen"] - reps["unseen"])}
        results["encoders"][name] = {"spec": spec, "arch": type(encoders[name].net).__name__,
                                     "encoder_params": sum(p.numel() for p in encoders[name].net.parameters()),
                                     "training": infos[name], "layers": layers, "readouts": readouts}

        info = infos[name]
        extra = f", seen ECGs in its pairs {info['seen_in_pairs']:.3f}" if "seen_in_pairs" in info else ""
        print(f"\n== {name}: {info.get('backbone', results['encoders'][name]['arch'])}, {info['trained_on']}{extra}  "
              f"({time.time() - t0:.0f}s)")
        print("   " + "  ".join(f"{l} {v['test']:.3f}" for l, v in layers.items()) + "   (test, per layer)")
        print("   " + "  ".join(f"{l} {v['val']:.3f}@{v['l2']:g}" for l, v in layers.items()) + "   (val @ l2)")
        for tag in ("h", "best"):
            r = readouts[tag]
            print(f"   {tag:<4} [{r['layer']}]  test {fmt(r['test'])}   seen {fmt(r['seen'])}   unseen {fmt(r['unseen'])}"
                  f"   gap {r['gap']['auc']:+.3f} [{r['gap']['lo']:+.3f} to {r['gap']['hi']:+.3f}]", flush=True)
        out_path.write_text(json.dumps(to_json(results), indent=2))

    results["encoders"] = {n: results["encoders"][n] for n in specs}     # command-line order
    out_path.write_text(json.dumps(to_json(results), indent=2))
    print(f"\nSaved {out_path}")
    if args.fig:
        plot(json.loads(out_path.read_text()), args.fig)


# ---------------------------------------------------------------- figure
# The same chrome and the first two categorical slots of the dataviz skill's documented palette as
# anatomie/memorization.py; everything that is not the matched pair is context, in the muted gray.
INK = {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
COLOR = {"cnn": "#2a78d6", "vit": "#eb6834"}     # color follows the entity, not its position
STAGES = ("stem /\npatch", "stage 1", "stage 2", "stage 3", "stage 4", "h")
NAMES = {"cnn": "CNN (matched)", "vit": "ViT (matched)", "cnn_random": "CNN, random init",
         "vit_random": "ViT, random init", "cnn_ep5": "CNN, epoch 5", "cnn_lowreg": "CNN, λ 0.005 / wd 0.01",
         "cnn_k1": "CNN, k=1", "vit_k1": "ViT, k=1, 405 ep", "vit_k1_p128": "ViT, k=1, predictor 128",
         "paper_dyn": "paper Dynamics (CNN)", "paper_inv": "paper Inverse (CNN)", "supervised": "supervised (CNN)"}


def plot(results, path, series=("cnn", "vit"), controls=("cnn_random", "vit_random")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    enc = results["encoders"]
    series = [s for s in series if s in enc]
    controls = [c for c in controls if c in enc]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK["axis"], "axes.labelcolor": INK["secondary"],
                         "xtick.color": INK["muted"], "ytick.color": INK["muted"], "axes.titlecolor": INK["primary"]})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, facecolor=INK["surface"],
                             gridspec_kw={"width_ratios": [1, 1.15]})
    lw, ms = 1.25, 5.5

    def frame(ax, title):
        ax.set_facecolor(INK["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=10)

    # (a) test macro AUROC by readout layer
    ax = axes[0]
    frame(ax, "a  Test macro AUROC of a linear probe, by layer")
    ax.grid(axis="y", color=INK["grid"], linewidth=0.5)
    x = np.arange(len(STAGES))
    for c in controls:
        ax.plot(x, [v["test"] for v in enc[c]["layers"].values()], color=INK["muted"], linewidth=lw, zorder=2)
    for s in series:
        color = COLOR[s]
        y = [v["test"] for v in enc[s]["layers"].values()]
        ax.plot(x, y, color=color, linewidth=lw * 1.6, marker="o", markersize=ms, markerfacecolor=color,
                markeredgecolor=INK["surface"], markeredgewidth=1.0, zorder=3)
        b = list(enc[s]["layers"]).index(enc[s]["readouts"]["best"]["layer"])
        ax.annotate(f"{y[b]:.3f}", (x[b], y[b]), xytext=(0, 8), textcoords="offset points", ha="center",
                    color=INK["secondary"], fontsize=8.5)
    ax.set_xticks(x, STAGES)
    ax.set_xlim(-0.3, len(STAGES) - 0.4)
    ax.set_xlabel("readout layer (CNN: stem, layer1-4; ViT: patch embedding, block1-4)")
    ax.set_ylabel("test macro AUROC (L2 chosen on val)")

    # (b) seen - unseen gap at the val-selected layer, every encoder
    ax = axes[1]
    frame(ax, "b  Probe advantage on ECGs the encoder trained on (layer h)")
    ax.grid(axis="x", color=INK["grid"], linewidth=0.5)
    names = list(enc)[::-1]
    for i, n in enumerate(names):
        g = enc[n]["readouts"]["h"]["gap"]
        color = COLOR[n] if n in series else INK["muted"]
        ax.plot([g["lo"], g["hi"]], [i, i], color=color, linewidth=lw, solid_capstyle="round", zorder=2)
        ax.plot(g["auc"], i, "o", color=color, markersize=ms, markeredgecolor=INK["surface"], markeredgewidth=1.0,
                zorder=3)
    ax.axvline(0, color=INK["axis"], linewidth=0.75, zorder=1)
    ax.set_yticks(range(len(names)), [NAMES.get(n, n) for n in names])
    ax.tick_params(axis="y", length=0, labelcolor=INK["secondary"])
    ax.set_xlabel("AUROC(seen) - AUROC(unseen) at h; 95% patient-level bootstrap interval")

    label = lambda s: (f"{s.upper()}: {enc[s]['training'].get('backbone', enc[s]['arch'])}, "
                       f"{enc[s]['encoder_params'] / 1e6:.2f}M params")
    handles = [Line2D([], [], color=COLOR[s], linewidth=lw * 1.6, marker="o", markersize=ms,
                      markerfacecolor=COLOR[s], markeredgecolor=INK["surface"], label=label(s)) for s in series]
    handles.append(Line2D([], [], color=INK["muted"], linewidth=lw, label="random-init controls and other encoders"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=9,
               labelcolor=INK["primary"], bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=INK["surface"])
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":   # spawned DataLoader workers re-import this file
    main()

"""
Beat-Space Action Ceiling
=========================
How much of the change between a patient's ECGs at consecutive stays does the diagnosis transition explain, in
signal space? anatomie/history_ceiling.py measures it on embeddings (<= 0.3%; <= 1.7% for label-trained encoders),
which depend on what the encoder learned. This measures it on features fixed by the signal: the median beat and
rhythm of anatomie/median_beats.py, in the input's per-lead z-units (what every encoder sees; absolute amplitudes are
not available to any model).

Pairs: k=1 cross-stay pairs of pairs.lance, fitted on train folds (<= 17), ridge strength chosen on fold 18,
scored on fold 19. Pairs where either ECG has no usable beats are dropped.

Targets, per ECG (the next ECG of a pair is predicted from the current one):
  beat        the 12-lead median beat at 100 Hz (12 x 100 means of 5 samples)
  hr, rr_irr  heart rate (bpm) and RR irregularity (normalized RMSSD)
  qrs_width   ms around the QRS fiducial where the lead-summed absolute slope of the beat exceeds 25% of its peak
              (searched -100..+150 ms): a proxy, checked against conduction-block labels at startup
  per lead    qrs (peak-to-peak, -50..+70 ms), st (mean +70..+110 ms) and t (signed extreme, +150..+450 ms), the
              last two relative to the PR baseline (-80..-40 ms)
Scalar features are clipped to their 0.1-99.9 train percentiles (detection failures would otherwise dominate).

Estimators, each with and without the action (onset / resolution indicators, 152 features):
  ridge   on the current values: the whole beat for beat, all 39 scalar features for every scalar target
  trees   gradient-boosted trees on the 39 current scalar features for the summary targets; they can use
          interactions between the action and the current state (fit on the change, scored on the next value)
  action share = (SSE without - SSE with) / SSE without on fold 19, i.e. (R2 with - R2 without) / (1 - R2 without):
                 the part of what the current ECG leaves unexplained that the transition explains. The beat's SSE is
                 pooled over its 1,200 samples in z-units (anatomie/signal_decoder.py's beat MSE); scalar targets are
                 standardized and also reported one by one
  shuffled     ridge with the actions permuted within each split: the share an uninformative action gets
Noise floor: two ECGs of the same stay share every label (a = 0). Var(change) of within-stay pairs over that of
cross-stay pairs, by time gap, is the part of the change seen without any diagnosis transition.
Label effects: mean change of each scalar feature on cross-stay pairs (all folds) with an onset (resolution) of a
label, minus that on pairs where the label is absent (present) at both stays, in SDs of the latter.

Example
-------
python anatomie/beat_action_ceiling.py      # builds anatomie/median_beats.py's cache on first use
"""

import argparse
import json
import sys
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import REPO, data_paths  # noqa: E402
from median_beats import FS, PRE, load_cache  # noqa: E402

OUT = REPO / "outputs" / "beat_action_ceiling"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SCALARS = (["hr", "rr_irr", "qrs_width"] + [f"qrs_{l}" for l in LEADS] + [f"st_{l}" for l in LEADS]
           + [f"t_{l}" for l in LEADS])
TREE_TARGETS = ["hr", "rr_irr", "qrs_width", "qrs_V1", "qrs_V5", "st_II", "st_V2", "t_II", "t_V5"]
EFFECT_LABELS = ["I48", "I44", "I45", "I21", "I47", "I49", "I50", "I25", "I10", "I42"]
LAMBDAS = [1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5]
ms = lambda t: PRE + int(round(t * FS / 1000))     # sample index of t ms after the QRS fiducial


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--workers", type=int, default=16, help="only used if the beat cache must be built")
    p.add_argument("--chunk", type=int, default=8192)
    return p.parse_args()


def features(beats):
    """beats (B, 12, 500) float32 -> (B, 12, 100) beat at 100 Hz, (B, 37) [qrs_width, qrs x12, st x12, t x12]."""
    B = len(beats)
    beat100 = beats.reshape(B, 12, 100, 5).mean(-1)
    base = beats[:, :, ms(-80):ms(-40)].mean(-1)
    qrs = beats[:, :, ms(-50):ms(70)].max(-1) - beats[:, :, ms(-50):ms(70)].min(-1)
    st = beats[:, :, ms(70):ms(110)].mean(-1) - base
    tw = beats[:, :, ms(150):ms(450)] - base[..., None]
    t = np.take_along_axis(tw, np.abs(tw).argmax(-1)[..., None], -1)[..., 0]
    lo, hi = ms(-100), ms(150)
    slope = np.abs(np.diff(beats[:, :, lo:hi], axis=-1)).sum(1)                   # (B, hi - lo - 1)
    above = slope >= 0.25 * slope.max(-1, keepdims=True)
    first = above.argmax(-1)
    last = above.shape[-1] - 1 - above[:, ::-1].argmax(-1)
    width = (last - first + 1) * 1000.0 / FS
    return beat100, np.concatenate([width[:, None], qrs, st, t], 1)


def load(paths, args):
    meta = lance.dataset(paths["lance_path"]).to_table(columns=["ecg_time", "icd"])
    labels = (meta.column("icd").combine_chunks().flatten().to_numpy(zero_copy_only=False)
              .reshape(meta.num_rows, 76).astype(np.int8))
    # Seconds as float64 (MIMIC's shifted dates, ~6.6e9 s); pd.to_datetime would read them as nanoseconds
    ecg_time = meta.column("ecg_time").to_numpy(zero_copy_only=False).astype(np.float64)
    pairs = lance.dataset(paths["pairs_path"]).to_table(
        columns=["idx_t", "idx_t1", "fold", "pair_type"]).to_pandas()
    cache = load_cache(paths["lance_path"], num_workers=args.workers)
    rows = np.unique(np.concatenate([pairs["idx_t"].values, pairs["idx_t1"].values]))
    rows = rows[cache["n_beats"][rows] > 0]
    beat100 = np.zeros((len(rows), 12, 100), np.float32)
    scal = np.zeros((len(rows), len(SCALARS)), np.float32)
    for s in range(0, len(rows), args.chunk):
        r = rows[s:s + args.chunk]
        b, f = features(np.asarray(cache["beats"][r], dtype=np.float32))
        beat100[s:s + len(r)] = b
        scal[s:s + len(r), 0:2] = cache["rhythm"][r]
        scal[s:s + len(r), 2:] = f
    ok = np.isfinite(scal).all(1)
    rows, beat100, scal = rows[ok], beat100[ok], scal[ok]
    pos = np.full(len(labels), -1)
    pos[rows] = np.arange(len(rows))
    pairs["p_t"], pairs["p_t1"] = pos[pairs["idx_t"].values], pos[pairs["idx_t1"].values]
    pairs = pairs[(pairs["p_t"] >= 0) & (pairs["p_t1"] >= 0)].reset_index(drop=True)
    pairs["gap_h"] = (ecg_time[pairs["idx_t1"].values] - ecg_time[pairs["idx_t"].values]) / 3600.0
    print(f"{len(rows):,} ECGs with usable beats; pairs kept: "
          f"{pairs.groupby('pair_type').size().to_dict()}", flush=True)
    return labels, rows, beat100.reshape(len(rows), -1), scal, pairs


# ---------------------------------------------------------------- estimators
def ridge_fit(X, Y, lam):
    X = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype, device=X.device)], 1)
    G = X.T @ X
    return torch.linalg.solve(G + lam * torch.eye(len(G), dtype=X.dtype, device=X.device), X.T @ Y)


def ridge_predict(X, W):
    return torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype, device=X.device)], 1) @ W


def ridge_sse(Xs, Ys):
    """Per-target test SSE, with the ridge strength chosen on val (pooled R2). Xs, Ys: {split: tensor}."""
    mu, sd = Xs["train"].mean(0), Xs["train"].std(0) + 1e-6
    X = {k: (v - mu) / sd for k, v in Xs.items()}
    best = None
    for lam in LAMBDAS:
        W = ridge_fit(X["train"], Ys["train"], lam * len(X["train"]) / 1e4)
        sse = ((ridge_predict(X["val"], W) - Ys["val"]) ** 2).sum().item()
        if best is None or sse < best[0]:
            best = (sse, W, lam)
    return ((ridge_predict(X["test"], best[1]) - Ys["test"]) ** 2).sum(0), best[2]


def share(sse_without, sse_with):
    return 1.0 - sse_with / sse_without


def main():
    args = parse_args()
    paths = data_paths(args.config)
    OUT.mkdir(parents=True, exist_ok=True)
    labels, rows, beat, scal, pairs = load(paths, args)
    vocab = json.load(open(paths["vocabulary_path"]))
    lab = labels[rows].astype(np.float32)                                         # labels per kept ECG

    # Sanity check of the QRS width proxy: wider with conduction blocks present
    w = scal[:, SCALARS.index("qrs_width")]
    for code in ("I44", "I45"):
        has = lab[:, vocab.index(code)] == 1
        print(f"qrs_width with {code} present {np.median(w[has]):.0f} ms (n={has.sum():,}) vs absent "
              f"{np.median(w[~has]):.0f} ms", flush=True)

    cross = pairs[pairs["pair_type"] == "cross_stay"].reset_index(drop=True)
    split = np.select([cross["fold"] <= 17, cross["fold"] == 18], ["train", "val"], "test")
    lo, hi = np.percentile(scal[cross["p_t"].values[split == "train"]], [0.1, 99.9], axis=0)
    scal = np.clip(scal, lo, hi)
    yt, yt1 = lab[cross["p_t"].values], lab[cross["p_t1"].values]
    a = yt1 - yt
    A = np.concatenate([(a == 1), (a == -1)], 1).astype(np.float64)
    dev = "cuda"
    T = lambda x, m: torch.tensor(x[m], dtype=torch.float64, device=dev)
    rng = np.random.default_rng(0)
    results = {"n_pairs": {s: int((split == s).sum()) for s in ("train", "val", "test")}}

    # ---------------- ridge: whole beat and scalar features, with / without / shuffled action
    ridge = {}
    for name, F in (("beat", beat), ("scalars", scal)):
        cur, nxt = F[cross["p_t"].values], F[cross["p_t1"].values]
        # The beat stays in z-units, so its pooled error is the decoder's beat MSE; scalars have different units
        ysd = np.ones(F.shape[1], np.float32) if name == "beat" else nxt[split == "train"].std(0) + 1e-6
        Ys = {s: T(nxt / ysd, split == s) for s in ("train", "val", "test")}
        X0 = {s: T(cur, split == s) for s in ("train", "val", "test")}
        XA = {s: torch.cat([X0[s], T(A, split == s)], 1) for s in X0}
        perm = {s: torch.tensor(rng.permutation(int((split == s).sum())), device=dev) for s in X0}
        XS = {s: torch.cat([X0[s], T(A, split == s)[perm[s]]], 1) for s in X0}
        sse0, l0 = ridge_sse(X0, Ys)
        sseA, lA = ridge_sse(XA, Ys)
        sseS, _ = ridge_sse(XS, Ys)
        sst = ((Ys["test"] - Ys["test"].mean(0)) ** 2).sum(0)
        ridge[name] = {"r2_without": (1 - sse0.sum() / sst.sum()).item(), "r2_with": (1 - sseA.sum() / sst.sum()).item(),
                       "share": share(sse0.sum(), sseA.sum()).item(), "share_shuffled": share(sse0.sum(), sseS.sum()).item(),
                       "lambda": [l0, lA]}
        if name == "scalars":
            ridge["per_feature"] = {f: {"r2_without": (1 - sse0[i] / sst[i]).item(), "share": share(sse0[i], sseA[i]).item(),
                                        "share_shuffled": share(sse0[i], sseS[i]).item()} for i, f in enumerate(SCALARS)}
        print(f"ridge {name:<8} R2 without {ridge[name]['r2_without']:.4f} with {ridge[name]['r2_with']:.4f} "
              f"share {ridge[name]['share']:+.4f} (shuffled {ridge[name]['share_shuffled']:+.4f})", flush=True)
    results["ridge"] = ridge

    # ---------------- trees on the summary targets (interactions with the current state)
    from sklearn.ensemble import HistGradientBoostingRegressor
    cur, nxt = scal[cross["p_t"].values], scal[cross["p_t1"].values]
    tr, te = split == "train", split == "test"
    trees = {}
    for f in TREE_TARGETS:
        i = SCALARS.index(f)
        d = nxt[:, i] - cur[:, i]
        sse = {}
        for tag, X in (("without", cur), ("with", np.concatenate([cur, A], 1))):
            m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, early_stopping=True,
                                              validation_fraction=0.1, random_state=0)
            m.fit(X[tr], d[tr])
            sse[tag] = float(((cur[te, i] + m.predict(X[te]) - nxt[te, i]) ** 2).sum())
        sst = float(((nxt[te, i] - nxt[te, i].mean()) ** 2).sum())
        trees[f] = {"r2_without": 1 - sse["without"] / sst, "r2_with": 1 - sse["with"] / sst,
                    "share": 1 - sse["with"] / sse["without"]}
        print(f"trees {f:<10} R2 without {trees[f]['r2_without']:.4f} with {trees[f]['r2_with']:.4f} "
              f"share {trees[f]['share']:+.4f}", flush=True)
    results["trees"] = trees

    # ---------------- noise floor: changes without any transition (same stay) vs across stays
    def change_var(df):
        d = scal[df["p_t1"].values] - scal[df["p_t"].values]
        db = beat[df["p_t1"].values] - beat[df["p_t"].values]
        return np.concatenate([[np.mean(db ** 2)], d.var(0)]), len(df)
    within = pairs[pairs["pair_type"] == "within_stay"]
    ref, n_ref = change_var(cross)
    floor = {"cross_stay_n": n_ref}
    for tag, m in (("within <1h", within["gap_h"] < 1), ("within 1-24h", (within["gap_h"] >= 1) & (within["gap_h"] < 24)),
                   ("within >=24h", within["gap_h"] >= 24)):
        v, n = change_var(within[m])
        floor[tag] = {"n": n, **{f: float(x) for f, x in zip(["beat"] + SCALARS, v / ref)}}
    results["noise_floor"] = floor
    print("\nchange variance relative to cross-stay pairs (beat, hr, rr_irr, qrs_width; median over leads qrs/st/t):")
    for tag in ("within <1h", "within 1-24h", "within >=24h"):
        r = floor[tag]
        lead = lambda k: np.median([r[f"{k}_{l}"] for l in LEADS])
        print(f"  {tag:<13} n={r['n']:>7,}  beat {r['beat']:.2f}  hr {r['hr']:.2f}  rr_irr {r['rr_irr']:.2f}  "
              f"qrs_width {r['qrs_width']:.2f}  qrs {lead('qrs'):.2f}  st {lead('st'):.2f}  t {lead('t'):.2f}", flush=True)

    # ---------------- label effects: mean change on onset / resolution pairs vs stable pairs, in SDs
    d = scal[cross["p_t1"].values] - scal[cross["p_t"].values]
    effects = {}
    print("\nlargest standardized effects (mean change vs stable pairs, in SDs of the stable pairs' change):")
    for code in EFFECT_LABELS:
        c = vocab.index(code)
        for kind, m_ev, m_ref in (("onset", (yt[:, c] == 0) & (yt1[:, c] == 1), (yt[:, c] == 0) & (yt1[:, c] == 0)),
                                  ("resolution", (yt[:, c] == 1) & (yt1[:, c] == 0), (yt[:, c] == 1) & (yt1[:, c] == 1))):
            if m_ev.sum() < 30:
                continue
            eff = (d[m_ev].mean(0) - d[m_ref].mean(0)) / (d[m_ref].std(0) + 1e-9)
            effects[f"{code} {kind}"] = {"n": int(m_ev.sum()), **{f: float(e) for f, e in zip(SCALARS, eff)}}
            top = np.argsort(-np.abs(eff))[:4]
            print(f"  {code} {kind:<10} n={m_ev.sum():>5,}  " + "  ".join(f"{SCALARS[i]} {eff[i]:+.2f}" for i in top),
                  flush=True)
    results["label_effects"] = effects
    with open(OUT / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved {OUT / 'results.json'}", flush=True)


if __name__ == "__main__":   # load_cache's (re)build uses spawned DataLoader workers
    main()

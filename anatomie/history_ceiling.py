"""
History Ceiling for Forward Predictors
======================================
Would a predictor that sees a patient's history (e.g. a causal transformer over visits) predict h(t+1) better, and
would the action then explain more? Upper-bound estimate with linear ridge predictors on frozen embeddings.

For every patient with >= 2 stays, the first ECG of each stay in time order gives a sequence h_1..h_L with labels
y_1..y_L. Each consecutive transition (s -> s+1) is one sample; the target is h_{s+1} (standardized on train).
Feature sets, fitted on train-fold patients and scored on val-fold patients (R^2 = 1 - SSE / SST, all dims):

  now       h_s                                          the information the current predictor has
  +history  + mean of h_1..h_{s-1}, has-history flag, log(1 + number of earlier stays)
  +time     + log(1 + days since the previous stay)
  each with and without the action a_s = y_{s+1} - y_s (separate onset / resolution indicators, 152 features)

  action share = (R^2 with action - R^2 without) / (1 - R^2 without): the share of what the context leaves
  unexplained that the action explains. If history doesn't raise it, no history model (causal transformer
  included) will make the action identifiable under MSE; it can only predict the patient better.

Example
-------
python anatomie/history_ceiling.py --ckpt vit_ms=runs/runs/20261006/044610/064d3fbda75f/checkpoints/last.ckpt \\
    --ckpt inverse=archive/inverse.ckpt --ckpt random=random
"""

import argparse
import sys
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import data_paths, embed, load_encoder  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True, metavar="NAME=SPEC",
                   help="encoder (repeatable); SPEC is 'random', 'random-vit' or a checkpoint path")
    p.add_argument("--layer", default="h")
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def transitions(lance_path):
    """Consecutive-stay transitions of train (folds 0-17) and val (fold 18) patients with >= 2 stays."""
    meta = lance.dataset(lance_path).to_table(
        columns=["subject_id", "ecg_time", "fold", "ecg_no_within_stay"]).to_pandas()
    meta["row"] = meta.index
    meta["ecg_time"] = pd.to_datetime(meta["ecg_time"])
    first = meta[(meta["ecg_no_within_stay"] == 0) & (meta["fold"] <= 18)].sort_values(["subject_id", "ecg_time"])
    first = first[first.groupby("subject_id")["row"].transform("size") >= 2]
    nxt = first.groupby("subject_id").shift(-1)
    t = first[nxt["row"].notna()].copy()
    t["row_next"] = nxt.loc[t.index, "row"].astype(np.int64)
    t["gap_days"] = (nxt.loc[t.index, "ecg_time"] - t["ecg_time"]).dt.total_seconds() / 86400
    return first, t


def ridge_r2(Xtr, Ytr, Xva, Yva, lam):
    Xtr, Xva = (torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1) for X in (Xtr, Xva))
    W = torch.linalg.solve(Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1], dtype=Xtr.dtype), Xtr.T @ Ytr)
    r2 = lambda X, Y: 1 - ((X @ W - Y) ** 2).sum().item() / ((Y - Y.mean(0)) ** 2).sum().item()
    return r2(Xtr, Ytr), r2(Xva, Yva)


def main():
    args = parse_args()
    lance_path = data_paths(args.config)["lance_path"]
    first, t = transitions(lance_path)
    split = np.where(t["fold"].values <= 17, "train", "val")
    print(f"transitions: train {np.sum(split == 'train'):,}, val {np.sum(split == 'val'):,}; "
          f"{len(first):,} stays of {first['subject_id'].nunique():,} patients", flush=True)

    specs = dict(c.split("=", 1) for c in args.ckpt)
    feats, labels = embed({n: load_encoder(s) for n, s in specs.items()}, first["row"].values, lance_path,
                          layers={args.layer}, num_workers=args.workers)
    pos = {r: i for i, r in enumerate(first["row"].values)}
    i_now = np.array([pos[r] for r in t["row"].values])
    i_next = np.array([pos[r] for r in t["row_next"].values])
    y = labels.double()
    a = (y[i_next] - y[i_now]).clamp(-1, 1)
    action = torch.cat([(a == 1).double(), (a == -1).double()], 1)
    gap = torch.tensor(np.log1p(np.clip(t["gap_days"].values, 0, None)), dtype=torch.float64).unsqueeze(1)
    tr, va = torch.tensor(split == "train"), torch.tensor(split == "val")

    for name in specs:
        H = feats[name][args.layer].double()
        H = (H - H[i_now][tr.numpy()].mean(0)) / (H[i_now][tr.numpy()].std(0) + 1e-6)
        # mean of the patient's earlier stays: cumulative sums within each patient's sequence
        subj = first["subject_id"].values
        prev_mean = torch.zeros_like(H)
        n_prev = np.zeros(len(H))
        start = 0
        for k in range(1, len(H) + 1):
            if k == len(H) or subj[k] != subj[start]:
                c = torch.cumsum(H[start:k], 0)
                idx = torch.arange(1, k - start)
                prev_mean[start + 1:k] = c[:-1] / idx.unsqueeze(1)
                n_prev[start:k] = np.arange(k - start)
                start = k
        hist = torch.cat([prev_mean, torch.tensor(n_prev > 0, dtype=torch.float64).unsqueeze(1),
                          torch.tensor(np.log1p(n_prev), dtype=torch.float64).unsqueeze(1)], 1)
        X = {"now": H[i_now]}
        X["+history"] = torch.cat([X["now"], hist[i_now]], 1)
        X["+time"] = torch.cat([X["+history"], gap], 1)
        Y = H[i_next]
        print(f"\n== {name} ({args.layer}, dim {H.shape[1]})")
        print(f"{'context':<10}{'R2 val':>9}{'R2 val +action':>16}{'action share':>14}{'   (train: R2, +action)':>24}")
        for ctx, Xc in X.items():
            r_tr, r_va = ridge_r2(Xc[tr], Y[tr], Xc[va], Y[va], args.ridge)
            ra_tr, ra_va = ridge_r2(torch.cat([Xc, action], 1)[tr], Y[tr], torch.cat([Xc, action], 1)[va], Y[va], args.ridge)
            print(f"{ctx:<10}{r_va:>9.4f}{ra_va:>16.4f}{(ra_va - r_va) / (1 - r_va):>14.4f}"
                  f"        ({r_tr:.4f}, {ra_tr:.4f})", flush=True)


if __name__ == "__main__":   # spawned DataLoader workers re-import this file
    main()

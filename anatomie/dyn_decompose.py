"""
Forward-Model Decomposition: Shrinkage vs Action Effect
========================================================
How much of a forward checkpoint's prediction of h(t+1) comes from the action, and does that part generalize?
On random train and val pairs, with the encoder's h (anatomie/encoders.py) and the trained Dyn (load_dynamics):

  nothing       ||h(t+1) - h(t)||^2, the do-nothing error; every MSE below is reported relative to it
  rho           1 - nothing / (2 var): correlation between the two embeddings of a pair
  (1+rho)/2     the error of pure regression to the mean (ignore the action, shrink h(t) toward the mean by rho)
  shrink        the best one-parameter shrinkage beta (h(t) - mean) + mean, fitted on the same pairs
  dyn0 / dyn    Dyn with the zero action / the true action (dyn is what eval/mse_ratio logs)
  action effect e = Dyn(h, a) - Dyn(h, 0) against the residual r = h(t+1) - Dyn(h, 0), on active pairs:
    R2_action   1 - sum||r - e||^2 / sum||r||^2: share of the residual the action explains (< 0: it hurts)
    cos         mean cosine(e, r); |e|/|r| its relative size
    moves       share of active pairs where the true action moves the prediction further from h(t) than the zero
                action (1 - eval/decoy_zero_nochange)

Interpretation
--------------
  dyn ~ shrink and R2_action ~ 0 on val    the model's only generalizing content is regression to the mean
  R2_action train >> val                   the action effects are memorized training transitions

Reference values (2026-10-13; ViT 128x4, see memory project_vit_backbone): multistep run 20261006/044610 on its own
pairs, val: shrink 0.856, dyn 0.919, R2_action -0.04 (train +0.19).

Example
-------
python anatomie/dyn_decompose.py --ckpt vit_ms=runs/runs/20261006/044610/064d3fbda75f/checkpoints/last.ckpt \\
    --pairs k1
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import REPO, MIMICLanceDataset, data_paths, load_dynamics, load_encoder  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH", help="forward checkpoint (repeatable)")
    p.add_argument("--pairs", choices=["own", "k1", "multistep"], default="own",
                   help="pairs to score on: the checkpoint's training pairs, k=1 pairs.lance, or multistep k<=12")
    p.add_argument("--n", type=int, default=8192, help="random pairs per split")
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--workers", type=int, default=12)
    return p.parse_args()


def pairs_path(choice, ckpt_path, paths):
    if choice == "own":
        hp = torch.load(REPO / ckpt_path, map_location="cpu", weights_only=False).get("hyper_parameters") or {}
        choice = "k1" if Path(str(hp.get("pairs_path", ""))).name == "pairs.lance" else "multistep"
    return choice, paths["pairs_path"] if choice == "k1" else paths["pairs_multistep_path"]


@torch.no_grad()
def decompose(enc, dyn, lance_path, pairs, split, n, workers):
    ds = MIMICLanceDataset(lance_path, split=split, mode="pair", pairs_path=pairs, pair_types=("cross_stay",))
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:n].tolist()
    H, H1, A = [], [], []
    for b in DataLoader(Subset(ds, idx), batch_size=1024, num_workers=workers, multiprocessing_context="spawn"):
        H.append(enc(b["xt"].cuda())["h"]); H1.append(enc(b["xt1"].cuda())["h"]); A.append(b["at"].cuda().float())
    h, h1, a = torch.cat(H), torch.cat(H1), torch.cat(A)
    p, p0 = dyn(h, a), dyn(h, torch.zeros_like(a))
    mse = lambda x: (x - h1).pow(2).mean().item()
    nothing, var = mse(h), h.var(0).mean().item()
    mu = h.mean(0)
    beta = (((h - mu) * (h1 - mu)).sum() / (h - mu).pow(2).sum()).item()
    act = (a != 0).any(1)
    e, r = (p - p0)[act], (h1 - p0)[act]
    return {"rho": 1 - nothing / (2 * var), "beta": beta, "shrink": mse(beta * (h - mu) + mu) / nothing,
            "dyn0": mse(p0) / nothing, "dyn": mse(p) / nothing,
            "r2": 1 - (r - e).pow(2).sum().item() / r.pow(2).sum().item(),
            "cos": F.cosine_similarity(e, r, dim=1).mean().item(), "rel": (e.norm(dim=1) / r.norm(dim=1)).mean().item(),
            "moves": ((p - h).pow(2).sum(1) > (p0 - h).pow(2).sum(1))[act].float().mean().item()}


def main():
    args = parse_args()
    paths = data_paths(args.config)
    print(f"{'checkpoint':<14}{'pairs':<10}{'split':<6}{'rho':>6}{'(1+rho)/2':>10}{'shrink':>8}{'dyn0':>7}{'dyn':>7}"
          f"{'R2_action':>10}{'cos':>7}{'|e|/|r|':>8}{'moves':>7}")
    for name, spec in (c.split("=", 1) for c in args.ckpt):
        enc, dyn = load_encoder(spec), load_dynamics(spec)
        choice, pairs = pairs_path(args.pairs, spec, paths)
        for split in ("train", "val"):
            m = decompose(enc, dyn, paths["lance_path"], pairs, split, args.n, args.workers)
            print(f"{name:<14}{choice:<10}{split:<6}{m['rho']:>6.3f}{(1 + m['rho']) / 2:>10.3f}{m['shrink']:>8.3f}"
                  f"{m['dyn0']:>7.3f}{m['dyn']:>7.3f}{m['r2']:>+10.4f}{m['cos']:>+7.3f}{m['rel']:>8.3f}{m['moves']:>7.3f}",
                  flush=True)


if __name__ == "__main__":   # spawned DataLoader workers re-import this file
    main()

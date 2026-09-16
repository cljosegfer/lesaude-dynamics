"""
Dynamics Model Headroom Check
==============================
Tests whether Dyn_phi's raw MSE prediction loss (against TRUE actions) is limited by capacity
(underfitting) or by a noise floor already reached during training, before investing in a
DynamicsPredictor architecture upgrade + retrain.

Compares Dyn_phi's mean squared prediction error on a TRAIN-split sample of longitudinal pairs against
the same metric on a VAL-split sample, plus the "zero-action" baseline (predicting no change) for
context. Mirrors the diagnostics built in demo/probe_action_subspace.ipynb Section 7, but restricted to
this one train-vs-val comparison and runnable standalone.

Interpretation
--------------
  train loss  ~= val loss -> the model can't beat the noise floor even on data it's allowed to memorize.
                              That's underfitting -- there's real headroom, and more DynamicsPredictor
                              capacity is a reasonable bet.
  train loss << val loss  -> a real generalization gap; the model already fits its own training data well
                              below the noise floor seen at val time. More capacity risks overfitting
                              harder, not improving discriminability -- the bottleneck looks aleatoric or
                              objective-related instead.

Example
-------
python demo/dynamics_headroom_check.py --batch-size 1024
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
import yaml

from dataset.dataset import MIMICLanceDataset
from models.resnet1d import ResNet1d
from models.dynamics import ActionProjector, DynamicsPredictor


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "src").is_dir() and (p / "configs").is_dir():
            return p
    raise RuntimeError("Could not locate repo root (looked for src/ and configs/)")


def load_dynamics_model(ckpt_path: Path, embedding_dim: int, action_dim: int, hidden_dim: int,
                         device: str):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

    backbone = ResNet1d(in_channels=12, embedding_dim=embedding_dim)
    backbone_state = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Backbone mismatch — missing: {missing}, unexpected: {unexpected}")

    proj = ActionProjector(action_dim=action_dim, embed_dim=embedding_dim)
    proj_state = {k.removeprefix("projector.proj."): v for k, v in state.items() if k.startswith("projector.proj.")}
    missing, unexpected = proj.load_state_dict(proj_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"ActionProjector mismatch — missing: {missing}, unexpected: {unexpected}")

    pred = DynamicsPredictor(embed_dim=embedding_dim, hidden_dim=hidden_dim)
    pred_state = {k.removeprefix("projector.pred."): v for k, v in state.items() if k.startswith("projector.pred.")}
    missing, unexpected = pred.load_state_dict(pred_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"DynamicsPredictor mismatch — missing: {missing}, unexpected: {unexpected}")

    for module in (backbone, proj, pred):
        module.eval().to(device)
        for p in module.parameters():
            p.requires_grad_(False)
    return backbone, proj, pred


@torch.no_grad()
def embed(backbone: torch.nn.Module, x: torch.Tensor, device: str, chunk: int = 256) -> torch.Tensor:
    outs = []
    for i in range(0, len(x), chunk):
        outs.append(backbone(x[i:i + chunk].to(device)).cpu())
    return torch.cat(outs)


@torch.no_grad()
def dyn_predict(proj: torch.nn.Module, pred: torch.nn.Module, ht: torch.Tensor, actions: torch.Tensor,
                 device: str) -> torch.Tensor:
    a_emb = proj(actions.float().to(device))
    return pred(ht.to(device), a_emb).cpu()


def sample_pairs(lance_path: str, pairs_path: str, split: str, batch_size: int, seed: int):
    ds = MIMICLanceDataset(lance_path, split=split, mode="pair", pairs_path=pairs_path)
    rng = np.random.default_rng(seed)
    n = min(batch_size, len(ds))
    idx = rng.choice(len(ds), size=n, replace=False).tolist()
    batch = ds.__getitems__(idx)
    xt = torch.stack([b["xt"] for b in batch])
    xt1 = torch.stack([b["xt1"] for b in batch])
    at = torch.stack([b["at"] for b in batch])
    return xt, xt1, at


def evaluate_split(name: str, xt: torch.Tensor, xt1: torch.Tensor, at: torch.Tensor,
                    backbone: torch.nn.Module, proj: torch.nn.Module, pred: torch.nn.Module,
                    device: str) -> pd.DataFrame:
    ht = embed(backbone, xt, device)
    ht1 = embed(backbone, xt1, device)
    zero_actions = torch.zeros_like(at, dtype=torch.float32)

    h_hat_true = dyn_predict(proj, pred, ht, at.float(), device)
    h_hat_zero = dyn_predict(proj, pred, ht, zero_actions, device)

    mse_true = (h_hat_true - ht1).pow(2).mean(dim=1).numpy()
    mse_zero = (h_hat_zero - ht1).pow(2).mean(dim=1).numpy()
    gt_active = at.abs().sum(dim=1).numpy() > 0

    return pd.DataFrame({"split": name, "mse_true": mse_true, "mse_zero": mse_zero, "gt_active": gt_active})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default="archive/pretrain_val.ckpt",
                        help="Dynamics checkpoint, relative to repo root unless absolute.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--predictor-hidden-dim", type=int, default=512)
    parser.add_argument("--action-dim", type=int, default=76)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(repo_root / "configs" / "data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    lance_path = data_cfg["lance_path"]
    pairs_path = data_cfg["pairs_path"]

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = repo_root / ckpt_path

    print(f"device:     {device}")
    print(f"checkpoint: {ckpt_path}")
    print(f"lance_path: {lance_path}")
    print(f"pairs_path: {pairs_path}\n")

    backbone, proj, pred = load_dynamics_model(
        ckpt_path, args.embedding_dim, args.action_dim, args.predictor_hidden_dim, device)
    print("Loaded backbone + ActionProjector + DynamicsPredictor.\n")

    results = []
    for split in ("train", "val"):
        print(f"Sampling {args.batch_size} pairs from {split}...")
        xt, xt1, at = sample_pairs(lance_path, pairs_path, split, args.batch_size, args.seed)
        results.append(evaluate_split(split, xt, xt1, at, backbone, proj, pred, device))
    results_df = pd.concat(results, ignore_index=True)

    print()
    print(results_df.groupby("split")[["mse_true", "mse_zero"]].agg(["mean", "std", "count"]).to_string())

    active_df = results_df[results_df["gt_active"]]
    print("\nGT-active pairs only:")
    print(active_df.groupby("split")[["mse_true", "mse_zero"]].agg(["mean", "std", "count"]).to_string())

    train_mse = results_df.loc[results_df["split"] == "train", "mse_true"].mean()
    val_mse = results_df.loc[results_df["split"] == "val", "mse_true"].mean()
    val_zero_mse = results_df.loc[results_df["split"] == "val", "mse_zero"].mean()
    gap = val_mse - train_mse
    gap_frac = gap / val_mse if val_mse else float("nan")

    print(f"\n{'=' * 60}")
    print("  Headroom check")
    print(f"{'=' * 60}")
    print(f"  train pred_loss (true action):  {train_mse:.4f}")
    print(f"  val   pred_loss (true action):  {val_mse:.4f}")
    print(f"  val   zero-action baseline:     {val_zero_mse:.4f}")
    print(f"  train-val gap:                  {gap:.4f}  ({gap_frac:.1%} of val loss)")
    print()
    if gap_frac < 0.10:
        print("  -> train ~= val: the model can't beat the noise floor even on data it's allowed to")
        print("     memorize. Looks like underfitting -- more DynamicsPredictor capacity is a")
        print("     reasonable bet.")
    elif gap_frac > 0.30:
        print("  -> train << val: a real generalization gap. The model already fits its own training")
        print("     data well below the val-time noise floor -- more capacity risks overfitting harder,")
        print("     not improving discriminability. The bottleneck looks aleatoric/objective-related,")
        print("     not capacity-related.")
    else:
        print("  -> ambiguous gap size -- rerun with a larger --batch-size or inspect the full")
        print("     distribution (results_df) before concluding either way.")


if __name__ == "__main__":
    main()

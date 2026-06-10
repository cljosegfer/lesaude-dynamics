"""
Inverse Dynamics Test-Set Evaluation
======================================
Loads an inverse-dynamics checkpoint and reports bootstrap AUROC on
the held-out test fold for three tasks:

  1. Onset         — sigmoid(onset_logits) vs (at_i == +1)
  2. Resolution    — sigmoid(res_logits)   vs (at_i == -1)
  3. Contextualized — P(yt+1_i=1|yt,d) vs true yt+1,
                      where P(yt+1_i=1) = yt_i*(1-res_prob_i) + (1-yt_i)*onset_prob_i

Example
-------
HYDRA_FULL_ERROR=1 python scripts/evaluate_inverse.py \\
    ckpt_path=checkpoints/inverse_.ckpt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from hydra.utils import get_original_cwd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import MIMICLanceDataset
from models.resnet1d import ResNet1d
from models.dynamics import ActionPredictor


def _load_model(ckpt_path, embedding_dim, predictor_hidden_dim, action_dim, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    backbone = ResNet1d(in_channels=12, embedding_dim=embedding_dim)
    backbone.load_state_dict(
        {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    )

    predictor = ActionPredictor(
        embed_dim=embedding_dim,
        hidden_dim=predictor_hidden_dim,
        action_dim=action_dim,
    )
    predictor.load_state_dict(
        {k[len("projector.pred."):]: v for k, v in sd.items() if k.startswith("projector.pred.")}
    )

    backbone.to(device).eval()
    predictor.to(device).eval()
    return backbone, predictor


@torch.no_grad()
def _run_inference(backbone, predictor, loader, device):
    onset_preds_list, res_preds_list, yt_list, at_list = [], [], [], []
    for batch in tqdm(loader, desc="Inference"):
        xt  = batch["xt"].to(device)
        xt1 = batch["xt1"].to(device)

        ht  = backbone(xt)
        ht1 = backbone(xt1)
        d   = ht1 - ht

        onset_logits, res_logits = predictor(d)
        onset_preds_list.append(torch.sigmoid(onset_logits).cpu().float().numpy())
        res_preds_list.append(torch.sigmoid(res_logits).cpu().float().numpy())
        yt_list.append(batch["yt"].float().numpy())
        at_list.append(batch["at"].float().numpy())

    return (
        np.concatenate(onset_preds_list),   # (N, 76)
        np.concatenate(res_preds_list),     # (N, 76)
        np.concatenate(yt_list),            # (N, 76)
        np.concatenate(at_list),            # (N, 76)
    )


def _bootstrap_auroc(preds, targets, n_bootstrap, seed, desc):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n, n_classes = targets.shape
    scores = []
    for _ in tqdm(range(n_bootstrap), desc=desc):
        idx = rng.integers(0, n, size=n)
        t, p = targets[idx], preds[idx]
        per_class = [
            roc_auc_score(t[:, c], p[:, c])
            for c in range(n_classes)
            if 0 < t[:, c].sum() < len(idx)
        ]
        if per_class:
            scores.append(np.mean(per_class))
    scores = np.array(scores)
    return scores.mean(), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


@hydra.main(version_base="1.3", config_path="../configs", config_name="inverse_pretrain")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(get_original_cwd()) / cfg.ckpt_path
    print(f"Checkpoint: {ckpt_path}")
    backbone, predictor = _load_model(
        ckpt_path, cfg.embedding_dim, cfg.predictor_hidden_dim, cfg.action_dim, device
    )

    ds = MIMICLanceDataset(
        cfg.lance_path,
        split="test",
        mode="pair",
        pairs_path=cfg.pairs_path,
        pair_types=tuple(cfg.pair_types),
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
        pin_memory=True,
    )
    print(f"Test pairs: {len(ds)}")

    onset_preds, res_preds, yt, at = _run_inference(backbone, predictor, loader, device)

    yt1 = np.clip(yt + at, 0.0, 1.0)                                        # (N, 76) true next labels

    onset_targets = (at == 1.0).astype(np.float32)                          # (N, 76)
    res_targets   = (at == -1.0).astype(np.float32)                         # (N, 76)
    ctx_preds     = yt * (1.0 - res_preds) + (1.0 - yt) * onset_preds      # (N, 76)

    n_bootstrap = cfg.evaluate.n_bootstrap
    seed        = cfg.evaluate.seed

    print()
    mean, lo, hi = _bootstrap_auroc(onset_preds, onset_targets, n_bootstrap, seed, "Onset bootstrap")
    print(f"Onset          AUROC: {mean:.4f}  (95% CI: {lo:.4f}–{hi:.4f})")

    mean, lo, hi = _bootstrap_auroc(res_preds, res_targets, n_bootstrap, seed, "Resolution bootstrap")
    print(f"Resolution     AUROC: {mean:.4f}  (95% CI: {lo:.4f}–{hi:.4f})")

    mean, lo, hi = _bootstrap_auroc(ctx_preds, yt1, n_bootstrap, seed, "Contextualized bootstrap")
    print(f"Contextualized AUROC: {mean:.4f}  (95% CI: {lo:.4f}–{hi:.4f})")


if __name__ == "__main__":
    main()

"""
CEM Planning for Action-Conditioned Cardiac World Model (LeJEPA)
================================================================
Instead of using the finetuned classifier, this script performs inference
via the Cross-Entropy Method (CEM) on the pre-trained dynamics model.

For each test pair (X_{t-1}, y_{t-1}) → X_t:
  1. Encode both:  h_{t-1} = Enc(X_{t-1}),  h_t = Enc(X_t)
  2. Run CEM to find a* = argmin_a  ||Dyn(Proj(h_{t-1}), ActionMLP(a)) - Proj(h_t)||^2
  3. Infer the label:  y_hat_t = clip(a* + y_{t-1}, 0, 1)

Usage:
    python scripts/cem_planning.py --checkpoint_path checkpoints/lejepa_best.pth
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import DynamicsDataset
from src.xresnet1d import xresnet1d50
from hparams import DATA_ROOT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 76


# ==============================================================================
# Model — mirrors dynamics_lejepa.py with action_mlp restored
# ==============================================================================

class MLP(nn.Sequential):
    def __init__(self, in_features, hidden_features, norm_layer=nn.BatchNorm1d):
        layers = []
        for hidden in hidden_features[:-1]:
            layers.extend([nn.Linear(in_features, hidden), norm_layer(hidden), nn.GELU()])
            in_features = hidden
        layers.append(nn.Linear(in_features, hidden_features[-1]))
        layers.append(norm_layer(hidden_features[-1]))
        super().__init__(*layers)


class LeJEPA_Model(nn.Module):
    def __init__(self, num_input_channels=12, num_action_classes=76, proj_dim=256):
        super().__init__()

        # A. Backbone
        self.backbone = xresnet1d50(input_channels=num_input_channels, num_classes=None)
        self.pool = nn.AdaptiveAvgPool1d(1)

        with torch.no_grad():
            dummy = torch.randn(2, num_input_channels, 100)
            out = self.backbone(dummy)
            self.enc_dim = out.shape[1]

        # B. Projector
        self.projector = MLP(self.enc_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

        # C. Action Encoder
        self.action_mlp = nn.Sequential(
            nn.Linear(num_action_classes, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, proj_dim),
        )

        # D. Predictor  (z_t || a_emb -> z_hat_{t+1})
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, proj_dim),
        )

    def encode(self, x):
        """Returns (h, z): backbone embedding and projected embedding."""
        h = self.pool(self.backbone(x)).flatten(1)
        z = self.projector(h)
        return h, z

    def predict(self, z_t, action):
        """Returns z_hat_{t+1} given z_t and a discrete action vector."""
        a_emb = self.action_mlp(action)
        return self.predictor(torch.cat([z_t, a_emb], dim=1))

    def forward(self, x_t, action):
        h_t, z_t = self.encode(x_t)
        z_hat_next = self.predict(z_t, action)
        return h_t, z_t, z_hat_next

    def encode_target(self, x_next):
        _, z_next = self.encode(x_next)
        return z_next


# ==============================================================================
# CEM Planner
# ==============================================================================

class CEMPlanner:
    """
    Cross-Entropy Method over the discrete action space {-1, 0, 1}^C.

    Each dimension c maintains a categorical distribution (p_{-1}, p_0, p_{+1}).
    We initialise with a prior biased toward stability (action=0) since ~73% of
    consecutive MIMIC-IV-ECG pairs are stable.
    """

    def __init__(
        self,
        num_classes: int = 76,
        n_samples: int = 128,
        n_elites: int = 24,
        n_iters: int = 8,
        smoothing: float = 0.1,
        prior_stable: float = 0.7,
    ):
        self.C = num_classes
        self.N = n_samples
        self.K = n_elites
        self.T = n_iters
        self.alpha = smoothing

        p_change = (1.0 - prior_stable) / 2.0
        # Shape (C, 3): columns correspond to actions {-1, 0, +1}
        self.init_probs = torch.tensor(
            [p_change, prior_stable, p_change], dtype=torch.float32
        ).unsqueeze(0).expand(num_classes, -1).clone()

    @torch.no_grad()
    def plan(self, model: LeJEPA_Model, z_prev: torch.Tensor, z_target: torch.Tensor):
        """
        Find the action a* minimising ||Dyn(z_prev, a*) - z_target||^2.

        Args:
            z_prev:   Projected previous state,  shape (1, proj_dim)  on DEVICE.
            z_target: Projected current state,   shape (1, proj_dim)  on DEVICE.

        Returns:
            best_action: float tensor (C,) with values in {-1., 0., +1.}  on CPU.
            best_score:  MSE of the winning candidate (scalar).
        """
        probs = self.init_probs.clone()          # (C, 3)  on CPU
        z_prev_batch = z_prev.expand(self.N, -1) # (N, proj_dim)

        best_action = None
        best_score  = float("inf")

        iter_best_scores = []  # best MSE found so far at each iteration
        iter_entropy     = []  # mean entropy of distribution at each iteration
        iter_confidence  = []  # fraction of dims where max_prob > 0.9

        for _ in range(self.T):
            # Sample N action vectors
            indices = torch.multinomial(probs, num_samples=self.N, replacement=True)  # (C, N)
            actions_int = indices - 1      # {0,1,2} -> {-1, 0, +1}
            actions = actions_int.T.float().to(DEVICE)   # (N, C)

            # Score via dynamics model
            z_hat = model.predict(z_prev_batch, actions)        # (N, proj_dim)
            scores = (z_hat - z_target).pow(2).mean(dim=1)     # (N,)

            # Elite selection
            elite_idx   = scores.argsort()[:self.K].cpu()
            elite_acts  = actions_int[:, elite_idx]             # (C, K)  on CPU

            # Track global best
            best_i = scores.argmin().item()
            if scores[best_i].item() < best_score:
                best_score  = scores[best_i].item()
                best_action = actions[best_i].cpu()             # (C,)

            iter_best_scores.append(best_score)

            # Update distribution from elites
            new_probs = torch.zeros_like(probs)
            for val, col in zip([-1, 0, 1], [0, 1, 2]):
                new_probs[:, col] = (elite_acts == val).float().mean(dim=1)

            probs = self.alpha * probs + (1.0 - self.alpha) * new_probs
            probs = probs / probs.sum(dim=1, keepdim=True)      # renormalise

            # Diagnostics on updated distribution
            entropy    = -(probs * (probs + 1e-8).log()).sum(dim=1).mean().item()
            confidence = (probs.max(dim=1).values > 0.9).float().mean().item()
            iter_entropy.append(entropy)
            iter_confidence.append(confidence)

        diagnostics = {
            'best_scores':    iter_best_scores,
            'entropy':        iter_entropy,
            'confidence':     iter_confidence,
            'final_p_zero':   probs[:, 1].mean().item(),
            'final_mean_min': probs.min(dim=1).values.mean().item(),  # avg peak prob per dim
            'final_n_active': (probs.argmax(dim=1) != 1).sum().item(), # dims where modal action != 0
            'final_p_nonzero': (1.0 - probs[:, 1]).mean().item(),     # mean p(any change) per dim
        }
        return best_action, best_score, diagnostics


# ==============================================================================
# Inference Loop
# ==============================================================================

@torch.no_grad()
def run_cem_inference(model, val_loader, planner, args):
    model.eval()

    all_soft    = []   # soft predicted label probabilities
    all_hard    = []   # hard {0,1} predictions
    all_targets = []   # ground truth y_t

    # Diagnostic accumulators (CEM mode only)
    diag_best_scores = np.zeros(planner.T)  # mean best-MSE per iteration
    diag_entropy     = np.zeros(planner.T)  # mean distribution entropy per iteration
    diag_confidence  = np.zeros(planner.T)  # mean high-confidence dim fraction per iteration
    diag_p_zero      = []   # final mean p(action=0) per sample
    diag_mean_min    = []   # final mean of per-dim min probability
    diag_n_active    = []   # final count of dims where modal action != 0
    diag_p_nonzero   = []   # final mean p(any change) per dim
    diag_final_mse   = []   # final best MSE per sample
    diag_zero_mse    = []   # MSE of Dyn(z_prev, 0) — zero-action baseline
    diag_gt_stable   = []   # True if GT action is all-zero
    all_pred_actions = []   # per-sample inferred best_action (C,)
    all_gt_actions   = []   # per-sample ground-truth action   (C,)
    n_cem_samples    = 0

    for batch in tqdm(val_loader, desc="CEM Planning"):
        x_prev    = batch['waveform'].to(DEVICE).float()        # (B, 12, T)
        x_curr    = batch['waveform_next'].to(DEVICE).float()   # (B, 12, T)
        y_prev    = batch['icd'].to(DEVICE).float()             # (B, C) — y_{t-1}
        action_gt = batch['action'].to(DEVICE).float()          # (B, C) — y_t - y_{t-1}
        y_curr    = (y_prev + action_gt).clamp(0, 1)            # (B, C) — ground truth y_t

        # Encode: only projected representations are needed for CEM scoring
        _, z_prev = model.encode(x_prev)   # (B, proj_dim)
        _, z_curr = model.encode(x_curr)   # (B, proj_dim)

        batch_soft = []
        batch_hard = []

        for i in range(x_prev.shape[0]):
            y_prev_i = y_prev[i].cpu()

            if args.carryforward:
                soft       = y_prev_i
                y_hat_hard = y_prev_i.clamp(0, 1)
            else:
                best_action, best_score, diag = planner.plan(
                    model,
                    z_prev[i].unsqueeze(0),
                    z_curr[i].unsqueeze(0),
                )
                y_hat_hard = (best_action + y_prev_i).clamp(0, 1)
                soft = torch.where(best_action == 0, y_prev_i, ((best_action + 1.0) / 2.0))

                diag_best_scores += np.array(diag['best_scores'])
                diag_entropy     += np.array(diag['entropy'])
                diag_confidence  += np.array(diag['confidence'])
                diag_p_zero.append(diag['final_p_zero'])
                diag_mean_min.append(diag['final_mean_min'])
                diag_n_active.append(diag['final_n_active'])
                diag_p_nonzero.append(diag['final_p_nonzero'])
                diag_final_mse.append(best_score)
                zero_act = torch.zeros(1, NUM_CLASSES, device=DEVICE)
                z_hat_zero = model.predict(z_prev[i].unsqueeze(0), zero_act)
                diag_zero_mse.append((z_hat_zero - z_curr[i].unsqueeze(0)).pow(2).mean().item())
                diag_gt_stable.append(action_gt[i].abs().sum().item() == 0)
                all_pred_actions.append(best_action.numpy())
                all_gt_actions.append(action_gt[i].cpu().numpy())
                n_cem_samples    += 1

            batch_soft.append(soft.numpy())
            batch_hard.append(y_hat_hard.numpy())

        all_soft.append(np.array(batch_soft))
        all_hard.append(np.array(batch_hard))
        all_targets.append(y_curr.cpu().numpy())

    all_soft    = np.concatenate(all_soft,    axis=0)
    all_hard    = np.concatenate(all_hard,    axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    valid_cls = [i for i in range(NUM_CLASSES) if len(np.unique(all_targets[:, i])) > 1]
    print(f"\nValid classes for AUROC: {len(valid_cls)} / {NUM_CLASSES}")

    auroc_soft = roc_auc_score(all_targets[:, valid_cls], all_soft[:, valid_cls], average='macro')
    auroc_hard = roc_auc_score(all_targets[:, valid_cls], all_hard[:, valid_cls], average='macro')

    print(f"\n{'='*52}")
    print(f"  CEM Planning Results")
    print(f"  Macro-AUROC (soft, action->prob): {auroc_soft:.4f}")
    print(f"  Macro-AUROC (hard, clipped):      {auroc_hard:.4f}")
    print(f"{'='*52}\n")

    if not args.carryforward and n_cem_samples > 0:
        diag_best_scores /= n_cem_samples
        diag_entropy     /= n_cem_samples
        diag_confidence  /= n_cem_samples

        print(f"  CEM Convergence Diagnostics (averaged over {n_cem_samples} samples)")
        print(f"  {'Iter':>4}  {'Best MSE':>10}  {'Entropy':>10}  {'Conf>0.9':>10}")
        for t in range(planner.T):
            print(f"  {t+1:>4}  {diag_best_scores[t]:>10.4f}  {diag_entropy[t]:>10.4f}  {diag_confidence[t]:>10.4f}")

        print(f"\n  Final MSE      — mean: {np.mean(diag_final_mse):.4f}  std: {np.std(diag_final_mse):.4f}")
        print(f"  Final p(a=0)   — mean: {np.mean(diag_p_zero):.4f}  std: {np.std(diag_p_zero):.4f}"
              f"  (prior: {planner.init_probs[0, 1].item():.2f})")
        print(f"  Mean-min p     — mean: {np.mean(diag_mean_min):.4f}  std: {np.std(diag_mean_min):.4f}")
        print(f"  Active dims    — mean: {np.mean(diag_n_active):.2f}  std: {np.std(diag_n_active):.2f}"
              f"  (modal action≠0, out of {NUM_CLASSES})")
        print(f"  Mean p(≠0)     — mean: {np.mean(diag_p_nonzero):.4f}  std: {np.std(diag_p_nonzero):.4f}"
              f"  (prior: {1 - planner.init_probs[0, 1].item():.2f})")
        print()

        pred_act    = np.stack(all_pred_actions, axis=0)          # (N, C)
        gt_act      = np.stack(all_gt_actions,   axis=0)          # (N, C)
        pred_stable = (np.abs(pred_act).sum(axis=1) == 0)         # (N,)
        gt_stable   = (np.abs(gt_act).sum(axis=1)   == 0)         # (N,)
        N           = len(pred_stable)

        tn = int(( pred_stable &  gt_stable).sum())
        fn = int(( pred_stable & ~gt_stable).sum())
        fp = int((~pred_stable &  gt_stable).sum())
        tp = int((~pred_stable & ~gt_stable).sum())

        print(f"  Action Collapse Diagnostics")
        print(f"  Predicted stable: {pred_stable.sum()}/{N}  ({pred_stable.mean():.3f})"
              f"  |  GT stable: {gt_stable.sum()}/{N}  ({gt_stable.mean():.3f})")
        print(f"  {'':30s}  GT stable  GT active")
        print(f"  Pred stable (all-zero)           {tn:>8d}  {fn:>8d}  <- FN: missed changes")
        print(f"  Pred active (non-zero)           {fp:>8d}  {tp:>8d}")
        if pred_stable.sum() > 0:
            print(f"  Precision [pred stable]: {tn / pred_stable.sum():.4f}")
        if (~pred_stable).sum() > 0:
            print(f"  Precision [pred active]: {tp / (~pred_stable).sum():.4f}")
        elem_stable = (pred_act[pred_stable] == gt_act[pred_stable]).mean() if pred_stable.any() else float('nan')
        elem_active = (pred_act[~pred_stable] == gt_act[~pred_stable]).mean() if (~pred_stable).any() else float('nan')
        print(f"  Element-wise action match [pred stable]: {elem_stable:.4f}")
        print(f"  Element-wise action match [pred active]: {elem_active:.4f}")
        print()

        zero_mse   = np.array(diag_zero_mse)
        best_mse   = np.array(diag_final_mse)
        gt_stable  = np.array(diag_gt_stable)
        gt_active  = ~gt_stable
        mse_gap    = zero_mse - best_mse           # >0 means CEM beat zero action

        print(f"  MSE Landscape  (zero-action MSE vs CEM best-action MSE)")
        print(f"  {'Group':20s}  {'N':>6}  {'zero-act MSE':>14}  {'best-act MSE':>14}  {'gap (z-b)':>10}")
        for mask, label in [(gt_stable, 'GT stable'), (gt_active, 'GT active')]:
            if mask.any():
                print(f"  {label:20s}  {mask.sum():>6d}"
                      f"  {zero_mse[mask].mean():>10.4f} ±{zero_mse[mask].std():>6.4f}"
                      f"  {best_mse[mask].mean():>10.4f} ±{best_mse[mask].std():>6.4f}"
                      f"  {mse_gap[mask].mean():>10.4f}")
        print()

        # carry-forward baseline: y_prev = y_curr - gt_action
        cf_preds = (all_targets - gt_act).clip(0, 1)

        print(f"  AUROC by GT Group  (CEM soft vs carry-forward y_prev baseline)")
        print(f"  {'Group':20s}  {'N':>6}  {'valid cls':>9}  {'CEM AUROC':>10}  {'CaryFwd AUROC':>14}")
        for mask, label in [(gt_stable, 'GT stable'), (gt_active, 'GT active')]:
            if not mask.any():
                continue
            sub_tgt = all_targets[mask]
            sub_sft = all_soft[mask]
            sub_cf  = cf_preds[mask]
            valid   = [i for i in range(NUM_CLASSES) if len(np.unique(sub_tgt[:, i])) > 1]
            if not valid:
                print(f"  {label:20s}  {mask.sum():>6d}  — no valid classes")
                continue
            auroc_cem = roc_auc_score(sub_tgt[:, valid], sub_sft[:, valid], average='macro')
            auroc_cf  = roc_auc_score(sub_tgt[:, valid], sub_cf[:, valid],  average='macro')
            print(f"  {label:20s}  {mask.sum():>6d}  {len(valid):>9d}  {auroc_cem:>10.4f}  {auroc_cf:>14.4f}")
        print()

    if args.verbose:
        per_class = roc_auc_score(
            all_targets[:, valid_cls], all_soft[:, valid_cls], average=None
        )
        for rank_i, cls_idx in enumerate(valid_cls):
            print(f"  Class {cls_idx:3d}: {per_class[rank_i]:.3f}")

    return auroc_soft, auroc_hard


# ==============================================================================
# Entry Point
# ==============================================================================

def main(args):
    print(f"Loading LeJEPA checkpoint: {args.checkpoint_path}")
    model = LeJEPA_Model(
        num_input_channels=12,
        num_action_classes=NUM_CLASSES,
        proj_dim=args.proj_dim,
    )
    ckpt = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print(f"  enc_dim={model.enc_dim}, proj_dim={args.proj_dim}")

    print(f"Building dataset (pair_mode={args.pair_mode})...")
    ds = DynamicsDataset(
        split='test',
        # split='train', # what if
        return_pairs=True,
        in_memory=args.in_memory,
        pair_mode=args.pair_mode,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"  pairs: {len(ds)}")

    planner = CEMPlanner(
        num_classes=NUM_CLASSES,
        n_samples=args.cem_samples,
        n_elites=args.cem_elites,
        n_iters=args.cem_iters,
        smoothing=args.cem_smoothing,
        prior_stable=args.cem_prior_stable,
    )

    run_cem_inference(model, loader, planner, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--proj_dim', type=int, default=256)

    # Dataset
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--in_memory', action='store_true')

    # CEM
    parser.add_argument('--cem_samples',      type=int,   default=128)
    parser.add_argument('--cem_elites',        type=int,   default=24)
    parser.add_argument('--cem_iters',         type=int,   default=16)
    parser.add_argument('--cem_smoothing',     type=float, default=0.1)
    parser.add_argument('--cem_prior_stable',  type=float, default=0.7)

    parser.add_argument('--pair_mode', type=str, default='monitoring',
                        choices=['monitoring', 'triage'],
                        help="'monitoring': consecutive ECG pairs (default); "
                             "'triage': first ECG of consecutive hospital stays.")
    parser.add_argument('--carryforward', action='store_true',
                        help="Skip CEM and predict y_hat = y_{t-1} as a trivial baseline.")
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    main(args)
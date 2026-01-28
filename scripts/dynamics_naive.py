import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import DynamicsDataset
from src.xresnet1d import xresnet1d50
from src.loss import SIGReg
from hparams import DATA_ROOT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 2. Model Architecture (LeJEPA Style)
# ==============================================================================
class MLP(nn.Sequential):
    """Helper to build the LeJEPA projector"""
    def __init__(self, in_features, hidden_features, norm_layer=nn.BatchNorm1d):
        layers = []
        for hidden in hidden_features[:-1]:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(norm_layer(hidden))
            layers.append(nn.GELU()) # LeJEPA uses GELU implicitly in timm or relu? Minimal uses MLP defaults.
            in_features = hidden
        layers.append(nn.Linear(in_features, hidden_features[-1]))
        # LeJEPA Minimal puts BatchNorm on the output of projector?
        # "self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)"
        # Usually projection heads end with BN in SSL (like VICReg).
        layers.append(norm_layer(hidden_features[-1])) 
        super().__init__(*layers)

class LeJEPA_Model(nn.Module):
    def __init__(self, num_input_channels=12, num_action_classes=76, proj_dim=128):
        super().__init__()
        
        # A. Backbone
        self.backbone = xresnet1d50(input_channels=num_input_channels, num_classes=None)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Detect backbone dim
        with torch.no_grad():
            dummy = torch.randn(2, 12, 100)
            out = self.backbone(dummy)
            self.enc_dim = out.shape[1] # Store this! (Likely 256)
            print(f"   > Detected Backbone Output Dim: {self.enc_dim}")
            
        # B. Projector (LeJEPA Spec: 3 layers, high dim expansion)
        # Minimal.md: MLP(512, [2048, 2048, proj_dim])
        self.projector = MLP(self.enc_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
        
        # # C. Action Encoder (Specific to Dynamics)
        # self.action_mlp = nn.Sequential(
        #     nn.Linear(num_action_classes, 512),
        #     nn.BatchNorm1d(512),
        #     nn.GELU(),
        #     nn.Linear(512, proj_dim) # Map action to Projector Space
        # )
        
        # D. Predictor (In Projected Space)
        # z_t (proj_dim) + action (proj_dim) -> z_t+1 (proj_dim)
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, proj_dim)
        )

    def forward(self, x_t, action):
        # 1. Encode
        h_t_feat = self.backbone(x_t)
        h_t = self.pool(h_t_feat).flatten(1)
        
        # 2. Project
        z_t = self.projector(h_t)
        
        # 3. Predict Dynamics
        # a_emb = self.action_mlp(action)
        z_hat_next = self.predictor(torch.cat([z_t, ], dim=1))
        
        return h_t, z_t, z_hat_next

    def encode_target(self, x_next):
        # No Gradient for target in LeJEPA logic? 
        # Minimal.md DOES compute gradient through target for SIGReg.
        # "inv_loss = (proj.mean(0) - proj).square().mean()" -> This implies target is just mean.
        # But in Dynamics, target is x_next. We allow gradients to flow to x_next encoder 
        # to enforce Gaussianity there too.
        h_next_feat = self.backbone(x_next)
        h_next = self.pool(h_next_feat).flatten(1)
        z_next = self.projector(h_next)
        return z_next

# ==============================================================================
# 3. Training Loop
# ==============================================================================
def save_checkpoint(model, probe, optimizer, scheduler, scaler, epoch, val_loss, val_auc, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'probe_state_dict': probe.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'val_loss': val_loss,
        'val_auc': val_auc
    }, path)

def train(args):
    wandb.init(project="lejepa-ecg", config=args)
    
    # --- Data ---
    print("Loading Dataset...")
    train_ds = DynamicsDataset(
        split='train', 
        return_pairs=True,
        data_fraction=args.data_fraction,
        in_memory=args.in_memory
        )
    val_ds = DynamicsDataset(
        split='val', 
        return_pairs=True,
        in_memory=args.in_memory
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)

    # --- Models ---
    model = LeJEPA_Model(proj_dim=args.proj_dim).to(DEVICE)
    sigreg = SIGReg().to(DEVICE)
    
    # Online Probe (On Backbone Output, usually 2048 dim)
    # Minimal.md uses LayerNorm before Linear
    print(f"Initializing Probe with input dim: {model.enc_dim}")
    probe = nn.Sequential(
        nn.LayerNorm(model.enc_dim), 
        nn.Linear(model.enc_dim, 76)
    ).to(DEVICE)

    # --- Optimizer (LeJEPA Spec) ---
    # Weight decay exclusion for biases/norms
    def get_param_groups(module):
        decay = []
        no_decay = []
        for name, param in module.named_parameters():
            if not param.requires_grad: continue
            if "bias" in name or "norm" in name or "bn" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0}
        ]

    params = get_param_groups(model) + get_param_groups(probe)
    optimizer = optim.AdamW(params, lr=args.lr)

    # --- Scheduler (Warmup + Cosine) ---
    warmup_steps = int(len(train_loader) * args.epochs * 0.05) # 5% warmup
    total_steps = len(train_loader) * args.epochs
    
    s1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler('cuda')
    best_val_auc = 0.0
    best_val_loss = float('inf')

    print("Starting LeJEPA Training...")
    start_epoch = 0
    if args.resume_from:
        if os.path.isfile(args.resume_from):
            print(f"Loading checkpoint '{args.resume_from}'")
            checkpoint = torch.load(args.resume_from)
            start_epoch = checkpoint['epoch'] + 1 # Start from next epoch
            
            model.load_state_dict(checkpoint['model_state_dict'])
            probe.load_state_dict(checkpoint['probe_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            best_val_auc = checkpoint.get('val_auc', 0.0)
            
            print(f"Resumed from Epoch {start_epoch} (Best Loss: {best_val_loss:.4f}, Best AUC: {best_val_auc:.4f})")
        else:
            print(f"No checkpoint found at '{args.resume_from}', starting from scratch.")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        probe.train()
        
        train_loss_accum = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in loop:
            x_t = batch['waveform'].to(DEVICE).float()
            x_next = batch['waveform_next'].to(DEVICE).float()
            action = batch['action'].to(DEVICE)
            label = batch['icd'].to(DEVICE)

            optimizer.zero_grad()
            
            with autocast('cuda'):
                # 1. Forward
                h_t, z_t, z_hat_next = model(x_t, action)
                z_next = model.encode_target(x_next)
                
                # 2. LeJEPA Losses
                # Prediction: MSE in Projected Space
                pred_loss = (z_hat_next - z_next).square().mean()
                
                # SIGReg: Apply to both target and prediction to enforce geometry
                # LeJEPA Minimal applies it to 'proj' which is the output.
                sigreg_loss = sigreg(z_t) + sigreg(z_next) 
                
                # Total Self-Supervised Loss
                lejepa_loss = (1 - args.lamb) * pred_loss + args.lamb * sigreg_loss
                
                # 3. Online Probe Loss
                # Detach h_t so probe doesn't affect backbone
                y_hat = probe(h_t.detach())
                probe_loss = F.binary_cross_entropy_with_logits(y_hat, label)
                
                loss = lejepa_loss + probe_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss_accum += loss.item()
            loop.set_postfix(loss=loss.item(), pred=pred_loss.item(), sig=sigreg_loss.item())
            
            wandb.log({
                "train/total_loss": loss.item(),
                "train/pred_loss": pred_loss.item(),
                "train/sigreg_loss": sigreg_loss.item(),
                "train/probe_loss": probe_loss.item(),
                "lr": scheduler.get_last_lr()[0]
            })

        # --- Validation ---
        model.eval()
        probe.eval()
        all_preds = []
        all_targets = []
        
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                x_t = batch['waveform'].to(DEVICE).float()
                x_next = batch['waveform_next'].to(DEVICE).float() # Needed for Loss
                action = batch['action'].to(DEVICE)
                # For probe validation, we only need current state
                label = batch['icd'].cpu().numpy()
                
                with autocast('cuda'):
                    # 1. Full Dynamics Forward (To calculate Val Loss)
                    h_t, z_t, z_hat_next = model(x_t, action)
                    z_next = model.encode_target(x_next)
                    
                    pred_loss = (z_hat_next - z_next).square().mean()
                    sigreg_loss = sigreg(z_t) + sigreg(z_next)
                    val_dyn_loss = (1 - args.lamb) * pred_loss + args.lamb * sigreg_loss
                    
                    # Accumulate
                    val_loss_accum += val_dyn_loss.item()
                    
                    # 2. Probe Forward (For AUROC)
                    logits = probe(h_t) # h_t is already computed above
                    probs = torch.sigmoid(logits).cpu().numpy()
                    
                all_preds.append(probs)
                all_targets.append(label)
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # Robust AUC
        valid_idxs = [i for i in range(76) if len(np.unique(all_targets[:, i])) > 1]
        val_auc = roc_auc_score(all_targets[:, valid_idxs], all_preds[:, valid_idxs], average='macro')

        avg_val_loss = val_loss_accum / len(val_loader)
        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val AUROC: {val_auc:.4f}")
        wandb.log({'val/loss': avg_val_loss, 'val/auc': val_auc, "epoch": epoch+1})
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_checkpoint(model, probe, optimizer, scheduler, scaler, epoch, best_val_loss, best_val_auc, "checkpoints/naive_probe.pth")
            print(">>> Saved probe Model")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, probe, optimizer, scheduler, scaler, epoch, best_val_loss, best_val_auc, "checkpoints/naive_loss.pth")
            print(">>> Saved loss Model")
        
        # 1. Save Latest (For resuming)
        save_checkpoint(model, probe, optimizer, scheduler, scaler, epoch, best_val_loss, best_val_auc, "checkpoints/naive_latest.pth")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3) # LeJEPA likes high LR (5e-4 to 2e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4) # ResNet default in Minimal
    
    # LeJEPA Params
    parser.add_argument('--proj_dim', type=int, default=256)
    parser.add_argument('--lamb', type=float, default=0.1) # Lambda trade-off

    parser.add_argument('--resume_from', type=str, default=None, help="Path to checkpoint to resume from")
    # dataset
    parser.add_argument('--data_fraction', type=float, default=1.0)
    parser.add_argument('--in_memory', action='store_true')
    
    args = parser.parse_args()
    train(args)

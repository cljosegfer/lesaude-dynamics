import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import DynamicsDataset
from src.models import MedicalLatentDynamics
from src.loss import VICRegLoss
from hparams import DATA_ROOT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- UTILS ---
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x = x.float()
        y = y.float()
        x = torch.clamp(x, min=-20, max=20)
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = -1 * (los_pos * torch.pow(1 - xs_pos, self.gamma_pos) + \
                     los_neg * torch.pow(1 - xs_neg, self.gamma_neg))
        return loss.sum()

class OnlineLinearHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Simple Linear Layer + BatchNorm for stability
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def save_checkpoint(model, probe, optimizer, scheduler, scaler, epoch, best_metric, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'probe_state_dict': probe.state_dict(), # Save probe too
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_auc': best_metric
    }, path)

def train(args):
    wandb.init(project="ecg-dynamics-online-probe", config=args)
    config = wandb.config

    # 1. Dataset
    print("Loading Dataset...")
    train_ds = DynamicsDataset(split='train', return_pairs=True)
    val_ds = DynamicsDataset(split='val', return_pairs=True)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)

    # 2. Models
    # A. The Main Dynamics Model
    model = MedicalLatentDynamics(
        num_input_channels=12,
        num_action_classes=76,
        latent_dim=config.latent_dim,
        projector_dim=config.projector_dim
    ).to(DEVICE)
    
    # B. The Online Linear Probe (Auxiliary)
    # Note: We assume latent_dim=2048 based on xresnet50 output
    probe = OnlineLinearHead(input_dim=2048, num_classes=76).to(DEVICE)

    # 3. Optimizers
    # Optimizer A: For Dynamics (Encoder + Predictor)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    
    # Optimizer B: For Linear Probe (Independent)
    # Probes often need higher LR than backbones
    probe_optimizer = optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, steps_per_epoch=len(train_loader), epochs=config.epochs)
    
    # Losses
    criterion_dynamics = VICRegLoss()
    criterion_class = AsymmetricLoss()
    
    scaler = GradScaler('cuda')

    # 4. Training Loop
    best_val_auc = 0.0
    
    for epoch in range(config.epochs):
        model.train()
        probe.train()
        
        loss_dyn_accum = 0.0
        loss_cls_accum = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in loop:
            # Inputs
            x_t = batch['waveform'].to(DEVICE).float()
            x_next = batch['waveform_next'].to(DEVICE).float()
            action = batch['action'].to(DEVICE)
            label = batch['icd'].to(DEVICE) # Needed for probe
            
            optimizer.zero_grad()
            probe_optimizer.zero_grad()
            
            with autocast('cuda'):
                # --- A. Dynamics Forward Pass ---
                outputs = model(x_t, action)
                
                # Get representations
                h_t = outputs['embedding']      # For Probe (features)
                z_hat = outputs['projection_hat'] # For Dynamics Loss
                
                # Get Target
                h_next = model.encode(x_next)
                z_next = model.vicreg_projector(h_next)
                
                # Dynamics Loss
                loss_dict = criterion_dynamics(z_hat, z_next)
                loss_dyn = loss_dict['loss']
                
                # --- B. Linear Probe Forward Pass ---
                # CRITICAL: .detach() stops gradients flowing back to Encoder
                # The probe learns from the encoder, but doesn't change it.
                logits = probe(h_t.detach()) 
                loss_cls = criterion_class(logits, label)

            # --- Optimization ---
            # 1. Update Dynamics Model
            scaler.scale(loss_dyn).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            
            # 2. Update Linear Probe (Separate graph due to detach)
            # We can use the same scaler or a separate one. 
            # Using simple backward here since head is fp32 safe usually, 
            # but safer to scale if using AMP.
            scaler.scale(loss_cls).backward()
            scaler.step(probe_optimizer)
            
            scaler.update()
            scheduler.step()
            
            loss_dyn_accum += loss_dyn.item()
            loss_cls_accum += loss_cls.item()
            loop.set_postfix(dyn=loss_dyn.item(), cls=loss_cls.item())
            
            wandb.log({
                "train/loss_dynamics": loss_dyn.item(),
                "train/loss_probe": loss_cls.item(),
                "lr": scheduler.get_last_lr()[0]
            })

        # --- Validation ---
        model.eval()
        probe.eval()
        
        val_dyn_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                x_t = batch['waveform'].to(DEVICE).float()
                x_next = batch['waveform_next'].to(DEVICE).float()
                action = batch['action'].to(DEVICE)
                label = batch['icd'].cpu().numpy()
                
                # Dynamics Val
                outputs = model(x_t, action)
                h_t = outputs['embedding']
                z_hat = outputs['projection_hat']
                h_next = model.encode(x_next)
                z_next = model.vicreg_projector(h_next)
                loss_d = criterion_dynamics(z_hat, z_next)
                val_dyn_loss += loss_d['loss'].item()
                
                # Probe Val
                logits = probe(h_t)
                probs = torch.sigmoid(logits).float().cpu().numpy()
                
                all_preds.append(probs)
                all_targets.append(label)
        
        # Metrics
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # Robust AUC Logic
        valid_class_indices = []
        for i in range(all_targets.shape[1]):
            if len(np.unique(all_targets[:, i])) > 1:
                valid_class_indices.append(i)
        
        if len(valid_class_indices) > 0:
            val_auc = roc_auc_score(all_targets[:, valid_class_indices], all_preds[:, valid_class_indices], average='macro')
        else:
            val_auc = 0.0

        avg_dyn_loss = loss_dyn_accum / len(train_loader)
        avg_val_dyn = val_dyn_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} | Dyn Loss: {avg_dyn_loss:.4f} | Probe AUROC: {val_auc:.4f}")
        wandb.log({
            "val/loss_dynamics": avg_val_dyn,
            "val/auc_probe": val_auc,
            "epoch": epoch+1
        })
        
        # Save Best based on AUROC (This is what we care about mostly)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_checkpoint(model, probe, optimizer, scheduler, scaler, epoch, avg_val_dyn, "checkpoints/best_model_online.pth")
            print(">>> Saved Best Model (based on Online Probe)")
        
        # 1. Save Latest (For resuming)
        save_checkpoint(model, probe, optimizer, scheduler, scaler, epoch, avg_val_dyn, "checkpoints/model_online_latest.pth")
        
        # 2. Save Epoch History (Optional, for safety)
        save_checkpoint(model, probe, optimizer, scheduler, scaler, epoch, avg_val_dyn, f"checkpoints/model_online_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4) # Main LR
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--latent_dim', type=int, default=2048)
    parser.add_argument('--projector_dim', type=int, default=8192)

    parser.add_argument('--sim_coeff', type=float, default=25.0)
    parser.add_argument('--std_coeff', type=float, default=25.0)
    parser.add_argument('--cov_coeff', type=float, default=1.0)
    
    args = parser.parse_args()
    train(args)
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import DynamicsDataset
from src.xresnet1d import xresnet1d50 
from src.utils import save_checkpoint
from hparams import DATA_ROOT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
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

def train(args):
    # Initialize W&B with resume capability
    wandb.init(project="ecg-supervised-baseline", config=args, resume="allow", id=args.run_id if args.run_id else None)
    config = wandb.config

    # 1. Dataset
    print("Loading Dataset...")
    train_ds = DynamicsDataset(
        split='train',
        return_pairs=False
    )
    val_ds = DynamicsDataset(
        split='val',
        return_pairs=False
    )
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    # 2. Model Setup
    print("Initializing ResNet1d50...")
    model = xresnet1d50(input_channels=12, num_classes=76).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, steps_per_epoch=len(train_loader), epochs=config.epochs)
    criterion = AsymmetricLoss()
    scaler = GradScaler('cuda')

    start_epoch = 0
    best_val_auc = 0.0

    # 3. Resume Logic
    if args.resume_from:
        if os.path.isfile(args.resume_from):
            print(f"Loading checkpoint '{args.resume_from}'")
            checkpoint = torch.load(args.resume_from)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            best_val_auc = checkpoint['best_val_auc']
            print(f"Resumed from Epoch {start_epoch} (Best AUC: {best_val_auc:.4f})")
        else:
            print(f"No checkpoint found at '{args.resume_from}', starting from scratch.")

    # 4. Loop
    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            x = batch['waveform'].to(DEVICE).float()
            y = batch['icd'].to(DEVICE)
            
            if torch.isnan(x).any(): continue

            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(x) 
                loss = criterion(logits, y)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if not torch.isnan(loss):
                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                x = batch['waveform'].to(DEVICE).float()
                y = batch['icd'].cpu().numpy()
                with autocast('cuda'):
                    logits = model(x)
                
                probs = torch.sigmoid(logits).float().cpu().numpy()
                all_preds.append(probs)
                all_targets.append(y)
                
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # --- ROBUST AUROC CALCULATION ---
        # 1. Identify classes that are present in the Validation Set
        # If a class has 0 positives, we cannot calculate AUC for it.
        valid_class_indices = []
        for i in range(all_targets.shape[1]):
            if len(np.unique(all_targets[:, i])) > 1:
                valid_class_indices.append(i)
        
        if len(valid_class_indices) > 0:
            # Calculate macro AUC only on the valid classes
            val_auc = roc_auc_score(
                all_targets[:, valid_class_indices], 
                all_preds[:, valid_class_indices], 
                average='macro'
            )
        else:
            val_auc = 0.0
            print("WARNING: No valid classes found in validation set!")

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val AUROC: {val_auc:.4f} (Computed on {len(valid_class_indices)} classes)")
        
        wandb.log({"train/loss": avg_train_loss, "val/auc": val_auc, "epoch": epoch+1})
        
        # --- SAVING ---
        os.makedirs("checkpoints", exist_ok=True)
        
        # 1. Save Latest (For resuming)
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, "checkpoints/supervised_latest.pth")
        
        # 2. Save Epoch History (Optional, for safety)
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, f"checkpoints/supervised_epoch_{epoch+1}.pth")
        
        # 3. Save Best
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_auc, "checkpoints/supervised_best.pth")
            print(">>> Saved Best Model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    # Resume arguments
    parser.add_argument('--resume_from', type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--run_id', type=str, default=None, help="WandB Run ID to resume logging")
    
    args = parser.parse_args()
    train(args)
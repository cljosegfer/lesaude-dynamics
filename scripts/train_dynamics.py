import sys
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import DynamicsDataset
from src.models import MedicalLatentDynamics
from src.loss import VICRegLoss
from hparams import DATA_ROOT

# ================= CONFIGURATION =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def train(args):
    # 1. Initialize W&B
    wandb.init(project="ecg-latent-dynamics", config=args)
    config = wandb.config

    # 2. Data Setup
    print("Loading Dataset...")
    full_dataset = DynamicsDataset(
        waveform_h5_path=os.path.join(DATA_ROOT, 'mimic_iv_ecg_waveforms.h5'),
        label_h5_path=os.path.join(DATA_ROOT, 'mimic_iv_ecg_icd.h5'),
        return_pairs=True
    )
    
    # Split Train/Val (90/10)
    # Note: A strict patient-wise split is better, but random_split is okay for a start 
    # since dataset indices are already grouped by patient.
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    # Optional: Weighted Sampling for Train
    # If using WeightedRandomSampler, you need to extract weights from the subset
    # For now, we use standard shuffling for simplicity.
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=4,      # NEW: Each worker pre-loads 4 batches. Keeps the pipe full.
        persistent_workers=True, # NEW: Don't kill workers after each epoch.
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=4,      # NEW: Each worker pre-loads 4 batches. Keeps the pipe full.
        persistent_workers=True, # NEW: Don't kill workers after each epoch.
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. Model Setup
    model = MedicalLatentDynamics(
        num_input_channels=12,
        num_action_classes=76,
        latent_dim=config.latent_dim,
        projector_dim=config.projector_dim
    ).to(DEVICE)
    
    # Log model gradients/parameters
    wandb.watch(model, log="all")

    # 4. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Scheduler: Warmup + Cosine Decay is standard for VICReg
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config.lr, 
        steps_per_epoch=len(train_loader), 
        epochs=config.epochs
    )
    
    criterion = VICRegLoss(
        sim_coeff=config.sim_coeff,
        std_coeff=config.std_coeff,
        cov_coeff=config.cov_coeff
    )
    
    scaler = GradScaler('cuda') # For Mixed Precision

    # 5. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        model.train()
        train_loss_accum = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for batch in loop:
            # Unpack Dictionary
            x_t = batch['waveform'].to(DEVICE).float()      # (B, 12, 5000)
            x_next = batch['waveform_next'].to(DEVICE).float() # (B, 12, 5000)
            action = batch['action'].to(DEVICE)     # (B, 76)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                # A. Forward Pass (Student/Predictor)
                # Model predicts: what should z_{t+1} look like given x_t and action?
                outputs = model(x_t, action)
                z_hat_next = outputs['projection_hat'] # The predicted projection
                
                # B. Target Pass (Teacher/Ground Truth)
                # Encode the ACTUAL future state x_{t+1}
                # Note: We do NOT detach x_next gradient. In VICReg, both branches train the encoder.
                h_next = model.encode(x_next)
                z_next = model.vicreg_projector(h_next)
                
                # C. Calculate Loss
                loss_dict = criterion(z_hat_next, z_next)
                loss = loss_dict['loss']

            # Backward
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Logging
            train_loss_accum += loss.item()
            loop.set_postfix(loss=loss.item())
            
            # W&B Step Log
            wandb.log({
                "train/loss": loss.item(),
                "train/sim_loss": loss_dict['sim_loss'].item(),
                "train/std_loss": loss_dict['std_loss'].item(),
                "train/cov_loss": loss_dict['cov_loss'].item(),
                "lr": scheduler.get_last_lr()[0]
            })

        avg_train_loss = train_loss_accum / len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        val_loss_accum = 0.0
        
        loop = tqdm(val_loader, desc=f"Validation")

        with torch.no_grad():
            for batch in loop:
                x_t = batch['waveform'].to(DEVICE).float()
                x_next = batch['waveform_next'].to(DEVICE).float()
                action = batch['action'].to(DEVICE)
                
                outputs = model(x_t, action)
                z_hat_next = outputs['projection_hat']
                
                h_next = model.encode(x_next)
                z_next = model.vicreg_projector(h_next)
                
                loss_dict = criterion(z_hat_next, z_next)
                val_loss_accum += loss_dict['loss'].item()
        
        avg_val_loss = val_loss_accum / len(val_loader)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        wandb.log({
            "val/loss": avg_val_loss, 
            "epoch": epoch+1
        })
        
        # Save Checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, avg_val_loss, "checkpoints/best_model.pth")
            print(">>> Saved Best Model")
            
        # Save Regular Checkpoint
        save_checkpoint(model, optimizer, epoch, avg_val_loss, f"checkpoints/epoch_{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64) # Adjust based on VRAM
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Architecture
    parser.add_argument('--latent_dim', type=int, default=2048)
    parser.add_argument('--projector_dim', type=int, default=8192)
    
    # Loss Weights
    parser.add_argument('--sim_coeff', type=float, default=25.0)
    parser.add_argument('--std_coeff', type=float, default=25.0)
    parser.add_argument('--cov_coeff', type=float, default=1.0)
    
    args = parser.parse_args()
    
    train(args)
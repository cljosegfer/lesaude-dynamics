import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

# Add src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import DynamicsDataset
from src.models import MedicalLatentDynamics
from hparams import DATA_ROOT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AsymmetricLoss(nn.Module):
    """
    ASL: Focuses on hard negatives. Crucial for imbalanced multi-label tasks.
    Ref: https://arxiv.org/abs/2009.14119
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        # x: logits, y: targets (multi-label binarized)
        
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Cross Entropy
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        
        # Asymmetric Focusing
        loss = -1 * (los_pos * torch.pow(1 - xs_pos, self.gamma_pos) + \
                     los_neg * torch.pow(1 - xs_neg, self.gamma_neg))
        
        return loss.sum()

class LinearProbe(nn.Module):
    def __init__(self, backbone, input_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.head = nn.Sequential(
            nn.BatchNorm1d(input_dim), # Normalize features before classifier
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone.encode(x) # Get h_t
        return self.head(features)

def evaluate(args):
    # 1. Load Pre-trained Model
    print(f"Loading Checkpoint: {args.checkpoint_path}")
    base_model = MedicalLatentDynamics(
        num_input_channels=12,
        num_action_classes=76,
        latent_dim=2048
    )
    
    # Load weights
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.to(DEVICE)
    base_model.eval()

    # 2. Setup Linear Probe
    # We detect output dim by running a dummy pass
    with torch.no_grad():
        dummy = torch.randn(2, 12, 5000).to(DEVICE)
        feat_dim = base_model.encode(dummy).shape[1]
        
    probe_model = LinearProbe(base_model, feat_dim, num_classes=76).to(DEVICE)
    
    # 3. Data (Classification Mode -> return_pairs=False)
    print("Loading Dataset (Classification Mode)...")
    dataset = DynamicsDataset(
        waveform_h5_path=args.waveform_h5_path,
        label_h5_path=args.label_h5_path,
        return_pairs=False 
    )
    
    # Use 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # 4. Optimizer
    optimizer = optim.AdamW(probe_model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = AsymmetricLoss()

    # 5. Training Loop (Linear Layer only)
    print("Starting Linear Probing...")
    for epoch in range(args.epochs):
        probe_model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x = batch['waveform'].to(DEVICE).float()
            y = batch['icd'].to(DEVICE)
            
            optimizer.zero_grad()
            logits = probe_model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        probe_model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['waveform'].to(DEVICE).float()
                y = batch['icd'].numpy()
                logits = probe_model(x)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                all_preds.append(probs)
                all_targets.append(y)
                
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # Calculate Macro AUROC
        try:
            auroc = roc_auc_score(all_targets, all_preds, average='macro')
        except ValueError:
            auroc = 0.0 # Handle edge case if a class has no positive samples
            
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val AUROC: {auroc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--waveform_h5_path', type=str, required=True)
    parser.add_argument('--label_h5_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10) # 10 epochs is enough for linear probe
    args = parser.parse_args()
    
    evaluate(args)
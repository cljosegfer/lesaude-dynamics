import sys
import os
import argparse
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import DynamicsDataset
from src.models import MedicalLatentDynamics
from hparams import DATA_ROOT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-using your ASL Class
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
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

def get_features(model, loader, desc="Extracting"):
    """Runs data through the frozen backbone and stores features in RAM."""
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            x = batch['waveform'].to(DEVICE).float()
            y = batch['icd'] # Keep on CPU for now to save GPU VRAM
            
            # Extract z_t (Encoder output)
            # Note: We use .encode() which returns the 2048-dim vector
            feats = model.encode(x)
            
            features_list.append(feats.cpu())
            labels_list.append(y)
            
    # Concatenate into one giant tensor
    features = torch.cat(features_list)
    labels = torch.cat(labels_list)
    return features, labels

def main(args):
    # 1. Load Backbone
    print(f"Loading Backbone: {args.checkpoint_path}")
    base_model = MedicalLatentDynamics(
        num_input_channels=12,
        num_action_classes=76,
        latent_dim=2048
    )
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.to(DEVICE)
    base_model.eval()

    # 2. Data Loader (For Extraction)
    print("Initializing I/O...")
    train_ds = DynamicsDataset(
        split='train',
        return_pairs=False
    )
    val_ds = DynamicsDataset(
        split='val',
        return_pairs=False
    )
    
    # We can use large batch size for inference since no gradients
    train_loader_io = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=8)
    val_loader_io = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=8)

    # 3. Extract Features (The Slow Part - Happens Once)
    print(">>> Phase 1: Extracting Training Features to RAM...")
    X_train, Y_train = get_features(base_model, train_loader_io, "Train Extract")
    
    print(">>> Phase 1: Extracting Validation Features to RAM...")
    X_val, Y_val = get_features(base_model, val_loader_io, "Val Extract")
    
    print(f"Features Cached! Train Shape: {X_train.shape} (~{X_train.element_size() * X_train.numel() / 1e9:.2f} GB)")

    # 4. Create RAM DataLoaders (Fast)
    # Move to GPU if VRAM allows (2.7GB fits in RTX 4090 easily)
    train_ds_ram = TensorDataset(X_train.to(DEVICE), Y_train.to(DEVICE))
    val_ds_ram = TensorDataset(X_val.to(DEVICE), Y_val.to(DEVICE))
    
    train_loader_fast = DataLoader(train_ds_ram, batch_size=1024, shuffle=True)
    val_loader_fast = DataLoader(val_ds_ram, batch_size=1024, shuffle=False)

    # 5. Linear Probe
    probe = nn.Sequential(
        nn.BatchNorm1d(2048),
        nn.Linear(2048, 76)
    ).to(DEVICE)
    
    optimizer = optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = AsymmetricLoss()

    # 6. Training Loop (Super Fast)
    print(">>> Phase 2: Training Linear Classifier (In Memory)...")
    best_val_auc = 0.0
    best_epoch = 0
    for epoch in range(20): # You can run 20 epochs in seconds now
        probe.train()
        train_loss = 0
        
        for x, y in (train_loader_fast):
            optimizer.zero_grad()
            logits = probe(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        probe.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, y in (val_loader_fast):
                logits = probe(x)
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        valid_class_indices = []
        for i in range(all_targets.shape[1]):
            if len(np.unique(all_targets[:, i])) > 1:
                valid_class_indices.append(i)
        
        if len(valid_class_indices) > 0:
            val_auc = roc_auc_score(all_targets[:, valid_class_indices], all_preds[:, valid_class_indices], average='macro')
        else:
            val_auc = 0.0
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            
        print(f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader_fast):.4f} | Val AUROC: {val_auc:.4f}")
    print(f"Best Val AUROC: {best_val_auc:.4f} at Epoch {best_epoch+1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best_model.pth')
    # parser.add_argument('--waveform_h5_path', type=str, required=True)
    # parser.add_argument('--label_h5_path', type=str, required=True)
    args = parser.parse_args()
    main(args)
import sys
import os
import argparse
import copy
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import DynamicsDataset
from dynamics_lejepa import LeJEPA_Model
# from dynamics_naive import LeJEPA_Model
from hparams import DATA_ROOT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Bootstrap Helper (Adapted from eval.py) ---
def compute_bootstrap_auroc(y_true, y_pred, n_bootstraps=1000, seed=42):
    """
    Computes 95% Confidence Interval using bootstrap resampling.
    """
    print(f"\n>>> Running Bootstrap Analysis ({n_bootstraps} iterations)...")
    rng = np.random.RandomState(seed)
    boot_scores = []
    n_samples = y_true.shape[0]
    indices = np.arange(n_samples)
    
    # Pre-calculate which columns were valid in the full set
    globally_valid_cols = [i for i in range(y_true.shape[1]) if len(np.unique(y_true[:, i])) > 1]
    
    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        # Sample indices with replacement
        boot_idx = resample(indices, replace=True, n_samples=n_samples, random_state=rng)
        
        y_t_boot = y_true[boot_idx]
        y_p_boot = y_pred[boot_idx]
        
        # Check for valid classes in THIS specific sample
        iteration_valid_cols = []
        for i in globally_valid_cols:
            if len(np.unique(y_t_boot[:, i])) > 1:
                iteration_valid_cols.append(i)
        
        if len(iteration_valid_cols) == 0:
            continue 
            
        # Calculate Macro AUC for this iteration
        score = roc_auc_score(
            y_t_boot[:, iteration_valid_cols], 
            y_p_boot[:, iteration_valid_cols], 
            average='macro'
        )
        boot_scores.append(score)
    
    boot_scores = np.array(boot_scores)
    
    # Calculate Percentiles
    lower = np.percentile(boot_scores, 2.5)
    upper = np.percentile(boot_scores, 97.5)
    mean_score = np.mean(boot_scores)
    
    return mean_score, lower, upper

# --- Loss Function ---
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
            
            # Extract features (h)
            h_feat = model.backbone(x)
            feats = model.pool(h_feat).flatten(1) # This is h (256-dim)
            
            features_list.append(feats.cpu())
            labels_list.append(y)
            
    # Concatenate into one giant tensor
    features = torch.cat(features_list)
    labels = torch.cat(labels_list)
    return features, labels

def main(args):
    # 1. Load Backbone
    print(f"Loading Backbone: {args.checkpoint_path}")
    base_model = LeJEPA_Model(proj_dim=256).to(DEVICE)
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.to(DEVICE)
    base_model.eval()

    # 2. Data Loader (For Extraction)
    print("Initializing I/O...")
    train_ds = DynamicsDataset(split='train', )
    val_ds = DynamicsDataset(split='val', return_pairs=args.return_pairs)
    test_ds = DynamicsDataset(split='test', return_pairs=args.return_pairs) # Add Test Set
    
    # We can use large batch size for inference since no gradients
    train_loader_io = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=8)
    val_loader_io = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=8)
    test_loader_io = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=8)

    # 3. Extract Features (The Slow Part - Happens Once)
    print(">>> Phase 1: Extracting Training Features to RAM...")
    X_train, Y_train = get_features(base_model, train_loader_io, "Train Extract")
    
    print(">>> Phase 1: Extracting Validation Features to RAM...")
    X_val, Y_val = get_features(base_model, val_loader_io, "Val Extract")

    print(">>> Phase 1: Extracting Test Features to RAM...")
    X_test, Y_test = get_features(base_model, test_loader_io, "Test Extract")
    
    print(f"Features Cached! Train Shape: {X_train.shape} (~{X_train.element_size() * X_train.numel() / 1e9:.2f} GB)")

    # 4. Create RAM DataLoaders (Fast)
    train_ds_ram = TensorDataset(X_train.to(DEVICE), Y_train.to(DEVICE))
    val_ds_ram = TensorDataset(X_val.to(DEVICE), Y_val.to(DEVICE))
    test_ds_ram = TensorDataset(X_test.to(DEVICE), Y_test.to(DEVICE))
    
    train_loader_fast = DataLoader(train_ds_ram, batch_size=1024, shuffle=True)
    val_loader_fast = DataLoader(val_ds_ram, batch_size=1024, shuffle=False)
    test_loader_fast = DataLoader(test_ds_ram, batch_size=1024, shuffle=False)

    # 5. Linear Probe
    # DETECT DIMENSION
    with torch.no_grad():
        dummy = torch.randn(2, 12, 100).to(DEVICE)
        dummy_feat = base_model.backbone(dummy)
        feat_dim = base_model.pool(dummy_feat).flatten(1).shape[1]
    print(f"Detected Feature Dimension: {feat_dim}") 

    probe = nn.Sequential(
        nn.BatchNorm1d(feat_dim),
        nn.Linear(feat_dim, 76)
    ).to(DEVICE)
    
    optimizer = optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = AsymmetricLoss()

    # 6. Training Loop (Super Fast)
    print(">>> Phase 2: Training Linear Classifier (In Memory)...")
    best_val_auc = 0.0
    best_epoch = 0
    best_probe_state = None

    for epoch in range(20): 
        probe.train()
        train_loss = 0
        
        for x, y in train_loader_fast:
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
            for x, y in val_loader_fast:
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
            best_probe_state = copy.deepcopy(probe.state_dict())
            
        print(f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader_fast):.4f} | Val AUROC: {val_auc:.4f}")
    
    print(f"Best Val AUROC: {best_val_auc:.4f} at Epoch {best_epoch+1}")

    # 7. Final Test Evaluation
    print("\n>>> Phase 3: Final Test Evaluation (Best Model)...")
    if best_probe_state is not None:
        probe.load_state_dict(best_probe_state)
    
    probe.eval()
    all_preds_test = []
    all_targets_test = []
    
    with torch.no_grad():
        for x, y in test_loader_fast:
            logits = probe(x)
            all_preds_test.append(torch.sigmoid(logits).cpu().numpy())
            all_targets_test.append(y.cpu().numpy())
            
    all_preds_test = np.concatenate(all_preds_test)
    all_targets_test = np.concatenate(all_targets_test)

    # Point Estimate
    valid_class_indices = []
    dropped_classes = []
    for i in range(all_targets_test.shape[1]):
        if len(np.unique(all_targets_test[:, i])) > 1:
            valid_class_indices.append(i)
        else:
            dropped_classes.append(i)

    if len(valid_class_indices) > 0:
        point_auc = roc_auc_score(all_targets_test[:, valid_class_indices], all_preds_test[:, valid_class_indices], average='macro')
    else:
        point_auc = 0.0
        
    # Bootstrap
    mean_boot, lower, upper = compute_bootstrap_auroc(all_targets_test, all_preds_test, n_bootstraps=args.n_bootstraps)

    print("-" * 60)
    print(f"TEST RESULT (Linear Probe on {os.path.basename(args.checkpoint_path)})")
    print(f"Split: Fold 19 | Return Pairs (All ECGs): {args.return_pairs}")
    print("-" * 60)
    print(f"Classes Evaluated: {len(valid_class_indices)} / 76")
    print(f"Classes Dropped: {len(dropped_classes)}")
    print("-" * 60)
    print(f"Macro AUROC (Point):     {point_auc:.4f}")
    print(f"Macro AUROC (Bootstrap): {mean_boot:.4f} [{lower:.4f} - {upper:.4f}]")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--n_bootstraps', type=int, default=1000, help="Number of bootstrap iterations for CI")
    parser.add_argument('--return_pairs', action='store_true', 
                        help="If set, uses ALL ECGs in test set. If not, uses only first ECG per stay.")
    args = parser.parse_args()
    main(args)
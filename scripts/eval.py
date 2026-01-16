import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import DynamicsDataset
from src.xresnet1d import xresnet1d50 
from hparams import DATA_ROOT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(args):
    print(f"--- EVALUATION ON TEST SET (Fold 19) ---")
    print(f"Checkpoint: {args.checkpoint_path}")
    
    # 1. Dataset Setup
    # Note: By default, DynamicsDataset applies 'ecg_no_within_stay == 0' 
    # for 'test' split when return_pairs=False.
    # We might need to override this if testing the "Monitoring" hypothesis.
    
    print("Loading Test Dataset...")
    # Hack: If we want ALL ECGs (Monitoring Task), we might need to bypass the dataset's internal filter.
    # For now, let's assume standard behavior (Triage Task).
    test_ds = DynamicsDataset(
        split='test',
        return_pairs=args.return_pairs 
    )
    
    # Optional: If you modified Dataset to accept a 'filter' arg, pass it here.
    # Otherwise, it defaults to the Baseline Filter (First ECG only).
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )

    # 2. Model Setup
    print("Initializing Model...")
    model = xresnet1d50(input_channels=12, num_classes=76)
    
    # Load Weights
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        
    ckpt = torch.load(args.checkpoint_path, map_location='cpu')
    
    # Handle both full checkpoint dicts and direct state_dicts
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
        
    # Load
    msg = model.load_state_dict(state_dict, strict=True)
    print(f"Weights Loaded. {msg}")
    
    model = model.to(DEVICE)
    model.eval()

    # 3. Inference Loop
    all_preds = []
    all_targets = []
    
    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x = batch['waveform'].to(DEVICE).float()
            y = batch['icd'].numpy() # Keep targets on CPU
            
            # Forward
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_preds.append(probs)
            all_targets.append(y)

    # Concatenate
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    print(f"Total Samples Evaluated: {len(all_targets)}")

    # 4. Robust Metrics Calculation
    print("Calculating Metrics...")
    
    # Identify valid classes (present in ground truth)
    valid_class_indices = []
    dropped_classes = []
    
    for i in range(all_targets.shape[1]):
        # We need at least one '0' and one '1' to calculate ROC
        unique_vals = np.unique(all_targets[:, i])
        if len(unique_vals) > 1:
            valid_class_indices.append(i)
        else:
            dropped_classes.append(i)
            
    if len(valid_class_indices) == 0:
        print("CRITICAL ERROR: No valid classes found in Test Set.")
        return

    # Filter arrays
    y_true_filtered = all_targets[:, valid_class_indices]
    y_pred_filtered = all_preds[:, valid_class_indices]
    
    # Calculate Macro AUROC
    test_auc = roc_auc_score(y_true_filtered, y_pred_filtered, average='macro')
    
    print("-" * 40)
    print(f"TEST RESULT checkpoint {args.checkpoint_path} (Fold 19) with return_pairs={args.return_pairs}")
    print("-" * 40)
    print(f"Macro AUROC: {test_auc:.4f}")
    print(f"Classes Evaluated: {len(valid_class_indices)} / 76")
    print(f"Classes Dropped (No positive samples in Test): {len(dropped_classes)}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help="Path to .pth file (e.g. checkpoints/finetuned_best.pth)")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--return_pairs', action='store_true')
    
    args = parser.parse_args()
    evaluate(args)
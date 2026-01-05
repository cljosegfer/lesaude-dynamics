import torch
import torch.nn as nn
import torch.nn.functional as F
from xresnet1d import xresnet1d50  # baseline

class MedicalLatentDynamics(nn.Module):
    def __init__(self, 
                 num_input_channels=12,
                 num_action_classes=76, 
                 latent_dim=2048,   # Default output of ResNet50
                 projector_dim=8192, # VICReg usually expands 4x
                 predictor_hidden_dim=512):
        super().__init__()
        
        # 1. ENCODER (h_theta)
        # We initialize xresnet1d50 with num_classes=None to get the backbone only.
        # But xresnet1d implementation might default to valid padding or specific stems.
        # We use the baseline parameters.
        self.encoder_backbone = xresnet1d50(input_channels=num_input_channels, num_classes=None)
        
        # The backbone output is usually (Batch, 2048, Length).
        # We need to pool it to (Batch, 2048).
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 2. ACTION PROJECTOR (Embeds the sparse difference vector)
        self.action_mlp = nn.Sequential(
            nn.Linear(num_action_classes, predictor_hidden_dim),
            nn.BatchNorm1d(predictor_hidden_dim),
            nn.ReLU(),
            nn.Linear(predictor_hidden_dim, latent_dim)
        )
        
        # 3. DYNAMICS PREDICTOR (f_theta)
        # Inputs: [z_t, action_embedding] -> Output: z_t+1_pred
        # Input dim = latent_dim * 2
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, predictor_hidden_dim),
            nn.BatchNorm1d(predictor_hidden_dim),
            nn.ReLU(),
            nn.Linear(predictor_hidden_dim, predictor_hidden_dim),
            nn.BatchNorm1d(predictor_hidden_dim),
            nn.ReLU(),
            nn.Linear(predictor_hidden_dim, latent_dim) # Predicts latent state
        )
        
        # 4. VICReg PROJECTOR (For Loss Calculation ONLY)
        # We don't use this for downstream tasks, only for the loss function.
        # It maps z -> high_dim representation where variance is enforced.
        self.vicreg_projector = nn.Sequential(
            nn.Linear(latent_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim)
        )

    def encode(self, x):
        """Returns the representation z (used for downstream tasks)."""
        feat = self.encoder_backbone(x) # (B, 2048, L)
        z = self.pool(feat).flatten(1)  # (B, 2048)
        return z

    def forward(self, x_t, action):
        """
        Forward pass for training dynamics.
        Returns:
            h_t: Representation of current state
            h_hat: Predicted representation of future state
            z_t: Projected z_t (for VICReg loss)
            z_hat: Projected prediction (for VICReg loss)
        """
        # A. Encode Current State
        h_t = self.encode(x_t)
        
        # B. Process Action
        a_emb = self.action_mlp(action)
        
        # C. Predict Future State
        # Concatenate: [h_t, action]
        combined = torch.cat([h_t, a_emb], dim=1)
        h_hat = self.predictor(combined)
        
        # D. Projections for Loss
        z_t = self.vicreg_projector(h_t)
        z_hat = self.vicreg_projector(h_hat)
        
        return {'embedding': h_t, 'embedding_hat': h_hat, 'projection': z_t, 'projection_hat': z_hat}
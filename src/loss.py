import torch
import torch.nn as nn
import torch.nn.functional as F

class VICRegLoss(nn.Module):
    def __init__(self, 
                 sim_coeff=25.0, 
                 std_coeff=25.0, 
                 cov_coeff=1.0, 
                 gamma=1.0):
        """
        VICReg Loss Module (Variance-Invariance-Covariance Regularization).
        
        Args:
            sim_coeff (float): Weight for the Invariance (MSE) loss (Lambda).
                               Standard default is 25.0.
            std_coeff (float): Weight for the Variance loss (Mu).
                               Standard default is 25.0.
            cov_coeff (float): Weight for the Covariance loss (Nu).
                               Standard default is 1.0.
            gamma (float): Target standard deviation for the variance loss. 
                           Usually 1.0.
        """
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma

    def off_diagonal(self, x):
        """Helper to extract off-diagonal elements of a square matrix."""
        n, m = x.shape
        assert n == m
        # Flatten the matrix and remove the diagonal elements
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z_pred, z_target):
        """
        Calculate the VICReg loss.

        Args:
            z_pred (Tensor): Projected prediction from the student/predictor branch.
                             Shape: (Batch_Size, Projector_Dim)
            z_target (Tensor): Projected target from the teacher/encoder branch.
                               Shape: (Batch_Size, Projector_Dim)
                               
        Returns:
            dict: Contains total 'loss' and individual components for logging.
        """
        batch_size = z_pred.size(0)
        embedding_dim = z_pred.size(1)

        # --- 1. Invariance Loss (Similarity) ---
        # MSE between the predicted projection and the actual future projection
        sim_loss = F.mse_loss(z_pred, z_target)

        # --- 2. Variance Loss (Hinge) ---
        # Force the batch to have std deviation >= gamma
        # We add epsilon for numerical stability of sqrt
        std_pred = torch.sqrt(z_pred.var(dim=0) + 0.0001)
        std_target = torch.sqrt(z_target.var(dim=0) + 0.0001)

        # ReLU(Gamma - Std) -> Penalizes if std is too low (Collapse)
        std_loss_pred = torch.mean(F.relu(self.gamma - std_pred))
        std_loss_target = torch.mean(F.relu(self.gamma - std_target))
        
        # Apply to both branches
        std_loss = std_loss_pred + std_loss_target

        # --- 3. Covariance Loss (Decorrelation) ---
        # Force off-diagonal elements of covariance matrix to be 0
        
        # Center the vectors (Zero Mean)
        z_pred_centered = z_pred - z_pred.mean(dim=0)
        z_target_centered = z_target - z_target.mean(dim=0)

        # Calculate Covariance Matrix: (D x D) = (D x B) @ (B x D) / (B - 1)
        cov_pred = (z_pred_centered.T @ z_pred_centered) / (batch_size - 1)
        cov_target = (z_target_centered.T @ z_target_centered) / (batch_size - 1)

        # Sum of squared off-diagonal elements
        cov_loss_pred = self.off_diagonal(cov_pred).pow(2).sum() / embedding_dim
        cov_loss_target = self.off_diagonal(cov_target).pow(2).sum() / embedding_dim
        
        # Apply to both branches
        cov_loss = cov_loss_pred + cov_loss_target

        # --- Total Loss ---
        loss = (self.sim_coeff * sim_loss) + \
               (self.std_coeff * std_loss) + \
               (self.cov_coeff * cov_loss)

        return {
            "loss": loss,
            "sim_loss": sim_loss,
            "std_loss": std_loss,
            "cov_loss": cov_loss
        }

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

import torch
import torch.nn as nn
import torch.nn.functional as F


class EppsPulley(nn.Module):
    """Epps-Pulley goodness-of-fit test for univariate normality.

    Evaluates the characteristic-function distance between the empirical
    distribution of 1D projections and N(0,1) using trapezoidal quadrature.

    Matches the spt LeJEPA reference implementation exactly.
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        assert n_points % 2 == 1

        self._is_ddp = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        self.world_size = torch.distributed.get_world_size() if self._is_ddp else 1

        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        self.register_buffer("t", t)

        phi = (-0.5 * t ** 2).exp()          # N(0,1) characteristic function
        self.register_buffer("phi", phi)

        weights = torch.full((n_points,), 2 * dt)
        weights[[0, -1]] = dt                 # trapezoidal end-point correction
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, S) — N samples, S random 1D projections
        Returns:
            per-slice statistic (S,)
        """
        N = x.size(0)
        x_t = x.unsqueeze(-1) * self.t        # (N, S, T)
        cos_mean = x_t.cos().mean(0)          # (S, T)
        sin_mean = x_t.sin().mean(0)          # (S, T)

        if self._is_ddp:
            torch.distributed.all_reduce(cos_mean, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(sin_mean, op=torch.distributed.ReduceOp.AVG)

        err = (cos_mean - self.phi).square() + sin_mean.square()  # (S, T)
        return (err @ self.weights) * N * self.world_size          # (S,)


class SlicedEppsPulley(nn.Module):
    """Sliced Epps-Pulley goodness-of-fit test — SIGReg from the LeJEPA paper.

    Projects embeddings onto random 1D directions (seeded by a step counter
    so all DDP ranks draw identical directions) and averages the univariate
    Epps-Pulley statistic across slices.

    Matches the spt LeJEPA reference implementation exactly.
    """

    def __init__(self, num_slices: int = 1024, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        self._is_ddp = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        self.num_slices = num_slices
        self.ep = EppsPulley(t_max=t_max, n_points=n_points)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, D) batch of embeddings
        Returns:
            scalar mean EP statistic
        """
        with torch.no_grad():
            step = self.global_step.clone()
            if self._is_ddp:
                # Safety broadcast against step drift from uneven batches
                torch.distributed.broadcast(step, src=0)
            g = torch.Generator(device=x.device).manual_seed(step.item())
            A = torch.randn(x.size(-1), self.num_slices, device=x.device, generator=g)
            A = A / A.norm(p=2, dim=0)    # unit-norm columns
            self.global_step.add_(1)

        proj = x @ A                      # (N, num_slices)
        return self.ep(proj).mean()


class ActionProjector(nn.Module):
    """Maps sparse action vector at ∈ {-1,0,1}^C to a continuous embedding.

    Input:  (B, action_dim)  — float-cast of the int8 transition vector
    Output: (B, embed_dim)
    """

    def __init__(self, action_dim: int = 76, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, at: torch.Tensor) -> torch.Tensor:
        return self.net(at)


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class DynamicsPredictor(nn.Module):
    """Residual MLP predicting ĥ_{t+1} from current state and action embedding.

    Input:  ht (B, embed_dim), action_emb (B, embed_dim)
    Output: ĥ_{t+1} (B, embed_dim)
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim * 2, hidden_dim)
        self.residual = _ResidualBlock(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, ht: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([ht, action_emb], dim=-1)    # (B, embed_dim*2)
        x = F.relu(self.input_proj(x))              # (B, hidden_dim)
        x = self.residual(x)                        # (B, hidden_dim)
        return self.output_proj(x)                  # (B, embed_dim)

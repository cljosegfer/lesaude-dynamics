import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _rope(n_tokens: int, head_dim: int, base: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(n_tokens, device=device, dtype=torch.float32), inv_freq)
    return freqs.cos(), freqs.sin()                      # (T, head_dim/2) each


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE on fp32 (B, H, T, head_dim)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qk_norm: bool):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        # QK-norm and RoPE in fp32 (the norm weights are fp32 under autocast), back to v's dtype for attention
        q = _apply_rope(self.q_norm(q.float()), cos, sin).to(v.dtype)
        k = _apply_rope(self.k_norm(k.float()), cos, sin).to(v.dtype)
        x = F.scaled_dot_product_attention(q, k, v)       # bidirectional: no causal mask
        return self.proj(x.transpose(1, 2).reshape(B, T, D))


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, qk_norm: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, num_heads, qk_norm)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        # 2/3 of the usual width keeps SwiGLU's three matrices at the parameter count of a 2-matrix MLP
        self.mlp = _SwiGLU(dim, int(2 / 3 * mlp_ratio * dim))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT1d(nn.Module):
    """1-D Vision Transformer for multi-lead ECGs, a drop-in alternative to ResNet1d.

    Input:  (B, in_channels, length)   e.g. (B, 12, 5000)
    Output: (B, embedding_dim)

    Non-overlapping Conv1d patches (patch_size=25 is 50 ms at 500 Hz) feed pre-norm transformer blocks
    with bidirectional attention, QK-norm and SwiGLU MLPs. Positions enter only through rotary embeddings
    (RoPE), which encode relative offsets: the embedding should not depend on where in the strip a beat
    falls. The final tokens are normalized, mean-pooled and mapped to embedding_dim by `fc`, named like
    ResNet1d's output layer.

    dim=192, depth=8 is LeNEPA's ECG ViT-XS layout (3.67M parameters) without its causal mask. The defaults
    (dim=128, depth=4, 0.87M) match the size of ResNet1d's stem..layer3 (0.96M of its 3.98M parameters).
    """

    def __init__(
        self,
        in_channels: int = 12,
        embedding_dim: int = 256,
        patch_size: int = 25,
        dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        rope_base: float = 10000.0,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.rope_base = rope_base
        self.grad_checkpoint = grad_checkpoint
        self.patch_embed = nn.Conv1d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([_Block(dim, num_heads, mlp_ratio, qk_norm) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc = nn.Linear(dim, embedding_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _tokens(self, x: torch.Tensor):
        """Yields the patch tokens, then the output of every block, each (B, T, dim)."""
        x = self.patch_embed(x).transpose(1, 2)                                   # (B, T, dim)
        cos, sin = _rope(x.shape[1], self.blocks[0].attn.qkv.in_features // self.blocks[0].attn.num_heads,
                         self.rope_base, x.device)
        yield x
        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
            yield x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *_, x = self._tokens(x)
        return self.fc(self.norm(x).mean(dim=1))

    def layer_features(self, x: torch.Tensor) -> dict:
        """Mean-pooled tokens after the patch embedding and after every block, plus h, for diagnostics."""
        out = {}
        for i, tokens in enumerate(self._tokens(x)):
            out["patch" if i == 0 else f"block{i}"] = tokens.mean(dim=1)
        out["h"] = self.fc(self.norm(tokens).mean(dim=1))
        return out

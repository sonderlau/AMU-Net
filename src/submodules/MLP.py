"""MLP prediction heads used in AdaMamba and related modules."""

import torch.nn as nn
from torch import Tensor
from einops import rearrange


class MLP(nn.Module):
    """Multi-headed MLP that expands features along a new dimension."""

    def __init__(self, dim_in: int, dim_out: int, factor: int = 1, hidden_dim: int = 2048):
        super().__init__()
        self.factor = factor
        self.dim_out = dim_out
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_out * factor),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project spatial features into (factor, dim_out) slices per token."""

        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> (b h w) c")
        x = self.mlp(x)
        x = rearrange(
            x,
            "(B H W) (N C) -> B N C H W",
            N=self.factor,
            C=self.dim_out,
            B=B,
            H=H,
            W=W,
        )
        return x


class MLPDeterministic(nn.Module):
    """Deterministic MLP head that preserves the spatial layout."""

    def __init__(self, dim_in: int, dim_out: int, hidden_dim: int = 2048):
        super().__init__()
        self.dim_out = dim_out
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project each spatial token to the output channel dimension."""

        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> (b h w) c").contiguous()
        x = self.mlp(x)
        x = rearrange(
            x,
            "(B H W) C -> B C H W",
            C=self.dim_out,
            B=B,
            H=H,
            W=W,
        ).contiguous()
        return x

"""Block-wise FiLM conditioning layer for temporal feature modulation."""

import torch
import torch.nn as nn
from einops import rearrange, repeat


class BFiLM(nn.Module):
    """Apply FiLM-style modulation derived from conditioning tensors."""

    def __init__(self, x_C_dim: int, cond_dim: int, mlp_hidden_dim: int = 32):
        super(BFiLM, self).__init__()

        self.x_C_dim = x_C_dim
        self.cond_dim = cond_dim

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 2 * x_C_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """Modulate the input tensor x using FiLM parameters derived from cond."""

        assert x.shape[0] == cond.shape[0], "Batch size of x and cond must be the same"

        if len(x.shape) == 5:
            time_shape = True
            B, T, C, H, W = x.shape
        else:
            B, C, H, W = x.shape
            x = repeat(x, "B C H W -> B T C H W", T=1)
            time_shape = False

        _, K, D = cond.shape

        assert self.x_C_dim == C, (
            f"Input tensor channel dimension {C} must equal x_C_dim {self.x_C_dim}"
        )
        assert D == self.cond_dim, (
            f"Condition tensor dim {D} must match initialized cond_dim {self.cond_dim}"
        )

        cond = rearrange(cond, "b k d -> (b k) d")

        params = self.mlp(cond)

        # After MLP, reshape to (B, K, 2 * C)
        params = rearrange(params, "(b k) C -> b k C", b=B, k=K)

        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.mean(dim=1)
        beta = beta.mean(dim=1)

        gamma = repeat(gamma, "b c -> b t c h w", t=1, h=1, w=1)
        beta = repeat(beta, "b c -> b t c h w", t=1, h=1, w=1)

        x = x * gamma + beta

        if time_shape:
            return x
        x = rearrange(x, "B T C H W -> B (T C) H W")
        return x

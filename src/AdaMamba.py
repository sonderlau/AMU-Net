import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp
from timm.layers import DropPath
import math

# Import the selective scan kernel when available; otherwise keep the placeholder None
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    print("Warning: mamba_ssm not installed. Please install it for MambaMixer.")
    selective_scan_fn = None


# =========================================================
# Level 1: Mamba Mixer (core SSM operations)
# =========================================================
class MambaMixer(nn.Module):
    """Adaptive SSM-based mixer that processes spatial tokens with a selective scan."""
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )

        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # Initialize dt bias
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D parameters for the selective scan
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(self, hidden_states):
        """Run the mixer on a batch of token sequences.

        Args:
            hidden_states: Tensor of shape (B, L, D) representing a batch of sequences.
        Returns:
            Tensor with the same shape as hidden_states after the Mamba mixing and gating.
        """
        batch, seqlen, dim = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)  # (B, L, D_inner)

        # 1. Conv1d (Causal / Local)
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv1d(x)[:, :, :seqlen]  # Causal padding
        x = x.transpose(1, 2)
        x = F.silu(x)

        # 2. SSM Parameters
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        dt = self.dt_proj(dt)  # (B, L, D)
        x_dbl = x.transpose(1, 2)  # (B, D, L) for selective scan input
        dt = dt.transpose(1, 2)

        # 3. Selective Scan
        A = -torch.exp(self.A_log.float())  # (D, N)
        B = B.transpose(1, 2).contiguous()  # (B, N, L)
        C = C.transpose(1, 2).contiguous()  # (B, N, L)

        # Ensure contiguous for CUDA kernel
        x_dbl = x_dbl.contiguous()
        dt = dt.contiguous()

        y = selective_scan_fn(
            x_dbl,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None,
        )

        # 4. Gating & Output
        y = y.transpose(1, 2)  # (B, L, D)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out


# Utility helpers for windowed processing.

def modulate(x, shift, scale):
    """Apply FiLM-style shift and scale to normalized features."""
    return x * (1 + scale) + shift


def window_partition(x, window_size):
    """Partition a feature map into flattened spatial windows.

    Args:
        x: Tensor of shape (B, C, H, W).
        window_size: Spatial size of the square windows.
    Returns:
        Tensor of shape (B * num_windows, window_area, C) representing flattened windows.
    """
    B, C, H, W = x.shape
    x = rearrange(
        x, "b c (h p1) (w p2) -> (b h w) (p1 p2) c", p1=window_size, p2=window_size
    )
    return x


def window_reverse(windows, window_size, H, W):
    """Restore the original feature map shape from flattened windows.

    Args:
        windows: Tensor produced by :func:`window_partition`.
        window_size: Spatial size of the square windows.
        H: Height of the padded feature map.
        W: Width of the padded feature map.
    Returns:
        Tensor of shape (B, C, H, W) assembled from the windows.
    """
    B_nw, L, C = windows.shape
    x = rearrange(
        windows,
        "(b h w) (p1 p2) c -> b c (h p1) (w p2)",
        h=H // window_size,
        w=W // window_size,
        p1=window_size,
        p2=window_size,
    )
    return x


class AdaMambaBlock(nn.Module):
    """Windowed AdaMamba block that applies adaptive normalization and mixing."""
    def __init__(
        self,
        dim,  # input feature dimension
        cond_dim,  # dimension of the global time vector
        window_size=8,
        mlp_ratio=4.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        # 1. Normalization layers
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # 2. AdaLN modulation controller produces shift/scale/gate per branch
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, 6 * dim, bias=True)
        )
        # Init zero for identity start
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        # 3. Mixer (Pure Mamba)
        self.mixer = MambaMixer(d_model=dim)

        # 4. MLP
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, cond):
        """Run a single windowed AdaMamba block with adaptive modulation.

        Args:
            x: Feature map of shape (B, C, H, W).
            cond: Conditioning tensor of shape (B, cond_dim).
        """
        B, C, H, W = x.shape

        # Padding if needed
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))

        Hp, Wp = x.shape[2], x.shape[3]

        # 1. Calculate modulation parameters
        # (B, 6*dim) -> 6 * (B, 1, 1, dim) for broadcasting
        # After window partition x becomes (B_nw, L, dim), so params must repeat.

        params = self.adaLN_modulation(cond)  # (B, 6D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(
            6, dim=1
        )

        # Window Partition: (B, C, H, W) -> (B*nw, L, C)
        x_win = window_partition(x, self.window_size)

        # Repeat modulation parameters for each window
        # x_win shape: (B * num_windows, L, C)
        num_windows = x_win.shape[0] // B

        def expand_to_win(t):
            return repeat(t, "b d -> (b nw) 1 d", nw=num_windows)

        shift_msa = expand_to_win(shift_msa)
        scale_msa = expand_to_win(scale_msa)
        gate_msa = expand_to_win(gate_msa)
        shift_mlp = expand_to_win(shift_mlp)
        scale_mlp = expand_to_win(scale_mlp)
        gate_mlp = expand_to_win(gate_mlp)

        # 2. Mamba mixer block with AdaLN
        x_norm = self.norm1(x_win)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)

        # Residual + Gate * Mixer
        x_win = x_win + gate_msa * self.drop_path(self.mixer(x_modulated))

        # 3. MLP block with AdaLN
        x_norm = self.norm2(x_win)
        x_modulated = modulate(x_norm, shift_mlp, scale_mlp)

        x_win = x_win + gate_mlp * self.drop_path(self.mlp(x_modulated))

        # 4. Reassemble
        x = window_reverse(x_win, self.window_size, Hp, Wp)

        # Remove padding to restore original resolution
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W]

        return x


class AdaMambaBackbone(nn.Module):
    """Stack of AdaMamba blocks with final normalization."""
    def __init__(
        self,
        channels,  # number of input channels (typically the downsampled dimension)
        depth=8,  # number of stacked blocks
        window_size=8,
        embedding_dim=256,  # time embedding dimension
        mlp_ratio=4.0,
        drop_path_rate=0.2,
    ):
        super().__init__()

        self.channels = channels
        self.blocks = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        for i in range(depth):
            self.blocks.append(
                AdaMambaBlock(
                    dim=channels,
                    cond_dim=embedding_dim,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                )
            )

        self.final_norm = nn.LayerNorm(channels)

    def forward(self, x, cond):
        """Process the feature map through all AdaMamba blocks and normalize.

        Args:
            x: Input tensor of shape (B, C, H, W).
            cond: Conditioning tensor of shape (B, D).
        """

        assert x.shape[1] == self.channels

        # 2. Pass through Blocks
        for block in self.blocks:
            x = block(x, cond)

        # 3. Final Norm (Optional, useful for stability)
        # LayerNorm expects (B, ..., C), so we move C to last
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.final_norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        return x

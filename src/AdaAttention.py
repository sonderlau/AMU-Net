import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp
from timm.layers import DropPath
# The attention mixer path reuses the local Attention implementation
from src.Attention import Attention


# Attention-based AdaMamba variant with adaptive normalization


def modulate(x, shift, scale):
    """Apply FiLM-style shift and scale to normalized features."""
    return x * (1 + scale) + shift

def window_partition(x, window_size):
    """Partition a feature map into flattened spatial windows for attention mixing.

    Args:
        x: Tensor of shape (B, C, H, W).
        window_size: Size of each square window.
    Returns:
        Tensor of shape (B * num_windows, window_area, C) representing flattened windows.
    """
    B, C, H, W = x.shape
    x = rearrange(
        x, "b c (h p1) (w p2) -> (b h w) (p1 p2) c", p1=window_size, p2=window_size
    )
    return x


def window_reverse(windows, window_size, H, W):
    """Reconstruct the padded feature map from flattened windows."""
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


class AdaAttentionBlock(nn.Module):
    """Adaptive attention block that mixes modulated tokens via Attention."""
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

        # 2. AdaLN modulation controller
        # Input: (B, cond_dim) -> Output: (B, 6 * dim)
        # Why 6? (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        # Similar to DiT, the gate controls the residual strength.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, 6 * dim, bias=True)
        )
        # Init zero for identity start
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        # 3. Mixer (Attention-based)
        self.mixer = Attention(dim=dim)

        # 4. MLP
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, cond):
        """Run a single windowed attention block with AdaLN modulation.

        Args:
            x: Tensor of shape (B, C, H, W).
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

        # Repeat params for windows
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

        # 2. Attention mixer block with AdaLN
        # Norm -> Modulate -> Mixer -> Gate -> Residual
        x_norm = self.norm1(x_win)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)

        # Residual + Gate * Mixer
        x_win = x_win + gate_msa * self.drop_path(self.mixer(x_modulated))

        # 3. MLP Block with AdaLN
        x_norm = self.norm2(x_win)
        x_modulated = modulate(x_norm, shift_mlp, scale_mlp)

        x_win = x_win + gate_mlp * self.drop_path(self.mlp(x_modulated))

        # 4. Reverse
        x = window_reverse(x_win, self.window_size, Hp, Wp)

        # Unpad
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W]

        return x


class AdaAttentionBackbone(nn.Module):
    """Stacked AdaAttention blocks with final normalization."""
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
                AdaAttentionBlock(
                    dim=channels,
                    cond_dim=embedding_dim,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                )
            )

        self.final_norm = nn.LayerNorm(channels)

    def forward(self, x, cond):
        """Process the feature map through attention blocks and normalize.

        Args:
            x: Tensor of shape (B, C, H, W).
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

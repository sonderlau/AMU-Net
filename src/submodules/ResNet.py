from torch.nn import (
    Module,
    Conv2d,
    ReLU,
    SiLU,
    Sequential,
    Linear,
    Identity,
    ModuleList,
    Dropout,
)
from torch import Tensor
from src.ChannelLayerNorm import ChannelLayerNorm
from utils.Shortcuts import default_value, check_exists
from einops import repeat

def modulate(x: Tensor, shift: Tensor, scale: Tensor):
    """Apply per-channel scaling and shifting to the activation map."""

    shift = repeat(shift, "b c -> b c h w", h=1, w=1)
    scale = repeat(scale, "b c -> b c h w", h=1, w=1)

    return x * (1 + scale) + shift
    

class Block(Module):
    """Basic convolution + channel normalization unit."""

    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = Conv2d(dim, dim_out, 3, padding=1)
        self.norm = ChannelLayerNorm(dim_out)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)

        return x


class ResnetBlock(Module):
    """Residual block with optional conditioning for AdaLN-style modulation."""

    def __init__(self, dim, dropout_rate: float = 0.1, dim_out=None, cond_dim=None):
        super().__init__()
        dim_out = default_value(dim_out, dim)

        self.use_cond = cond_dim is not None

        self.block1 = Block(dim, dim_out)

        if self.use_cond:
            self.cond_proj = Sequential(
                SiLU(),
                Linear(cond_dim, dim_out * 2),
            )

        self.block2 = Block(dim_out, dim_out)
        self.res_conv = Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()
        self.drop_out = Dropout(p=dropout_rate)
        self.activation = ReLU()

    def forward(self, x: Tensor, cond: Tensor = None):
        """Process the input through two convolutional blocks plus optional modulation."""

        h = self.block1(x)

        if self.use_cond and cond is not None:
            params = self.cond_proj(cond)
            scale, shift = params.chunk(2, dim=1)

            h = modulate(x=h, shift=shift, scale=scale)

        h = self.activation(h)
        h = self.drop_out(h)
        h = self.block2(h)
        h = self.activation(h)

        return h + self.res_conv(x)


class ResnetBlocks(Module):
    """Stack multiple residual blocks for deeper representations."""

    def __init__(
        self, dim_out, dropout_rate: float = 0.1, dim_in=None, depth=1, cond_dim=None
    ):
        super().__init__()
        curr_dim = default_value(dim_in, dim_out)

        blocks = []
        for _ in range(depth):
            blocks.append(
                ResnetBlock(
                    dim=curr_dim,
                    dim_out=dim_out,
                    dropout_rate=dropout_rate,
                    cond_dim=cond_dim,
                )
            )
            curr_dim = dim_out

        self.blocks = ModuleList(blocks)

    def forward(self, x: Tensor, cond: Tensor = None):
        """Propagate the input through every residual block in sequence."""

        for block in self.blocks:
            x = block(x, cond)

        return x

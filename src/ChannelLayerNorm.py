"""Channel-wise LayerNorm implemented via GroupNorm."""

import torch.nn as nn


class ChannelLayerNorm(nn.Module):
    """Normalize each channel independently using GroupNorm."""

    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_channels, num_channels=num_channels, eps=eps)

    def forward(self, x):
        """Apply channel normalization to the input tensor."""
        return self.norm(x)

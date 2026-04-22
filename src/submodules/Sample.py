"""Sampling utilities for down-/up-sampling and spatial padding/cropping."""

from torch.nn import MaxPool2d, ConvTranspose2d
import torch.nn.functional as func
import torch.nn as nn


def downsample_2x():
    """Return a 2x downsampling max-pool module."""

    return MaxPool2d(kernel_size=2, stride=2, padding=0)


def upsample_2x(dim_in: int, dim_out: int):
    """Return a 2x transpose convolutional upsampling block."""

    return ConvTranspose2d(in_channels=dim_in, out_channels=dim_out, kernel_size=2, stride=2, padding=0)


def upsample_4x(dim_in: int, dim_out: int):
    """Return a 4x transpose convolutional upsampling block."""

    return ConvTranspose2d(in_channels=dim_in, out_channels=dim_out, kernel_size=4, stride=4, padding=0)


class Downsample(nn.Module):
    """Convolutional downsampling block with optional channel keeping."""

    def __init__(self, dim, keep_dim=False):
        """Initialize the downsampling block.

        Args:
            dim: Feature dimension before reduction.
            keep_dim: Maintain the same channel dimension if True.
        """

        super().__init__()
        dim_out = dim if keep_dim else 2 * dim
        self.reduction = nn.Sequential(nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False))

    def forward(self, x):
        """Apply convolutional reduction to the input tensor."""

        return self.reduction(x)


class CenterPad(nn.Module):
    """Pad spatial dimensions symmetrically to reach the target size."""

    def __init__(self, target_dim):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, x):
        """Pad x evenly around height and width to match target_dim."""

        target_dim = self.target_dim
        *_, height, width = x.shape
        assert target_dim >= height and target_dim >= width

        height_pad = target_dim - height
        width_pad = target_dim - width
        left_height_pad = height_pad // 2
        left_width_pad = width_pad // 2

        return func.pad(
            x,
            (
                left_width_pad,
                width_pad - left_width_pad,
                left_height_pad,
                height_pad - left_height_pad,
            ),
            value=0.0,
        )


class CenterCrop(nn.Module):
    """Crop the input tensor to the specified spatial dimension."""

    def __init__(self, crop_dim):
        """Initialize the center crop module.

        Args:
            crop_dim: Length of the square crop on each side.
        """

        super().__init__()
        self.crop_dim = crop_dim

    def forward(self, x):
        """Return the centered crop for the input tensor."""

        crop_dim = self.crop_dim
        *_, height, width = x.shape
        assert height >= crop_dim and width >= crop_dim

        cropped_height_start_idx = (height - crop_dim) // 2
        cropped_width_start_idx = (width - crop_dim) // 2

        height_slice = slice(cropped_height_start_idx, cropped_height_start_idx + crop_dim)
        width_slice = slice(cropped_width_start_idx, cropped_width_start_idx + crop_dim)
        return x[..., height_slice, width_slice]

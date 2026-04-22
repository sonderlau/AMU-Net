"""PatchGAN discriminator implementation used by AMU-Net."""

import functools
import torch
import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator that produces localized authenticity maps."""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Initialize the PatchGAN discriminator layers.

        Args:
            input_nc (int): Number of input channels (commonly radar reflectivity).
            ndf (int): Base number of filters.
            n_layers (int): Number of convolutional blocks.
            norm_layer (callable): Normalization layer to apply after convolutions.
        """
        super(NLayerDiscriminator, self).__init__()

        # Discriminator can operate on conditional inputs or just the future frames.
        # Here we assume input_nc already includes all required channels.

        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        # Final convolution produces a single-channel prediction map.
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Run input through the PatchGAN model."""

        return self.model(input)

"""Loss modules used by the AMU-Net generator and discriminator."""

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """Abstract common GAN loss variants to avoid manual label creation."""

    def __init__(
        self, gan_mode="vanilla", target_real_label=1.0, target_fake_label=0.0
    ):
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode

        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class SpectralLoss(nn.Module):
    """Spectral frequency loss that compares amplitude magnitudes."""

    def __init__(self, loss_weight=1.0, log_matrix=True):
        super(SpectralLoss, self).__init__()
        self.loss_weight = loss_weight
        self.log_matrix = log_matrix
        self.criterion = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.float().contiguous()
        target = target.float().contiguous()

        # 1. Apply FFT (rfft2 returns the non-redundant half for real inputs)
        z_pred = torch.fft.rfft2(pred, norm="ortho")
        z_target = torch.fft.rfft2(target, norm="ortho")

        # 2. Compare magnitudes between predictions and targets
        mag_pred = torch.abs(z_pred)
        mag_target = torch.abs(z_target)

        # 3. Optional log scaling can be enabled via log_matrix
        # if self.log_matrix:
        #     mag_pred = torch.log(mag_pred + 1e-8)
        #     mag_target = torch.log(mag_target + 1e-8)

        loss = self.criterion(mag_pred, mag_target)

        return loss * self.loss_weight

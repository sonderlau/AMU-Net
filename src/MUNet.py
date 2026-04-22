from typing import Any

from lightning.pytorch import LightningModule
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torch.nn import Identity, Dropout, Sequential, MSELoss
from timm.layers import LayerNorm2d, trunc_normal_
import torch.nn as nn
from torch import cat


from src.AdaMamba import AdaMambaBackbone
from src.submodules.Sample import CenterCrop, downsample_2x, upsample_2x
from src.submodules.ResNet import ResnetBlocks
from src.submodules.MLP import MLPDeterministic
from src.TimeInfoEmbedding import TimeInfoEmbedding

from src.Loss import SpectralLoss, GANLoss
from utils.Metrics import CSIMean, RMSE, SSIM, CSI, HeidkeSkillScore
from torchmetrics import MetricCollection

from src.Discriminator import NLayerDiscriminator

class MUNet(LightningModule):
    """Lightning module implementing a GAN-based MUNet for precipitation nowcasting."""
    def __init__(
        self,
        
        dim: int,
        embedding_dim: int,
        elapsed_time_num: int,
        lead_time_max: int,
        
        input_channels: int,
        resnet_depth: int = 2,
        dropout_rate: float = 0.1,
        
        mamba_depth: int = 4,
        mamba_window_size: int = 8,
        mamba_mlp_ratio: int = 3,
        mamba_drop_path_rate: float = 0.2,
        
        # Optimizer
        
        lambda_gan: float = 0.2,
        lambda_pixel: float = 0.5,
        lambda_spectral: float = 0.2,
        
        lr_model: float = 1e-4,
        lr_gan: float = 2e-4,
        
        gan_start_epoch: int = 10,
        

    ):
        """Initialize the MUNet GAN module with its encoder, backbone, and schedulers.

        Args:
            dim: Base channel dimension used throughout the ResNet stages.
            embedding_dim: Size of the embeddings produced by the time-conditioning module.
            elapsed_time_num: Number of elapsed time intervals encoded in the time embedding.
            lead_time_max: Maximum lead time that the embedding needs to support.
            input_channels: Number of radar channels (temporal slices) in the input tensor.
            resnet_depth: Number of ResNet blocks in each down- and up-sampling stage.
            dropout_rate: Dropout probability applied to the skip connection paths.
            mamba_depth: Number of layers in the AdaMamba backbone.
            mamba_window_size: Spatial window size used by AdaMamba mixing.
            mamba_mlp_ratio: Expansion ratio of the AdaMamba MLP layers.
            mamba_drop_path_rate: Drop path rate applied within the AdaMamba blocks.
            lambda_gan: Weight for the adversarial loss term.
            lambda_pixel: Weight for the pixel-wise reconstruction loss.
            lambda_spectral: Weight for the spectral consistency loss.
            lr_model: Learning rate for the generator optimizer.
            lr_gan: Learning rate for the discriminator optimizer.
            gan_start_epoch: Epoch at which the discriminator begins participating in training.
        """
        super().__init__()
        
        self.lambda_gan = lambda_gan
        self.lambda_pixel = lambda_pixel
        self.lambda_spectral = lambda_spectral
        self.lr_model = lr_model
        self.lr_gan = lr_gan
        
        self.gan_start = gan_start_epoch


        self.save_hyperparameters()

        self.channels = input_channels
        
        self.dim = dim

        self.time_embedding = TimeInfoEmbedding(
            embedding_dim=embedding_dim,
            elapsed_num=elapsed_time_num,
            lead_time_max=lead_time_max,
        )

        self.stage_1_down2x = downsample_2x()
        self.stage_1_down_resnet = ResnetBlocks(
            dim_in=input_channels,
            dim_out=dim,
            depth=resnet_depth,
            cond_dim=embedding_dim,
        )

        self.skip_connection_stage_1 = Sequential(
            Identity(), CenterCrop(crop_dim=192), Dropout(p=dropout_rate)
        )

        self.stage_2_down_resnet = ResnetBlocks(
            dim_in=dim,
            dim_out=dim,
            depth=resnet_depth,
            cond_dim=embedding_dim,
        )

        self.stage_2_down2x = downsample_2x()

        self.skip_connection_stage_2 = Sequential(
            Identity(), CenterCrop(crop_dim=96), Dropout(p=dropout_rate)
        )

        self.mamba_vision = AdaMambaBackbone(
            channels=dim,
            depth=mamba_depth,
            window_size=mamba_window_size,
            embedding_dim= embedding_dim,
            mlp_ratio=mamba_mlp_ratio,
            drop_path_rate=mamba_drop_path_rate
        )



        self.stage_2_up2x = upsample_2x(
            dim_in= 2 * dim,
            dim_out= 2 * dim,
        )
        self.stage_2_up_resnet = ResnetBlocks(
            dim_in=2 * dim,
            dim_out=dim,
            depth=resnet_depth,
            cond_dim=embedding_dim,
        )

        self.stage_1_up2x = upsample_2x(
            dim_in=dim + input_channels,
            dim_out=dim + input_channels,
        )

        self.stage_1_up_resnet = ResnetBlocks(
            dim_in=dim + input_channels,
            dim_out=dim,
            depth=resnet_depth,
            cond_dim=embedding_dim,
        )

        # Historical spelling is preserved so loading checkpoints/state_dicts keeps compatibility.
        self.predcition_header = MLPDeterministic(dim_in=dim, dim_out=1)
        
        self.criterionGAN = GANLoss()
        self.criterionSpectral = SpectralLoss()
        self.criterionPixel = MSELoss()
        
        self.discriminator = NLayerDiscriminator(input_nc=1)

        # Metrics
        
        self.metric_template = MetricCollection(
            {
                "HSS": HeidkeSkillScore(),
                "CSIMean": CSIMean(),
                "CSI-219": CSI(threshold=219),
                "CSI-181": CSI(threshold=181),
                "RMSE": RMSE(),
                "SSIM": SSIM()
            }
        )
        
        self.metric_train = self.metric_template.clone(prefix="train/")
        self.metric_val = self.metric_template.clone(prefix="val/")
        

        # init weights
        self.apply(self._init_weights)

        self.automatic_optimization = False

    @property
    def prediction_header(self) -> MLPDeterministic:
        """Expose the registered prediction head without creating a second module."""
        return self.predcition_header

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


    def forward(self, data: dict) -> Any:


        x = data["sequence"] # B, T, H, W

        time_embed = self.time_embedding(
            lead_time=data["lead_time"],
            elapsed = data["elapsed_time"]
        )



        skip_connection_s1 = self.skip_connection_stage_1(
            x
        )  # (B, T, 192, 192)

        x = self.stage_1_down2x(x)

        x = self.stage_1_down_resnet(x, cond=time_embed)  # (B, DIM, 192, 192)


        skip_connection_s2 = self.skip_connection_stage_2(
            x
        )  # (B, DIM, 96, 96)

        x = self.stage_2_down2x(x)

        x = self.stage_2_down_resnet(x, cond=time_embed)  # (B, DIM, 96, 96)


        x = self.mamba_vision(x, cond=time_embed)  # (B, DIM, H, W)

        x = cat((x, skip_connection_s2), dim=1)  # (B, DIM + DIM, 96, 96)

        x = self.stage_2_up2x(x)  # (B, M_DIM + DIM, 192, 192)
        
        x = self.stage_2_up_resnet(x, cond=time_embed)  # (B, DIM, 192, 192)

        x = cat((x, skip_connection_s1), dim=1) # (B, DIM + T, 192, 192)
        
        x = self.stage_1_up2x(x) # (B, DIM + T, 384, 384)

        x = self.stage_1_up_resnet(x, cond=time_embed)  # (B, DIM + T, 384, 384)



        pred = self.prediction_header(x)  # (B, 1, 384, 384)

        return {
            "pred": pred,
            "lead_time": data["lead_time"],
        }

    def training_step(self, batch, batch_idx) -> Any:

        opt_g, opt_d = self.optimizers()

        pred = self.forward(batch)["pred"]
        y_true = batch["target"]
        
        is_warmup = self.current_epoch < self.gan_start

        # ==================================================
        # Evaluate Discriminator
        # ==================================================

        if not is_warmup:

            pred_real = self.discriminator(y_true)
            loss_d_real = self.criterionGAN(pred_real, True)

            # 2. Fake loss: discriminator should mark generated samples as False.
            pred_fake = self.discriminator(pred.detach())
            loss_d_fake = self.criterionGAN(pred_fake, False)

            # Combined D Loss
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            
            loss_d = loss_d.contiguous()

            # Update D
            opt_d.zero_grad()
            self.manual_backward(loss_d)
            opt_d.step()
        else:
            loss_d = 0.0

        # ==================================================
        #  Evaluate Generator (G)
        # ==================================================

        # 2. Pixel-wise Loss (MAE/MSE)
        loss_g_pixel = self.criterionPixel(pred, y_true)

        # 3. Spectral loss: encourage sharpness and accurate extremes in the prediction.
        loss_g_spectral = self.criterionSpectral(pred, y_true)
        
        if not is_warmup:
            # 1. GAN loss (adversarial): encourage the discriminator to accept generated samples as True.
            pred_fake_g = self.discriminator(pred)
            loss_g_gan = self.criterionGAN(pred_fake_g, True).contiguous()
        else:
            loss_g_gan = torch.tensor(0.0, device=self.device)
            

        # Combined G Loss
        loss_g = (
            self.lambda_gan * loss_g_gan
            + self.lambda_pixel * loss_g_pixel
            + self.lambda_spectral * loss_g_spectral
        )

        # Update G
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()
        
        
        # Loss logging
        
        self.log("train/loss_disc", loss_d, sync_dist=True)
        self.log("train/loss_gan", loss_g_gan, sync_dist=True)
        self.log("train/loss_pixel", loss_g_pixel, sync_dist=True)
        self.log("train/loss_spectral", loss_g_spectral, sync_dist=True)
        self.log("train/loss", loss_g, sync_dist=True)

        self.metric_train(pred * 255, y_true * 255)
        
        self.log_dict(self.metric_train, sync_dist=True)
        
    def validation_step(self, batch, batch_idx) -> Any:
        

        pred = self.forward(batch)["pred"]
        y_true = batch["target"]

        # ==================================================
        # Train Discriminator
        # ==================================================

        pred_real = self.discriminator(y_true)
        loss_d_real = self.criterionGAN(pred_real, True)

        # 2. Fake loss: encourage the discriminator to classify generated samples as False.
        pred_fake = self.discriminator(pred.detach())
        loss_d_fake = self.criterionGAN(pred_fake, False)

        # Combined D Loss
        loss_d = (loss_d_real + loss_d_fake) * 0.5



        # ==================================================
        #  Train Generator (G)
        # ==================================================
        # 1. GAN loss (adversarial): encourage the discriminator to accept the generator output as True.
        pred_fake_g = self.discriminator(pred)
        loss_g_gan = self.criterionGAN(pred_fake_g, True)

        # 2. Pixel-wise Loss (MAE/MSE)
        loss_g_pixel = self.criterionPixel(pred, y_true)

        # 3. Spectral loss: encourage accurate extremes and sharp features.
        loss_g_spectral = self.criterionSpectral(pred, y_true)

        # Combined G Loss
        loss_g = (
            self.lambda_gan * loss_g_gan
            + self.lambda_pixel * loss_g_pixel
            + self.lambda_spectral * loss_g_spectral
        )
        
        self.log("val/loss_disc", loss_d, sync_dist=True)
        self.log("val/loss_gan", loss_g_gan, sync_dist=True)
        self.log("val/loss_pixel", loss_g_pixel, sync_dist=True)
        self.log("val/loss_spectral", loss_g_spectral, sync_dist=True)
        self.log("val/loss", loss_g, sync_dist=True)


        self.metric_val(pred * 255, y_true * 255)
        
        self.log_dict(self.metric_val, sync_dist=True)
        


    def configure_optimizers(self):
        # Separate generator and discriminator parameters.
        # The generator gets every parameter except the discriminator modules defined above.
        d_params = list(self.discriminator.parameters())
        g_params = [
            p for n, p in self.named_parameters() if not n.startswith("discriminator")
        ]

        # Define optimizers with the standard GAN beta values.
        opt_g = AdamW(g_params, lr=self.lr_model, betas=(0.5, 0.999), weight_decay=1e-2)
        opt_d = AdamW(d_params, lr=self.lr_gan, betas=(0.5, 0.999), weight_decay=1e-2)

        # Define cosine annealing schedulers for both optimizers.
        sch_g = CosineAnnealingLR(opt_g, T_max=300, eta_min=1e-8)
        sch_d = CosineAnnealingLR(opt_d, T_max=300, eta_min=1e-8)

        # Return format expected by Lightning: list of optimizers and list of schedulers.
        return [opt_g, opt_d], [sch_g, sch_d]

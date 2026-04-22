"""Training entrypoint that configures and launches AMU-Net training."""

from lightning import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from torch import set_float32_matmul_precision

from src.MUNet import MUNet
from data.sevir_type_dataset import SEVIRDataModule


set_float32_matmul_precision("medium")

# 42 is the answer to everything
seed_everything(42)


dataset = SEVIRDataModule(
    catalog_path="/data3/SEVIR/CATALOG.csv",
    data_root="/data3/SEVIR",
    event_types="",  # include all event types
    split_date="2019-06-01",  # data before 2019 is used for training
    batch_size=12,
    num_workers=128,
    seed=42,
    input_frames=12,
    lead_times=(10, 20, 30, 60, 120),
)


callbacks = [
    # EarlyStopping("val/loss", min_delta=1e-7, patience=10),
    ModelSummary(max_depth=3),
    ModelCheckpoint(
        dirpath="weights",
        verbose=True,
        filename="MUNet-{epoch}",
        save_top_k=1,
        save_on_train_epoch_end=True,
        every_n_epochs=1,
        monitor="val/loss",
        save_last=True,
    ),
]


logger = CSVLogger(save_dir="exps", name="MUNet")


trainer = Trainer(
    accelerator="cuda",
    precision="bf16-mixed",
    strategy=DDPStrategy(find_unused_parameters=True),
    devices=[0, 1, 2, 3],
    callbacks=callbacks,
    log_every_n_steps=50,
    max_epochs=300,
    logger=logger,
    # profiler="advanced",
    # fast_dev_run=True,
    # detect_anomaly=True,
)


model = MUNet(
    dim=256,
    embedding_dim=128,
    elapsed_time_num=2,
    lead_time_max=125,
    input_channels=6,
    resnet_depth=2,
    mamba_depth=6,
    mamba_window_size=8,
    mamba_mlp_ratio=3,
    lambda_gan=0.02,
    lambda_pixel=1.0,
    lambda_spectral=0.5,
    lr_model=1e-4,
    lr_gan=2e-4,
)


trainer.fit(model=model, datamodule=dataset)

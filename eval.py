"""Evaluate deterministic baselines on SEVIR with standard metrics."""

import torch
from torch import set_float32_matmul_precision

from lightning.pytorch import seed_everything
from torchmetrics import MetricCollection
from rich.console import Console

from utils.Metrics import CSI, CSIMean, RMSE, SSIM, HeidkeSkillScore, PeakRatio
from utils.model_factory import ModelFactory
from utils.statistic import print_result
from data.sevir_type_dataset import SEVIRDataModule


set_float32_matmul_precision("high")
console = Console()
LEAD_TIMES = (10, 20, 30, 60, 120)


seed_everything(42)


DEVICE = "cuda:7"

CHECKPOINTS = {
    # "UNet": "checkpoints/UNet/last.ckpt",
    # "ConvLSTM": "checkpoints/ConvLSTM/last.ckpt",
    "SimVP": "checkpoints/SimVP-10/last.ckpt",
    "SmaAt": "checkpoints/SmaAt-UNet/last.ckpt",
    # "MUNet": "",
    # "pySTEPS": "",
}

METRICS_NAME = (
    "rmse",
    "ssim",
    "hss",
    "peak",
    "csi_mean",
    "csi_h",
    "csi_e",
)


for model_name, ckpt_path in CHECKPOINTS.items():
    model = ModelFactory(
        model_name,
        ckpt_path=ckpt_path,
        device=DEVICE if model_name != "pySTEPS" else "",
    )

    metric_collection = {
        f"lead_{h}": (
            MetricCollection(
                {
                    "rmse": RMSE(),
                    "ssim": SSIM(),
                    "hss": HeidkeSkillScore(),
                    "csi_mean": CSIMean(),
                    "peak": PeakRatio(),
                    "csi_h": CSI(threshold=181),
                    "csi_e": CSI(threshold=219),
                }
            ).to(DEVICE)
        )
        for h in LEAD_TIMES
    }

    for lt in LEAD_TIMES:
        dataset = SEVIRDataModule(
            catalog_path="/data3/SEVIR/CATALOG.csv",
            data_root="/data3/SEVIR",
            event_types="",
            split_date="2019-06-01",
            batch_size=15,
            num_workers=128,
            seed=42,
            input_frames=12,
            lead_times=(lt, ),
        )

        dataset.setup("fit")

        with torch.no_grad():
            for batch in dataset.val_dataloader():
                sequence = batch["sequence"].to(DEVICE)
                target = batch["target"].to(DEVICE)

                for _ in range(int(lt / 10)):
                    pred = model(sequence)
                    sequence = torch.cat([sequence, pred], dim=1)[:, 1:, :, :]

                collection = metric_collection[f"lead_{lt}"]
                for metric_value in collection.values():
                    metric_value.update(pred * 255, target * 255)

    table = print_result(
        model_name=model_name,
        metric_collection=metric_collection,
        metric_names=METRICS_NAME,
        lead_times=LEAD_TIMES,
    )
    console.print(table)
    print("")

    del model
    torch.cuda.empty_cache()

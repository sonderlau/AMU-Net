"""Inference script for ProbMamba that logs CSV metrics for US weather data."""

from datetime import timedelta

import torch
from model.ProbMamba import ProbMamba
from US_Dataset import USWeatherData
from utils.CSVWriter import CSVWriter
from utils.Transform import pred_topk_mae, probability_to_value
from utils.csi import compute_csi, compute_mae

WINDOWS = [10]  # 20, 30, 60, 120, 180
MRMS_WINDOW = [0, 2, 4, 6, 8, 30]
ASOS_WINDOW = [0, 5, 10, 15, 20, 30]
DEVICE = "cuda:4"

model = ProbMamba.load_from_checkpoint("./weights/ProbMamba-v1.ckpt").to(DEVICE)
model.eval()

writer = CSVWriter(
    csv_file="ProbMamba.csv",
    columns=["10", "20", "30", "60", "120", "180"],
    append=True,
)

for w in WINDOWS:


    PREDICTION_WINDOW = [w]
    dataset = USWeatherData(
        date_start="2022-06-02",
        date_range=timedelta(days=180),
        batch_size=10,
        num_workers=128,
        MRMS_look_back_window=MRMS_WINDOW,
        ASOS_look_back_window=ASOS_WINDOW,
        prediction_window= PREDICTION_WINDOW
    )





    with torch.no_grad():
        for data in dataset.val_dataloader():
            
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(DEVICE)

            pred = model(data)
            
            data_list = [
                {
                    "name": "mrms_rate",
                    "pred": pred["mrms_512_bins"][:, 0, :, :, :],
                    "truth": data["mrms_1km_output"][:, 0:1, :, :],
                    "bin_size": 0.2,
                },
                # {
                #     "name": "mrms_accumulation",
                #     "pred": pred["mrms_512_bins"][:, 1, :, :, :],
                #     "truth": data["mrms_1km_output"][:, 1:2, :, :],
                #     "bin_size": 0.2,
                # },
                # {
                #     "name": "asos_wind_speed",
                #     "pred": pred["asos_256_bins"][:, 0, :, :, :],
                #     "truth": data["asos_output"][:, 1:2, :, :],
                #     "bin_size": 0.1,
                # },
                # {
                #     "name": "asos_air_temperature",
                #     "pred": pred["asos_256_bins"][:, 1, :, :, :],
                #     "truth": data["asos_output"][:, 2:3, :, :],
                #     "bin_size": 1,
                # },
                # {
                #     "name": "asos_dew_point_temperature",
                #     "pred": pred["asos_256_bins"][:, 2, :, :, :],
                #     "truth": data["asos_output"][:, 3:4, :, :],
                #     "bin_size": 0.1,
                # },
            ]
            
            for prediction in data_list:

                truth = prediction["truth"]
                pred_value = prediction["pred"]

                raw_mask = (truth >= 0) & (truth != float("inf"))
                mask = raw_mask.float()

                # mae = pred_topk_mae(
                #     1, pred_value, truth, mask, prediction["bin_size"]
                # )
                mae = compute_mae(
                    pred=probability_to_value(pred_value, bin_size=prediction["bin_size"], bins=pred_value.shape[1]),
                    truth=truth
                )
                
                mae_3 = pred_topk_mae(
                    3, pred_value, truth, mask, prediction["bin_size"]
                )

                lead_time = data["lead_time"][0].item()
                
                writer.write_csv(f"{lead_time}", f"mae_1/{mae}")
                writer.write_csv(f"{lead_time}", f"mae_3/{mae_3}")

                
                if prediction["name"] == "mrms_rate":
                    pred_mrms_rate = probability_to_value(pred=prediction["pred"], bin_size=prediction["bin_size"], bins=512)
                    
                    print(pred_mrms_rate.max(), pred_mrms_rate.min(), truth.max())
                    
                    csi_1 = compute_csi(pred=pred_mrms_rate, truth=prediction["truth"], threshold=1.0)
                    csi_4 = compute_csi(pred=pred_mrms_rate, truth=prediction["truth"], threshold=4.0)
                    csi_8 = compute_csi(pred=pred_mrms_rate, truth=prediction["truth"], threshold=8.0)
                    
                    writer.write_csv(f"{lead_time}", f"csi_1/{csi_1.item()}")
                    writer.write_csv(f"{lead_time}", f"csi_4/{csi_4.item()}")
                    writer.write_csv(f"{lead_time}", f"csi_8/{csi_8.item()}")

                break
    print("Done! ", w)
        

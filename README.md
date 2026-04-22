# AMU-Net

## Overview

AMU-Net is a Lightning-based generative adversarial U‑Net model for nowcasting weather radar reflectivity frames on the SEVIR dataset. It integrates a ResNet encoder–decoder architecture with an AdaMamba adaptive mixing backbone and performs manual adversarial optimization during training.

## Highlights

- **Hybrid U‑Net and AdaMamba backbone**: combines skip‑connected ResNet feature extraction with selective AdaMamba mixing for spatiotemporal modeling.
- **Manual GAN optimization**: uses separate generator and discriminator optimizers with cosine annealing schedulers and manual optimization steps.
- **Multiple lead times**: supports deterministic nowcasting at 10, 20, 30, 60, and 120‑minute horizons.
- **Lightweight research snapshot**: focuses on code readability and reproducibility without altering core model semantics.

## Method Overview

AMU-Net’s forward pass applies a time embedding through a sequence of downsampling ResNet blocks, an AdaMamba backbone that adaptively mixes features with a selective-scan Mamba mixer, and skip-connected upsampling stages. The model’s generator and discriminator are trained with alternating manual steps controlled by a warmup period (`gan_start_epoch`).

## Repository Structure

```
train.py            # Training entrypoint (Lightning)
eval.py             # Baseline evaluation script
test.py             # ProbMamba inference & metrics (external dependencies)
src/
  MUNet.py          # Core LightningModule implementation
  AdaMamba.py       # AdaMamba backbone and Mamba mixer
  AdaAttention.py   # Alternate attention‑based backbone variant
  TimeInfoEmbedding.py  # Time conditioning embedding module
  Loss.py           # Loss definitions for generator and discriminator
  Discriminator.py  # Discriminator network for GAN training
  Attention.py      # Attention utilities for AdaAttention
  BFiLM.py          # Block‑wise FiLM conditioning layer
  ChannelLayerNorm.py  # Channel‑wise LayerNorm implementation
  submodules/
    ResNet.py       # ResNet building blocks
    Sample.py       # Sampling utilities
    MLP.py          # MLP prediction head
```

## Environment and Dependencies

- PyTorch, Lightning, timm, einops, and torchmetrics are used by the checked-in model code.
- External dataset and utility modules are **not** included in this snapshot, including `data.sevir_type_dataset.SEVIRDataModule`, `utils.Metrics`, `utils.model_factory`, `utils.statistic`, `model.ProbMamba`, `US_Dataset`, `utils.CSVWriter`, `utils.Transform`, and `utils.csi`.
- `mamba_ssm` is required for the real AdaMamba path at runtime; the import location only emits a warning when the package is missing.

## Data and Checkpoints

- SEVIR data catalog (external): `/data3/SEVIR/CATALOG.csv`
- SEVIR data root directory: `/data3/SEVIR`
- Training writes checkpoints to `weights/` and logs to `exps/`
- Evaluation expects checkpoints under `checkpoints/...`

## Usage

Run the verified entrypoints directly (assuming required external modules and data are available):
```bash
python train.py    # training with bf16‑mixed precision on devices [0,1,2,3]
python eval.py     # baseline evaluation on cuda:7
python test.py     # ProbMamba inference on cuda:4
```

## Reproducibility Notes

- Seeds are fixed to 42 for training and evaluation for deterministic runs.
- Hard‑coded device and path settings preserve original experiment configurations.
- This snapshot omits external dependencies and datasets; full reproduction requires restoring data.sevir_type_dataset, utils.Metrics, and other modules.

## License

This repository is provided for research and benchmarking purposes under the MIT License. See `LICENSE` for details.

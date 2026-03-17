# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flow Matching + DiT (Diffusion Transformer) training on ImageNet subset and CIFAR-10.

## Project Origin

This project is primarily built by **Claude Code** + **GLM-4.7**.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `config.py` - Training configuration (eval_interval, wandb settings, etc.)
- `model.py` - DiT model implementation
- `flow_matching.py` - Flow Matching loss and sampling
- `dataset.py` - ImageNet subset and CIFAR-10 data loader
- `train.py` - Main training script with wandb logging
- `utils.py` - Utility functions

## Key Features

- **Flow Matching**: Uses rectified flow for image generation
- **DiT Architecture**: Transformer-based diffusion model
- **WandB Monitoring**: Tracks train_loss, val_loss, and evaluation metrics
- **Eval Interval**: Configurable evaluation during training via `eval_interval`
- **CIFAR-10 Support**: Real small dataset for quick validation
- **ImageNet Subset**: Smaller subset for faster training/testing

## Training

### Using CIFAR-10 (recommended for quick validation)

```bash
python train.py --use_cifar10 --wandb_project your_project
```

### Using ImageNet subset

```bash
python train.py --wandb_project your_project
```

### Using synthetic data (for testing)

```bash
python train.py --use_synthetic --wandb_project your_project
```

### Resume training from checkpoint

```bash
python train.py --use_cifar10 --resume checkpoints/checkpoint_epoch_50.pt
```

## Training Configuration

Key configuration options in `config.py`:

- `eval_interval` - Run evaluation every N steps (default: 1000)
- `save_interval` - Save checkpoint every N steps (default: 5000)
- `log_interval` - Log training metrics every N steps (default: 100)
- `num_epochs` - Total number of training epochs (default: 100)
- `batch_size` - Batch size for training (default: 32)
- `learning_rate` - Initial learning rate (default: 1e-4)
- `warmup_steps` - Learning rate warmup steps (default: 5000)

## Training Results

When training on CIFAR-10 with 100 epochs:
- Initial loss: ~1.15
- Final loss: ~0.24
- Loss reduction: ~79%

## WandB Monitoring

The following metrics are logged to WandB:
- `train_loss` - Training loss per step
- `train_loss_avg` - Average training loss
- `val_loss` - Validation loss
- `learning_rate` - Current learning rate
- `samples` - Generated sample images

## Development Notes

- The project was developed iteratively with Claude Code for architecture design and implementation
- GLM-4.7 was used for code assistance and debugging
- All flow matching and DiT implementations follow standard research practices
- CIFAR-10 dataset was used as a quick validation baseline

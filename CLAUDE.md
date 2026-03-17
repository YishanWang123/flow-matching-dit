# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flow Matching + DiT (Diffusion Transformer) training on ImageNet subset.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `config.py` - Training configuration (eval_interval, wandb settings, etc.)
- `model.py` - DiT model implementation
- `flow_matching.py` - Flow Matching loss and sampling
- `dataset.py` - ImageNet subset data loader
- `train.py` - Main training script with wandb logging
- `utils.py` - Utility functions

## Key Features

- **Flow Matching**: Uses rectified flow for image generation
- **DiT Architecture**: Transformer-based diffusion model
- **WandB Monitoring**: Tracks train_loss, val_loss, and evaluation metrics
- **Eval Interval**: Configurable evaluation during training via `eval_interval`
- **ImageNet Subset**: Smaller subset for faster training/testing

## Training

```bash
python train.py --config config.yaml --wandb_project your_project
```

import torch
import os
import json
import logging
from typing import Dict, Any
import wandb


def setup_logging(log_file: str = "training.log"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss,
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: str = "cuda",
):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_model_info(model: torch.nn.Module, logger):
    """Log model information."""
    num_params = count_parameters(model)
    logger.info(f"Model has {num_params:,} trainable parameters")

    # Log model architecture summary
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()

    logger.info(f"Total parameters: {total_params:,}")


def init_wandb(
    config: Dict[str, Any],
    project: str,
    entity: str = None,
    run_name: str = None,
):
    """Initialize WandB logging."""
    wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
    )
    return wandb


def save_config(config: Dict[str, Any], save_path: str = "config.json"):
    """Save configuration to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    return save_path


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize images from [-1, 1] to [0, 1]."""
    return (tensor + 1.0) / 2.0


def save_images(
    images: torch.Tensor,
    save_path: str,
    nrow: int = 8,
    denorm: bool = True,
):
    """Save a grid of images."""
    from torchvision.utils import save_image

    if denorm:
        images = denormalize(images)

    save_image(images.clamp(0, 1), save_path, nrow=nrow)
    return save_path


def cosine_lr_schedule(
    optimizer: torch.optim.Optimizer,
    step: int,
    warmup_steps: int,
    max_steps: int,
    base_lr: float,
):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        lr = base_lr * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

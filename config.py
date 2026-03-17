import dataclasses
from typing import Optional


@dataclasses.dataclass
class TrainingConfig:
    """Training configuration for Flow Matching + DiT."""

    # Model
    model_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    patch_size: int = 4
    num_classes: int = 1000
    latent_dim: int = 4

    # Flow Matching
    num_steps: int = 1000
    sigma_min: float = 0.002
    sigma_max: float = 80.0

    # Training
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 5000
    eval_interval: int = 1000  # Evaluate every N steps
    save_interval: int = 5000  # Save checkpoint every N steps
    log_interval: int = 100  # Log training metrics every N steps

    # Data
    image_size: int = 32  # Use smaller images for faster training
    data_path: str = "./imagenet_subset"
    num_workers: int = 4

    # WandB
    wandb_project: str = "flow-matching-dit"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Misc
    device: str = "cuda"
    seed: int = 42
    checkpoint_dir: str = "./checkpoints"

    def __post_init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

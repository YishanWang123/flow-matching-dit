import argparse
import os
import sys
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import wandb
from config import TrainingConfig
from model import DiT
from flow_matching import FlowMatching
from dataset import create_dataloaders, create_synthetic_dataset, create_cifar10_dataloaders
from utils import (
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    log_model_info,
    init_wandb,
    save_config,
    denormalize,
    save_images,
    cosine_lr_schedule,
    AverageMeter,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Flow Matching + DiT")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--wandb_project", type=str, default="flow-matching-dit", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--use_synthetic", action="store_true", help="Use synthetic data instead of ImageNet")
    parser.add_argument("--use_cifar10", action="store_true", help="Use CIFAR-10 dataset (real small dataset)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def evaluate(
    model: nn.Module,
    flow_matching: FlowMatching,
    val_loader: DataLoader,
    device: str,
    logger: logging.Logger,
    step: int,
    config: TrainingConfig,
):
    """Evaluate the model on validation set."""
    model.eval()

    val_loss_meter = AverageMeter()
    num_batches = 0

    logger.info("Running validation...")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            x = x.to(device)
            y = y.to(device) if config.num_classes > 0 else None

            # Compute loss
            loss, info = flow_matching.compute_loss(model, x, y)

            val_loss_meter.update(loss.item(), x.size(0))
            num_batches += 1

            # Limit validation for faster training
            if num_batches >= 50:
                break

    val_loss = val_loss_meter.avg

    logger.info(f"Validation - Step {step}: Loss = {val_loss:.6f}")

    # Log to WandB
    wandb.log({
        "val_loss": val_loss,
        "step": step,
    })

    return val_loss


def train_epoch(
    model: nn.Module,
    flow_matching: FlowMatching,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    config: TrainingConfig,
    logger: logging.Logger,
):
    """Train for one epoch."""
    model.train()

    train_loss_meter = AverageMeter()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (x, y) in enumerate(progress_bar):
        x = x.to(device)
        y = y.to(device) if config.num_classes > 0 else None

        # Forward pass
        loss, info = flow_matching.compute_loss(model, x, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        train_loss_meter.update(loss.item(), x.size(0))
        global_step = epoch * len(train_loader) + batch_idx

        # Update learning rate
        if config.warmup_steps > 0:
            lr = cosine_lr_schedule(
                optimizer,
                global_step,
                config.warmup_steps,
                config.num_epochs * len(train_loader),
                config.learning_rate,
            )

        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss.item():.6f}",
            "avg_loss": f"{train_loss_meter.avg:.6f}",
        })

        # Log training metrics
        if global_step % config.log_interval == 0:
            logger.info(f"Step {global_step}: Loss = {loss.item():.6f}, Avg Loss = {train_loss_meter.avg:.6f}")

            wandb.log({
                "train_loss": loss.item(),
                "train_loss_avg": train_loss_meter.avg,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "step": global_step,
            })

        # Evaluation during training
        if global_step % config.eval_interval == 0 and global_step > 0:
            logger.info(f"Starting evaluation at step {global_step}")
            val_loss = evaluate(model, flow_matching, train_loader, device, logger, global_step, config)
            model.train()

        # Save checkpoint
        if global_step % config.save_interval == 0 and global_step > 0:
            checkpoint_path = save_checkpoint(
                model,
                optimizer,
                epoch,
                global_step,
                train_loss_meter.avg,
                config.checkpoint_dir,
                f"checkpoint_step_{global_step}.pt",
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            wandb.save(checkpoint_path)

    return train_loss_meter.avg, global_step


def sample_and_log(
    model: nn.Module,
    flow_matching: FlowMatching,
    device: str,
    step: int,
    config: TrainingConfig,
    logger: logging.Logger,
    num_samples: int = 16,
):
    """Sample images and log to WandB."""
    model.eval()

    logger.info(f"Generating {num_samples} samples at step {step}...")

    with torch.no_grad():
        samples = flow_matching.sample(
            model,
            shape=(num_samples, 3, config.image_size, config.image_size),
            y=None,
            num_steps=50,  # Use fewer steps for faster sampling during training
            device=device,
            progress=False,
        )

        # Denormalize to [0, 1]
        samples = denormalize(samples)

        # Create grid and log to WandB
        grid = make_grid(samples.clamp(0, 1), nrow=4)
        wandb.log({
            "samples": wandb.Image(grid),
            "step": step,
        })

        # Also save locally
        save_dir = os.path.join(config.checkpoint_dir, "samples")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"samples_step_{step}.png")
        save_images(samples, save_path, nrow=4, denorm=False)
        logger.info(f"Saved samples to {save_path}")

    model.train()


def main():
    # Parse arguments
    args = parse_args()

    # Setup configuration
    config = TrainingConfig()
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # Override with command line arguments
    config.wandb_project = args.wandb_project
    config.wandb_entity = args.wandb_entity
    config.wandb_run_name = args.wandb_run_name

    # Setup logging
    logger = setup_logging()
    logger.info("Starting training...")
    logger.info(f"Configuration: {config}")

    # Set random seed
    set_seed(config.seed)

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Save configuration
    save_config(vars(config), os.path.join(config.checkpoint_dir, "config.json"))

    # Initialize WandB
    init_wandb(
        config=vars(config),
        project=config.wandb_project,
        entity=config.wandb_entity,
        run_name=config.wandb_run_name,
    )
    logger.info(f"WandB initialized: {config.wandb_project}")

    # Create dataloaders
    if args.use_cifar10:
        logger.info("Using CIFAR-10 dataset (real small dataset)")
        config.num_classes = 10  # CIFAR-10 has 10 classes
        config.image_size = 32  # CIFAR-10 images are 32x32
        train_loader, val_loader = create_cifar10_dataloaders(
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            num_train_samples=10000,  # Limit to 10k samples for faster training
            num_val_samples=1000,     # Limit to 1k samples for faster validation
        )
    elif args.use_synthetic:
        logger.info("Using synthetic dataset")
        train_loader, val_loader = create_synthetic_dataset(
            num_samples=10000,
            image_size=config.image_size,
            batch_size=config.batch_size,
        )
    else:
        logger.info(f"Loading dataset from {config.data_path}")
        try:
            train_loader, val_loader = create_dataloaders(
                data_path=config.data_path,
                image_size=config.image_size,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                num_train_samples=None,
                num_val_samples=None,
            )
        except Exception as e:
            logger.warning(f"Failed to load dataset: {e}")
            logger.info("Falling back to synthetic dataset...")
            train_loader, val_loader = create_synthetic_dataset(
                num_samples=10000,
                image_size=config.image_size,
                batch_size=config.batch_size,
            )

    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Val loader: {len(val_loader)} batches")

    # Create model
    model = DiT(
        input_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=3,
        hidden_dim=config.model_dim,
        depth=config.num_layers,
        num_heads=config.num_heads,
        num_classes=config.num_classes,
        learn_sigma=False,
    ).to(config.device)

    log_model_info(model, logger)

    # Create Flow Matching
    flow_matching = FlowMatching(
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        num_steps=config.num_steps,
    )

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    start_step = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, start_step, _ = load_checkpoint(
            model,
            optimizer,
            args.resume,
            config.device,
        )
        logger.info(f"Resumed from epoch {start_epoch}, step {start_step}")

    # Log initial samples
    logger.info("Generating initial samples...")
    sample_and_log(model, flow_matching, config.device, 0, config, logger)

    # Training loop
    logger.info("Starting training loop...")
    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"{'='*50}")
        logger.info(f"Epoch {epoch}/{config.num_epochs}")

        train_loss, global_step = train_epoch(
            model,
            flow_matching,
            train_loader,
            optimizer,
            config.device,
            epoch,
            config,
            logger,
        )

        logger.info(f"Epoch {epoch} completed. Average loss: {train_loss:.6f}")

        # Save epoch checkpoint
        checkpoint_path = save_checkpoint(
            model,
            optimizer,
            epoch,
            global_step,
            train_loss,
            config.checkpoint_dir,
            f"checkpoint_epoch_{epoch}.pt",
        )
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        wandb.save(checkpoint_path)

        # Sample and log images at end of epoch
        sample_and_log(model, flow_matching, config.device, global_step, config, logger)

    # Final evaluation
    logger.info("Running final evaluation...")
    val_loss = evaluate(model, flow_matching, val_loader, config.device, logger, global_step, config)

    # Final samples
    logger.info("Generating final samples...")
    sample_and_log(model, flow_matching, config.device, global_step, config, logger, num_samples=64)

    # Save final model
    final_checkpoint_path = save_checkpoint(
        model,
        optimizer,
        config.num_epochs,
        global_step,
        val_loss,
        config.checkpoint_dir,
        "final_model.pt",
    )
    logger.info(f"Saved final model: {final_checkpoint_path}")
    wandb.save(final_checkpoint_path)

    logger.info("Training completed!")
    wandb.finish()


if __name__ == "__main__":
    main()

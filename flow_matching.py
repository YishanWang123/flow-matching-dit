import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from tqdm import tqdm


class FlowMatching:
    """Rectified Flow for Flow Matching."""

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        num_steps: int = 1000,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_steps = num_steps

    def compute_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Flow Matching loss.

        Args:
            model: The DiT model
            x: Input images (B, C, H, W)
            y: Class labels (optional)

        Returns:
            loss: The flow matching loss
            info: Dictionary with additional info
        """
        B = x.shape[0]
        device = x.device

        # Sample random timestep
        t = torch.rand(B, device=device)  # (B,)

        # Sample noise
        noise = torch.randn_like(x)  # (B, C, H, W)

        # Compute interpolation
        x_t = (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * x  # (B, C, H, W)

        # Compute velocity target
        v_target = x - noise  # (B, C, H, W)

        # Predict velocity
        v_pred = model(x_t, t, y)  # (B, C, H, W)

        # Compute loss (MSE)
        loss = F.mse_loss(v_pred, v_target)

        # Additional info
        info = {
            "t": t.mean().item(),
            "v_norm": v_pred.norm().item(),
        }

        return loss, info

    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, int, int, int],
        y: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        device: str = "cuda",
        progress: bool = False,
    ) -> torch.Tensor:
        """
        Sample from the model using Euler solver.

        Args:
            model: The DiT model
            shape: Shape of the samples (B, C, H, W)
            y: Class labels (optional)
            num_steps: Number of sampling steps (default: self.num_steps)
            device: Device to sample on
            progress: Whether to show progress bar

        Returns:
            samples: Generated images (B, C, H, W)
        """
        num_steps = num_steps or self.num_steps
        B, C, H, W = shape

        # Start from noise
        x = torch.randn(shape, device=device)

        # Sampling loop
        dt = 1.0 / num_steps
        iterator = range(num_steps)

        if progress:
            iterator = tqdm(iterator, desc="Sampling")

        for i in iterator:
            t = (i * dt) * torch.ones(B, device=device)
            with torch.no_grad():
                v = model(x, t, y)
            x = x + v * dt

        return x

    def sample_with_ode(
        self,
        model: nn.Module,
        shape: Tuple[int, int, int, int],
        y: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        device: str = "cuda",
        progress: bool = False,
    ) -> torch.Tensor:
        """
        Sample using ODE solver (more accurate but slower).

        Args:
            model: The DiT model
            shape: Shape of the samples (B, C, H, W)
            y: Class labels (optional)
            num_steps: Number of sampling steps
            device: Device to sample on
            progress: Whether to show progress bar

        Returns:
            samples: Generated images (B, C, H, W)
        """
        num_steps = num_steps or self.num_steps
        B, C, H, W = shape

        # Start from noise
        x = torch.randn(shape, device=device)

        # Sampling loop
        dt = 1.0 / num_steps
        iterator = range(num_steps)

        if progress:
            iterator = tqdm(iterator, desc="ODE Sampling")

        for i in iterator:
            t = (i * dt) * torch.ones(B, device=device)

            with torch.no_grad():
                v = model(x, t, y)

            # Half step
            x_mid = x + v * dt / 2
            t_mid = t + dt / 2

            with torch.no_grad():
                v_mid = model(x_mid, t_mid, y)

            x = x + v_mid * dt

        return x

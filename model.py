import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Attention(nn.Module):
    """Multi-head attention layer."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """Feed-forward network layer."""

    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class DiTBlock(nn.Module):
    """DiT Transformer block."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DiT(nn.Module):
    """Diffusion Transformer (DiT) model."""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        num_classes: int = 1000,
        learn_sigma: bool = False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.num_classes = num_classes
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Class embedding (optional)
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])

        # Final layer norm
        self.final_layer = nn.LayerNorm(hidden_dim)

        # Output projection
        out_channels = in_channels * (2 if learn_sigma else 1)
        self.output_proj = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)

        # Patchify/unpatchify
        self.patch_size = patch_size
        self.out_channels = out_channels

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

        self.apply(_init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            t: Timestep tensor of shape (B,)
            y: Class labels of shape (B,)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # (B, hidden_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_dim)

        # Add position embeddings
        x = x + self.pos_embed

        # Add time embedding
        t_emb = self.time_embed(t)  # (B, hidden_dim)
        x = x + t_emb[:, None, :]

        # Add class embedding (if provided)
        if y is not None and self.num_classes > 0:
            y_emb = self.class_embed(y)  # (B, hidden_dim)
            x = x + y_emb[:, None, :]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.final_layer(x)

        # Output projection
        x = self.output_proj(x)  # (B, num_patches, patch_size^2 * out_channels)

        # Reshape to image
        x = x.reshape(B, self.num_patches, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 4, 1, 2, 3)  # (B, out_channels, num_patches, patch_size, patch_size)
        x = x.reshape(B, self.out_channels, H, W)

        return x

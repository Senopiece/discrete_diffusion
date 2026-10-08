"""A small timestep-conditioned U-Net for image denoising experiments."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def make_binary_embeddings(embedding_dim: int, device: torch.device) -> torch.Tensor:
    """Fixed, orthonormal unit vectors for the two maze pixel classes."""
    if embedding_dim < 2:
        raise ValueError("embedding_dim must be >= 2 for binary labels")
    embeddings = torch.zeros(2, embedding_dim, device=device)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    return embeddings


def embed_binary(maze: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    """Map (B,H,W) labels to (B,D,H,W) continuous clean data."""
    return embeddings[maze.long()].permute(0, 3, 1, 2).contiguous()


def decode_probabilities(
    x: torch.Tensor, embeddings: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Decode embedding dot products into two-class probabilities."""
    logits = torch.einsum("bdhw,kd->bkhw", x, embeddings) / temperature
    return logits.softmax(dim=1)


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for integer or continuous scalar timesteps."""
    half = dim // 2
    scale = math.log(10_000) / max(half - 1, 1)
    frequencies = torch.exp(
        -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    result = torch.cat((angles.sin(), angles.cos()), dim=1)
    if dim % 2:
        result = F.pad(result, (0, 1))
    return result


class ResBlock(nn.Module):
    """Residual convolution block with additive timestep conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Denoiser(nn.Module):
    """Predict per-pixel image targets from an input image and timestep."""

    def __init__(
        self,
        embedding_dim: int = 16,
        output_dim: int | None = None,
        base_channels: int = 64,
        time_dim: int = 128,
    ):
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be >= 2 for binary labels")
        if output_dim is None:
            output_dim = embedding_dim
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        if base_channels % 8:
            raise ValueError("base_channels must be divisible by 8")
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.input = nn.Conv2d(embedding_dim, c1, kernel_size=3, padding=1)
        self.enc32 = ResBlock(c1, c1, time_dim)
        self.down16 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.enc16 = ResBlock(c2, c2, time_dim)
        self.down8 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.mid1 = ResBlock(c3, c3, time_dim)
        self.mid2 = ResBlock(c3, c3, time_dim)

        self.up16 = nn.Conv2d(c3, c2, kernel_size=3, padding=1)
        self.dec16 = ResBlock(c2 * 2, c2, time_dim)
        self.up32 = nn.Conv2d(c2, c1, kernel_size=3, padding=1)
        self.dec32 = ResBlock(c1 * 2, c1, time_dim)
        self.output_norm = nn.GroupNorm(8, c1)
        self.output = nn.Conv2d(c1, output_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_mlp(
            sinusoidal_timestep_embedding(timesteps, self.time_dim)
        )

        skip32 = self.enc32(self.input(x), time_embedding)
        skip16 = self.enc16(self.down16(skip32), time_embedding)
        h = self.mid2(self.mid1(self.down8(skip16), time_embedding), time_embedding)

        h = F.interpolate(h, size=skip16.shape[-2:], mode="nearest")
        h = self.up16(h)
        h = self.dec16(torch.cat((h, skip16), dim=1), time_embedding)
        h = F.interpolate(h, size=skip32.shape[-2:], mode="nearest")
        h = self.up32(h)
        h = self.dec32(torch.cat((h, skip32), dim=1), time_embedding)
        return self.output(F.silu(self.output_norm(h)))

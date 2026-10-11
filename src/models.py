from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskAutoEncoder(nn.Module):
    def __init__(self, latent_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(128, latent_channels, 3, 1, 1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class ImageAutoEncoder(nn.Module):
    """RGB autoencoder defining the image latent space for conditional diffusion."""

    def __init__(self, latent_channels: int):
        super().__init__()
        self.enc1 = ConvBlock(3, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.to_latent = nn.Conv2d(128, latent_channels, 3, 1, 1)
        self.from_latent = ConvBlock(latent_channels, 128)
        self.up2 = UpsampleConv(128, 64)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = UpsampleConv(64, 32)
        self.dec1 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, 3, 3, 1, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        return self.to_latent(F.avg_pool2d(e3, 2))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = self.up2(h)
        h = self.dec2(torch.cat([h, torch.zeros_like(h)], dim=1))
        h = self.up1(h)
        h = self.dec1(torch.cat([h, torch.zeros_like(h)], dim=1))
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        return torch.tanh(self.out(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000.0) / max(half - 1, 1)))
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.net(emb)


class LatentDenoiser(nn.Module):
    def __init__(self, channels: int, time_dim: int = 128):
        super().__init__()
        self.time = TimeEmbedding(time_dim)
        self.in_conv = nn.Conv2d(channels, 64, 3, 1, 1)
        self.down1 = nn.Conv2d(64, 128, 4, 2, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.up1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.out = nn.Conv2d(64, channels, 3, 1, 1)
        self.t64 = nn.Linear(time_dim, 64)
        self.t128 = nn.Linear(time_dim, 128)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time(t)
        h1 = F.silu(self.in_conv(x) + self.t64(emb)[:, :, None, None])
        h2 = F.silu(self.down1(h1) + self.t128(emb)[:, :, None, None])
        h2 = F.silu(self.down2(h2))
        h = F.silu(self.up1(h2) + h1)
        return self.out(h)


class ConditionalLatentDenoiser(nn.Module):
    """Denoises RGB image latents conditioned on encoded anatomy masks."""

    def __init__(self, image_channels: int, condition_channels: int, time_dim: int = 128):
        super().__init__()
        self.time = TimeEmbedding(time_dim)
        in_channels = image_channels + condition_channels
        self.in_conv = nn.Conv2d(in_channels, 96, 3, 1, 1)
        self.down1 = nn.Conv2d(96, 160, 4, 2, 1)
        self.mid = ConvBlock(160, 160)
        self.up1 = UpsampleConv(160, 96)
        self.out = nn.Conv2d(96, image_channels, 3, 1, 1)
        self.t96 = nn.Linear(time_dim, 96)
        self.t160 = nn.Linear(time_dim, 160)

    def forward(self, z: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if cond.shape[-2:] != z.shape[-2:]:
            cond = F.interpolate(cond, size=z.shape[-2:], mode="bilinear", align_corners=False)
        emb = self.time(t)
        h1 = F.silu(self.in_conv(torch.cat([z, cond], dim=1)) + self.t96(emb)[:, :, None, None])
        h2 = F.silu(self.down1(h1) + self.t160(emb)[:, :, None, None])
        h2 = self.mid(h2)
        h = self.up1(h2) + h1
        return self.out(h)


class MaskConditionedImageUNet(nn.Module):
    """Small conditional renderer: binary mask/noise condition -> RGB medical image."""

    def __init__(self, condition_channels: int = 2):
        super().__init__()
        self.condition_channels = condition_channels
        self.enc1 = nn.Sequential(nn.Conv2d(condition_channels, 32, 3, 1, 1), nn.SiLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.SiLU())
        self.down1 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.SiLU())
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(), nn.Conv2d(128, 128, 3, 1, 1), nn.SiLU())
        self.mid = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.SiLU(), nn.Conv2d(128, 128, 3, 1, 1), nn.SiLU())
        self.up2 = UpsampleConv(128, 64)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.SiLU())
        self.up1 = UpsampleConv(64, 32)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.SiLU(), nn.Conv2d(32, 3, 3, 1, 1))

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(mask)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        mid = self.mid(e3)
        d2 = self.up2(mid)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        return torch.tanh(self.dec1(torch.cat([d1, e1], dim=1)))


class DDPM:
    def __init__(self, timesteps: int, device: torch.device):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        abar = self.alpha_bar[t][:, None, None, None]
        return torch.sqrt(abar) * z0 + torch.sqrt(1.0 - abar) * noise

    def predict_z0(self, zt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        abar = self.alpha_bar[t][:, None, None, None]
        return (zt - torch.sqrt(1.0 - abar) * eps) / torch.sqrt(abar)

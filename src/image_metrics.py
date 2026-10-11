from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def ssim_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Global SSIM for images in [-1, 1]."""
    x = (x + 1.0) / 2.0
    y = (y + 1.0) / 2.0
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mux = x.mean(dim=(1, 2, 3))
    muy = y.mean(dim=(1, 2, 3))
    sigx = ((x - mux[:, None, None, None]) ** 2).mean(dim=(1, 2, 3))
    sigy = ((y - muy[:, None, None, None]) ** 2).mean(dim=(1, 2, 3))
    sigxy = ((x - mux[:, None, None, None]) * (y - muy[:, None, None, None])).mean(dim=(1, 2, 3))
    score = ((2 * mux * muy + c1) * (2 * sigxy + c2)) / ((mux**2 + muy**2 + c1) * (sigx + sigy + c2))
    return score.mean()


def image_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x - y))


@torch.no_grad()
def inception_features(images: torch.Tensor, device: torch.device, batch_size: int = 16) -> np.ndarray:
    try:
        from torchvision.models import Inception_V3_Weights, inception_v3
    except Exception as exc:
        raise RuntimeError("TorchVision is required for Inception FID features.") from exc

    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, aux_logits=True, transform_input=False).to(device)
    model.fc = torch.nn.Identity()
    model.eval()
    feats = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device)
        batch = (batch + 1.0) / 2.0
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = weights.transforms()(batch)
        feats.append(model(batch).detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def covariance(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    return (x.T @ x) / max(len(x) - 1, 1)


def sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)[None, :]) @ vecs.T


def fid_from_features(real: np.ndarray, fake: np.ndarray) -> float:
    mu1, mu2 = real.mean(axis=0), fake.mean(axis=0)
    c1, c2 = covariance(real), covariance(fake)
    c1_sqrt = sqrtm_psd(c1)
    covmean = sqrtm_psd(c1_sqrt @ c2 @ c1_sqrt)
    fid = float(np.sum((mu1 - mu2) ** 2) + np.trace(c1 + c2 - 2 * covmean))
    return max(fid, 0.0)


@torch.no_grad()
def lpips_distance(real: torch.Tensor, fake: torch.Tensor, device: torch.device) -> Optional[float]:
    try:
        import lpips
    except Exception:
        return None
    loss_fn = lpips.LPIPS(net="alex").to(device)
    vals = []
    for i in range(len(real)):
        vals.append(float(loss_fn(real[i:i + 1].to(device), fake[i:i + 1].to(device)).item()))
    return float(np.mean(vals))

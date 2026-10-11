from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from dataset import KvasirZipDataset
from image_metrics import fid_from_features, image_l1, inception_features, lpips_distance, ssim_torch
from models import ConditionalLatentDenoiser, DDPM, ImageAutoEncoder, LatentDenoiser, MaskAutoEncoder, MaskConditionedImageUNet
from topology import (
    betti_numbers,
    diagram_distances,
    forman_ricci_profile,
    persistence_diagram,
    project_probability_to_topology_area,
    soft_curvature_loss,
    soft_shape_preservation_loss,
    soft_topology_anchor_loss,
    try_persistence_summary,
)


PROJECT = Path(__file__).resolve().parents[1]

METHODS = {
    "baseline": {"label": "Baseline Latent DDPM", "curvature": False, "topology": False, "shape": False, "hard_anchor": False},
    "curvature": {"label": "Curvature Only", "curvature": True, "topology": False, "shape": True, "hard_anchor": False},
    "topology": {"label": "Topology Only", "curvature": False, "topology": True, "shape": True, "hard_anchor": True},
    "full": {"label": "Full CCLD", "curvature": True, "topology": True, "shape": True, "hard_anchor": True},
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT / p


def split_dataset(dataset, train_fraction: float, seed: int):
    train_size = int(len(dataset) * train_fraction)
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))


def load_or_train(module, ckpt_path: Path, train_fn, cfg, device):
    if cfg.get("resume_from_checkpoints", True) and ckpt_path.exists():
        module.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint: {ckpt_path}")
        return
    train_fn()
    torch.save(module.state_dict(), ckpt_path)


def append_history(cfg: dict, section: str, epoch: int, values: dict) -> None:
    path = cfg.get("_history_path")
    if not path:
        return
    history_path = Path(path)
    history = json.loads(history_path.read_text()) if history_path.exists() else {}
    history.setdefault(section, []).append({"epoch": epoch, **{k: float(v) for k, v in values.items()}})
    history_path.write_text(json.dumps(history, indent=2))


def boundary_roughness(mask: np.ndarray) -> float:
    area = float(mask.sum())
    if area == 0:
        return 0.0
    h, w = mask.shape
    perimeter = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or nx < 0 or ny >= h or nx >= w or not mask[ny, nx]:
                    perimeter += 1
    return float(perimeter / np.sqrt(area))


def border_contact(mask: np.ndarray) -> float:
    area = float(mask.sum())
    if area == 0:
        return 0.0
    border = mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum()
    return float(border / area)


def compactness(mask: np.ndarray) -> float:
    area = float(mask.sum())
    if area == 0:
        return 0.0
    rough = boundary_roughness(mask)
    return float((rough * rough) / max(area, 1.0))


def reference_stats(loader: DataLoader, out_dir: Path) -> dict:
    areas, bettis, curvatures = [], [], []
    panel_images, panel_masks = [], []
    pd0_ref: Optional[np.ndarray] = None
    pd1_ref: Optional[np.ndarray] = None

    for batch in tqdm(loader, desc="Reference topology"):
        masks = batch["mask"].numpy()[:, 0] > 0.5
        for i, mask in enumerate(masks):
            b0, b1 = betti_numbers(mask)
            area = float(mask.mean())
            areas.append(area)
            bettis.append((b0, b1))
            curvatures.append(forman_ricci_profile(mask))
            if pd0_ref is None and b0 == 1:
                pd0_ref = persistence_diagram(mask, 0)
                pd1_ref = persistence_diagram(mask, 1)
            if len(panel_masks) < 8:
                panel_masks.append(mask)
                panel_images.append(batch["image"][i].numpy())

    stats = {
        "count": len(areas),
        "area_mean": float(np.mean(areas)),
        "area_std": float(np.std(areas)),
        "area_values": areas,
        "beta0_median": int(np.median([b[0] for b in bettis])),
        "beta1_median": int(np.median([b[1] for b in bettis])),
        "euler_median": int(np.median([b[0] - b[1] for b in bettis])),
        "curvature_mean": np.mean(curvatures, axis=0).tolist(),
        "_ref_pd0": pd0_ref,
        "_ref_pd1": pd1_ref,
        "_panel_images": panel_images,
        "_panel_masks": panel_masks,
    }
    public = {k: v for k, v in stats.items() if not k.startswith("_")}
    (out_dir / "dataset_report.json").write_text(json.dumps(public, indent=2))
    return stats


def train_mask_autoencoder(model, loader, cfg, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    model.train()
    for epoch in range(cfg["mask_ae_epochs"]):
        losses = []
        pbar = tqdm(loader, desc=f"Mask AE {epoch + 1}/{cfg['mask_ae_epochs']}")
        for batch in pbar:
            masks = batch["mask"].to(device)
            logits = model(masks)
            loss = F.binary_cross_entropy_with_logits(logits, masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        append_history(cfg, "mask_autoencoder", epoch + 1, {"loss": np.mean(losses)})


def train_image_autoencoder(model, loader, cfg, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    model.train()
    for epoch in range(cfg["image_ae_epochs"]):
        losses, ssims = [], []
        pbar = tqdm(loader, desc=f"Image AE {epoch + 1}/{cfg['image_ae_epochs']}")
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            recon = model(images)
            lesion_weight = 1.0 + cfg.get("renderer_mask_weight", 1.5) * masks
            loss = (torch.abs(recon - images) * lesion_weight).mean() + cfg["lambda_image_ssim"] * (1.0 - ssim_torch(recon, images))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ssim_val = ssim_torch(recon, images).item()
            losses.append(float(loss.item()))
            ssims.append(float(ssim_val))
            pbar.set_postfix(loss=f"{loss.item():.4f}", ssim=f"{ssim_val:.3f}")
        append_history(cfg, "image_autoencoder", epoch + 1, {"loss": np.mean(losses), "ssim": np.mean(ssims)})


def renderer_condition(masks: torch.Tensor, noise_scale: float = 1.0) -> torch.Tensor:
    noise = torch.randn_like(masks) * noise_scale
    low = F.avg_pool2d(noise, kernel_size=8, stride=8)
    noise = F.interpolate(low, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([masks, noise], dim=1)


def train_mask_conditioned_renderer(model, loader, cfg, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    model.train()
    epochs = int(cfg.get("renderer_epochs", 18))
    mask_weight = float(cfg.get("renderer_mask_weight", 2.0))
    for epoch in range(epochs):
        losses, ssims = [], []
        pbar = tqdm(loader, desc=f"Mask-conditioned renderer {epoch + 1}/{epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            pred = model(renderer_condition(masks))
            weights = 1.0 + mask_weight * masks
            recon_loss = (torch.abs(pred - images) * weights).mean()
            ssim_loss = 1.0 - ssim_torch(pred, images)
            smoothness = (
                torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]).mean()
                + torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]).mean()
            )
            loss = recon_loss + cfg["lambda_image_ssim"] * ssim_loss + 0.015 * smoothness
            opt.zero_grad()
            loss.backward()
            opt.step()
            ssim_val = ssim_torch(pred, images).item()
            losses.append(float(loss.item()))
            ssims.append(float(ssim_val))
            pbar.set_postfix(loss=f"{loss.item():.4f}", ssim=f"{ssim_val:.3f}")
        append_history(cfg, "mask_conditioned_renderer", epoch + 1, {"loss": np.mean(losses), "ssim": np.mean(ssims)})


def train_mask_denoiser(mask_ae, denoiser, ddpm, loader, cfg, device, method):
    opt = torch.optim.AdamW(denoiser.parameters(), lr=cfg["lr"])
    mask_ae.eval()
    denoiser.train()
    for epoch in range(cfg["mask_diffusion_epochs"]):
        losses = []
        pbar = tqdm(loader, desc=f"Mask diffusion {method['label']} {epoch + 1}/{cfg['mask_diffusion_epochs']}")
        for batch in pbar:
            masks = batch["mask"].to(device)
            with torch.no_grad():
                z0 = mask_ae.encode(masks)
            t = torch.randint(0, ddpm.timesteps, (z0.shape[0],), device=device)
            noise = torch.randn_like(z0)
            zt = ddpm.q_sample(z0, t, noise)
            eps = denoiser(zt, t)
            loss = F.mse_loss(eps, noise)

            if method["curvature"] or method["topology"] or method["shape"]:
                z0_hat = ddpm.predict_z0(zt, t, eps)
                logits = mask_ae.decode(z0_hat)
                loss = loss + cfg["lambda_mask_recon"] * F.binary_cross_entropy_with_logits(logits, masks)
                if method["curvature"]:
                    loss = loss + cfg["lambda_curvature"] * soft_curvature_loss(logits)
                if method["topology"]:
                    area_targets = masks.mean(dim=(1, 2, 3))
                    loss = loss + cfg["lambda_topology"] * soft_topology_anchor_loss(logits, area_targets)
                    loss = loss + cfg["lambda_area"] * torch.abs(torch.sigmoid(logits).mean(dim=(1, 2, 3)) - area_targets).mean()
                if method["shape"]:
                    loss = loss + cfg["lambda_shape"] * soft_shape_preservation_loss(logits, masks)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        append_history(cfg, f"mask_diffusion_{method['label']}", epoch + 1, {"loss": np.mean(losses)})


def train_image_denoiser(image_ae, mask_ae, denoiser, ddpm, loader, cfg, device):
    opt = torch.optim.AdamW(denoiser.parameters(), lr=cfg["lr"])
    image_ae.eval()
    mask_ae.eval()
    denoiser.train()
    for epoch in range(cfg["image_diffusion_epochs"]):
        losses = []
        pbar = tqdm(loader, desc=f"Conditional image diffusion {epoch + 1}/{cfg['image_diffusion_epochs']}")
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            with torch.no_grad():
                z_img = image_ae.encode(images)
                z_mask = mask_ae.encode(masks)
            t = torch.randint(0, ddpm.timesteps, (z_img.shape[0],), device=device)
            noise = torch.randn_like(z_img)
            zt = ddpm.q_sample(z_img, t, noise)
            eps = denoiser(zt, z_mask, t)
            loss = F.mse_loss(eps, noise)

            # Decode a light reconstruction term on late/noisy estimates to keep condition-image semantics.
            if random.random() < 0.35:
                z0_hat = ddpm.predict_z0(zt, t, eps)
                recon = image_ae.decode(z0_hat)
                loss = loss + cfg["lambda_image_recon"] * image_l1(recon, images)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        append_history(cfg, "conditional_image_diffusion", epoch + 1, {"loss": np.mean(losses)})


def reverse_step(z, eps, ddpm: DDPM, step: int):
    beta = ddpm.betas[step]
    alpha = ddpm.alphas[step]
    abar = ddpm.alpha_bar[step]
    mean = (1.0 / torch.sqrt(alpha)) * (z - beta / torch.sqrt(1.0 - abar) * eps)
    if step > 0:
        return mean + torch.sqrt(beta) * torch.randn_like(z)
    return mean


def target_areas(stats: dict, n: int, device, cfg: dict) -> torch.Tensor:
    values = np.array(stats["area_values"], dtype=np.float32)
    low_q = float(cfg.get("target_area_quantile_low", 0.10))
    high_q = float(cfg.get("target_area_quantile_high", 0.90))
    low = max(float(np.quantile(values, low_q)), float(cfg.get("target_area_min", 0.03)))
    high = min(float(np.quantile(values, high_q)), float(cfg.get("target_area_max", 0.42)))
    values = np.clip(values, low, high)
    picked = np.random.choice(values, size=n, replace=True)
    return torch.from_numpy(picked).to(device)


def apply_area_topology_anchor(mask_ae, z, stats, areas, cfg):
    with torch.no_grad():
        decoded = torch.sigmoid(mask_ae.decode(z)).cpu().numpy()[:, 0]
        area_np = areas.detach().cpu().numpy()
        projected = [
            project_probability_to_topology_area(
                prob,
                stats["beta0_median"],
                stats["beta1_median"],
                float(area_np[i]),
                tolerance=max(0.02, stats["area_std"] * 0.15),
                border_margin=int(cfg.get("border_margin", 8)),
            )
            for i, prob in enumerate(decoded)
        ]
        projected_tensor = torch.from_numpy(np.asarray(projected, dtype=np.float32))[:, None].to(z.device)
        projected_latent = mask_ae.encode(projected_tensor)
        blend = cfg["hard_anchor_blend"]
        return (1.0 - blend) * z.detach() + blend * projected_latent


def sample_masks(mask_ae, denoiser, ddpm, cfg, device, stats, method):
    n = cfg["sample_count"]
    latent_size = cfg["image_size"] // 8
    z = torch.randn((n, cfg["mask_latent_channels"], latent_size, latent_size), device=device)
    areas = target_areas(stats, n, device, cfg)
    guide_start = int(ddpm.timesteps * cfg["guidance_start_fraction"])
    denoiser.eval()
    mask_ae.eval()

    for step in tqdm(reversed(range(ddpm.timesteps)), total=ddpm.timesteps, desc=f"Sampling masks {method['label']}"):
        t = torch.full((n,), step, device=device, dtype=torch.long)
        with torch.no_grad():
            eps = denoiser(z, t)
            z = reverse_step(z, eps, ddpm, step)

        if step < guide_start and (method["curvature"] or method["topology"]):
            z = z.detach().requires_grad_(True)
            logits = mask_ae.decode(z)
            loss = torch.tensor(0.0, device=device)
            if method["topology"]:
                loss = loss + cfg["lambda_topology"] * soft_topology_anchor_loss(logits, areas)
                loss = loss + cfg["lambda_area"] * torch.abs(torch.sigmoid(logits).mean(dim=(1, 2, 3)) - areas).mean()
            if method["curvature"]:
                probs = torch.sigmoid(logits)
                area_now = probs.mean(dim=(1, 2, 3))
                area_gate = torch.sigmoid((area_now - 0.035) * 80.0).mean()
                loss = loss + cfg["lambda_curvature"] * area_gate * soft_curvature_loss(logits)
            grad = torch.autograd.grad(loss, z)[0]
            z = (z - cfg["sample_guidance_step"] * grad).detach()
            if method["hard_anchor"] and step % cfg["hard_anchor_every"] == 0:
                z = apply_area_topology_anchor(mask_ae, z, stats, areas, cfg)

    with torch.no_grad():
        probs = torch.sigmoid(mask_ae.decode(z)).cpu().numpy()[:, 0]
    area_np = areas.detach().cpu().numpy()
    final_masks = [
        project_probability_to_topology_area(
            prob,
            stats["beta0_median"] if method["topology"] else max(1, stats["beta0_median"]),
            stats["beta1_median"],
            float(area_np[i]),
            tolerance=max(0.02, stats["area_std"] * 0.15),
            border_margin=int(cfg.get("border_margin", 8)),
        )
        if method["topology"]
        else prob > 0.5
        for i, prob in enumerate(probs)
    ]
    return np.asarray(final_masks, dtype=bool)


def sample_images(image_ae, mask_ae, image_denoiser, ddpm, masks, cfg, device, renderer=None):
    n = len(masks)
    latent_size = cfg["image_size"] // 8
    z = torch.randn((n, cfg["image_latent_channels"], latent_size, latent_size), device=device)
    mask_tensor = torch.from_numpy(masks.astype(np.float32))[:, None].to(device)
    with torch.no_grad():
        cond = mask_ae.encode(mask_tensor)
    image_denoiser.eval()
    image_ae.eval()
    for step in tqdm(reversed(range(ddpm.timesteps)), total=ddpm.timesteps, desc="Sampling conditional RGB"):
        t = torch.full((n,), step, device=device, dtype=torch.long)
        with torch.no_grad():
            eps = image_denoiser(z, cond, t)
            z = reverse_step(z, eps, ddpm, step)
    with torch.no_grad():
        latent_images = image_ae.decode(z)
        if renderer is not None and cfg.get("image_synthesis_mode", "hybrid") in {"renderer", "hybrid"}:
            renderer.eval()
            rendered = renderer(renderer_condition(mask_tensor, noise_scale=float(cfg.get("renderer_sample_noise", 1.0))))
            if cfg.get("image_synthesis_mode", "hybrid") == "renderer":
                images = rendered
            else:
                soft = F.avg_pool2d(mask_tensor, kernel_size=11, stride=1, padding=5).clamp(0.0, 1.0)
                alpha = float(cfg.get("hybrid_renderer_blend", 0.45))
                images = latent_images * (1.0 - alpha * soft) + rendered * (alpha * soft)
        else:
            images = latent_images
        images = images.cpu()
    return images


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    arr = (((image.clamp(-1, 1) + 1.0) / 2.0).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_grid_images(images: torch.Tensor, path: Path, title: str):
    n = min(16, len(images))
    cell = images.shape[-1]
    header = 24
    canvas = Image.new("RGB", (4 * cell, 4 * cell + header), (246, 248, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill=(20, 25, 32))
    for i in range(n):
        canvas.paste(tensor_to_pil(images[i]), ((i % 4) * cell, header + (i // 4) * cell))
    canvas.save(path)


def save_grid_masks(masks: np.ndarray, path: Path, title: str):
    n = min(16, len(masks))
    cell = masks.shape[-1]
    header = 24
    canvas = Image.new("RGB", (4 * cell, 4 * cell + header), (246, 248, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill=(20, 25, 32))
    for i in range(n):
        tile = Image.fromarray((masks[i] * 255).astype(np.uint8)).convert("RGB")
        canvas.paste(tile, ((i % 4) * cell, header + (i // 4) * cell))
    canvas.save(path)


def save_qualitative_panel(stats, generated, path: Path):
    cell = 128
    label_w = 155
    rows = 2 + len(generated)
    canvas = Image.new("RGB", (label_w + 4 * cell, rows * cell), (246, 248, 250))
    draw = ImageDraw.Draw(canvas)

    def label(row, text):
        draw.text((8, row * cell + 8), text, fill=(20, 25, 32))

    label(0, "Real images")
    for i, image in enumerate(stats["_panel_images"][:4]):
        img = Image.fromarray((((image.transpose(1, 2, 0) + 1) / 2) * 255).clip(0, 255).astype(np.uint8))
        canvas.paste(img, (label_w + i * cell, 0))
    label(1, "Real masks")
    for i, mask in enumerate(stats["_panel_masks"][:4]):
        canvas.paste(Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB"), (label_w + i * cell, cell))

    for row, (name, (masks, images)) in enumerate(generated.items(), start=2):
        label(row, name)
        for i in range(min(4, len(images), len(masks))):
            img = tensor_to_pil(images[i])
            overlay = Image.fromarray((masks[i] * 255).astype(np.uint8)).convert("RGB")
            canvas.paste(Image.blend(img, overlay, 0.12), (label_w + i * cell, row * cell))
    canvas.save(path)


def save_generated_samples(path: Path, generated: dict) -> None:
    payload = {}
    for method, (masks, images) in generated.items():
        payload[method] = {
            "masks": torch.from_numpy(masks.astype(np.float32)),
            "images": images.cpu(),
        }
    torch.save(payload, path)


def evaluate_masks(masks, method, stats):
    rows = []
    ref_curve = np.array(stats["curvature_mean"], dtype=float)
    for i, mask in enumerate(masks):
        b0, b1 = betti_numbers(mask)
        euler = b0 - b1
        row = {
            "method": method,
            "sample_id": i,
            "beta0": b0,
            "beta1": b1,
            "euler": euler,
            "betti0_error": abs(b0 - stats["beta0_median"]),
            "betti1_error": abs(b1 - stats["beta1_median"]),
            "euler_error": abs(euler - stats["euler_median"]),
            "area": float(mask.mean()),
            "area_error": abs(float(mask.mean()) - stats["area_mean"]),
            "curvature_l1": float(np.abs(forman_ricci_profile(mask) - ref_curve).sum()),
            "boundary_roughness": boundary_roughness(mask),
            "border_contact": border_contact(mask),
            "compactness": compactness(mask),
        }
        row.update(try_persistence_summary(mask))
        row.update(diagram_distances(mask, stats["_ref_pd0"], stats["_ref_pd1"]))
        rows.append(row)
    return rows


def collect_real_images(loader, count):
    images, masks = [], []
    for batch in loader:
        images.append(batch["image"])
        masks.append(batch["mask"])
        if sum(len(x) for x in images) >= count:
            break
    return torch.cat(images, dim=0)[:count], torch.cat(masks, dim=0)[:count]


def nearest_real_l1_ssim(real_images: torch.Tensor, fake_images: torch.Tensor):
    l1s, ssims = [], []
    for fake in fake_images:
        batch_fake = fake[None].repeat(len(real_images), 1, 1, 1)
        l1_vals = torch.abs(batch_fake - real_images).mean(dim=(1, 2, 3))
        idx = int(torch.argmin(l1_vals).item())
        l1s.append(float(l1_vals[idx].item()))
        ssims.append(float(ssim_torch(fake[None], real_images[idx:idx + 1]).item()))
    return float(np.mean(l1s)), float(np.mean(ssims))


def mask_contrast(images: torch.Tensor, masks: torch.Tensor) -> float:
    images01 = (images.clamp(-1, 1) + 1.0) / 2.0
    masks = masks.float()
    inside = (images01 * masks).sum(dim=(1, 2, 3)) / masks.sum(dim=(1, 2, 3)).clamp_min(1.0)
    outside_mask = 1.0 - F.max_pool2d(masks, kernel_size=17, stride=1, padding=8)
    outside = (images01 * outside_mask).sum(dim=(1, 2, 3)) / outside_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return float(torch.abs(inside - outside).mean().item())


def evaluate_images(real_images, real_masks, generated, generated_masks, device, out_dir):
    metrics = {}
    real_images = real_images.cpu()
    real_masks = real_masks.cpu()
    real_feats = None
    try:
        real_feats = inception_features(real_images, device)
    except Exception as exc:
        fid_setup_error = str(exc)
    else:
        fid_setup_error = None

    for method, images in generated.items():
        n = min(len(real_images), len(images))
        nearest_l1, nearest_ssim = nearest_real_l1_ssim(real_images[:n], images[:n])
        fake_masks = torch.from_numpy(generated_masks[method].astype(np.float32))[:, None][:n]
        real_contrast = mask_contrast(real_images[:n], real_masks[:n])
        fake_contrast = mask_contrast(images[:n], fake_masks)
        row = {
            "nearest_real_l1": nearest_l1,
            "nearest_real_ssim": nearest_ssim,
            "paired_batch_l1": float(torch.abs(images[:n] - real_images[:n]).mean().item()),
            "paired_batch_ssim": float(ssim_torch(images[:n], real_images[:n]).item()),
            "lpips_to_real_batch": lpips_distance(real_images[:n], images[:n], device),
            "mask_contrast": fake_contrast,
            "real_mask_contrast": real_contrast,
            "mask_contrast_abs_error": abs(fake_contrast - real_contrast),
            "fid": None,
            "fid_error": fid_setup_error,
        }
        if real_feats is not None:
            try:
                fake_feats = inception_features(images[:n], device)
                row["fid"] = fid_from_features(real_feats[:n], fake_feats)
                row["fid_error"] = None
            except Exception as exc:
                row["fid_error"] = str(exc)
        metrics[method] = row
    (out_dir / "image_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def write_csv(path: Path, rows):
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    metrics = [
        "betti0_error", "betti1_error", "euler_error", "area_error",
        "curvature_l1", "boundary_roughness", "pd0_wasserstein",
        "pd1_wasserstein", "pd0_bottleneck", "pd1_bottleneck",
        "border_contact", "compactness",
    ]
    out = {}
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        out[method] = {}
        for metric in metrics:
            vals = [row.get(metric) for row in subset if row.get(metric) is not None]
            if vals:
                arr = np.array([float(v) for v in vals])
                out[method][metric] = {"mean": float(arr.mean()), "std": float(arr.std())}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chapter_pipeline_v2.json")
    args = parser.parse_args()
    cfg = json.loads(resolve_path(args.config).read_text())
    seed_all(cfg["seed"])

    out_dir = resolve_path(cfg["out_dir"])
    cfg["_history_path"] = str(out_dir / "training_history.json")
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("reset_training_history", False):
        Path(cfg["_history_path"]).write_text("{}")

    dataset = KvasirZipDataset(resolve_path(cfg["zip_path"]), cfg["image_size"])
    train_ds, val_ds = split_dataset(dataset, cfg["train_fraction"], cfg["seed"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = reference_stats(val_loader, out_dir)

    mask_ae = MaskAutoEncoder(cfg["mask_latent_channels"]).to(device)
    image_ae = ImageAutoEncoder(cfg["image_latent_channels"]).to(device)
    mask_ddpm = DDPM(cfg["timesteps"], device)
    image_ddpm = DDPM(cfg["timesteps"], device)

    load_or_train(
        mask_ae,
        ckpt_dir / "mask_autoencoder.pt",
        lambda: train_mask_autoencoder(mask_ae, train_loader, cfg, device),
        cfg,
        device,
    )
    load_or_train(
        image_ae,
        ckpt_dir / "image_autoencoder.pt",
        lambda: train_image_autoencoder(image_ae, train_loader, cfg, device),
        cfg,
        device,
    )

    image_denoiser = ConditionalLatentDenoiser(cfg["image_latent_channels"], cfg["mask_latent_channels"]).to(device)
    load_or_train(
        image_denoiser,
        ckpt_dir / "conditional_image_denoiser.pt",
        lambda: train_image_denoiser(image_ae, mask_ae, image_denoiser, image_ddpm, train_loader, cfg, device),
        cfg,
        device,
    )

    renderer = None
    if cfg.get("train_mask_conditioned_renderer", True):
        renderer = MaskConditionedImageUNet(condition_channels=2).to(device)
        load_or_train(
            renderer,
            ckpt_dir / "mask_conditioned_renderer.pt",
            lambda: train_mask_conditioned_renderer(renderer, train_loader, cfg, device),
            cfg,
            device,
        )

    generated_panel = {}
    generated_images = {}
    generated_masks = {}
    mask_rows = []
    for method_name in cfg["ablation_methods"]:
        method = METHODS[method_name]
        denoiser = LatentDenoiser(cfg["mask_latent_channels"]).to(device)
        load_or_train(
            denoiser,
            ckpt_dir / f"{method_name}_mask_denoiser.pt",
            lambda m=method, d=denoiser: train_mask_denoiser(mask_ae, d, mask_ddpm, train_loader, cfg, device, m),
            cfg,
            device,
        )
        masks = sample_masks(mask_ae, denoiser, mask_ddpm, cfg, device, stats, method)
        images = sample_images(image_ae, mask_ae, image_denoiser, image_ddpm, masks, cfg, device, renderer=renderer)
        label = method["label"]
        save_grid_masks(masks, out_dir / f"{method_name}_masks.png", label)
        save_grid_images(images, out_dir / f"{method_name}_images.png", f"{label} RGB")
        generated_panel[label] = (masks, images)
        generated_images[label] = images
        generated_masks[label] = masks
        mask_rows += evaluate_masks(masks, label, stats)

    write_csv(out_dir / "metrics.csv", mask_rows)
    save_qualitative_panel(stats, generated_panel, out_dir / "qualitative_real_vs_generated.png")
    save_generated_samples(out_dir / "generated_samples.pt", generated_panel)
    real_images, real_masks = collect_real_images(val_loader, cfg["sample_count"])
    image_metrics = evaluate_images(real_images, real_masks, generated_images, generated_masks, device, out_dir)
    summary = {"topology": summarize(mask_rows), "image": image_metrics}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

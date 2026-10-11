from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from dataset import KvasirZipDataset
from image_metrics import fid_from_features, image_l1, inception_features, lpips_distance, ssim_torch
from models import DDPM, LatentDenoiser, MaskAutoEncoder, MaskConditionedImageUNet
from topology import (
    betti_numbers,
    diagram_distances,
    forman_ricci_profile,
    persistence_diagram,
    project_to_betti_anchor,
    soft_curvature_loss,
    soft_shape_preservation_loss,
    soft_topology_anchor_loss,
    try_persistence_summary,
)


PROJECT = Path(__file__).resolve().parents[1]


METHODS = {
    "baseline": {"label": "Baseline Latent DDPM", "curvature": False, "topology": False, "shape": False, "sample_guidance": False},
    "curvature": {"label": "Curvature Only", "curvature": True, "topology": False, "shape": True, "sample_guidance": True},
    "topology": {"label": "Topology Only", "curvature": False, "topology": True, "shape": True, "sample_guidance": True},
    "full": {"label": "Full CCLD", "curvature": True, "topology": True, "shape": True, "sample_guidance": True},
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_project_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT / p


def split_dataset(dataset, train_fraction: float, seed: int):
    train_size = int(len(dataset) * train_fraction)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


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


def dataset_reference_stats(loader: DataLoader, out_dir: Path) -> dict:
    areas = []
    bettis = []
    curvatures = []
    ids = []
    pd0_candidates = []
    pd1_candidates = []
    masks_for_panel = []
    images_for_panel = []

    for batch in tqdm(loader, desc="Reference topology"):
        masks = batch["mask"].numpy()[:, 0] > 0.5
        images = batch["image"]
        for sample_id, mask, image in zip(batch["id"], masks, images):
            b0, b1 = betti_numbers(mask)
            areas.append(float(mask.mean()))
            bettis.append((b0, b1))
            curvatures.append(forman_ricci_profile(mask))
            ids.append(sample_id)
            if len(pd0_candidates) < 12:
                pd0_candidates.append(persistence_diagram(mask, 0))
                pd1_candidates.append(persistence_diagram(mask, 1))
            if len(masks_for_panel) < 16:
                masks_for_panel.append(mask)
                images_for_panel.append(image.numpy())

    stats = {
        "count": len(areas),
        "area_mean": float(np.mean(areas)),
        "area_std": float(np.std(areas)),
        "beta0_median": int(np.median([b[0] for b in bettis])),
        "beta1_median": int(np.median([b[1] for b in bettis])),
        "euler_median": int(np.median([b[0] - b[1] for b in bettis])),
        "curvature_mean": np.mean(curvatures, axis=0).tolist(),
        "ids_used": ids[:20],
    }

    medoid_idx = int(np.argmin(np.abs(np.array(areas) - stats["area_mean"])))
    medoid_idx = min(medoid_idx, len(pd0_candidates) - 1) if pd0_candidates else 0
    stats["_ref_pd0"] = pd0_candidates[medoid_idx] if pd0_candidates else None
    stats["_ref_pd1"] = pd1_candidates[medoid_idx] if pd1_candidates else None
    stats["_panel_masks"] = masks_for_panel
    stats["_panel_images"] = images_for_panel

    public_stats = {k: v for k, v in stats.items() if not k.startswith("_")}
    (out_dir / "dataset_report.json").write_text(json.dumps(public_stats, indent=2))
    return stats


def train_autoencoder(ae, loader, cfg, device, ckpt_dir):
    ckpt_path = ckpt_dir / "autoencoder.pt"
    if cfg.get("resume_from_checkpoints", True) and ckpt_path.exists():
        ae.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded existing autoencoder: {ckpt_path}")
        return
    optimizer = torch.optim.AdamW(ae.parameters(), lr=cfg["lr"])
    ae.train()
    for epoch in range(cfg["ae_epochs"]):
        pbar = tqdm(loader, desc=f"Autoencoder {epoch + 1}/{cfg['ae_epochs']}")
        for batch in pbar:
            masks = batch["mask"].to(device)
            logits = ae(masks)
            loss = F.binary_cross_entropy_with_logits(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    torch.save(ae.state_dict(), ckpt_path)


def train_renderer(renderer, loader, cfg, device, ckpt_dir):
    ckpt_path = ckpt_dir / "mask_to_image_renderer.pt"
    if cfg.get("resume_from_checkpoints", True) and ckpt_path.exists():
        renderer.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded existing renderer: {ckpt_path}")
        return
    optimizer = torch.optim.AdamW(renderer.parameters(), lr=cfg["lr"])
    renderer.train()
    for epoch in range(cfg["renderer_epochs"]):
        pbar = tqdm(loader, desc=f"Mask->Image renderer {epoch + 1}/{cfg['renderer_epochs']}")
        for batch in pbar:
            masks = batch["mask"].to(device)
            images = batch["image"].to(device)
            pred = renderer(masks)
            loss = image_l1(pred, images) + cfg["lambda_ssim"] * (1.0 - ssim_torch(pred, images))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}", ssim=f"{ssim_torch(pred, images).item():.3f}")
    torch.save(renderer.state_dict(), ckpt_path)


def train_denoiser(ae, denoiser, ddpm, loader, cfg, device, ckpt_path: Path, method: dict, area_target: torch.Tensor):
    if cfg.get("resume_from_checkpoints", True) and ckpt_path.exists():
        denoiser.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded existing denoiser: {ckpt_path}")
        return
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=cfg["lr"])
    ae.eval()
    denoiser.train()
    label = method["label"]
    for epoch in range(cfg["diffusion_epochs"]):
        pbar = tqdm(loader, desc=f"{label} {epoch + 1}/{cfg['diffusion_epochs']}")
        for batch in pbar:
            masks = batch["mask"].to(device)
            with torch.no_grad():
                z0 = ae.encode(masks)
            t = torch.randint(0, ddpm.timesteps, (z0.shape[0],), device=device)
            noise = torch.randn_like(z0)
            zt = ddpm.q_sample(z0, t, noise)
            pred_noise = denoiser(zt, t)
            loss = F.mse_loss(pred_noise, noise)

            if method["curvature"] or method["topology"] or method["shape"]:
                z0_hat = ddpm.predict_z0(zt, t, pred_noise)
                mask_logits = ae.decode(z0_hat)
                loss = loss + cfg["lambda_recon"] * F.binary_cross_entropy_with_logits(mask_logits, masks)
                if method["curvature"]:
                    loss = loss + cfg["lambda_curvature"] * soft_curvature_loss(mask_logits)
                if method["topology"]:
                    loss = loss + cfg["lambda_topology"] * soft_topology_anchor_loss(mask_logits, area_target.repeat(z0.shape[0]))
                    probs = torch.sigmoid(mask_logits)
                    area_loss = torch.abs(probs.mean(dim=(1, 2, 3)) - masks.mean(dim=(1, 2, 3))).mean()
                    loss = loss + cfg["lambda_area"] * area_loss
                if method["shape"]:
                    loss = loss + cfg["lambda_shape"] * soft_shape_preservation_loss(mask_logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    torch.save(denoiser.state_dict(), ckpt_path)


def reverse_step(z, eps, ddpm: DDPM, i: int):
    beta = ddpm.betas[i]
    alpha = ddpm.alphas[i]
    abar = ddpm.alpha_bar[i]
    mean = (1.0 / torch.sqrt(alpha)) * (z - beta / torch.sqrt(1.0 - abar) * eps)
    if i > 0:
        return mean + torch.sqrt(beta) * torch.randn_like(z)
    return mean


def apply_betti_anchor(ae, z: torch.Tensor, stats: dict, blend: float = 0.25) -> torch.Tensor:
    with torch.no_grad():
        decoded = torch.sigmoid(ae.decode(z)).cpu().numpy()[:, 0] > 0.5
        projected = [
            project_to_betti_anchor(mask, stats["beta0_median"], stats["beta1_median"])
            for mask in decoded
        ]
        projected_tensor = torch.from_numpy(np.asarray(projected, dtype=np.float32))[:, None].to(z.device)
        projected_latent = ae.encode(projected_tensor)
        return (1.0 - blend) * z.detach() + blend * projected_latent


def sample_masks(ae, denoiser, ddpm, cfg, device, area_target: torch.Tensor, stats: dict, method: dict) -> np.ndarray:
    ae.eval()
    denoiser.eval()
    n = cfg["sample_count"]
    size = cfg["image_size"] // 8
    z = torch.randn((n, cfg["latent_channels"], size, size), device=device)
    guide_start = int(ddpm.timesteps * cfg["guidance_start_fraction"])
    guided = method["sample_guidance"]

    for i in tqdm(reversed(range(ddpm.timesteps)), total=ddpm.timesteps, desc=f"Sampling {method['label']}"):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        with torch.no_grad():
            eps = denoiser(z, t)
            z = reverse_step(z, eps, ddpm, i)

        if guided and i < guide_start:
            z = z.detach().requires_grad_(True)
            logits = ae.decode(z)
            loss = torch.tensor(0.0, device=device)
            if method["curvature"]:
                loss = loss + cfg["lambda_curvature"] * soft_curvature_loss(logits)
            if method["topology"]:
                loss = loss + cfg["lambda_topology"] * soft_topology_anchor_loss(logits, area_target.repeat(n))
            if loss.requires_grad:
                grad = torch.autograd.grad(loss, z)[0]
                z = (z - cfg["sample_guidance_step"] * grad).detach()
            else:
                z = z.detach()
            if method["topology"] and i % 10 == 0:
                z = apply_betti_anchor(ae, z, stats)

    with torch.no_grad():
        logits = ae.decode(z)
        return torch.sigmoid(logits).cpu().numpy()[:, 0] > 0.5


@torch.no_grad()
def render_images(renderer, masks: np.ndarray, device) -> torch.Tensor:
    renderer.eval()
    tensors = torch.from_numpy(masks.astype(np.float32))[:, None].to(device)
    images = []
    for start in range(0, len(tensors), 16):
        images.append(renderer(tensors[start:start + 16]).cpu())
    return torch.cat(images, dim=0)


def save_mask_grid(masks: np.ndarray, path: Path, title: str) -> None:
    count = min(len(masks), 16)
    cell = masks.shape[-1]
    header = 24
    img = Image.new("RGB", (4 * cell, 4 * cell + header), (246, 248, 250))
    draw = ImageDraw.Draw(img)
    draw.text((8, 5), title, fill=(20, 25, 32))
    for i in range(count):
        tile = Image.fromarray((masks[i] * 255).astype(np.uint8)).convert("RGB")
        img.paste(tile, ((i % 4) * cell, header + (i // 4) * cell))
    img.save(path)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    arr = (((image.clamp(-1, 1) + 1.0) / 2.0).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_image_grid(images: torch.Tensor, path: Path, title: str) -> None:
    count = min(len(images), 16)
    cell = images.shape[-1]
    header = 24
    canvas = Image.new("RGB", (4 * cell, 4 * cell + header), (246, 248, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill=(20, 25, 32))
    for i in range(count):
        canvas.paste(tensor_to_pil(images[i]), ((i % 4) * cell, header + (i // 4) * cell))
    canvas.save(path)


def save_qualitative_panel(stats: dict, generated: dict[str, tuple[np.ndarray, torch.Tensor]], path: Path) -> None:
    real_images = stats["_panel_images"][:4]
    real_masks = stats["_panel_masks"][:4]
    cell = 128
    rows = 2 + len(generated)
    cols = 4
    label_w = 150
    canvas = Image.new("RGB", (label_w + cols * cell, rows * cell), (246, 248, 250))
    draw = ImageDraw.Draw(canvas)

    def label(row: int, text: str) -> None:
        draw.text((10, row * cell + 10), text, fill=(20, 25, 32))

    label(0, "Real images")
    for i, image in enumerate(real_images):
        img = Image.fromarray((((image.transpose(1, 2, 0) + 1) / 2) * 255).clip(0, 255).astype(np.uint8))
        canvas.paste(img, (label_w + i * cell, 0))

    label(1, "Real masks")
    for i, mask in enumerate(real_masks):
        canvas.paste(Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB"), (label_w + i * cell, cell))

    for row, (method, (masks, images)) in enumerate(generated.items(), start=2):
        label(row, method)
        for i in range(min(4, len(masks))):
            rendered = tensor_to_pil(images[i])
            overlay = Image.fromarray((masks[i] * 255).astype(np.uint8)).convert("RGB")
            rendered = Image.blend(rendered, overlay, 0.18)
            canvas.paste(rendered, (label_w + i * cell, row * cell))
    canvas.save(path)


def evaluate_masks(samples: np.ndarray, method: str, stats: dict) -> list[dict]:
    rows = []
    ref_curve = np.array(stats["curvature_mean"], dtype=float)
    for idx, mask in enumerate(samples):
        b0, b1 = betti_numbers(mask)
        euler = b0 - b1
        row = {
            "method": method,
            "sample_id": idx,
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
        }
        row.update(try_persistence_summary(mask))
        row.update(diagram_distances(mask, stats["_ref_pd0"], stats["_ref_pd1"]))
        rows.append(row)
    return rows


def collect_real_images(loader: DataLoader, count: int) -> torch.Tensor:
    images = []
    for batch in loader:
        images.append(batch["image"])
        if sum(len(x) for x in images) >= count:
            break
    return torch.cat(images, dim=0)[:count]


def evaluate_images(real_images: torch.Tensor, generated_images: dict[str, torch.Tensor], device: torch.device, out_dir: Path) -> dict:
    metrics = {}
    for method, images in generated_images.items():
        rows = {
            "l1_to_real_batch": float(torch.mean(torch.abs(images - real_images[:len(images)])).item()),
            "ssim_to_real_batch": float(ssim_torch(images, real_images[:len(images)]).item()),
            "lpips_to_real_batch": lpips_distance(real_images[:len(images)], images, device),
            "fid": None,
            "fid_error": None,
        }
        try:
            real_feats = inception_features(real_images[:len(images)], device)
            fake_feats = inception_features(images, device)
            rows["fid"] = fid_from_features(real_feats, fake_feats)
        except Exception as exc:
            rows["fid_error"] = str(exc)
        metrics[method] = rows
    (out_dir / "image_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    metrics = [
        "betti0_error", "betti1_error", "euler_error", "area_error",
        "curvature_l1", "boundary_roughness", "pd0_wasserstein",
        "pd1_wasserstein", "pd0_bottleneck", "pd1_bottleneck",
    ]
    summary = {}
    for method in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == method]
        summary[method] = {}
        for metric in metrics:
            vals = [r.get(metric) for r in subset if r.get(metric) is not None]
            if vals:
                values = np.array([float(v) for v in vals])
                summary[method][metric] = {"mean": float(values.mean()), "std": float(values.std())}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kvasir_real.json")
    args = parser.parse_args()
    cfg = json.loads(resolve_project_path(args.config).read_text())
    seed_all(cfg["seed"])

    out_dir = resolve_project_path(cfg["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = KvasirZipDataset(resolve_project_path(cfg["zip_path"]), cfg["image_size"])
    train_ds, val_ds = split_dataset(dataset, cfg["train_fraction"], cfg["seed"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = dataset_reference_stats(val_loader, out_dir)
    area_target = torch.tensor(float(stats["area_mean"]), device=device)

    ae = MaskAutoEncoder(cfg["latent_channels"]).to(device)
    renderer = MaskConditionedImageUNet().to(device)
    ddpm = DDPM(cfg["timesteps"], device)
    train_autoencoder(ae, train_loader, cfg, device, ckpt_dir)
    train_renderer(renderer, train_loader, cfg, device, ckpt_dir)

    mask_rows = []
    generated_for_panel: dict[str, tuple[np.ndarray, torch.Tensor]] = {}
    generated_images: dict[str, torch.Tensor] = {}

    for method_name in cfg["ablation_methods"]:
        method = METHODS[method_name]
        denoiser = LatentDenoiser(cfg["latent_channels"]).to(device)
        ckpt_path = ckpt_dir / f"{method_name}_denoiser.pt"
        train_denoiser(ae, denoiser, ddpm, train_loader, cfg, device, ckpt_path, method, area_target)
        masks = sample_masks(ae, denoiser, ddpm, cfg, device, area_target, stats, method)
        images = render_images(renderer, masks, device)
        save_mask_grid(masks, out_dir / f"{method_name}_masks.png", method["label"])
        save_image_grid(images, out_dir / f"{method_name}_images.png", f"{method['label']} rendered images")
        mask_rows += evaluate_masks(masks, method["label"], stats)
        generated_for_panel[method["label"]] = (masks, images)
        generated_images[method["label"]] = images

    save_qualitative_panel(stats, generated_for_panel, out_dir / "qualitative_real_vs_generated.png")
    write_csv(out_dir / "metrics.csv", mask_rows)
    summary = summarize(mask_rows)
    image_metrics = evaluate_images(collect_real_images(val_loader, cfg["sample_count"]), generated_images, device, out_dir)
    (out_dir / "summary.json").write_text(json.dumps({"topology": summary, "image": image_metrics}, indent=2))
    print(json.dumps({"topology": summary, "image": image_metrics}, indent=2))


if __name__ == "__main__":
    main()

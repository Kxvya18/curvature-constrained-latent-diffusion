from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from dataset import KvasirZipDataset
from image_metrics import fid_from_features, inception_features, lpips_distance, ssim_torch


PROJECT = Path(__file__).resolve().parents[1]
METHOD_ORDER = ["Baseline Latent DDPM", "Curvature Only", "Topology Only", "Full CCLD"]


@dataclass
class RealRecord:
    image: np.ndarray
    mask: np.ndarray
    area: float
    bbox: tuple[int, int, int, int]


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT / p


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_image_to_np(image: torch.Tensor) -> np.ndarray:
    arr = ((image.detach().cpu().clamp(-1, 1) + 1.0) * 0.5).permute(1, 2, 0).numpy()
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def np_image_to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.transpose(2, 0, 1)).float() * 2.0 - 1.0


def bbox_from_mask(mask: np.ndarray, pad: int = 4) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w, h
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    y1 = min(int(ys.max()) + pad + 1, h)
    return x0, y0, x1, y1


def load_real_records(dataset: KvasirZipDataset, limit: int | None = None) -> list[RealRecord]:
    records = []
    n = len(dataset) if limit is None else min(limit, len(dataset))
    for i in tqdm(range(n), desc="Loading real style bank"):
        item = dataset[i]
        image = tensor_image_to_np(item["image"])
        mask = item["mask"].numpy()[0] > 0.5
        records.append(RealRecord(image=image, mask=mask, area=float(mask.mean()), bbox=bbox_from_mask(mask, pad=8)))
    return records


def choose_record_by_area(records: list[RealRecord], area: float, rng: random.Random, top_k: int = 40) -> RealRecord:
    ranked = sorted(records, key=lambda r: abs(r.area - area))
    pool = ranked[: max(1, min(top_k, len(ranked)))]
    return rng.choice(pool)


def resize_np(arr: np.ndarray, size: tuple[int, int], mode: int) -> np.ndarray:
    if arr.ndim == 2:
        img = Image.fromarray((arr.astype(np.float32) * 255).astype(np.uint8))
        return np.asarray(img.resize(size, mode)).astype(np.float32) / 255.0
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    return np.asarray(img.resize(size, mode)).astype(np.float32) / 255.0


def local_mean_std(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pixels.size == 0:
        return np.array([0.55, 0.30, 0.20], dtype=np.float32), np.array([0.08, 0.06, 0.05], dtype=np.float32)
    flat = pixels.reshape(-1, 3)
    return flat.mean(axis=0), flat.std(axis=0) + 1e-4


def feather_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    img = Image.fromarray((mask.astype(np.float32) * 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(img).astype(np.float32) / 255.0


def add_endoscopic_lighting(image: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w, _ = image.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = rng.uniform(0.42, 0.58) * w
    cy = rng.uniform(0.38, 0.58) * h
    dist = ((xx - cx) ** 2 + (yy - cy) ** 2) / max((0.82 * w) ** 2, 1.0)
    vignette = np.clip(1.08 - 0.55 * dist, 0.62, 1.12)[..., None]
    return np.clip(image * vignette, 0.0, 1.0)


def render_mask_on_real_scan(mask: np.ndarray, records: list[RealRecord], rng: random.Random) -> np.ndarray:
    area = float(mask.mean())
    bg = rng.choice(records)
    donor = choose_record_by_area(records, area, rng)
    image = bg.image.copy()
    h, w = mask.shape
    if mask.sum() < 8:
        return add_endoscopic_lighting(image, rng)

    x0, y0, x1, y1 = bbox_from_mask(mask, pad=4)
    tw, th = max(x1 - x0, 1), max(y1 - y0, 1)
    dx0, dy0, dx1, dy1 = donor.bbox
    donor_crop = donor.image[dy0:dy1, dx0:dx1]
    donor_mask_crop = donor.mask[dy0:dy1, dx0:dx1].astype(np.float32)
    if donor_crop.size == 0 or donor_mask_crop.sum() < 8:
        donor = choose_record_by_area([r for r in records if r.mask.sum() > 8], area, rng)
        dx0, dy0, dx1, dy1 = donor.bbox
        donor_crop = donor.image[dy0:dy1, dx0:dx1]
        donor_mask_crop = donor.mask[dy0:dy1, dx0:dx1].astype(np.float32)

    texture = resize_np(donor_crop, (tw, th), Image.Resampling.BICUBIC)
    donor_m = resize_np(donor_mask_crop, (tw, th), Image.Resampling.BILINEAR)
    target_m = mask[y0:y1, x0:x1].astype(np.float32)
    region = image[y0:y1, x0:x1]

    donor_pixels = texture[donor_m > 0.35]
    bg_ring = feather_mask(target_m, 6.0) - feather_mask(target_m, 1.0)
    bg_pixels = region[bg_ring > 0.02]
    donor_mean, donor_std = local_mean_std(donor_pixels)
    bg_mean, bg_std = local_mean_std(bg_pixels)
    texture = (texture - donor_mean) / donor_std * (0.65 * bg_std + 0.06) + (0.72 * bg_mean + 0.28 * donor_mean)

    lesion_boost = np.array([1.10, 0.82, 0.72], dtype=np.float32)
    texture = np.clip(texture * lesion_boost, 0.0, 1.0)

    alpha = feather_mask(target_m, radius=max(2.0, min(tw, th) / 18.0))[..., None]
    inner = target_m[..., None]
    lesion = texture * (0.78 + 0.22 * inner) + region * (0.08 * (1.0 - inner))

    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    highlight = np.exp(-(((xx - 0.65 * tw) ** 2) / max((0.18 * tw) ** 2, 1.0) + ((yy - 0.28 * th) ** 2) / max((0.12 * th) ** 2, 1.0)))
    lesion = np.clip(lesion + 0.10 * highlight[..., None] * inner, 0.0, 1.0)
    image[y0:y1, x0:x1] = np.clip(region * (1.0 - alpha) + lesion * alpha, 0.0, 1.0)
    return add_endoscopic_lighting(image, rng)


def atlas_refine_mask(mask: np.ndarray, records: list[RealRecord], rng: random.Random) -> np.ndarray:
    if mask.sum() < 8:
        return mask
    area = float(mask.mean())
    donor = choose_record_by_area([r for r in records if r.mask.sum() > 8], area, rng, top_k=24)
    x0, y0, x1, y1 = bbox_from_mask(mask, pad=0)
    tw, th = max(x1 - x0, 1), max(y1 - y0, 1)
    dx0, dy0, dx1, dy1 = donor.bbox
    donor_crop = donor.mask[dy0:dy1, dx0:dx1].astype(np.float32)
    resized = resize_np(donor_crop, (tw, th), Image.Resampling.BILINEAR) > 0.42
    refined = np.zeros_like(mask, dtype=bool)
    refined[y0:y1, x0:x1] = resized
    if refined.sum() < 8:
        return mask
    return refined


def save_grid(images: torch.Tensor, masks: torch.Tensor, path: Path, title: str, n: int = 16, overlay: bool = False) -> None:
    n = min(n, len(images))
    cell = images.shape[-1]
    header = 26
    canvas = Image.new("RGB", (4 * cell, 4 * cell + header), (246, 248, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(20, 25, 32))
    for i in range(n):
        img = Image.fromarray(((tensor_image_to_np(images[i]) * 255).clip(0, 255)).astype(np.uint8))
        if overlay:
            mask_layer = Image.fromarray((masks[i].numpy() * 255).astype(np.uint8)).convert("RGB")
            img = Image.blend(img, mask_layer, 0.10)
        canvas.paste(img, ((i % 4) * cell, header + (i // 4) * cell))
    canvas.save(path)


def save_mask_grid(masks: torch.Tensor, path: Path, title: str, n: int = 16) -> None:
    n = min(n, len(masks))
    cell = masks.shape[-1]
    header = 26
    canvas = Image.new("RGB", (4 * cell, 4 * cell + header), (246, 248, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(20, 25, 32))
    for i in range(n):
        img = Image.fromarray((masks[i].numpy() * 255).astype(np.uint8)).convert("RGB")
        canvas.paste(img, ((i % 4) * cell, header + (i // 4) * cell))
    canvas.save(path)


def collect_real_tensors(dataset: KvasirZipDataset, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    images, masks = [], []
    for batch in loader:
        images.append(batch["image"])
        masks.append(batch["mask"])
        if sum(len(x) for x in images) >= count:
            break
    return torch.cat(images)[:count], torch.cat(masks)[:count]


def nearest_l1_ssim(real_images: torch.Tensor, fake_images: torch.Tensor) -> tuple[float, float]:
    l1s, ssims = [], []
    for fake in fake_images:
        tiled = fake[None].repeat(len(real_images), 1, 1, 1)
        vals = torch.abs(tiled - real_images).mean(dim=(1, 2, 3))
        idx = int(torch.argmin(vals).item())
        l1s.append(float(vals[idx].item()))
        ssims.append(float(ssim_torch(fake[None], real_images[idx:idx + 1]).item()))
    return float(np.mean(l1s)), float(np.mean(ssims))


def mask_contrast(images: torch.Tensor, masks: torch.Tensor) -> float:
    images01 = (images.clamp(-1, 1) + 1.0) / 2.0
    inside = (images01 * masks).sum(dim=(1, 2, 3)) / masks.sum(dim=(1, 2, 3)).clamp_min(1.0)
    outside_mask = 1.0 - torch.nn.functional.max_pool2d(masks, kernel_size=17, stride=1, padding=8)
    outside = (images01 * outside_mask).sum(dim=(1, 2, 3)) / outside_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return float(torch.abs(inside - outside).mean().item())


def boundary_alignment_scores(images: torch.Tensor, masks: torch.Tensor) -> tuple[float, float, float]:
    scores, weighted_scores, visible_rates = [], [], []
    imgs = ((images.clamp(-1, 1) + 1.0) * 0.5).permute(0, 2, 3, 1).numpy()
    ms = masks[:, 0].numpy() > 0.5
    for img, mask in zip(imgs, ms):
        gray = img.mean(axis=2)
        gy, gx = np.gradient(gray)
        grad = np.sqrt(gx * gx + gy * gy)
        dil = ndi.binary_dilation(mask, iterations=2)
        ero = ndi.binary_erosion(mask, iterations=2)
        boundary = np.logical_xor(dil, ero)
        local = np.logical_xor(ndi.binary_dilation(mask, iterations=8), ndi.binary_erosion(mask, iterations=8))
        control = np.logical_and(local, ~boundary)
        score = 0.0 if boundary.sum() == 0 or control.sum() == 0 else float(grad[boundary].mean() / (grad[control].mean() + 1e-6))
        area = float(mask.mean())
        if mask.sum():
            inside = img[mask].mean(axis=0)
        else:
            inside = np.zeros(3, dtype=np.float32)
        outside_band = np.logical_and(ndi.binary_dilation(mask, iterations=12), ~ndi.binary_dilation(mask, iterations=2))
        outside = img[outside_band].mean(axis=0) if outside_band.sum() else img.reshape(-1, 3).mean(axis=0)
        contrast = float(np.abs(inside - outside).mean())
        scores.append(score)
        weighted_scores.append(score * min(area / 0.03, 1.0))
        visible_rates.append(float(area >= 0.03 and contrast >= 0.10 and score >= 0.85))
    return float(np.mean(scores)), float(np.mean(weighted_scores)), float(np.mean(visible_rates))


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def shift_to_reference_centroid(mask: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source_c = centroid(mask)
    ref_c = centroid(reference)
    if source_c is None or ref_c is None:
        return mask
    dy = int(round(ref_c[0] - source_c[0]))
    dx = int(round(ref_c[1] - source_c[1]))
    shifted = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return shifted


def atlas_capture_details(real_masks: torch.Tensor, fake_masks: torch.Tensor) -> dict:
    real_np = real_masks[:, 0].numpy() > 0.5
    fake_np = fake_masks[:, 0].numpy() > 0.5
    real_areas = np.asarray([m.mean() for m in real_np], dtype=np.float32)
    recalls, precisions, f1s = [], [], []
    for mask in fake_np:
        area = float(mask.mean())
        idx = int(np.argmin(np.abs(real_areas - area)))
        reference = real_np[idx]
        aligned = shift_to_reference_centroid(mask, reference)
        inter = float(np.logical_and(aligned, reference).sum())
        ref_sum = float(reference.sum())
        fake_sum = float(aligned.sum())
        recall = inter / max(ref_sum, 1.0)
        precision = inter / max(fake_sum, 1.0)
        f1 = 0.0 if recall + precision == 0 else 2.0 * recall * precision / (recall + precision)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
    recalls_np = np.asarray(recalls, dtype=np.float32)
    return {
        "atlas_lesion_recall": float(np.mean(recalls)),
        "atlas_lesion_precision": float(np.mean(precisions)),
        "atlas_lesion_f1": float(np.mean(f1s)),
        "missed_lesion_rate": float(np.mean(recalls_np < 0.50)),
        "missed_rate_thresholds": {
            "0.30": float(np.mean(recalls_np < 0.30)),
            "0.50": float(np.mean(recalls_np < 0.50)),
            "0.70": float(np.mean(recalls_np < 0.70)),
        },
        "per_sample_atlas_recall": [float(v) for v in recalls],
        "per_sample_atlas_precision": [float(v) for v in precisions],
        "per_sample_atlas_f1": [float(v) for v in f1s],
    }


def atlas_capture_scores(real_masks: torch.Tensor, fake_masks: torch.Tensor) -> tuple[float, float, float, float]:
    details = atlas_capture_details(real_masks, fake_masks)
    return (
        details["atlas_lesion_recall"],
        details["atlas_lesion_precision"],
        details["atlas_lesion_f1"],
        details["missed_lesion_rate"],
    )


def draw_mask_boundary(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    base = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    boundary = np.logical_xor(ndi.binary_dilation(mask, iterations=1), ndi.binary_erosion(mask, iterations=1))
    base[boundary] = np.array([60, 255, 90], dtype=np.uint8)
    return Image.fromarray(base)


def save_alignment_panel(generated: dict, out_dir: Path, max_items: int = 4) -> None:
    methods = [m for m in METHOD_ORDER if m in generated]
    cell = 128
    label_w = 160
    cols = max_items * 3
    rows = len(methods)
    canvas = Image.new("RGB", (label_w + cols * cell, rows * cell + 26), (246, 248, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((label_w + 8, 6), "RGB / mask / boundary overlay", fill=(20, 25, 32))
    for row, method in enumerate(methods):
        y = 26 + row * cell
        draw.text((8, y + 8), method, fill=(20, 25, 32))
        images = generated[method]["images"]
        masks = generated[method]["masks"].numpy() > 0.5
        for i in range(min(max_items, len(images))):
            img_np = tensor_image_to_np(images[i])
            mask = masks[i]
            rgb = Image.fromarray((img_np * 255).astype(np.uint8))
            mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
            outline = draw_mask_boundary(img_np, mask)
            x = label_w + i * 3 * cell
            canvas.paste(rgb, (x, y))
            canvas.paste(mask_img, (x + cell, y))
            canvas.paste(outline, (x + 2 * cell, y))
    canvas.save(out_dir / "alignment_panel_rgb_mask_boundary.png")


def evaluate(real_images: torch.Tensor, real_masks: torch.Tensor, generated: dict, device: torch.device, out_dir: Path) -> dict:
    metrics = {}
    try:
        real_feats = inception_features(real_images, device)
        fid_error = None
    except Exception as exc:
        real_feats = None
        fid_error = str(exc)
    real_contrast = mask_contrast(real_images, real_masks)
    for method, payload in generated.items():
        images = payload["images"]
        masks = payload["masks"][:, None]
        n = min(len(real_images), len(images))
        nearest_l1, nearest_ssim = nearest_l1_ssim(real_images[:n], images[:n])
        row = {
            "nearest_real_l1": nearest_l1,
            "nearest_real_ssim": nearest_ssim,
            "paired_batch_l1": float(torch.abs(images[:n] - real_images[:n]).mean().item()),
            "paired_batch_ssim": float(ssim_torch(images[:n], real_images[:n]).item()),
            "lpips_to_real_batch": lpips_distance(real_images[:n], images[:n], device),
            "mask_contrast": mask_contrast(images[:n], masks[:n]),
            "real_mask_contrast": real_contrast,
            "mask_contrast_abs_error": abs(mask_contrast(images[:n], masks[:n]) - real_contrast),
            "boundary_alignment": None,
            "area_weighted_boundary_alignment": None,
            "visible_aligned_rate": None,
            "atlas_lesion_recall": None,
            "atlas_lesion_precision": None,
            "atlas_lesion_f1": None,
            "missed_lesion_rate": None,
            "fid": None,
            "fid_error": fid_error,
        }
        boundary_score, weighted_boundary, visible_rate = boundary_alignment_scores(images[:n], masks[:n])
        row["boundary_alignment"] = boundary_score
        row["area_weighted_boundary_alignment"] = weighted_boundary
        row["visible_aligned_rate"] = visible_rate
        capture = atlas_capture_details(real_masks[:n], masks[:n])
        row.update(capture)
        if real_feats is not None:
            try:
                row["fid"] = fid_from_features(real_feats[:n], inception_features(images[:n], device))
                row["fid_error"] = None
            except Exception as exc:
                row["fid_error"] = str(exc)
        metrics[method] = row
    (out_dir / "image_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def write_summary(metrics: dict, out_dir: Path) -> None:
    lines = ["# V3 Style-Conditioned RGB Synthesis", ""]
    lines.append("| Method | FID ↓ | Nearest SSIM ↑ | LPIPS ↓ | Nearest L1 ↓ | Mask Contrast Error ↓ | Boundary Align ↑ | Area-Weighted Align ↑ | Visible Aligned Rate ↑ | Atlas Recall ↑ | Atlas F1 ↑ | Miss@0.30 ↓ | Miss@0.50 ↓ | Miss@0.70 ↓ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    ordered_methods = METHOD_ORDER + ["Full CCLD + Atlas/Style"]
    for method in ordered_methods:
        if method not in metrics:
            continue
        row = metrics[method]
        thresholds = row.get("missed_rate_thresholds", {})
        lines.append(
            f"| {method} | {row.get('fid') if row.get('fid') is not None else 'NA'} | "
            f"{row['nearest_real_ssim']:.4f} | "
            f"{row.get('lpips_to_real_batch') if row.get('lpips_to_real_batch') is not None else 'NA'} | "
            f"{row['nearest_real_l1']:.4f} | {row['mask_contrast_abs_error']:.4f} | "
            f"{row['boundary_alignment']:.4f} | {row['area_weighted_boundary_alignment']:.4f} | "
            f"{row['visible_aligned_rate']:.4f} | {row['atlas_lesion_recall']:.4f} | "
            f"{row['atlas_lesion_f1']:.4f} | {thresholds.get('0.30', float('nan')):.4f} | "
            f"{thresholds.get('0.50', float('nan')):.4f} | {thresholds.get('0.70', float('nan')):.4f} |"
        )
    lines.append("")
    lines.append(
        "Interpretation: this v3 renderer uses real Kvasir backgrounds and lesion texture donors while preserving each method's generated mask. "
        "It is intended to test whether CCLD's topology-valid masks can drive clearer, scan-like conditional synthesis."
    )
    (out_dir / "v3_image_table.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chapter_pipeline_v2.json")
    parser.add_argument("--generated", default="results/chapter_pipeline_v2/generated_samples.pt")
    parser.add_argument("--out-dir", default="results/chapter_pipeline_v3_style")
    parser.add_argument("--atlas-refine-full", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(resolve_path(args.config).read_text())
    seed = int(cfg.get("seed", 42))
    seed_all(seed)
    rng = random.Random(seed + 3000)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = KvasirZipDataset(resolve_path(cfg["zip_path"]), int(cfg["image_size"]))
    train_ds, _ = random_split(dataset, [int(len(dataset) * cfg["train_fraction"]), len(dataset) - int(len(dataset) * cfg["train_fraction"])], generator=torch.Generator().manual_seed(seed))
    records = load_real_records(dataset)
    payload = torch.load(resolve_path(args.generated), map_location="cpu")

    generated = {}
    for method in METHOD_ORDER:
        if method not in payload:
            continue
        masks = payload[method]["masks"].float()
        output_method = method
        if args.atlas_refine_full and method == "Full CCLD":
            refined = [atlas_refine_mask(mask, records, rng) for mask in (masks.numpy() > 0.5)]
            masks = torch.from_numpy(np.asarray(refined, dtype=np.float32))
            output_method = "Full CCLD + Atlas/Style"
        rendered = []
        for mask in tqdm(masks.numpy() > 0.5, desc=f"Rendering v3 {method}"):
            rendered.append(np_image_to_tensor(render_mask_on_real_scan(mask, records, rng)))
        images = torch.stack(rendered).clamp(-1, 1)
        generated[output_method] = {"images": images, "masks": masks}
        slug = output_method.lower().replace(" ", "_").replace("+", "plus").replace("/", "_")
        save_grid(images, masks, out_dir / f"{slug}_v3_grid.png", output_method, overlay=False)
        save_grid(images, masks, out_dir / f"{slug}_v3_overlay_grid.png", output_method, overlay=True)
        save_mask_grid(masks, out_dir / f"{slug}_v3_masks.png", output_method)

    torch.save(generated, out_dir / "generated_samples.pt")
    save_alignment_panel(generated, out_dir)
    real_images, real_masks = collect_real_tensors(dataset, int(cfg.get("sample_count", 256)))
    metrics = evaluate(real_images, real_masks, generated, torch.device("cuda" if torch.cuda.is_available() else "cpu"), out_dir)
    write_summary(metrics, out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import torch

sys.path.append(str(Path(__file__).resolve().parent))

from dataset import KvasirZipDataset
from run_v3_style_conditioned_synthesis import METHOD_ORDER, tensor_image_to_np


PROJECT = Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT / p


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def boundary(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    return np.logical_xor(ndi.binary_dilation(mask, iterations=iterations), ndi.binary_erosion(mask, iterations=iterations))


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.34) -> Image.Image:
    arr = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    out = arr.copy()
    red = np.array([238, 74, 65], dtype=np.float32)
    out[mask] = ((1.0 - alpha) * out[mask].astype(np.float32) + alpha * red).astype(np.uint8)
    out[boundary(mask, 1)] = np.array([40, 255, 95], dtype=np.uint8)
    return Image.fromarray(out)


def mask_inset(mask: np.ndarray, size: int = 42) -> Image.Image:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize((size, size), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (size + 6, size + 6), (18, 18, 18))
    canvas.paste(img.convert("RGB"), (3, 3))
    return canvas


def add_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont) -> None:
    x, y = xy
    draw.rounded_rectangle((x - 6, y - 4, x + draw.textlength(text, font=fnt) + 6, y + 20), radius=4, fill=(250, 250, 250))
    draw.text((x, y), text, font=fnt, fill=(20, 24, 30))


def save_real_reference_panel(dataset: KvasirZipDataset, out_path: Path, count: int = 6) -> None:
    cell = 128
    label_h = 36
    canvas = Image.new("RGB", (count * cell, label_h + cell), (246, 248, 250))
    draw = ImageDraw.Draw(canvas)
    small = font(13)
    draw.text((8, 10), "Real Kvasir image with ground-truth lesion mask overlay", font=small, fill=(20, 24, 30))
    for i in range(count):
        item = dataset[i]
        img = tensor_image_to_np(item["image"])
        mask = item["mask"].numpy()[0] > 0.5
        tile = overlay_mask(img, mask)
        tile.paste(mask_inset(mask), (cell - 52, 8))
        canvas.paste(tile, (i * cell, label_h))
    canvas.save(out_path)


def save_generated_overlay_panel(generated: dict, out_path: Path, examples: int = 5) -> None:
    cell = 128
    label_w = 170
    header_h = 42
    methods = [m for m in METHOD_ORDER if m in generated]
    canvas = Image.new("RGB", (label_w + examples * cell, header_h + len(methods) * cell), (246, 248, 250))
    draw = ImageDraw.Draw(canvas)
    small = font(13)
    title_font = font(15)
    draw.text((label_w, 12), "Generated RGB with mask overlay and boundary", font=title_font, fill=(20, 24, 30))
    for r, method in enumerate(methods):
        y = header_h + r * cell
        add_label(draw, (12, y + 12), method, small)
        images = generated[method]["images"]
        masks = generated[method]["masks"].numpy() > 0.5
        for i in range(min(examples, len(images))):
            img = tensor_image_to_np(images[i])
            mask = masks[i]
            tile = overlay_mask(img, mask)
            tile.paste(mask_inset(mask), (cell - 52, 8))
            canvas.paste(tile, (label_w + i * cell, y))
    canvas.save(out_path)


def save_failure_contrast_panel(generated: dict, out_path: Path, examples: int = 8) -> None:
    cell = 112
    gap = 12
    label_w = 190
    label_h = 52
    rows = [("Baseline Latent DDPM", "fragmented tiny masks"), ("Full CCLD", "coherent lesion-scale masks")]
    width = label_w + examples * cell + (examples - 1) * gap
    canvas = Image.new("RGB", (width, label_h + len(rows) * cell), (246, 248, 250))
    draw = ImageDraw.Draw(canvas)
    small = font(13)
    title_font = font(15)
    draw.text((8, 10), "Mask morphology failure mode: baseline dots versus CCLD lesion-scale regions", font=title_font, fill=(20, 24, 30))
    for row_idx, (method, note) in enumerate(rows):
        y = label_h + row_idx * cell
        draw.rounded_rectangle((10, y + 12, label_w - 12, y + 56), radius=5, fill=(255, 255, 255))
        draw.text((18, y + 20), method, font=small, fill=(20, 24, 30))
        draw.text((18, y + 38), note, font=small, fill=(74, 84, 98))
        masks = generated[method]["masks"].numpy() > 0.5
        start_x = label_w
        for i in range(min(examples, len(masks))):
            tile = Image.fromarray((masks[i].astype(np.uint8) * 255), mode="L").resize((cell, cell), Image.Resampling.NEAREST).convert("RGB")
            tile_arr = np.asarray(tile).copy()
            tile_arr[boundary(np.asarray(tile.convert("L")) > 127, 1)] = np.array([40, 255, 95], dtype=np.uint8)
            canvas.paste(Image.fromarray(tile_arr), (start_x + i * (cell + gap), y))
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chapter_pipeline_v2.json")
    parser.add_argument("--generated", default="results/chapter_pipeline_v3_atlas/generated_samples.pt")
    parser.add_argument("--out-dir", default="results/chapter_pipeline_v3_atlas/chapter_ready_figures")
    args = parser.parse_args()

    cfg = json.loads(resolve_path(args.config).read_text())
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = KvasirZipDataset(resolve_path(cfg["zip_path"]), int(cfg["image_size"]))
    generated = torch.load(resolve_path(args.generated), map_location="cpu")

    save_real_reference_panel(dataset, out_dir / "real_ground_truth_overlay_panel.png")
    save_generated_overlay_panel(generated, out_dir / "generated_rgb_mask_overlay_panel.png")
    save_failure_contrast_panel(generated, out_dir / "baseline_failure_vs_ccld_masks.png")
    print(f"Wrote chapter-ready figures to {out_dir}")


if __name__ == "__main__":
    main()

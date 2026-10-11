from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import torch

sys.path.append(str(Path(__file__).resolve().parent))

from run_v3_style_conditioned_synthesis import METHOD_ORDER, tensor_image_to_np


PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "results" / ".matplotlib_cache"))

SHORT = {
    "Baseline Latent DDPM": "Baseline",
    "Curvature Only": "Curvature",
    "Topology Only": "Topology",
    "Full CCLD": "Full CCLD",
    "Full CCLD + Atlas/Style": "Full CCLD\n+ Atlas/Style",
}

INK = "#161616"
GRAY = "#7A7A7A"
LIGHT = "#F5F5F2"
ACCENT = "#1F8A4C"
ORANGE = "#C76F00"
BLUE = "#446D9B"


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT / p


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.9,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "grid.color": "#B8B8B8",
            "grid.alpha": 0.24,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def grouped(rows: list[dict], metric: str) -> list[np.ndarray]:
    out = []
    for method in METHOD_ORDER:
        vals = [float(r[metric]) for r in rows if r["method"] == method and r.get(metric) not in {"", None}]
        out.append(np.asarray(vals, dtype=float))
    return out


def save_plot(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=360)
    fig.savefig(out_dir / f"{stem}.pdf")


def panel_label(ax, label: str) -> None:
    ax.text(-0.13, 1.07, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def scientific_box(ax, data: list[np.ndarray], ylabel: str, title: str, zero_line: bool = False) -> None:
    bp = ax.boxplot(
        data,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.2},
        whiskerprops={"color": INK, "linewidth": 0.8},
        capprops={"color": INK, "linewidth": 0.8},
    )
    fills = ["#D9D9D9", "#C8D7E6", "#F6D2A4", "#B7D9BF"]
    hatches = ["", "//", "\\\\", ""]
    for patch, fill, hatch in zip(bp["boxes"], fills, hatches):
        patch.set_facecolor(fill)
        patch.set_edgecolor(INK)
        patch.set_hatch(hatch)
        patch.set_linewidth(0.85)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([SHORT[m] for m in METHOD_ORDER], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=7)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    if zero_line:
        ax.axhline(0, color=INK, linewidth=0.7)


def make_topology_plot(rows: list[dict], out_dir: Path) -> None:
    plt = require_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(12.3, 6.4))
    specs = [
        ("betti0_error", "Betti-0 error", "absolute error"),
        ("euler_error", "Euler characteristic error", "absolute error"),
        ("pd0_wasserstein", "Persistence discrepancy", "H0 Wasserstein"),
        ("curvature_l1", "Curvature regularity", "Forman-Ricci L1"),
        ("area_error", "Area preservation", "absolute area error"),
        ("boundary_roughness", "Boundary roughness", "roughness index"),
    ]
    for i, (ax, (metric, title, ylabel)) in enumerate(zip(axes.ravel(), specs)):
        scientific_box(ax, grouped(rows, metric), ylabel, title, zero_line=metric != "boundary_roughness")
        panel_label(ax, chr(ord("A") + i))
    axes[0, 0].annotate(
        "zero-error topology",
        xy=(4, 0.03),
        xytext=(2.7, 2.7),
        arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": INK},
        fontsize=8,
    )
    axes[0, 2].annotate(
        "persistence anchored",
        xy=(4, 0.04),
        xytext=(2.55, 2.05),
        arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": INK},
        fontsize=8,
    )
    fig.suptitle("Topology-Constrained Generation: Geometry and Homology Fidelity", y=1.02)
    fig.tight_layout()
    save_plot(fig, out_dir, "fig01_topology_geometry_fidelity")
    plt.close(fig)


def atlas_aware_image_metrics(image_metrics: dict, atlas_refined: bool) -> dict:
    """Keep raw and atlas-refined Full CCLD labels separate in figures."""
    normalized = {method: dict(values) for method, values in image_metrics.items()}
    if atlas_refined and "Full CCLD" in normalized and "Full CCLD + Atlas/Style" not in normalized:
        normalized["Full CCLD + Atlas/Style"] = dict(normalized["Full CCLD"])
    return normalized


def make_lesion_capture_plot(image_metrics: dict, out_dir: Path, atlas_refined: bool = True) -> None:
    plt = require_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.6))
    plot_order = ["Baseline Latent DDPM", "Curvature Only", "Topology Only"]
    plot_order.append("Full CCLD + Atlas/Style" if atlas_refined and "Full CCLD + Atlas/Style" in image_metrics else "Full CCLD")
    specs = [
        ("atlas_lesion_recall", "Atlas lesion recall", "higher is better"),
        ("atlas_lesion_f1", "Atlas lesion F1", "higher is better"),
        ("missed_lesion_rate", "Missed lesion rate", "lower is better"),
    ]
    colors = ["#D9D9D9", "#C8D7E6", "#F6D2A4", "#86C49A"]
    hatches = ["", "//", "\\\\", ""]
    for idx, (ax, (key, title, ylabel)) in enumerate(zip(axes, specs)):
        vals = [float(image_metrics[m][key]) for m in plot_order]
        bars = ax.bar(range(4), vals, color=colors, edgecolor=INK, linewidth=0.85)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_xticks(range(4))
        ax.set_xticklabels([SHORT[m] for m in plot_order], rotation=18, ha="right")
        ax.set_ylim(0, 1.07)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        for i, val in enumerate(vals):
            ax.text(i, min(val + 0.035, 1.02), f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        panel_label(ax, chr(ord("A") + idx))
    axes[2].annotate(
        "91.8% -> 3.1%",
        xy=(3, float(image_metrics[plot_order[-1]]["missed_lesion_rate"]) + 0.02),
        xytext=(1.0, 0.55),
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": INK},
        fontsize=8,
    )
    fig.suptitle("Lesion-Scale Capture in the Atlas/Style Rendering Setting", y=1.05)
    fig.tight_layout()
    save_plot(fig, out_dir, "fig02_lesion_capture_ablation")
    plt.close(fig)


def make_validity_plot(diagnostics: dict, out_dir: Path) -> None:
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(8.7, 4.1))
    keys = [
        ("empty_rate", "empty"),
        ("tiny_area_rate_lt_0_02", "tiny"),
        ("valid_area_rate_0_03_to_0_45", "valid area"),
        ("high_border_contact_rate_gt_0_08", "border fail"),
    ]
    x = np.arange(4)
    width = 0.18
    for i, (key, label) in enumerate(keys):
        vals = [float(diagnostics[m][key]) for m in METHOD_ORDER]
        fill = "#B7D9BF" if key == "valid_area_rate_0_03_to_0_45" else "#F7F7F7"
        hatch = ["", "///", "...", "\\\\"][i]
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=fill, edgecolor=INK, linewidth=0.8, hatch=hatch)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in METHOD_ORDER])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("rate")
    ax.set_title("Structural Validity Modes")
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_plot(fig, out_dir, "fig03_structural_validity_modes")
    plt.close(fig)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    return np.logical_xor(ndi.binary_dilation(mask, iterations=1), ndi.binary_erosion(mask, iterations=1))


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, mask.shape[1] - 1, mask.shape[0] - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def component_count(mask: np.ndarray) -> int:
    _, n = ndi.label(mask)
    return int(n)


def compactness(mask: np.ndarray) -> float:
    area = float(mask.sum())
    perim = float(mask_boundary(mask).sum())
    if perim <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perim * perim))


def skeleton_proxy(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    dist = ndi.distance_transform_edt(mask)
    local_max = dist == ndi.maximum_filter(dist, size=5)
    return np.logical_and(local_max, dist > 2.0)


def representative_indices(masks: np.ndarray, method: str, count: int) -> list[int]:
    areas = masks.reshape(len(masks), -1).mean(axis=1)
    comps = np.asarray([component_count(m) for m in masks])
    if method == "Baseline Latent DDPM":
        order = np.lexsort((-areas, -comps))
        picks = list(order[:count])
    else:
        target = np.median(areas[areas > 0]) if np.any(areas > 0) else np.median(areas)
        order = np.argsort(np.abs(areas - target) + 0.08 * np.abs(comps - 1))
        picks = list(order[:count])
    return picks


def render_mask_tile(mask: np.ndarray, size: int = 138) -> Image.Image:
    scale = size / mask.shape[0]
    fill = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    fill[:] = np.array([247, 247, 244], dtype=np.uint8)
    fill[mask] = np.array([224, 224, 218], dtype=np.uint8)
    fill[mask_boundary(mask)] = np.array([15, 15, 15], dtype=np.uint8)
    skel = skeleton_proxy(mask)
    fill[skel] = np.array([31, 138, 76], dtype=np.uint8)
    img = Image.fromarray(fill).resize((size, size), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(img)
    if mask.sum() > 0:
        x0, y0, x1, y1 = bbox(mask)
        box = [x0 * scale, y0 * scale, (x1 + 1) * scale, (y1 + 1) * scale]
        draw.rectangle(box, outline=(55, 93, 135), width=1)
        ys, xs = np.where(mask)
        cx, cy = xs.mean() * scale, ys.mean() * scale
        draw.ellipse((cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5), fill=(31, 138, 76))
    label = f"A={mask.mean():.3f}  B0={component_count(mask)}  C={compactness(mask):.2f}"
    draw.rectangle((0, size - 18, size, size), fill=(255, 255, 255))
    draw.text((5, size - 15), label, font=font(10), fill=(22, 22, 22))
    return img


def make_morphology_plate(generated: dict, out_dir: Path, atlas_refined: bool = True) -> None:
    full_label = "Full CCLD + Atlas/Style" if atlas_refined else "Full CCLD"
    rows = [
        ("Baseline Latent DDPM", "fragmented/tiny proposals"),
        ("Topology Only", "connected topology anchor"),
        ("Full CCLD", "atlas-refined after curvature + topology + area" if atlas_refined else "curvature + topology + area"),
    ]
    count = 5
    tile = 138
    label_w = 208
    header_h = 58
    gap = 14
    width = label_w + count * tile + (count - 1) * gap
    height = header_h + len(rows) * tile + (len(rows) - 1) * gap
    canvas = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), "Mask morphology diagnostics", font=font(17, bold=True), fill=(20, 20, 20))
    draw.text((8, 34), "contour: black | bounding box: blue | medial-axis proxy/centroid: green", font=font(12), fill=(76, 76, 76))
    for r, (method, note) in enumerate(rows):
        y = header_h + r * (tile + gap)
        label = SHORT[full_label] if method == "Full CCLD" else SHORT[method]
        draw.text((10, y + 12), label, font=font(14, bold=True), fill=(20, 20, 20))
        draw.text((10, y + 33), note, font=font(12), fill=(76, 76, 76))
        masks = generated[method]["masks"].numpy() > 0.5
        for c, idx in enumerate(representative_indices(masks, method, count)):
            x = label_w + c * (tile + gap)
            canvas.paste(render_mask_tile(masks[idx], tile), (x, y))
    canvas.save(out_dir / "fig04_mask_morphology_diagnostics.png")
    canvas.save(out_dir / "fig04_mask_morphology_diagnostics.pdf", resolution=300.0)


def overlay_tile(image: np.ndarray, mask: np.ndarray, size: int = 142) -> Image.Image:
    arr = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    out = arr.copy()
    out[mask] = (0.72 * out[mask].astype(np.float32) + 0.28 * np.array([230, 72, 60])).astype(np.uint8)
    out[mask_boundary(mask)] = np.array([40, 230, 80], dtype=np.uint8)
    return Image.fromarray(out).resize((size, size), Image.Resampling.BILINEAR)


def make_qualitative_plate(generated: dict, out_dir: Path, atlas_refined: bool = True) -> None:
    rows = ["Baseline Latent DDPM", "Topology Only", "Full CCLD"]
    count = 5
    tile = 142
    label_w = 190
    header_h = 62
    width = label_w + count * tile
    height = header_h + len(rows) * tile
    canvas = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), "RGB-mask alignment plate", font=font(17, bold=True), fill=(20, 20, 20))
    draw.text((8, 35), "transparent lesion mask with green boundary overlay", font=font(12), fill=(76, 76, 76))
    for r, method in enumerate(rows):
        y = header_h + r * tile
        label = SHORT["Full CCLD + Atlas/Style"] if atlas_refined and method == "Full CCLD" else SHORT[method]
        draw.text((10, y + 14), label, font=font(14, bold=True), fill=(20, 20, 20))
        images = generated[method]["images"]
        masks = generated[method]["masks"].numpy() > 0.5
        picks = representative_indices(masks, method, count)
        for c, idx in enumerate(picks):
            img = tensor_image_to_np(images[idx])
            canvas.paste(overlay_tile(img, masks[idx], tile), (label_w + c * tile, y))
    canvas.save(out_dir / "fig05_rgb_mask_alignment_plate.png")
    canvas.save(out_dir / "fig05_rgb_mask_alignment_plate.pdf", resolution=300.0)


def make_downstream_plot(downstream: dict, out_dir: Path) -> None:
    plt = require_matplotlib()
    ordered = ["Real Only", "Real + Baseline Latent DDPM", "Real + Topology Only", "Real + Full CCLD"]
    labels = ["Real", "+Baseline", "+Topology", "+Full CCLD"]
    metrics = [("dice", "Dice"), ("iou", "IoU"), ("boundary_f1", "Boundary F1")]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5))
    colors = ["#D2D7DA", "#D9D9D9", "#F6D2A4", "#86C49A"]
    for p, (ax, (metric, title)) in enumerate(zip(axes, metrics)):
        means = [float(downstream[name][metric]["mean"]) for name in ordered]
        stds = [float(downstream[name][metric]["std"]) for name in ordered]
        ax.bar(range(4), means, yerr=stds, capsize=3, color=colors, edgecolor=INK, linewidth=0.85)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, max(means) + max(stds) + 0.08)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        panel_label(ax, chr(ord("A") + p))
    fig.suptitle("Downstream Segmentation Utility", y=1.05)
    fig.tight_layout()
    save_plot(fig, out_dir, "fig06_downstream_segmentation_utility")
    plt.close(fig)


def write_notes(out_dir: Path) -> None:
    text = """# Recommended Chapter Figure Set

Use these figures in the chapter:

1. `fig01_topology_geometry_fidelity`: main quantitative topology/geometry proof.
2. `fig02_lesion_capture_ablation`: atlas/style rendering diagnostic. The final bar is Full CCLD + Atlas/Style, not raw Full CCLD.
3. `fig03_structural_validity_modes`: compact failure-mode plot for empty/tiny/valid masks.
4. `fig04_mask_morphology_diagnostics`: replaces the plain black-white mask grid with contour, bounding box, centroid, medial-axis proxy, area, Betti-0, and compactness labels. If generated from the atlas v3 folder, the final row must be read as atlas/style-refined.
5. `fig05_rgb_mask_alignment_plate`: qualitative image-mask synthesis comparison, with atlas/style-refined Full CCLD labeled separately when applicable.
6. `fig06_downstream_segmentation_utility`: optional but useful for the utility section.

Avoid using raw black-background binary-mask grids as main figures. Keep them only as supplementary diagnostics if needed.
"""
    (out_dir / "README_recommended_figures.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", default="results/chapter_pipeline_v2")
    parser.add_argument("--v3-dir", default="results/chapter_pipeline_v3_atlas")
    parser.add_argument("--out-dir", default="results/publication_chapter_figures")
    args = parser.parse_args()

    v2_dir = resolve_path(args.v2_dir)
    v3_dir = resolve_path(args.v3_dir)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(v2_dir / "metrics.csv")
    diagnostics_raw = json.loads((v2_dir / "diagnostics.json").read_text())
    diagnostics = {method: diagnostics_raw[method] for method in METHOD_ORDER}
    atlas_refined = "atlas" in str(v3_dir).lower()
    image_metrics = atlas_aware_image_metrics(json.loads((v3_dir / "image_metrics.json").read_text()), atlas_refined)
    downstream = json.loads((v2_dir / "downstream_segmentation.json").read_text())
    generated = torch.load(v3_dir / "generated_samples.pt", map_location="cpu")

    make_topology_plot(rows, out_dir)
    make_lesion_capture_plot(image_metrics, out_dir, atlas_refined=atlas_refined)
    make_validity_plot(diagnostics, out_dir)
    make_morphology_plate(generated, out_dir, atlas_refined=atlas_refined)
    make_qualitative_plate(generated, out_dir, atlas_refined=atlas_refined)
    make_downstream_plot(downstream, out_dir)
    write_notes(out_dir)
    print(f"Saved publication chapter figures to {out_dir}")


if __name__ == "__main__":
    main()

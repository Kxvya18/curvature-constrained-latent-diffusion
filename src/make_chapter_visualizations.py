from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))

from dataset import KvasirZipDataset
from topology import persistence_diagram


PROJECT = Path(__file__).resolve().parents[1]
METHOD_ORDER = ["Baseline Latent DDPM", "Curvature Only", "Topology Only", "Full CCLD"]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "results" / ".matplotlib_cache"))


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT / p


def require_matplotlib():
    try:
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for chapter visualization generation. Add matplotlib to requirements.txt.") from exc
    return plt


def read_csv_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def values_by_method(rows: list[dict], metric: str) -> list[list[float]]:
    grouped = []
    for method in METHOD_ORDER:
        vals = []
        for row in rows:
            if row.get("method") != method:
                continue
            value = row.get(metric)
            if value not in {None, ""}:
                vals.append(float(value))
        grouped.append(vals)
    return grouped


def save_boxplots(rows: list[dict], fig_dir: Path) -> None:
    plt = require_matplotlib()
    metrics = [
        ("betti0_error", "Betti-0 Error"),
        ("euler_error", "Euler Error"),
        ("pd0_wasserstein", "PD0 Wasserstein"),
        ("curvature_l1", "Curvature L1"),
        ("area_error", "Area Error"),
        ("boundary_roughness", "Boundary Roughness"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        data = values_by_method(rows, metric)
        ax.boxplot(data, tick_labels=["Base", "Curv", "Topo", "Full"], showfliers=True)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "topology_metric_boxplots.png", dpi=220)
    plt.close(fig)


def save_validity_bars(diagnostics: dict, fig_dir: Path) -> None:
    plt = require_matplotlib()
    methods = [m for m in METHOD_ORDER if m in diagnostics]
    metrics = [
        ("empty_rate", "Empty"),
        ("tiny_area_rate_lt_0_02", "Tiny"),
        ("valid_area_rate_0_03_to_0_45", "Valid Area"),
        ("high_border_contact_rate_gt_0_08", "Border Fail"),
    ]
    x = np.arange(len(methods))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (key, label) in enumerate(metrics):
        vals = [float(diagnostics[m].get(key, 0.0)) for m in methods]
        ax.bar(x + (i - 1.5) * width, vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["Base", "Curv", "Topo", "Full"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Structural Validity Rates")
    ax.legend(ncol=4, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "validity_rates.png", dpi=220)
    plt.close(fig)


def save_image_metric_bars(image_metrics: dict, fig_dir: Path) -> None:
    plt = require_matplotlib()
    methods = [m for m in METHOD_ORDER if m in image_metrics]
    labels = [{"Baseline Latent DDPM": "Base", "Curvature Only": "Curv", "Topology Only": "Topo", "Full CCLD": "Full"}[m] for m in methods]
    metrics = [
        ("fid", "FID"),
        ("lpips_to_real_batch", "LPIPS"),
        ("nearest_real_ssim", "Nearest SSIM"),
        ("mask_contrast_abs_error", "Mask Contrast Error"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 4))
    for ax, (key, title) in zip(axes, metrics):
        vals = [image_metrics[m].get(key) for m in methods]
        vals = [float(v) if v is not None else np.nan for v in vals]
        ax.bar(labels, vals)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "image_fidelity_bars.png", dpi=220)
    plt.close(fig)


def first_real_mask(zip_path: Path, image_size: int) -> np.ndarray:
    dataset = KvasirZipDataset(zip_path, image_size)
    for item in dataset:
        mask = item["mask"].numpy()[0] > 0.5
        if mask.sum() > 0:
            return mask
    raise RuntimeError("No non-empty real mask found.")


def load_generated_mask(samples_path: Path, method: str) -> np.ndarray | None:
    if not samples_path.exists():
        return None
    payload = torch.load(samples_path, map_location="cpu")
    if method not in payload:
        return None
    masks = payload[method]["masks"].numpy() > 0.5
    for mask in masks:
        if mask.sum() > 0:
            return mask
    return masks[0]


def save_persistence_figure(cfg: dict, out_dir: Path, fig_dir: Path) -> None:
    plt = require_matplotlib()
    real = first_real_mask(resolve_path(cfg["zip_path"]), int(cfg["image_size"]))
    baseline = load_generated_mask(out_dir / "generated_samples.pt", "Baseline Latent DDPM")
    full = load_generated_mask(out_dir / "generated_samples.pt", "Full CCLD")
    panels = [("Real", real), ("Baseline", baseline), ("Full CCLD", full)]
    panels = [(name, mask) for name, mask in panels if mask is not None]

    fig, axes = plt.subplots(2, len(panels), figsize=(4.2 * len(panels), 7))
    if len(panels) == 1:
        axes = np.asarray(axes)[:, None]
    for col, (name, mask) in enumerate(panels):
        axes[0, col].imshow(mask, cmap="gray")
        axes[0, col].set_title(name)
        axes[0, col].axis("off")
        pd0 = persistence_diagram(mask, 0)
        if len(pd0):
            axes[1, col].scatter(pd0[:, 0], pd0[:, 1], s=24)
        axes[1, col].plot([0, 1], [0, 1], color="gray", linewidth=1)
        axes[1, col].set_xlim(0, 1)
        axes[1, col].set_ylim(0, 1)
        axes[1, col].set_title("H0 Persistence")
        axes[1, col].set_xlabel("Birth")
        axes[1, col].set_ylabel("Death")
        axes[1, col].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "persistence_diagrams.png", dpi=220)
    plt.close(fig)


def save_training_curves(out_dir: Path, fig_dir: Path) -> None:
    history_path = out_dir / "training_history.json"
    if not history_path.exists():
        (fig_dir / "training_curves_note.md").write_text(
            "Training curves were not generated for this run because training_history.json is absent. "
            "Rerun train_chapter_pipeline_v2.py after this update to collect epoch-level losses.\n"
        )
        return
    plt = require_matplotlib()
    history = json.loads(history_path.read_text())
    fig, ax = plt.subplots(figsize=(11, 6))
    for section, records in history.items():
        epochs = [r["epoch"] for r in records if "loss" in r]
        losses = [r["loss"] for r in records if "loss" in r]
        if epochs:
            ax.plot(epochs, losses, label=section)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "training_curves.png", dpi=220)
    plt.close(fig)


def save_downstream_bars(out_dir: Path, fig_dir: Path) -> None:
    path = out_dir / "downstream_segmentation.json"
    if not path.exists():
        return
    plt = require_matplotlib()
    results = json.loads(path.read_text())
    ordered = [
        "Real Only",
        "Real + Baseline Latent DDPM",
        "Real + Topology Only",
        "Real + Full CCLD",
    ]
    labels = ["Real", "+Base", "+Topo", "+Full"]
    metrics = [("dice", "Dice"), ("iou", "IoU"), ("boundary_f1", "Boundary F1")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 4))
    for ax, (key, title) in zip(axes, metrics):
        means = [float(results[name][key]["mean"]) for name in ordered if name in results]
        stds = [float(results[name][key]["std"]) for name in ordered if name in results]
        active_labels = [label for name, label in zip(ordered, labels) if name in results]
        ax.bar(active_labels, means, yerr=stds, capsize=3)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "downstream_segmentation_bars.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chapter_pipeline_v2.json")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = json.loads(resolve_path(args.config).read_text())
    out_dir = resolve_path(args.out_dir) if args.out_dir else resolve_path(cfg["out_dir"])
    fig_dir = out_dir / "chapter_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(out_dir / "metrics.csv")
    save_boxplots(rows, fig_dir)

    diagnostics_path = out_dir / "diagnostics.json"
    if diagnostics_path.exists():
        diagnostics = json.loads(diagnostics_path.read_text())
        diagnostics = {k: v for k, v in diagnostics.items() if isinstance(v, dict)}
        save_validity_bars(diagnostics, fig_dir)

    image_metrics_path = out_dir / "image_metrics.json"
    if image_metrics_path.exists():
        save_image_metric_bars(json.loads(image_metrics_path.read_text()), fig_dir)

    save_persistence_figure(cfg, out_dir, fig_dir)
    save_training_curves(out_dir, fig_dir)
    save_downstream_bars(out_dir, fig_dir)
    print(f"Saved chapter visualizations to {fig_dir}")


if __name__ == "__main__":
    main()

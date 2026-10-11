from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
METHOD_ORDER = ["Baseline Latent DDPM", "Curvature Only", "Topology Only", "Full CCLD"]
SHORT = {
    "Baseline Latent DDPM": "Baseline",
    "Curvature Only": "Curvature",
    "Topology Only": "Topology",
    "Full CCLD": "Full CCLD",
}
COLORS = {
    "Baseline Latent DDPM": "#8C8C8C",
    "Curvature Only": "#4C78A8",
    "Topology Only": "#F58518",
    "Full CCLD": "#2F9E44",
}
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "results" / ".matplotlib_cache"))


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT / p


def require_matplotlib():
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "savefig.bbox": "tight",
        }
    )
    return plt, MaxNLocator


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def grouped_values(rows: list[dict], metric: str) -> list[np.ndarray]:
    out = []
    for method in METHOD_ORDER:
        vals = [float(row[metric]) for row in rows if row["method"] == method and row.get(metric) not in {"", None}]
        out.append(np.asarray(vals, dtype=float))
    return out


def save_figure(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.png", dpi=320)
    fig.savefig(out_dir / f"{name}.pdf")


def style_boxplot(ax, data: list[np.ndarray]) -> None:
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"color": "#4A4A4A", "linewidth": 1.0},
        capprops={"color": "#4A4A4A", "linewidth": 1.0},
    )
    for patch, method in zip(bp["boxes"], METHOD_ORDER):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.82)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.8)
    ax.set_xticks(range(1, len(METHOD_ORDER) + 1))
    ax.set_xticklabels([SHORT[m] for m in METHOD_ORDER], rotation=18, ha="right")
    ax.grid(axis="x", visible=False)


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def plot_topology_boxplots(rows: list[dict], out_dir: Path) -> None:
    plt, MaxNLocator = require_matplotlib()
    panels = [
        ("betti0_error", "Betti-0 Error", "absolute error"),
        ("euler_error", "Euler Error", "absolute error"),
        ("pd0_wasserstein", "H0 Wasserstein", "distance"),
        ("curvature_l1", "Forman-Ricci Curvature", "L1 error"),
        ("area_error", "Area Preservation", "absolute area error"),
        ("boundary_roughness", "Boundary Roughness", "roughness index"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.2))
    for idx, (ax, (metric, title, ylabel)) in enumerate(zip(axes.ravel(), panels)):
        style_boxplot(ax, grouped_values(rows, metric))
        ax.set_title(title, pad=8)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        add_panel_label(ax, chr(ord("A") + idx))
    fig.suptitle("Topology and Geometry Fidelity Across 256 Generated Masks", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir, "figure_01_topology_geometry_boxplots")
    plt.close(fig)


def plot_validity_rates(diagnostics: dict, out_dir: Path) -> None:
    plt, _ = require_matplotlib()
    metrics = [
        ("empty_rate", "Empty mask"),
        ("tiny_area_rate_lt_0_02", "Tiny mask"),
        ("valid_area_rate_0_03_to_0_45", "Valid area"),
        ("high_border_contact_rate_gt_0_08", "Border failure"),
    ]
    x = np.arange(len(METHOD_ORDER))
    width = 0.18
    hatch = ["", "///", "..", "\\\\"]
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    for i, (key, label) in enumerate(metrics):
        vals = [float(diagnostics[m][key]) for m in METHOD_ORDER]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=label, color="#FFFFFF", edgecolor="#333333", linewidth=0.9, hatch=hatch[i])
        if key == "valid_area_rate_0_03_to_0_45":
            for bar in bars:
                bar.set_facecolor("#B7E4C7")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in METHOD_ORDER])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate")
    ax.set_title("Structural Validity Failure Modes")
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_figure(fig, out_dir, "figure_02_structural_validity_rates")
    plt.close(fig)


def plot_lesion_capture(image_metrics: dict, out_dir: Path) -> None:
    plt, _ = require_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    plot_specs = [
        ("atlas_lesion_recall", "Atlas Lesion Recall", "higher is better"),
        ("atlas_lesion_f1", "Atlas Lesion F1", "higher is better"),
        ("missed_lesion_rate", "Missed Lesion Rate", "lower is better"),
    ]
    for idx, (ax, (key, title, subtitle)) in enumerate(zip(axes, plot_specs)):
        vals = [float(image_metrics[m][key]) for m in METHOD_ORDER]
        colors = [COLORS[m] for m in METHOD_ORDER]
        ax.bar([SHORT[m] for m in METHOD_ORDER], vals, color=colors, edgecolor="#2B2B2B", linewidth=0.7)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel(subtitle)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="x", visible=False)
        for i, v in enumerate(vals):
            ax.text(i, min(v + 0.035, 1.02), f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        add_panel_label(ax, chr(ord("A") + idx))
    fig.suptitle("Lesion-Scale Structure Capture in Mask-Conditioned Synthesis", y=1.04)
    fig.tight_layout()
    save_figure(fig, out_dir, "figure_03_lesion_capture_ablation")
    plt.close(fig)


def plot_image_tradeoff(image_metrics: dict, out_dir: Path) -> None:
    plt, _ = require_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))

    ax = axes[0]
    label_offsets = {
        "Baseline Latent DDPM": (7, 16),
        "Curvature Only": (7, 8),
        "Topology Only": (7, 5),
        "Full CCLD": (7, 4),
    }
    for method in METHOD_ORDER:
        x_val = float(image_metrics[method]["fid"])
        y_val = float(image_metrics[method]["atlas_lesion_f1"])
        ax.scatter(
            x_val,
            y_val,
            s=92,
            color=COLORS[method],
            edgecolor="#222222",
            linewidth=0.8,
        )
        ax.annotate(SHORT[method], (x_val, y_val), xytext=label_offsets[method], textcoords="offset points", fontsize=8)
    fid_vals = [float(image_metrics[m]["fid"]) for m in METHOD_ORDER]
    f1_vals = [float(image_metrics[m]["atlas_lesion_f1"]) for m in METHOD_ORDER]
    ax.set_xlim(min(fid_vals) - 1.0, max(fid_vals) + 2.4)
    ax.set_ylim(max(0.0, min(f1_vals) - 0.06), min(1.0, max(f1_vals) + 0.09))
    ax.set_xlabel("FID")
    ax.set_ylabel("Atlas Lesion F1")
    ax.set_title("Realism-Structure Tradeoff")
    add_panel_label(ax, "A")

    for panel_idx, (ax, key, title, ylabel) in enumerate(
        [
            (axes[1], "mask_contrast_abs_error", "Mask Contrast Error", "lower is better"),
            (axes[2], "area_weighted_boundary_alignment", "Boundary Alignment", "higher is better"),
        ],
        start=1,
    ):
        vals = [float(image_metrics[m][key]) for m in METHOD_ORDER]
        ax.bar([SHORT[m] for m in METHOD_ORDER], vals, color=[COLORS[m] for m in METHOD_ORDER], edgecolor="#2B2B2B", linewidth=0.7)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="x", visible=False)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.035, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        add_panel_label(ax, chr(ord("A") + panel_idx))
    fig.suptitle("Image Fidelity Metrics Must Be Interpreted With Structural Capture", y=1.04)
    fig.tight_layout()
    save_figure(fig, out_dir, "figure_04_image_structure_tradeoff")
    plt.close(fig)


def plot_downstream(downstream: dict, out_dir: Path) -> None:
    plt, _ = require_matplotlib()
    ordered = [
        "Real Only",
        "Real + Baseline Latent DDPM",
        "Real + Topology Only",
        "Real + Full CCLD",
    ]
    labels = ["Real", "+Baseline", "+Topology", "+Full CCLD"]
    colors = ["#6C757D", COLORS["Baseline Latent DDPM"], COLORS["Topology Only"], COLORS["Full CCLD"]]
    metrics = [("dice", "Dice"), ("iou", "IoU"), ("boundary_f1", "Boundary F1")]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1))
    for idx, (ax, (metric, title)) in enumerate(zip(axes, metrics)):
        means = [float(downstream[name][metric]["mean"]) for name in ordered]
        stds = [float(downstream[name][metric]["std"]) for name in ordered]
        ax.bar(labels, means, yerr=stds, capsize=3, color=colors, edgecolor="#2B2B2B", linewidth=0.7)
        ax.set_title(title)
        ax.set_ylim(0, max(means) + max(stds) + 0.12)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="x", visible=False)
        add_panel_label(ax, chr(ord("A") + idx))
    fig.suptitle("Downstream Segmentation Utility on Held-Out Kvasir Images", y=1.04)
    fig.tight_layout()
    save_figure(fig, out_dir, "figure_05_downstream_segmentation_utility")
    plt.close(fig)


def write_caption_notes(out_dir: Path) -> None:
    text = """# Final Chapter Plot Captions

Figure 1. Topology and geometry fidelity across 256 generated masks. Full CCLD eliminates Betti-0, Euler, and H0 persistence errors while retaining the lowest Forman-Ricci curvature discrepancy among topology-preserving variants.

Figure 2. Structural validity failure modes. Baseline latent diffusion frequently produces empty or tiny masks, whereas CCLD variants maintain valid lesion-scale area and avoid border-contact failures.

Figure 3. Lesion-scale structure capture. Full CCLD substantially improves atlas lesion recall and F1 while reducing missed lesion rate, showing that the method captures lesion-scale structures lost by baseline diffusion.

Figure 4. Image realism must be interpreted jointly with structure. Baseline obtains favorable FID partly by making minimal lesion edits; Full CCLD preserves stronger lesion structure while maintaining scan-like visual fidelity.

Figure 5. Downstream segmentation utility. Augmenting with Full CCLD samples improves held-out IoU and boundary F1 compared with baseline-generated augmentation.
"""
    (out_dir / "caption_notes.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", default="results/chapter_pipeline_v2")
    parser.add_argument("--v3-dir", default="results/chapter_pipeline_v3_atlas")
    parser.add_argument("--out-dir", default="results/final_chapter_plots")
    args = parser.parse_args()

    v2_dir = resolve_path(args.v2_dir)
    v3_dir = resolve_path(args.v3_dir)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(v2_dir / "metrics.csv")
    diagnostics = json.loads((v2_dir / "diagnostics.json").read_text())
    diagnostics = {k: diagnostics[k] for k in METHOD_ORDER}
    image_metrics = json.loads((v3_dir / "image_metrics.json").read_text())
    downstream = json.loads((v2_dir / "downstream_segmentation.json").read_text())

    plot_topology_boxplots(rows, out_dir)
    plot_validity_rates(diagnostics, out_dir)
    plot_lesion_capture(image_metrics, out_dir)
    plot_image_tradeoff(image_metrics, out_dir)
    plot_downstream(downstream, out_dir)
    write_caption_notes(out_dir)
    print(f"Saved final chapter plots to {out_dir}")


if __name__ == "__main__":
    main()

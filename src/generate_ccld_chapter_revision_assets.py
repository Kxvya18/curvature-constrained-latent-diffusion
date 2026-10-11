#!/usr/bin/env python3
"""Generate chapter-revision assets from existing CCLD experiment outputs."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "publication_chapter_figures"


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, mean, mean
    sem = float(values.std(ddof=1) / math.sqrt(n))
    delta = 1.96 * sem
    return mean, mean - delta, mean + delta


def load_metrics_csv(path: Path) -> dict[str, dict[str, tuple[float, float, float]]]:
    wanted = [
        "betti0_error",
        "euler_error",
        "curvature_l1",
        "pd0_wasserstein",
        "area_error",
        "boundary_roughness",
    ]
    rows: dict[str, dict[str, list[float]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method = row["method"]
            rows.setdefault(method, {key: [] for key in wanted})
            for key in wanted:
                rows[method][key].append(float(row[key]))
    return {
        method: {key: mean_ci(np.asarray(vals)) for key, vals in cols.items()}
        for method, cols in rows.items()
    }


def write_ci_table() -> None:
    stats = load_metrics_csv(RESULTS / "chapter_pipeline_v2" / "metrics.csv")
    order = ["Baseline Latent DDPM", "Curvature Only", "Topology Only", "Full CCLD"]
    keys = [
        ("betti0_error", "Betti-0 error"),
        ("euler_error", "Euler error"),
        ("curvature_l1", "Curvature L1"),
        ("pd0_wasserstein", "PD0 Wasserstein"),
        ("area_error", "Area error"),
        ("boundary_roughness", "Boundary roughness"),
    ]
    lines = [
        "# CCLD Sample-Level Confidence Intervals",
        "",
        "Intervals are normal-approximation 95% confidence intervals over generated samples.",
        "They quantify sample-level uncertainty, not repeated training-seed uncertainty.",
        "",
    ]
    for key, label in keys:
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| Method | Mean | 95% CI |")
        lines.append("|---|---:|---:|")
        for method in order:
            mean, lo, hi = stats[method][key]
            lines.append(f"| {method} | {mean:.4f} | [{lo:.4f}, {hi:.4f}] |")
        lines.append("")
    (RESULTS / "CCLD_SAMPLE_CI.md").write_text("\n".join(lines), encoding="utf-8")


def draw_conceptual_pipeline() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.1, 3.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    colors = {
        "data": "#f4f4f4",
        "latent": "#e8f0f7",
        "energy": "#f8efe3",
        "projection": "#e8f4ec",
        "image": "#f2eaf4",
        "line": "#2b2b2b",
    }

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        patch = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=colors["line"], linewidth=0.9)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.27, title, ha="center", va="top", fontweight="bold", fontsize=9)
        ax.text(x + w / 2, y + h / 2 - 0.18, body, ha="center", va="center", fontsize=8, linespacing=1.25)

    def arrow(x0: float, y0: float, x1: float, y1: float, label: str = "") -> None:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.0, color=colors["line"]))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.18, label, ha="center", va="bottom", fontsize=7)

    box(0.35, 3.7, 1.7, 1.25, "Real pairs", "$x_i, m_i$\nKvasir masks", colors["data"])
    box(2.55, 3.7, 1.75, 1.25, "Latent prior", "$z_T \\rightarrow z_0$\nDDPM denoiser", colors["latent"])
    box(4.9, 3.7, 1.95, 1.25, "Energy guidance", "$\\nabla E_{str}$\ncurvature + area", colors["energy"])
    box(7.35, 3.7, 2.0, 1.25, "Hard projection", "$P_{\\beta,a}$\nBetti + area feasible", colors["projection"])

    box(2.2, 1.25, 2.2, 1.2, "What is learned", "latent mask distribution\nand stochastic variation", colors["latent"])
    box(5.0, 1.25, 2.2, 1.2, "What is enforced", "$\\beta_0,\\beta_1$, area\nat projection steps", colors["projection"])
    box(7.95, 1.25, 1.85, 1.2, "RGB branch", "atlas/style\nconditioned rendering", colors["image"])

    arrow(2.05, 4.32, 2.55, 4.32)
    arrow(4.3, 4.32, 4.9, 4.32)
    arrow(6.85, 4.32, 7.35, 4.32)
    arrow(8.35, 3.7, 8.75, 2.45, "mask condition")
    arrow(4.4, 1.85, 5.0, 1.85)
    arrow(7.2, 1.85, 7.95, 1.85)

    ax.text(
        5.0,
        5.58,
        "Curvature-Constrained Latent Diffusion: sampler-level structural control",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        5.0,
        0.55,
        "The denoiser proposes plausible masks; structural guidance shapes the trajectory; projection enforces feasible topology.",
        ha="center",
        va="center",
        fontsize=8.5,
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig00_ccld_conceptual_pipeline.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig00_ccld_conceptual_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    write_ci_table()
    draw_conceptual_pipeline()
    print(f"Wrote {RESULTS / 'CCLD_SAMPLE_CI.md'}")
    print(f"Wrote {OUT / 'fig00_ccld_conceptual_pipeline.pdf'}")


if __name__ == "__main__":
    main()

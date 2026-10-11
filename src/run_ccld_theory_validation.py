from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib_cache").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from topology import (
    betti_numbers,
    connected_components,
    forman_ricci_profile,
    project_probability_to_topology_area,
    project_to_betti_anchor,
)


OUT = Path("results/ccld_theory_validation")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(31415)

    rows: list[dict] = []
    rows.extend(validate_projection_betti(rng))
    rows.extend(validate_area_projection(rng))
    rows.extend(validate_curvature_fragmentation(rng))
    rows.extend(validate_energy_descent(rng))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "theory_validation_metrics.csv", index=False)
    summary = summarize(df)
    (OUT / "theory_validation_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(df, summary)
    make_figures(df)
    print(json.dumps(summary, indent=2))
    print(f"Wrote CCLD theory validation to {OUT}")


def validate_projection_betti(rng: np.random.Generator, n: int = 320) -> list[dict]:
    rows = []
    for i in range(n):
        mask = random_fragmented_mask(rng, size=64)
        projected = project_to_betti_anchor(mask, 1, 0)
        b0_before, b1_before = betti_numbers(mask)
        b0_after, b1_after = betti_numbers(projected)
        rows.append(
            {
                "experiment": "betti_projection",
                "trial": i,
                "target_beta0": 1,
                "target_beta1": 0,
                "beta0_before": b0_before,
                "beta1_before": b1_before,
                "beta0_after": b0_after,
                "beta1_after": b1_after,
                "success": float(b0_after == 1 and b1_after == 0 and projected.any()),
                "area_before": float(mask.mean()),
                "area_after": float(projected.mean()),
            }
        )

    for i in range(n):
        mask = random_two_hole_mask(rng, size=64)
        projected = project_to_betti_anchor(mask, 1, 1)
        b0_before, b1_before = betti_numbers(mask)
        b0_after, b1_after = betti_numbers(projected)
        rows.append(
            {
                "experiment": "betti_projection_with_hole",
                "trial": i,
                "target_beta0": 1,
                "target_beta1": 1,
                "beta0_before": b0_before,
                "beta1_before": b1_before,
                "beta0_after": b0_after,
                "beta1_after": b1_after,
                "success": float(b0_after == 1 and b1_after == 1 and projected.any()),
                "area_before": float(mask.mean()),
                "area_after": float(projected.mean()),
            }
        )
    return rows


def validate_area_projection(rng: np.random.Generator, n: int = 360) -> list[dict]:
    rows = []
    targets = rng.uniform(0.035, 0.42, size=n)
    tolerance = 0.025
    for i, target in enumerate(targets):
        prob = random_probability_field(rng, size=64)
        projected = project_probability_to_topology_area(
            prob,
            target_beta0=1,
            target_beta1=0,
            target_area=float(target),
            tolerance=tolerance,
            border_margin=8,
        )
        b0, b1 = betti_numbers(projected)
        area = float(projected.mean())
        err = abs(area - float(target))
        rows.append(
            {
                "experiment": "area_topology_projection",
                "trial": i,
                "target_area": float(target),
                "area_after": area,
                "area_error": err,
                "tolerance": tolerance,
                "beta0_after": b0,
                "beta1_after": b1,
                "success": float(err <= tolerance and b0 == 1 and b1 == 0 and projected.any()),
                "topology_success": float(b0 == 1 and b1 == 0 and projected.any()),
                "area_success": float(err <= tolerance),
            }
        )
    return rows


def validate_curvature_fragmentation(rng: np.random.Generator, n: int = 260) -> list[dict]:
    rows = []
    ref_masks = [ellipse_mask(64, rng.uniform(0.17, 0.24), rng.uniform(0.10, 0.18), rng) for _ in range(80)]
    ref_curve = np.mean([forman_ricci_profile(m) for m in ref_masks], axis=0)
    for i in range(n):
        coherent = ellipse_mask(64, rng.uniform(0.15, 0.25), rng.uniform(0.10, 0.18), rng)
        fragmented = area_matched_fragments(rng, coherent, size=64)
        coh_l1 = float(np.abs(forman_ricci_profile(coherent) - ref_curve).sum())
        frag_l1 = float(np.abs(forman_ricci_profile(fragmented) - ref_curve).sum())
        coh_b0, coh_b1 = betti_numbers(coherent)
        frag_b0, frag_b1 = betti_numbers(fragmented)
        rows.append(
            {
                "experiment": "curvature_fragmentation",
                "trial": i,
                "coherent_curvature_l1": coh_l1,
                "fragmented_curvature_l1": frag_l1,
                "delta_fragment_minus_coherent": frag_l1 - coh_l1,
                "coherent_beta0": coh_b0,
                "fragmented_beta0": frag_b0,
                "coherent_area": float(coherent.mean()),
                "fragmented_area": float(fragmented.mean()),
                "success": float(frag_l1 > coh_l1),
            }
        )
    return rows


def validate_energy_descent(rng: np.random.Generator, n: int = 120) -> list[dict]:
    rows = []
    for i in range(n):
        p = np.clip(random_probability_field(rng, size=18), 1e-3, 1 - 1e-3)
        target = float(rng.uniform(0.08, 0.32))
        eta = 0.04
        e0, grad = structural_energy_and_grad(p, target)
        p1 = np.clip(p - eta * grad, 1e-4, 1 - 1e-4)
        e1, _ = structural_energy_and_grad(p1, target)
        rows.append(
            {
                "experiment": "first_order_energy_descent",
                "trial": i,
                "energy_before": e0,
                "energy_after": e1,
                "energy_delta": e1 - e0,
                "target_area": target,
                "success": float(e1 <= e0),
            }
        )
    return rows


def random_fragmented_mask(rng: np.random.Generator, size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for _ in range(int(rng.integers(2, 9))):
        cy = int(rng.integers(8, size - 8))
        cx = int(rng.integers(8, size - 8))
        ry = float(rng.uniform(2.5, 8.0))
        rx = float(rng.uniform(2.5, 8.0))
        yy, xx = np.mgrid[:size, :size]
        mask |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    if rng.random() < 0.6:
        add_hole(mask, rng)
    return mask


def random_two_hole_mask(rng: np.random.Generator, size: int) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    for _ in range(200):
        mask = ellipse_mask(size, rng.uniform(0.30, 0.36), rng.uniform(0.28, 0.34), rng)
        for dx in (-9, 9):
            cy = size // 2 + int(rng.integers(-2, 3))
            cx = size // 2 + dx + int(rng.integers(-2, 3))
            r = float(rng.uniform(3.0, 4.5))
            mask[((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r] = False
        b0, b1 = betti_numbers(mask)
        if b0 == 1 and b1 >= 2:
            return mask
    raise RuntimeError("could not construct feasible two-hole mask")


def add_hole(mask: np.ndarray, rng: np.random.Generator) -> None:
    comps = connected_components(mask, True)
    if not comps:
        return
    comp = max(comps, key=len)
    y, x = comp[int(rng.integers(0, len(comp)))]
    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
    r = float(rng.uniform(2.0, 4.0))
    mask[((yy - y) ** 2 + (xx - x) ** 2) <= r * r] = False


def ellipse_mask(size: int, ry_frac: float, rx_frac: float, rng: np.random.Generator) -> np.ndarray:
    cy = size / 2 + rng.uniform(-5, 5)
    cx = size / 2 + rng.uniform(-5, 5)
    ry = ry_frac * size
    rx = rx_frac * size
    angle = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[:size, :size]
    y = yy - cy
    x = xx - cx
    ca, sa = np.cos(angle), np.sin(angle)
    xr = ca * x + sa * y
    yr = -sa * x + ca * y
    return (yr / ry) ** 2 + (xr / rx) ** 2 <= 1.0


def area_matched_fragments(rng: np.random.Generator, target_mask: np.ndarray, size: int) -> np.ndarray:
    target_pixels = int(target_mask.sum())
    out = np.zeros((size, size), dtype=bool)
    attempts = 0
    while out.sum() < target_pixels and attempts < 500:
        attempts += 1
        cy = int(rng.integers(4, size - 4))
        cx = int(rng.integers(4, size - 4))
        r = float(rng.uniform(1.5, 4.0))
        yy, xx = np.mgrid[:size, :size]
        blob = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
        out |= blob
    if out.sum() > target_pixels:
        coords = np.argwhere(out)
        remove_n = int(out.sum() - target_pixels)
        remove_idx = rng.choice(len(coords), size=remove_n, replace=False)
        for y, x in coords[remove_idx]:
            out[y, x] = False
    return out


def random_probability_field(rng: np.random.Generator, size: int) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    field = np.zeros((size, size), dtype=float)
    for _ in range(int(rng.integers(1, 5))):
        cy = rng.uniform(8, size - 8)
        cx = rng.uniform(8, size - 8)
        sy = rng.uniform(4, 14)
        sx = rng.uniform(4, 14)
        amp = rng.uniform(0.45, 1.1)
        field += amp * np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
    field += 0.12 * rng.normal(size=(size, size))
    field = 1.0 / (1.0 + np.exp(-3.0 * (field - np.percentile(field, 65))))
    return np.clip(field, 0.0, 1.0)


def structural_energy_and_grad(p: np.ndarray, target_area: float) -> tuple[float, np.ndarray]:
    def energy(q: np.ndarray) -> float:
        dy = q[1:, :] - q[:-1, :]
        dx = q[:, 1:] - q[:, :-1]
        tv = np.sqrt(dx * dx + 1e-5).sum() + np.sqrt(dy * dy + 1e-5).sum()
        area = float(q.mean())
        return float(tv / q.size + 6.0 * (area - target_area) ** 2)

    e0 = energy(p)
    grad = np.zeros_like(p)
    eps = 1e-4
    for y in range(p.shape[0]):
        for x in range(p.shape[1]):
            old = p[y, x]
            p[y, x] = min(old + eps, 1.0)
            ep = energy(p)
            p[y, x] = max(old - eps, 0.0)
            em = energy(p)
            p[y, x] = old
            grad[y, x] = (ep - em) / (2 * eps)
    norm = float(np.linalg.norm(grad))
    if norm > 0:
        grad = grad / norm
    return e0, grad


def summarize(df: pd.DataFrame) -> dict:
    summary: dict[str, dict] = {}
    for exp, sub in df.groupby("experiment"):
        row = {
            "n": int(len(sub)),
            "success_rate": float(sub["success"].mean()) if "success" in sub else None,
        }
        for col in [
            "area_error",
            "delta_fragment_minus_coherent",
            "energy_delta",
            "coherent_curvature_l1",
            "fragmented_curvature_l1",
        ]:
            if col in sub:
                vals = pd.to_numeric(sub[col], errors="coerce").dropna()
                if len(vals):
                    row[col + "_mean"] = float(vals.mean())
                    row[col + "_std"] = float(vals.std())
        if "topology_success" in sub:
            vals = pd.to_numeric(sub["topology_success"], errors="coerce").dropna()
            if len(vals):
                row["topology_success_rate"] = float(vals.mean())
        if "area_success" in sub:
            vals = pd.to_numeric(sub["area_success"], errors="coerce").dropna()
            if len(vals):
                row["area_success_rate"] = float(vals.mean())
        summary[exp] = row
    return summary


def write_report(df: pd.DataFrame, summary: dict) -> None:
    lines = [
        "# CCLD Theory Validation Report",
        "",
        "These are controlled mathematical stress tests for the CCLD operators.",
        "They are not substitutes for real Kvasir synthesis experiments.",
        "",
        "## Claim Status",
        "",
        "- Betti projection theorem: directly tested by projecting fragmented and holed masks to target Betti anchors.",
        "- Area projection bound: tested on random probability fields with target lesion-area ranges.",
        "- Curvature fragmentation proposition: tested by comparing coherent ellipses with area-matched fragmented masks.",
        "- First-order energy descent: tested on a differentiable structural surrogate using finite-difference gradients.",
        "",
        "## Summary",
        "",
        json.dumps(summary, indent=2),
        "",
        "## Important Interpretation",
        "",
        "Exact zero Betti errors are expected when hard projection succeeds. This validates the projection operator under its feasibility assumptions; it does not mean an unconstrained latent diffusion denoiser learned exact topology.",
        "",
        "Area projection is an iterative discrete operator, so failures or residual errors identify where the theorem's termination/tolerance assumptions are not met.",
        "",
    ]
    (OUT / "THEORY_VALIDATION_REPORT.md").write_text("\n".join(lines))


def make_figures(df: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), constrained_layout=True)
    order = [
        "betti_projection",
        "betti_projection_with_hole",
        "area_topology_projection",
        "curvature_fragmentation",
        "first_order_energy_descent",
    ]
    labels = ["Betti\n(1,0)", "Betti\n(1,1)", "Area+\nTopology", "Curvature\nFragment", "Energy\nDescent"]
    success = [df[df["experiment"] == exp]["success"].mean() for exp in order]
    bars = axes[0].bar(labels, success, color=["#222222", "#444444", "#1f77b4", "#1b9e77", "#7570b3"])
    for bar, val in zip(bars, success):
        axes[0].text(bar.get_x() + bar.get_width() / 2, min(val + 0.025, 1.03), f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("success rate")
    axes[0].set_title("Theorem stress tests")
    axes[0].tick_params(axis="x", labelrotation=25)

    area = df[df["experiment"] == "area_topology_projection"]
    axes[1].hist(area["area_error"], bins=24, color="#1f77b4", alpha=0.85)
    axes[1].axvline(float(area["tolerance"].iloc[0]), color="crimson", ls="--", lw=1.2, label="tolerance")
    axes[1].set_xlabel("area error")
    axes[1].set_ylabel("count")
    axes[1].set_title("Area projection bound")
    axes[1].legend(frameon=False)

    curv = df[df["experiment"] == "curvature_fragmentation"]
    axes[2].boxplot(
        [curv["coherent_curvature_l1"], curv["fragmented_curvature_l1"]],
        tick_labels=["coherent", "fragmented"],
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": "#f2f2f2"},
        medianprops={"color": "black"},
    )
    axes[2].set_ylabel("curvature L1 to coherent reference")
    axes[2].set_title("Fragmentation changes curvature")
    fig.savefig(OUT / "fig01_theory_validation_summary.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig01_theory_validation_summary.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

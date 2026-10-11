#!/usr/bin/env python3
"""Generate review-fix diagnostics from non-torch CCLD artifacts.

This script intentionally avoids torch so it can run on machines where the
training environment is unavailable. It uses existing CSV/JSON metrics, raw
image-metric summaries, and the original zip dataset for topology auditing.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))

from topology import betti_numbers


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
OUT = RESULTS / "review_fix_diagnostics"
METHODS = ["Baseline Latent DDPM", "Curvature Only", "Topology Only", "Full CCLD"]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT / path


def load_cfg() -> dict:
    return json.loads((PROJECT / "configs" / "chapter_pipeline_v2.json").read_text())


def zip_pair_members(zip_path: Path) -> list[tuple[str, str]]:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    images = [n for n in names if n.startswith("segmented-images/images/") and n.lower().endswith((".jpg", ".jpeg", ".png"))]
    masks = [n for n in names if n.startswith("segmented-images/masks/") and n.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_by_stem = {Path(n).stem: n for n in images}
    mask_by_stem = {Path(n).stem: n for n in masks}
    stems = sorted(set(image_by_stem) & set(mask_by_stem))
    return [(image_by_stem[s], mask_by_stem[s]) for s in stems]


def read_mask_from_zip(zip_path: Path, member: str, image_size: int) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(member)
    from io import BytesIO

    mask = Image.open(BytesIO(data)).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
    return np.asarray(mask) > 127


def dataset_topology_audit(cfg: dict) -> dict:
    zip_path = resolve(cfg["zip_path"])
    pairs = zip_pair_members(zip_path)
    masks = [read_mask_from_zip(zip_path, mask_member, int(cfg["image_size"])) for _, mask_member in pairs]
    bettis = [betti_numbers(m) for m in masks]
    counts = Counter(f"{b0},{b1}" for b0, b1 in bettis)
    areas = np.asarray([m.mean() for m in masks], dtype=float)
    train_count = int(math.floor(len(masks) * float(cfg["train_fraction"])))
    return {
        "zip_path": cfg["zip_path"],
        "image_size": int(cfg["image_size"]),
        "mask_resize_interpolation": "PIL Image.Resampling.NEAREST",
        "image_resize_interpolation": "PIL Image.Resampling.BILINEAR",
        "paired_samples": len(pairs),
        "train_fraction": float(cfg["train_fraction"]),
        "estimated_train_count": train_count,
        "estimated_validation_count": len(pairs) - train_count,
        "target_topology": "(1,0)",
        "target_topology_fraction": float(sum(b == (1, 0) for b in bettis) / max(len(bettis), 1)),
        "betti_distribution": dict(sorted(counts.items())),
        "area_mean": float(areas.mean()),
        "area_std": float(areas.std(ddof=0)),
        "area_min": float(areas.min()),
        "area_max": float(areas.max()),
    }


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def group_metric(rows: list[dict], metric: str) -> dict[str, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(float(row[metric]))
    return {k: np.asarray(v, dtype=float) for k, v in grouped.items()}


def curvature_failure_diagnostics(rows: list[dict]) -> dict:
    stats = {}
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        beta0 = np.asarray([float(r["beta0"]) for r in subset])
        curvature = np.asarray([float(r["curvature_l1"]) for r in subset])
        area = np.asarray([float(r["area"]) for r in subset])
        rough = np.asarray([float(r["boundary_roughness"]) for r in subset])
        valid_area = np.asarray([float(r["valid_area"]) if "valid_area" in r else float(0.035 <= float(r["area"]) <= 0.42) for r in subset])
        stats[method] = {
            "n": int(len(subset)),
            "beta0_mean": float(beta0.mean()),
            "beta0_median": float(np.median(beta0)),
            "curvature_l1_mean": float(curvature.mean()),
            "area_mean": float(area.mean()),
            "valid_area_rate": float(valid_area.mean()),
            "boundary_roughness_mean": float(rough.mean()),
            "interpretation": "Curvature-only has low-ish curvature L1 but high beta0, indicating fragmented area-valid masks."
            if method == "Curvature Only" else "",
        }

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    colors = ["#B03A2E", "#D68910", "#2E86C1", "#2E8B57"]
    beta0_vals = group_metric(rows, "beta0")
    curv_vals = group_metric(rows, "curvature_l1")
    axes[0].boxplot([beta0_vals[m] for m in METHODS], labels=["Base", "Curv", "Topo", "Full"], showfliers=False)
    axes[0].set_ylabel(r"$\beta_0$")
    axes[0].set_title("Connected-component count")
    axes[0].grid(axis="y", alpha=0.25)
    for method, color in zip(METHODS, colors):
        xs = np.asarray([float(r["beta0"]) for r in rows if r["method"] == method])
        ys = np.asarray([float(r["curvature_l1"]) for r in rows if r["method"] == method])
        axes[1].scatter(xs, ys, s=10, alpha=0.45, label=method, color=color)
    axes[1].set_xlabel(r"$\beta_0$")
    axes[1].set_ylabel("Curvature L1")
    axes[1].set_title("Curvature-only degeneracy diagnostic")
    axes[1].legend(fontsize=6, frameon=False)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_review_curvature_only_failure.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_review_curvature_only_failure.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return stats


def combined_v3_table() -> dict:
    raw = json.loads((RESULTS / "chapter_pipeline_v3_style" / "image_metrics.json").read_text())
    atlas = json.loads((RESULTS / "chapter_pipeline_v3_atlas" / "image_metrics.json").read_text())
    combined = {method: dict(raw[method]) for method in METHODS}
    combined["Full CCLD + Atlas/Style"] = dict(atlas["Full CCLD"])
    for method in METHODS:
        for key in [
            "boundary_alignment",
            "area_weighted_boundary_alignment",
            "visible_aligned_rate",
            "atlas_lesion_recall",
            "atlas_lesion_precision",
            "atlas_lesion_f1",
            "missed_lesion_rate",
        ]:
            if method == "Full CCLD":
                combined[method][key] = None
            else:
                combined[method][key] = atlas.get(method, {}).get(key, None)
    combined["review_note"] = {
        "raw_full_ccld_available": True,
        "raw_full_ccld_atlas_capture_not_available_without_loading_pt_masks": True,
        "atlas_full_ccld_capture_is_computed_after_atlas_refinement": True,
        "required_fix_for_future_runs": "Save per-sample atlas recall/precision/F1 and threshold curves for raw and atlas-refined masks separately.",
    }
    return combined


def write_review_tables(audit: dict, curvature: dict, combined: dict) -> None:
    lines = ["# CCLD Review-Fix Diagnostics", ""]
    lines += [
        "## Dataset and Resizing Audit",
        "",
        f"- Paired segmented samples loaded from zip: `{audit['paired_samples']}`",
        f"- Image size used by experiments: `{audit['image_size']} x {audit['image_size']}`",
        f"- Mask interpolation: `{audit['mask_resize_interpolation']}`",
        f"- Train/validation split: approximately `{audit['estimated_train_count']}/{audit['estimated_validation_count']}` using train fraction `{audit['train_fraction']}`",
        f"- Fraction with target topology `(1,0)` after resizing: `{audit['target_topology_fraction']:.4f}`",
        f"- Betti distribution after resizing: `{audit['betti_distribution']}`",
        "",
        "## Curvature-Only Failure Diagnosis",
        "",
        "| Method | beta0 mean | beta0 median | curvature L1 mean | area mean | valid area rate | boundary roughness mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = curvature[method]
        lines.append(
            f"| {method} | {row['beta0_mean']:.4f} | {row['beta0_median']:.4f} | "
            f"{row['curvature_l1_mean']:.4f} | {row['area_mean']:.4f} | "
            f"{row['valid_area_rate']:.4f} | {row['boundary_roughness_mean']:.4f} |"
        )
    lines += [
        "",
        "Interpretation: Curvature Only can satisfy area better than baseline while fragmenting into many components. Therefore curvature guidance alone must not be claimed to preserve topology; Full CCLD relies on projection for topology and curvature for secondary local geometry.",
        "",
        "## Raw-vs-Atlas Image Metrics",
        "",
        "| Method | FID | SSIM | LPIPS | Contrast error | Atlas F1 | Missed rate | Note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for method in ["Baseline Latent DDPM", "Curvature Only", "Topology Only", "Full CCLD", "Full CCLD + Atlas/Style"]:
        row = combined[method]
        atlas_f1 = row.get("atlas_lesion_f1")
        missed = row.get("missed_lesion_rate")
        note = "raw render" if method != "Full CCLD + Atlas/Style" else "atlas-refined; not raw CCLD evidence"
        lines.append(
            f"| {method} | {row['fid']:.4f} | {row['nearest_real_ssim']:.4f} | "
            f"{row['lpips_to_real_batch']:.4f} | {row['mask_contrast_abs_error']:.4f} | "
            f"{atlas_f1:.4f} | {missed:.4f} | {note} |"
            if atlas_f1 is not None
            else
            f"| {method} | {row['fid']:.4f} | {row['nearest_real_ssim']:.4f} | "
            f"{row['lpips_to_real_batch']:.4f} | {row['mask_contrast_abs_error']:.4f} | NA | NA | {note} |"
        )
    lines += [
        "",
        "Important: atlas F1 and missed rate for `Full CCLD + Atlas/Style` are downstream of atlas mask replacement. They must be interpreted as atlas/style rendering plausibility, not as raw CCLD mask-generation proof.",
        "",
        "## Remaining Experiment Needed for Complete Closure",
        "",
        "A future torch-enabled rerun should save per-sample atlas recalls for raw Full CCLD and atlas-refined Full CCLD, then report missed-rate curves at thresholds 0.3, 0.5, and 0.7.",
        "",
    ]
    (RESULTS / "CCLD_REVIEW_FIX_TABLES.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "review_fix_metrics.json").write_text(
        json.dumps({"dataset_audit": audit, "curvature_failure": curvature, "combined_v3": combined}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    audit = dataset_topology_audit(cfg)
    rows = load_rows(RESULTS / "chapter_pipeline_v2" / "metrics.csv")
    curvature = curvature_failure_diagnostics(rows)
    combined = combined_v3_table()
    write_review_tables(audit, curvature, combined)
    print(json.dumps({"dataset_audit": audit, "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()

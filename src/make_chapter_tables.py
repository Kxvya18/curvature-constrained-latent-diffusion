from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(entry: dict | None) -> str:
    if not entry:
        return "NA"
    return f"{entry['mean']:.4f} +/- {entry['std']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/kvasir_real/summary.json")
    parser.add_argument("--out", default="results/kvasir_real/chapter_tables.md")
    parser.add_argument("--downstream", default=None)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    topology = summary.get("topology", summary)
    image = summary.get("image", {})

    lines = [
        "# Chapter Tables",
        "",
        "## Topology Fidelity",
        "",
        "| Method | Betti-0 Error ↓ | Betti-1 Error ↓ | Euler Error ↓ | Curvature L1 ↓ | PD0 Wasserstein ↓ | PD1 Wasserstein ↓ | Boundary Roughness ↓ | Border Contact ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in topology.items():
        lines.append(
            f"| {method} | {fmt(row.get('betti0_error'))} | {fmt(row.get('betti1_error'))} | "
            f"{fmt(row.get('euler_error'))} | {fmt(row.get('curvature_l1'))} | "
            f"{fmt(row.get('pd0_wasserstein'))} | {fmt(row.get('pd1_wasserstein'))} | "
            f"{fmt(row.get('boundary_roughness'))} | {fmt(row.get('border_contact'))} |"
        )

    lines.extend([
        "",
        "## Image Fidelity",
        "",
        "| Method | FID ↓ | Nearest SSIM ↑ | LPIPS ↓ | Nearest L1 ↓ | Mask Contrast Error ↓ | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for method, row in image.items():
        fid = "NA" if row.get("fid") is None else f"{row['fid']:.4f}"
        ssim_value = row.get("nearest_real_ssim", row.get("ssim_to_real_batch"))
        l1_value = row.get("nearest_real_l1", row.get("l1_to_real_batch"))
        ssim = "NA" if ssim_value is None else f"{ssim_value:.4f}"
        lpips = "NA" if row.get("lpips_to_real_batch") is None else f"{row['lpips_to_real_batch']:.4f}"
        l1 = "NA" if l1_value is None else f"{l1_value:.4f}"
        contrast = "NA" if row.get("mask_contrast_abs_error") is None else f"{row['mask_contrast_abs_error']:.4f}"
        note = row.get("fid_error") or ""
        lines.append(f"| {method} | {fid} | {ssim} | {lpips} | {l1} | {contrast} | {note} |")

    downstream_path = Path(args.downstream) if args.downstream else summary_path.parent / "downstream_segmentation.json"
    if downstream_path.exists():
        downstream = json.loads(downstream_path.read_text())
        lines.extend([
            "",
            "## Downstream Segmentation Utility",
            "",
            "| Training Condition | Dice ↑ | IoU ↑ | Precision ↑ | Recall ↑ | Boundary F1 ↑ |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for condition, metrics in downstream.items():
            lines.append(
                f"| {condition} | {fmt(metrics.get('dice'))} | {fmt(metrics.get('iou'))} | "
                f"{fmt(metrics.get('precision'))} | {fmt(metrics.get('recall'))} | {fmt(metrics.get('boundary_f1'))} |"
            )

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

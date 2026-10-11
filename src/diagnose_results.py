from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="results/chapter_pipeline_v2")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    metrics_path = result_dir / "metrics.csv"
    summary_path = result_dir / "summary.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    rows = list(csv.DictReader(metrics_path.open()))
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    report = {}
    for method, subset in by_method.items():
        areas = np.array([float(r["area"]) for r in subset])
        beta0 = np.array([float(r["beta0"]) for r in subset])
        area_error = np.array([float(r["area_error"]) for r in subset])
        border = np.array([float(r.get("border_contact", 0.0) or 0.0) for r in subset])
        compact = np.array([float(r.get("compactness", 0.0) or 0.0) for r in subset])
        report[method] = {
            "n": int(len(subset)),
            "empty_rate": float(np.mean(areas == 0.0)),
            "tiny_area_rate_lt_0_02": float(np.mean(areas < 0.02)),
            "area_mean": float(areas.mean()),
            "area_min": float(areas.min()),
            "area_max": float(areas.max()),
            "area_error_mean": float(area_error.mean()),
            "beta0_mean": float(beta0.mean()),
            "valid_area_rate_0_03_to_0_45": float(np.mean((areas >= 0.03) & (areas <= 0.45))),
            "border_contact_mean": float(border.mean()),
            "high_border_contact_rate_gt_0_08": float(np.mean(border > 0.08)),
            "compactness_mean": float(compact.mean()),
        }

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        report["summary_image_metrics"] = summary.get("image", {})

    out = result_dir / "diagnostics.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

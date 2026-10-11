from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    theory_path = Path("results/ccld_theory_validation/theory_validation_summary.json")
    v2_path = Path("results/chapter_pipeline_v2/summary.json")
    v3_path = Path("results/chapter_pipeline_v3_atlas/image_metrics.json")
    review_path = Path("results/review_fix_diagnostics/review_fix_metrics.json")

    theory = json.loads(theory_path.read_text()) if theory_path.exists() else {}
    v2 = json.loads(v2_path.read_text()) if v2_path.exists() else {}
    v3 = json.loads(v3_path.read_text()) if v3_path.exists() else {}
    review = json.loads(review_path.read_text()) if review_path.exists() else {}

    lines = [
        "# CCLD Claim Status",
        "",
        "This report separates mathematical operator claims from generative synthesis claims.",
        "",
        "## Mathematically Validated Claims",
        "",
    ]

    betti = theory.get("betti_projection", {})
    betti_hole = theory.get("betti_projection_with_hole", {})
    area = theory.get("area_topology_projection", {})
    curv = theory.get("curvature_fragmentation", {})
    energy = theory.get("first_order_energy_descent", {})

    lines.extend(
        [
            f"- Hard Betti projection to `(beta0,beta1)=(1,0)`: success rate `{betti.get('success_rate', 'NA')}` over `{betti.get('n', 'NA')}` feasible stress tests.",
            f"- Hard Betti projection to `(beta0,beta1)=(1,1)`: success rate `{betti_hole.get('success_rate', 'NA')}` over `{betti_hole.get('n', 'NA')}` feasible stress tests.",
            f"- Topology+area projection: topology success `{area.get('topology_success_rate', 'NA')}`, area success `{area.get('area_success_rate', 'NA')}`, mean area error `{area.get('area_error_mean', 'NA')}`.",
            f"- Curvature fragmentation proposition: success rate `{curv.get('success_rate', 'NA')}`, fragmented-minus-coherent curvature L1 mean `{curv.get('delta_fragment_minus_coherent_mean', 'NA')}`.",
            f"- First-order structural energy descent surrogate: success rate `{energy.get('success_rate', 'NA')}`, mean energy change `{energy.get('energy_delta_mean', 'NA')}`.",
        ]
    )

    lines.extend(["", "## Empirically Supported Generative Claims", ""])
    top = v2.get("topology", {})
    full = top.get("Full CCLD", {})
    base = top.get("Baseline Latent DDPM", {})
    if full and base:
        lines.extend(
            [
                f"- Full CCLD Betti-0 error mean `{full['betti0_error']['mean']}` versus baseline `{base['betti0_error']['mean']}`.",
                f"- Full CCLD Euler error mean `{full['euler_error']['mean']}` versus baseline `{base['euler_error']['mean']}`.",
                f"- Full CCLD curvature L1 mean `{full['curvature_l1']['mean']}` versus baseline `{base['curvature_l1']['mean']}`.",
                f"- Full CCLD PD0 Wasserstein mean `{full['pd0_wasserstein']['mean']}` versus baseline `{base['pd0_wasserstein']['mean']}`.",
            ]
        )

    combined_v3 = review.get("combined_v3", {})
    full_raw_v3 = combined_v3.get("Full CCLD", {})
    full_v3 = combined_v3.get("Full CCLD + Atlas/Style", v3.get("Full CCLD", {}))
    base_v3 = v3.get("Baseline Latent DDPM", {})
    if full_raw_v3:
        lines.extend(
            [
                f"- Raw Full CCLD v3 render FID `{full_raw_v3.get('fid', 'NA')}` and SSIM `{full_raw_v3.get('nearest_real_ssim', 'NA')}`.",
                "- Raw Full CCLD atlas F1/missed-rate should be recomputed from raw saved masks in a torch-enabled rerun; do not borrow atlas-refined values.",
            ]
        )
    if full_v3 and base_v3:
        lines.extend(
            [
                f"- Atlas/style-conditioned lesion capture: Full CCLD + Atlas/Style atlas F1 `{full_v3['atlas_lesion_f1']}` versus baseline `{base_v3['atlas_lesion_f1']}`.",
                f"- Missed lesion rate: Full CCLD + Atlas/Style `{full_v3['missed_lesion_rate']}` versus baseline `{base_v3['missed_lesion_rate']}`.",
                f"- RGB image synthesis remains atlas/style-conditioned rendering, not pure clinical-grade RGB diffusion.",
            ]
        )

    lines.extend(
        [
            "",
            "## Still Not Fully Satisfied",
            "",
            "- Raw latent diffusion alone has not learned exact topology; exact Betti results depend on the hard projection operator.",
            "- Full CCLD v3 visual synthesis must be separated into raw Full CCLD and Full CCLD + Atlas/Style.",
            "- The RGB image branch is presentable and useful for chapter evidence, but not a standalone photorealistic clinical diffusion generator.",
            "- Atlas-refined lesion-capture metrics are not raw CCLD mask-generation metrics because atlas refinement replaces mask shape.",
            "- Persistence-diagram zero errors in the final generated masks are consequences of projected topology under the current binary-mask evaluation; they should not be described as unconstrained learned topology.",
            "- Larger multi-dataset validation and stronger learned image-generation baselines would be required for broader claims.",
            "",
            "## Best Defensible Chapter Claim",
            "",
            "CCLD is best defended as a constrained mask-synthesis framework whose hard projection enforces feasible Betti topology, whose area projection preserves lesion-scale support with high success, and whose curvature guidance improves local geometry only when paired with projection. On real Kvasir/HyperKvasir-derived experiments, these operators improve topology-faithful lesion mask synthesis and support atlas/style-conditioned image-mask generation when raw and atlas-refined claims are kept separate.",
            "",
        ]
    )

    out = Path("results/CCLD_CLAIM_STATUS.md")
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()

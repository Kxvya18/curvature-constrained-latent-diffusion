# CCLD Claim Status

This report separates mathematical operator claims from generative synthesis claims.

## Mathematically Validated Claims

- Hard Betti projection to `(beta0,beta1)=(1,0)`: success rate `1.0` over `320` feasible stress tests.
- Hard Betti projection to `(beta0,beta1)=(1,1)`: success rate `1.0` over `320` feasible stress tests.
- Topology+area projection: topology success `1.0`, area success `0.9972222222222222`, mean area error `0.009308026296110774`.
- Curvature fragmentation proposition: success rate `1.0`, fragmented-minus-coherent curvature L1 mean `0.8959628199322618`.
- First-order structural energy descent surrogate: success rate `1.0`, mean energy change `-0.0076743817016906805`.

## Empirically Supported Generative Claims

- Full CCLD Betti-0 error mean `0.0` versus baseline `1.484375`.
- Full CCLD Euler error mean `0.0` versus baseline `1.484375`.
- Full CCLD curvature L1 mean `0.08782797545657492` versus baseline `0.7803401198278335`.
- Full CCLD PD0 Wasserstein mean `0.0` versus baseline `1.0628670903636324`.
- Raw Full CCLD v3 render FID `72.28010559082031` and SSIM `0.8433836828917265`.
- Raw Full CCLD atlas F1/missed-rate should be recomputed from raw saved masks in a torch-enabled rerun; do not borrow atlas-refined values.
- Atlas/style-conditioned lesion capture: Full CCLD + Atlas/Style atlas F1 `0.7870067140332023` versus baseline `0.17082727790396826`.
- Missed lesion rate: Full CCLD + Atlas/Style `0.03125` versus baseline `0.91796875`.
- RGB image synthesis remains atlas/style-conditioned rendering, not pure clinical-grade RGB diffusion.

## Still Not Fully Satisfied

- Raw latent diffusion alone has not learned exact topology; exact Betti results depend on the hard projection operator.
- Full CCLD v3 visual synthesis must be separated into raw Full CCLD and Full CCLD + Atlas/Style.
- The RGB image branch is presentable and useful for chapter evidence, but not a standalone photorealistic clinical diffusion generator.
- Atlas-refined lesion-capture metrics are not raw CCLD mask-generation metrics because atlas refinement replaces mask shape.
- Persistence-diagram zero errors in the final generated masks are consequences of projected topology under the current binary-mask evaluation; they should not be described as unconstrained learned topology.
- Larger multi-dataset validation and stronger learned image-generation baselines would be required for broader claims.

## Best Defensible Chapter Claim

CCLD is best defended as a constrained mask-synthesis framework whose hard projection enforces feasible Betti topology, whose area projection preserves lesion-scale support with high success, and whose curvature guidance improves local geometry only when paired with projection. On real Kvasir/HyperKvasir-derived experiments, these operators improve topology-faithful lesion mask synthesis and support atlas/style-conditioned image-mask generation when raw and atlas-refined claims are kept separate.

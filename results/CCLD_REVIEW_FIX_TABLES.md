# CCLD Review-Fix Diagnostics

## Dataset and Resizing Audit

- Paired segmented samples loaded from zip: `1000`
- Image size used by experiments: `128 x 128`
- Mask interpolation: `PIL Image.Resampling.NEAREST`
- Train/validation split: approximately `850/150` using train fraction `0.85`
- Fraction with target topology `(1,0)` after resizing: `0.9430`
- Betti distribution after resizing: `{'1,0': 943, '1,1': 5, '1,2': 1, '1,3': 1, '10,0': 1, '2,0': 41, '3,0': 7, '4,0': 1}`

## Curvature-Only Failure Diagnosis

| Method | beta0 mean | beta0 median | curvature L1 mean | area mean | valid area rate | boundary roughness mean |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Latent DDPM | 2.3750 | 2.0000 | 0.7803 | 0.0111 | 0.0195 | 6.6049 |
| Curvature Only | 8.9023 | 9.0000 | 0.3578 | 0.0914 | 0.9961 | 14.2685 |
| Topology Only | 1.0000 | 1.0000 | 0.1050 | 0.1506 | 1.0000 | 8.5713 |
| Full CCLD | 1.0000 | 1.0000 | 0.0878 | 0.1521 | 1.0000 | 7.9065 |

Interpretation: Curvature Only can satisfy area better than baseline while fragmenting into many components. Therefore curvature guidance alone must not be claimed to preserve topology; Full CCLD relies on projection for topology and curvature for secondary local geometry.

## Raw-vs-Atlas Image Metrics

| Method | FID | SSIM | LPIPS | Contrast error | Atlas F1 | Missed rate | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline Latent DDPM | 53.1913 | 0.8616 | 0.4850 | 0.3115 | 0.1708 | 0.9180 | raw render |
| Curvature Only | 65.5244 | 0.8611 | 0.4828 | 0.4070 | 0.1841 | 0.9961 | raw render |
| Topology Only | 70.5910 | 0.8367 | 0.4875 | 0.3312 | 0.5655 | 0.3398 | raw render |
| Full CCLD | 72.2801 | 0.8434 | 0.4881 | 0.3435 | NA | NA | raw render |
| Full CCLD + Atlas/Style | 66.8626 | 0.8340 | 0.4897 | 0.2897 | 0.7870 | 0.0312 | atlas-refined; not raw CCLD evidence |

Important: atlas F1 and missed rate for `Full CCLD + Atlas/Style` are downstream of atlas mask replacement. They must be interpreted as atlas/style rendering plausibility, not as raw CCLD mask-generation proof.

## Remaining Experiment Needed for Complete Closure

A future torch-enabled rerun should save per-sample atlas recalls for raw Full CCLD and atlas-refined Full CCLD, then report missed-rate curves at thresholds 0.3, 0.5, and 0.7.

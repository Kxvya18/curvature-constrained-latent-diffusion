# V3 Style-Conditioned RGB Synthesis

| Method | FID ↓ | Nearest SSIM ↑ | LPIPS ↓ | Nearest L1 ↓ | Mask Contrast Error ↓ | Boundary Align ↑ | Area-Weighted Align ↑ | Visible Aligned Rate ↑ | Atlas Recall ↑ | Atlas F1 ↑ | Missed Lesion Rate ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Latent DDPM | 53.19130325317383 | 0.8616 | 0.48504213683190756 | 0.1780 | 0.3115 | 1.1726 | 0.4353 | 0.0039 | 0.1439 | 0.1708 | 0.9180 |
| Curvature Only | 65.5243911743164 | 0.8611 | 0.4827537122182548 | 0.1737 | 0.4070 | 1.0479 | 1.0479 | 0.0273 | 0.1738 | 0.1841 | 0.9961 |
| Topology Only | 70.59104919433594 | 0.8367 | 0.48751263646408916 | 0.1914 | 0.3312 | 1.1778 | 1.1778 | 0.0391 | 0.5563 | 0.5655 | 0.3398 |
| Full CCLD | 66.86264038085938 | 0.8340 | 0.48973122402094305 | 0.1949 | 0.2897 | 1.1761 | 1.1408 | 0.0352 | 0.7835 | 0.7870 | 0.0312 |

Interpretation: this v3 renderer uses real Kvasir backgrounds and lesion texture donors while preserving each method's generated mask. It is intended to test whether CCLD's topology-valid masks can drive clearer, scan-like conditional synthesis.

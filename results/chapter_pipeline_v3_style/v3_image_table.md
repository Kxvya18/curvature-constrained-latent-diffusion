# V3 Style-Conditioned RGB Synthesis

| Method | FID ↓ | Nearest SSIM ↑ | LPIPS ↓ | Nearest L1 ↓ | Mask Contrast Error ↓ |
|---|---:|---:|---:|---:|---:|
| Baseline Latent DDPM | 53.19130325317383 | 0.8616 | 0.48504213683190756 | 0.1780 | 0.3115 |
| Curvature Only | 65.5243911743164 | 0.8611 | 0.4827537122182548 | 0.1737 | 0.4070 |
| Topology Only | 70.59104919433594 | 0.8367 | 0.48751263646408916 | 0.1914 | 0.3312 |
| Full CCLD | 72.28010559082031 | 0.8434 | 0.48808407923206687 | 0.1896 | 0.3435 |

Interpretation: this v3 renderer uses real Kvasir backgrounds and lesion texture donors while preserving each method's generated mask. It is intended to test whether CCLD's topology-valid masks can drive clearer, scan-like conditional synthesis.

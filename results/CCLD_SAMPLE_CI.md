# CCLD Sample-Level Confidence Intervals

Intervals are normal-approximation 95% confidence intervals over generated samples.
They quantify sample-level uncertainty, not repeated training-seed uncertainty.

## Betti-0 error

| Method | Mean | 95% CI |
|---|---:|---:|
| Baseline Latent DDPM | 1.4844 | [1.3425, 1.6262] |
| Curvature Only | 7.9023 | [7.6532, 8.1515] |
| Topology Only | 0.0000 | [0.0000, 0.0000] |
| Full CCLD | 0.0000 | [0.0000, 0.0000] |

## Euler error

| Method | Mean | 95% CI |
|---|---:|---:|
| Baseline Latent DDPM | 1.4844 | [1.3425, 1.6262] |
| Curvature Only | 7.8984 | [7.6485, 8.1484] |
| Topology Only | 0.0000 | [0.0000, 0.0000] |
| Full CCLD | 0.0000 | [0.0000, 0.0000] |

## Curvature L1

| Method | Mean | 95% CI |
|---|---:|---:|
| Baseline Latent DDPM | 0.7803 | [0.7230, 0.8377] |
| Curvature Only | 0.3578 | [0.3491, 0.3664] |
| Topology Only | 0.1050 | [0.0982, 0.1119] |
| Full CCLD | 0.0878 | [0.0809, 0.0948] |

## PD0 Wasserstein

| Method | Mean | 95% CI |
|---|---:|---:|
| Baseline Latent DDPM | 1.0629 | [0.9633, 1.1624] |
| Curvature Only | 5.5795 | [5.4038, 5.7552] |
| Topology Only | 0.0000 | [0.0000, 0.0000] |
| Full CCLD | 0.0000 | [0.0000, 0.0000] |

## Area error

| Method | Mean | 95% CI |
|---|---:|---:|
| Baseline Latent DDPM | 0.1495 | [0.1484, 0.1505] |
| Curvature Only | 0.0691 | [0.0663, 0.0720] |
| Topology Only | 0.0831 | [0.0774, 0.0888] |
| Full CCLD | 0.0826 | [0.0771, 0.0882] |

## Boundary roughness

| Method | Mean | 95% CI |
|---|---:|---:|
| Baseline Latent DDPM | 6.6049 | [6.3331, 6.8767] |
| Curvature Only | 14.2685 | [14.0873, 14.4496] |
| Topology Only | 8.5713 | [8.3165, 8.8260] |
| Full CCLD | 7.9065 | [7.6929, 8.1201] |

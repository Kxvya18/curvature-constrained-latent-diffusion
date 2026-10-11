# Curvature-Constrained Latent Diffusion for Real Kvasir-SEG Masks

This project implements the real-data experiment for the book chapter. It uses the downloaded Kvasir/HyperKvasir segmented endoscopy dataset and does not include synthetic-data fallback code.

## Dataset

Expected ZIP:

```text
data/hyper-kvasir-segmented-images.zip
```

The archive should contain:

```text
segmented-images/images/*.jpg
segmented-images/masks/*.jpg
```

## Method

The code trains:

1. A convolutional autoencoder on real expert polyp masks to define a compact latent space.
2. A mask-conditioned RGB image renderer trained on paired real endoscopy images.
3. Four latent DDPM ablations on encoded masks:
   - baseline
   - curvature only
   - topology only
   - full CCLD
4. A curvature/topology-constrained latent DDPM using:
   - diffusion noise-prediction loss
   - decoded reconstruction consistency
   - differentiable boundary/curvature regularization
   - real-data topology anchors from Betti statistics and area distribution
   - area and shape-moment preservation losses
5. A guided reverse sampler that applies curvature/topology gradients and Betti projection anchors in later denoising steps.
6. Conditional RGB synthesis from generated masks.

## Run

```bash
pip install -r requirements.txt
python src/inspect_dataset.py --zip-path data/hyper-kvasir-segmented-images.zip
python src/train_chapter_pipeline_v2.py --config configs/chapter_pipeline_v2.json
```

For a faster first run, use:

```bash
python src/train_chapter_pipeline_v2.py --config configs/chapter_pipeline_v2_smoke.json
```

## Outputs

Results are saved to:

```text
results/kvasir_real/
```

Files:

- `dataset_report.json`
- `metrics.csv`
- `summary.json`
- `image_metrics.json`
- `baseline_masks.png`, `curvature_masks.png`, `topology_masks.png`, `full_masks.png`
- `baseline_images.png`, `curvature_images.png`, `topology_images.png`, `full_images.png`
- `qualitative_real_vs_generated.png`
- `checkpoints/autoencoder.pt`
- `checkpoints/mask_to_image_renderer.pt`
- `checkpoints/baseline_denoiser.pt`
- `checkpoints/curvature_denoiser.pt`
- `checkpoints/topology_denoiser.pt`
- `checkpoints/full_denoiser.pt`

## Metrics

Topology:

- Betti-0 error
- Betti-1 error
- Euler characteristic error
- Forman-Ricci curvature profile L1 distance
- Boundary roughness
- Persistence diagram Wasserstein distance, when `gudhi` and `POT` are installed
- Bottleneck distance, when `gudhi` is installed

Image quality:

- L1 against a held-out real batch
- SSIM against a held-out real batch
- LPIPS, when `lpips` is installed
- FID, when TorchVision can load Inception weights

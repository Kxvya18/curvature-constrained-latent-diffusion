# Run Commands

From this folder:

```bash
cd "/Users/boolu/Desktop/simulations/ALL PROJECTS/curvature_constrained_latent_diffusion"
```

Verify the real dataset:

```bash
python3 src/inspect_dataset.py --zip-path data/hyper-kvasir-segmented-images.zip
```

Create a clean local environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the full real-data experiment:

```bash
python src/train_ccld_kvasir.py --config configs/kvasir_real.json
```

Because this upgraded pipeline trains four ablations and an RGB renderer, it will run longer than the earlier mask-only version. Existing checkpoints are reused when `resume_from_checkpoints` is true in the config.

Sync from the writable space-folder copy to your target underscore folder:

```bash
rsync -av --exclude ".venv" --exclude "results" "/Users/boolu/Desktop/simulations/ALL PROJECTS/curvature_constrained_latent_diffusion/" "/Users/boolu/Desktop/simulations/ALL_PROJECTS/curvature_constrained_latent_diffusion/"
```

If you want a faster first test, edit `configs/kvasir_real.json`:

```json
{
  "timesteps": 50,
  "ae_epochs": 1,
  "diffusion_epochs": 1,
  "sample_count": 8
}
```

Then run the same training command again.

## Chapter-Grade V2 Pipeline

Use this redesigned pipeline for the chapter claim:

```bash
python src/train_chapter_pipeline_v2.py --config configs/chapter_pipeline_v2.json
python src/make_chapter_tables.py --summary results/chapter_pipeline_v2/summary.json --out results/chapter_pipeline_v2/chapter_tables.md
```

What this version changes:

- Generates topology-aware masks with area-preserving anchors.
- Trains RGB image autoencoder latents.
- Trains conditional RGB latent diffusion from masks.
- Evaluates four ablations: baseline, curvature only, topology only, full CCLD.
- Reports topology metrics plus image metrics.

Quick runtime smoke test:

```bash
python src/train_chapter_pipeline_v2.py --config configs/chapter_pipeline_v2_smoke.json
```

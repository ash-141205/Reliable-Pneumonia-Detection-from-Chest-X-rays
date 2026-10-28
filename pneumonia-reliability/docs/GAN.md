# GAN Augmentation Protocol

## Goal

Test whether GAN-generated NORMAL chest X-rays improve **external** reliability
and calibration of the pneumonia classifier, or only the internal (Kermany)
scores.  This is an experiment with an honest answer: a negative result
(GAN helps internally but not externally) is valid and worth reporting.

---

## Leakage rules (hard, enforced in code)

| Rule | Where enforced |
|------|----------------|
| The GAN trains **only** on the `train` rows of a given run's `split.csv`. | `gan.py` → asserts no "synthetic" in path before training |
| One GAN is trained **per seed** (splits differ by seed). | `run_gan.sh` loops over seeds |
| Synthetic images are added to **training only**; they never appear in val, test, or external. | `data.py → assert_no_synthetic_outside_train` raises loudly |
| Every synthetic image is tagged `source=synthetic` in the manifest. | `data.py → save_manifest` |
| Evaluation is on **real images only**. | `evaluate.py` unchanged; validation and temperature fit are real-only |

---

## Architecture: Class-conditional DCGAN at 64 × 64 grayscale

```
Noise z ∈ ℝ^128  ──┐
                    ├─► concat ─► ConvTranspose stack (512→256→128→64→1) ─► Tanh ─► 64×64 image
Class emb ∈ ℝ^2  ──┘

Image (1ch) ──┐
              ├─► concat class-channel ─► Spectral-norm Conv stack (2→64→128→256→512→1) ─► logit
Class emb   ──┘
```

Stabilisers used:
- **Spectral normalisation** on all discriminator Conv layers.
- **Instance noise** on discriminator inputs (annealed from σ=0.1 to 0 over training).
- **Label smoothing** on real targets (0.9 instead of 1.0).
- **Adam(lr=2e-4, betas=(0.5, 0.999))** for both G and D.

> **Limitation:** 64×64 generation means the model cannot capture fine lung
> texture at full resolution. Images are upsampled to 224×224 with bicubic
> interpolation for the classifier. GAN-based augmentation gains are often small
> or absent; this is a known limitation, not a bug.

---

## Flags

### `python -m pneumreliab.gan train`

| Flag | Default | Description |
|------|---------|-------------|
| `--run-dir` | (required) | Path to `results/<tag>_<arch>_s<seed>` directory (must contain `split.csv`). |
| `--epochs` | 50 | Number of GAN training epochs. Use `--epochs 1` for smoke tests. |
| `--batch-size` | 64 | Mini-batch size. |
| `--seed` | 0 | Random seed for reproducibility. |
| `--sample-every` | 5 | Save a fixed-seed sample grid every N epochs. |
| `--num-workers` | 2 | DataLoader workers. |

Outputs saved to `--run-dir`:
- `generator.pt` — saved generator state dict.
- `gan_loss.json` + `gan_loss.png` — per-epoch G/D losses.
- `gan_grid_ep<N>.png` — fixed-seed sample grids.

### `python -m pneumreliab.gan sample`

| Flag | Default | Description |
|------|---------|-------------|
| `--run-dir` | (required) | Run directory containing `generator.pt`. |
| `--cls` | (required) | `NORMAL` or `PNEUMONIA`. |
| `--n` | (required) | Number of images to generate. |
| `--out` | (required) | Output directory. |
| `--seed` | 0 | Sampling seed. |
| `--size` | 224 | Upsample size in pixels (bicubic). |

If `torch-fidelity` or `pytorch-fid` is installed, FID is computed between the
real train images and the synthetic directory; otherwise a message is logged and
training continues.

### `python -m pneumreliab.train` (new GAN flags)

| Flag | Default | Description |
|------|---------|-------------|
| `--synthetic-dir` | None | Directory of synthetic images. If omitted, training is unchanged. |
| `--synthetic-ratio` | 1.0 | Fraction of minority-class count to add (0.5 = 50 %, 1.0 = 100 %). |
| `--synthetic-cls` | `NORMAL` | Which class the synthetic images belong to. |

---

## How to run

### Quick smoke test (no GPU needed)

```bash
export PYTHONPATH=src
# 1. Run a tiny baseline training to generate split.csv
python -m pneumreliab.train \
  --internal /data/chest_xray --external /data/external \
  --models smallcnn --seeds 0 --epochs 1 --batch-size 8 \
  --num-workers 0 --tag smoke --out results

# 2. Train GAN for 1 epoch
python -m pneumreliab.gan train \
  --run-dir results/smoke_smallcnn_s0 --epochs 1 --batch-size 8 \
  --seed 0 --num-workers 0

# 3. Sample 5 images
python -m pneumreliab.gan sample \
  --run-dir results/smoke_smallcnn_s0 --cls NORMAL --n 5 \
  --out results/smoke_smallcnn_s0/synthetic/NORMAL --seed 0

# 4. Train with synthetic augmentation
python -m pneumreliab.train \
  --internal /data/chest_xray --external /data/external \
  --models smallcnn --seeds 0 --epochs 1 --batch-size 8 \
  --num-workers 0 --tag smoke_gan --out results \
  --synthetic-dir results/smoke_smallcnn_s0/synthetic/NORMAL \
  --synthetic-ratio 0.5 --synthetic-cls NORMAL
```

### Full GPU run (3 seeds, ResNet50, 100 GAN epochs)

```bash
export INTERNAL=/data/chest_xray
export EXTERNAL=/data/external
export SEEDS="0 1 2"
export EPOCHS=10
export GAN_EPOCHS=100
export GAN_BATCH=64
export N_SYNTH=2000
export MODELS=resnet50

bash scripts/run_gan.sh
```

Results will be written to `results/` with the tags:
- `baseline_resnet50_s{0,1,2}` — no augmentation
- `gan_r05_resnet50_s{0,1,2}` — 50 % synthetic NORMAL
- `gan_r10_resnet50_s{0,1,2}` — 100 % synthetic NORMAL

The aggregate table at `results/summary.md` reports mean ± std over seeds for
each tag × split × calibration combination.

---

## Optional FID quality check

If either `torch-fidelity` or `pytorch-fid` is installed, running
`python -m pneumreliab.gan sample` will automatically compute FID between
the real training images and the generated directory and save
`results/<run>/fid_NORMAL.json`.

```bash
pip install torch-fidelity    # or: pip install pytorch-fid
```

If neither is installed, the tool prints:

```
[GAN] FID skipped -- install torch-fidelity or pytorch-fid to enable.
```

and continues normally.

---

## What to log for the paper

The following information should be recorded for each experimental condition
(not invented here — all values come from real runs):

- `results/summary.md` → AUROC, sensitivity, specificity, ECE (raw + calibrated),
  mean ± std over seeds, for internal_test and external_test.
- `results/*/gan_loss.json` → G/D loss curves (confirm training stability).
- `results/*/fid_NORMAL.json` → FID score if computed.
- `results/*/config.json` → `n_real_train`, `n_synthetic_train` (real vs synthetic
  image counts per class).
- Sample grids (`gan_grid_ep*.png`) for qualitative inspection.

---

## Limits and caveats

1. **64×64 resolution.** The GAN operates at 64×64 and images are upsampled
   to 224×224 for the classifier. High-frequency texture will be blurred.
2. **GAN gains are often small.** Many papers report minimal benefit from
   GAN augmentation on medical imaging tasks. A null or negative result here is
   scientifically valid.
3. **Class imbalance.** Only NORMAL images are generated by default (to
   up-sample the minority class). Change `--synthetic-cls` to generate PNEUMONIA
   if you want to test that direction.
4. **No external leakage.** The GAN never sees external images. Improvements
   on external metrics are solely due to better generalisation of the classifier,
   not data contamination.
5. **FID is indicative, not causal.** A lower FID does not guarantee improved
   classifier performance.

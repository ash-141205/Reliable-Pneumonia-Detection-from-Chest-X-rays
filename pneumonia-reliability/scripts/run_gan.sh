#!/usr/bin/env bash
# GAN-augmentation experiment suite.
#
# Usage:
#   INTERNAL=/data/chest_xray EXTERNAL=/data/external bash scripts/run_gan.sh
#
# Environment variables (all optional, with defaults):
#   SEEDS   – space-separated seeds  (default: "0 1 2")
#   EPOCHS  – classifier epochs       (default: 10)
#   GAN_EPOCHS – GAN epochs           (default: 100)
#   GAN_BATCH  – GAN batch size       (default: 64)
#   N_SYNTH    – synthetic images per class per seed (default: 2000)
#   MODELS     – arch(s) to train     (default: "resnet50")
#   RESULTS    – output root          (default: results)
#
# What it does (per seed):
#   1. Train a baseline classifier (tag=baseline) and run evaluate.
#   2. Train a GAN on that seed's train split.
#   3. Sample N_SYNTH NORMAL images.
#   4. Re-train the classifier with synthetic ratio 0.5 (tag=gan_r05).
#   5. Re-train the classifier with synthetic ratio 1.0 (tag=gan_r10).
#   6. Run aggregate to produce summary.md.
#
# Hard rules enforced by the code:
#   - The GAN reads only the "train" rows of split.csv (no val/test/external).
#   - Synthetic images are appended to training only; evaluation is real-only.
#   - A separate GAN is trained per seed.
set -euo pipefail

: "${INTERNAL:?set INTERNAL=/path/to/chest_xray}"
: "${EXTERNAL:?set EXTERNAL=/path/to/external}"

SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-10}"
GAN_EPOCHS="${GAN_EPOCHS:-100}"
GAN_BATCH="${GAN_BATCH:-64}"
N_SYNTH="${N_SYNTH:-2000}"
MODELS="${MODELS:-resnet50}"
RESULTS="${RESULTS:-results}"

export PYTHONPATH=src

# Common classifier flags shared across all runs
BASE_FLAGS="--internal $INTERNAL --external $EXTERNAL \
  --models $MODELS --epochs $EPOCHS"

echo "=== GAN Experiment Suite ==="
echo "SEEDS=$SEEDS  EPOCHS=$EPOCHS  GAN_EPOCHS=$GAN_EPOCHS"
echo "N_SYNTH=$N_SYNTH  MODELS=$MODELS  RESULTS=$RESULTS"

for SEED in $SEEDS; do
  echo ""
  echo "──────────────── Seed $SEED ────────────────"

  # ── 1. Baseline classifier ────────────────────────────────────────────────
  echo "[1/5] Training baseline classifier (seed=$SEED) ..."
  python -m pneumreliab.train $BASE_FLAGS \
    --seeds "$SEED" --tag baseline --out "$RESULTS"

  BASELINE_RUN="$RESULTS/baseline_${MODELS}_s${SEED}"

  # ── 2. Train GAN on this seed's train split ───────────────────────────────
  echo "[2/5] Training GAN on $BASELINE_RUN/split.csv (train only) ..."
  python -m pneumreliab.gan train \
    --run-dir "$BASELINE_RUN" \
    --epochs "$GAN_EPOCHS" \
    --batch-size "$GAN_BATCH" \
    --seed "$SEED" \
    --num-workers 4

  # ── 3. Sample synthetic NORMAL images ─────────────────────────────────────
  SYN_DIR="$BASELINE_RUN/synthetic/NORMAL"
  echo "[3/5] Sampling $N_SYNTH synthetic NORMAL images -> $SYN_DIR ..."
  python -m pneumreliab.gan sample \
    --run-dir "$BASELINE_RUN" \
    --cls NORMAL \
    --n "$N_SYNTH" \
    --out "$SYN_DIR" \
    --seed "$SEED"

  # ── 4. Train with synthetic ratio 0.5 ─────────────────────────────────────
  echo "[4/5] Training gan_r05 (ratio=0.5, seed=$SEED) ..."
  python -m pneumreliab.train $BASE_FLAGS \
    --seeds "$SEED" --tag gan_r05 --out "$RESULTS" \
    --synthetic-dir "$SYN_DIR" \
    --synthetic-ratio 0.5 \
    --synthetic-cls NORMAL

  # ── 5. Train with synthetic ratio 1.0 ─────────────────────────────────────
  echo "[5/5] Training gan_r10 (ratio=1.0, seed=$SEED) ..."
  python -m pneumreliab.train $BASE_FLAGS \
    --seeds "$SEED" --tag gan_r10 --out "$RESULTS" \
    --synthetic-dir "$SYN_DIR" \
    --synthetic-ratio 1.0 \
    --synthetic-cls NORMAL

done   # end seed loop

# ── Aggregate results ─────────────────────────────────────────────────────────
echo ""
echo "=== Aggregating results ==="
python -m pneumreliab.aggregate --results "$RESULTS"

echo ""
echo "Done. Results:"
echo "  $RESULTS/summary.md      -- mean ± std table (baseline vs gan_r05 vs gan_r10)"
echo "  $RESULTS/summary.csv     -- same data as CSV"
echo "  $RESULTS/*/plots.png     -- ROC, reliability, abstention curves"
echo "  $RESULTS/*/gan_*.png     -- GAN sample grids and loss curve"
echo ""
echo "Full GPU run commands are printed in docs/GAN.md."

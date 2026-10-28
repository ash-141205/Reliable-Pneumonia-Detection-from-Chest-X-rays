#!/usr/bin/env bash
# Full experiment suite. Usage: INTERNAL=/data/chest_xray EXTERNAL=/data/external bash scripts/run_all.sh
set -euo pipefail
: "${INTERNAL:?set INTERNAL}"; : "${EXTERNAL:?set EXTERNAL}"
SEEDS="${SEEDS:-0 1 2}"; EPOCHS="${EPOCHS:-10}"
export PYTHONPATH=src
C="--internal $INTERNAL --external $EXTERNAL --seeds $SEEDS --epochs $EPOCHS"

# 1. Main experiments: from-scratch baseline vs fine-tuned models
python -m pneumreliab.train $C --models smallcnn resnet18 resnet50 --tag main
# 2. Ablations (resnet50): remove one ingredient at a time
python -m pneumreliab.train $C --models resnet50 --no-augment      --tag abl_noaug
python -m pneumreliab.train $C --models resnet50 --no-class-weight --tag abl_nocw
python -m pneumreliab.train $C --models resnet50 --no-pretrained   --tag abl_nopre
# 3. Shortcut audit and summary tables (mean +/- std over seeds)
python -m pneumreliab.shortcut --internal "$INTERNAL" --external "$EXTERNAL"
python -m pneumreliab.aggregate --results results
echo "Done. See results/summary.md, results/*/plots.png, results/shortcut_audit.json"

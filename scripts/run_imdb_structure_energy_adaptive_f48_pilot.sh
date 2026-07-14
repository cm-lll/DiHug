#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

SEED="${SEED:-0}"
OUT_DIR="${OUT_DIR:-results/imdb_structure_energy_adaptive_f48_pilot_seed${SEED}}"
mkdir -p "${OUT_DIR}"

python scripts/run_structure_energy_gates.py \
  --output-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --null-chains 4 \
  --null-steps 100000 \
  --null-burn-in 20000 \
  --thin 5000 \
  --max-features 48 \
  --nce-l2 0.1 \
  --nce-max-weight 8.0 \
  --moment-rounds 24 \
  --moment-steps-per-round 4000 \
  --moment-samples-per-round 4 \
  --moment-learning-rate 1.0 \
  --moment-l2 0.01 \
  --moment-tolerance 0.10 \
  --moment-max-batches-per-round 8 \
  --moment-min-samples-per-chain 16 \
  --moment-min-ess-per-chain 8 \
  --moment-rhat-threshold 1.10 \
  --moment-drift-z-threshold 2.0 \
  --moment-warmup-steps-per-round 4000 \
  --moment-step-decay 0.60 \
  --moment-update-norm-clip 0.5 \
  --moment-diagnostic-quantile 0.90 \
  --moment-max-retained-samples-per-chain 256 \
  --moment-min-successful-updates 10 \
  --moment-pilot-only \
  --structure-tolerance 0.10 \
  --rhat-threshold 1.10 \
  --min-ess-per-chain 20 \
  --chain-drift-z-threshold 2.0 \
  --chain-max-extensions 4 \
  --chain-extension-steps 250000 \
  --diagnostic-window 64 \
  --curveball-probability 0.0 \
  2>&1 | tee "${OUT_DIR}/run.log"

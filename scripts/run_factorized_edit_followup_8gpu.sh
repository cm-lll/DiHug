#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/lyh/gnn_project/DiHug}"
DATASET_DIR="${DATASET_DIR:-data/IMDB_movie_subgraphs/subgraph_000}"
TAG="${TAG:-imdb_factorized_edit_followup}"
SAMPLE_STEPS="${SAMPLE_STEPS:-20000}"
SAMPLE_CANDIDATES="${SAMPLE_CANDIDATES:-64}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/$TAG}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/$TAG}"
CONDA_SH="${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
STARTUP_STAGGER_SECONDS="${STARTUP_STAGGER_SECONDS:-15}"

source "$CONDA_SH"
conda activate sparse_block
cd "$ROOT"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

names=(c32_a00_s0 c32_a10_s0 c64_a00_s0 c64_a10_s0 c64_a10_s1 c64_a10_s2 c64_a25_s0 c128_a10_s0)
train_candidates=(32 32 64 64 64 64 64 128)
aux_weights=(0.0 0.1 0.0 0.1 0.1 0.1 0.25 0.1)
seeds=(0 0 0 0 1 2 0 0)
pids=()

for gpu in 0 1 2 3 4 5 6 7; do
  name="${names[$gpu]}"
  candidates="${train_candidates[$gpu]}"
  aux="${aux_weights[$gpu]}"
  seed="${seeds[$gpu]}"
  (
    sleep "$((gpu * STARTUP_STAGGER_SECONDS))"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_mixed_edit_diffusion.py \
      --mode train --denoiser factorized --kernel bdrs \
      --dataset-dir "$DATASET_DIR" --output-dir "$OUT_ROOT/$name" \
      --seed "$seed" --device cuda \
      --trajectories 32 --forward-steps 2000 --record-every 20 \
      --train-candidates "$candidates" --epochs 100 --batch-size 64 \
      --hidden-dim 256 --layers 3 --fixed-t-trials 256 \
      --balanced-action-aux-weight "$aux" \
      > "$LOG_ROOT/train_${name}.log" 2>&1
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_mixed_edit_diffusion.py \
      --mode sample --denoiser factorized --kernel bdrs \
      --output-dir "$OUT_ROOT/${name}_sample" \
      --checkpoint "$OUT_ROOT/$name/factorized_edit_diffusion.pt" \
      --seed "$seed" --device cuda \
      --sample-steps "$SAMPLE_STEPS" \
      --sample-candidates "$SAMPLE_CANDIDATES" \
      --sample-temperature 0 --metrics-every 1000 \
      > "$LOG_ROOT/sample_${name}.log" 2>&1
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"

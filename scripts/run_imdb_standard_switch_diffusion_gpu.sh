#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPU="${GPU:-7}"
SEED="${SEED:-0}"
DATASET_DIR="${DATASET_DIR:-$ROOT/data/IMDB_movie_subgraphs/subgraph_000}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/imdb_standard_switch_diffusion_gpu${GPU}}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imdb_standard_switch_diffusion_gpu${GPU}}"
STRUCTURE_CKPT="${STANDARD_STRUCTURE_CKPT:-$OUT_ROOT/structure_checkpoint.json}"

source /data2/lyh/miniconda3/etc/profile.d/conda.sh
conda activate sparse_block
cd "$ROOT"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

if [[ ! -f "$STRUCTURE_CKPT" ]]; then
  python scripts/build_switch_structure_checkpoint.py \
    --dataset-dir "$DATASET_DIR" --output "$STRUCTURE_CKPT" \
    --seed 0 --null-chains 4 --null-steps 5000 --null-thin 500 \
    --max-features 64 \
    --feature-prefix triangle/ \
    --feature-prefix root_triangle/ \
    --feature-prefix root_triangle_total/ \
    --reserve-prefix triangle/=-1 \
    --reserve-prefix root_triangle_total/=-1 \
    > "$LOG_ROOT/structure_checkpoint.log" 2>&1
fi

COMMON_SCHEDULE=(
  --diffusion-steps 100
  --switches-per-diffusion-step 200
)

echo "[STANDARD-SWITCH] stage A: target-free reverse-kernel training"
CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_switch_diffusion.py \
  --dataset-dir "$DATASET_DIR" --structure-checkpoint "$STRUCTURE_CKPT" \
  --output-dir "$OUT_ROOT/base_train" --seed "$SEED" --device cuda \
  --conditioning-mode state_only --training-objective inverse_ce \
  "${COMMON_SCHEDULE[@]}" \
  --trajectories 4 --record-every 50 --train-candidates 32 \
  --epochs 100 --batch-size 64 --hidden-dim 256 --layers 3 --dropout 0.1 \
  --train-only > "$LOG_ROOT/base_train.log" 2>&1

echo "[STANDARD-SWITCH] stage B: matched stochastic and deterministic reverse chains"
CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_switch_diffusion.py \
  --dataset-dir "$DATASET_DIR" --structure-checkpoint "$STRUCTURE_CKPT" \
  --output-dir "$OUT_ROOT/base_stochastic" \
  --run-label standard_switch_diffusion_base \
  --seed "$SEED" --device cuda \
  --test-only "$OUT_ROOT/base_train/switch_diffusion.pt" \
  "${COMMON_SCHEDULE[@]}" --randomization-steps 20000 \
  --sample-candidates 64 --sample-temperature 1.0 \
  --sampling-mode model --metrics-every 2000 --switch-kernel tensor \
  > "$LOG_ROOT/base_stochastic.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_switch_diffusion.py \
  --dataset-dir "$DATASET_DIR" --structure-checkpoint "$STRUCTURE_CKPT" \
  --output-dir "$OUT_ROOT/base_deterministic" \
  --run-label standard_switch_diffusion_argmax \
  --seed "$SEED" --device cuda \
  --test-only "$OUT_ROOT/base_train/switch_diffusion.pt" \
  "${COMMON_SCHEDULE[@]}" --randomization-steps 20000 \
  --sample-candidates 64 --sample-temperature 0 \
  --sampling-mode model --metrics-every 2000 --switch-kernel tensor \
  > "$LOG_ROOT/base_deterministic.log" 2>&1

echo "[STANDARD-SWITCH] stage C: learned-condition guidance on the same time layers"
CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_switch_diffusion.py \
  --dataset-dir "$DATASET_DIR" --structure-checkpoint "$STRUCTURE_CKPT" \
  --dataset-cache "$OUT_ROOT/base_train/switch_dataset.npz" \
  --output-dir "$OUT_ROOT/guided_train" --seed "$SEED" --device cuda \
  --conditioning-mode state_only --training-objective utility_kl \
  --utility-temperature 1 --learned-condition --learned-condition-ridge 1e-6 \
  "${COMMON_SCHEDULE[@]}" \
  --epochs 1 --batch-size 64 --hidden-dim 256 --layers 3 --dropout 0 \
  --train-only > "$LOG_ROOT/guided_train.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_switch_diffusion.py \
  --dataset-dir "$DATASET_DIR" --structure-checkpoint "$STRUCTURE_CKPT" \
  --output-dir "$OUT_ROOT/guided_stochastic" \
  --run-label standard_switch_diffusion_guided \
  --seed "$SEED" --device cuda \
  --test-only "$OUT_ROOT/guided_train/switch_diffusion.pt" \
  "${COMMON_SCHEDULE[@]}" --randomization-steps 20000 \
  --sample-candidates 64 --sample-temperature 1.0 \
  --sampling-mode model --metrics-every 2000 --switch-kernel tensor \
  > "$LOG_ROOT/guided_stochastic.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_switch_diffusion.py \
  --dataset-dir "$DATASET_DIR" --structure-checkpoint "$STRUCTURE_CKPT" \
  --output-dir "$OUT_ROOT/guided_noop_quality" \
  --run-label guided_noop_quality_upper \
  --seed "$SEED" --device cuda \
  --test-only "$OUT_ROOT/guided_train/switch_diffusion.pt" \
  "${COMMON_SCHEDULE[@]}" --randomization-steps 20000 \
  --sample-candidates 64 --sample-temperature 0 \
  --sampling-mode model --metrics-every 2000 --switch-kernel tensor \
  --learned-condition-noop --learned-condition-noop-patience 1000 \
  > "$LOG_ROOT/guided_noop_quality.log" 2>&1

python scripts/summarize_switch_paper_matrix.py \
  --dataset imdb --dataset-dir "$DATASET_DIR" --root "$OUT_ROOT" \
  --raw-output "$OUT_ROOT/paper_raw.tsv" \
  --aggregate-output "$OUT_ROOT/paper_aggregate.tsv" \
  | tee "$LOG_ROOT/summary.log"

echo "[STANDARD-SWITCH] all stages completed"

#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET_NAME="${DATASET_NAME:?set DATASET_NAME to imdb or pubmed}"
DATASET_DIR="${DATASET_DIR:?set DATASET_DIR}"
STRUCTURE_CKPT="${STRUCTURE_CKPT:?set STRUCTURE_CKPT}"
OUT_ROOT="${OUT_ROOT:?set OUT_ROOT}"
LOG_ROOT="${LOG_ROOT:?set LOG_ROOT}"
SEED_OFFSET="${SEED_OFFSET:-0}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
read -r -a GPUS <<< "$GPU_LIST"
GPU_COUNT="${#GPUS[@]}"
if (( GPU_COUNT < 7 )); then
  echo "PAPER-MATRIX requires at least 7 GPUs, got: $GPU_LIST" >&2
  exit 1
fi

source /data2/lyh/miniconda3/etc/profile.d/conda.sh
conda activate sparse_block
cd "$ROOT"
mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$(dirname "$STRUCTURE_CKPT")"

if [[ ! -f "$STRUCTURE_CKPT" ]]; then
  echo "[PAPER-MATRIX] building identity-free relation-degree/graphlet checkpoint"
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

train_learned() {
  local gpu="$1" seed="$2" tag="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_switch_diffusion.py \
    --dataset-dir "$DATASET_DIR" --structure-checkpoint "$STRUCTURE_CKPT" \
    --output-dir "$OUT_ROOT/${tag}_s${seed}_train" \
    --seed "$seed" --device cuda --conditioning-mode state_only \
    --training-objective utility_kl --utility-temperature 1.0 \
    --learned-condition --learned-condition-ridge 1e-6 \
    --trajectories 4 --forward-steps 20000 --record-every 100 \
    --train-candidates 16 --epochs 1 --batch-size 64 \
    --hidden-dim 256 --layers 3 --dropout 0 --train-only \
    > "$LOG_ROOT/${tag}_s${seed}_train.log" 2>&1
}

sample() {
  local gpu="$1" checkpoint="$2" out_name="$3" seed="$4" label="$5"
  local mode="$6" steps="$7" candidates="$8" temperature="$9"
  shift 9
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_switch_diffusion.py \
    --dataset-dir "$DATASET_DIR" --structure-checkpoint "$STRUCTURE_CKPT" \
    --output-dir "$OUT_ROOT/$out_name" --run-label "$label" \
    --seed "$seed" --device cuda --test-only "$checkpoint" \
    --randomization-steps 20000 --sample-steps "$steps" \
    --sample-candidates "$candidates" --sample-temperature "$temperature" \
    --sampling-mode "$mode" --metrics-every 2000 --switch-kernel tensor \
    "$@" > "$LOG_ROOT/$out_name.log" 2>&1
}

train_and_sample_state_only() {
  local gpu="$1" seed="$2" objective="$3" tag="$4"
  local train_dir="$OUT_ROOT/${tag}_s${seed}_train"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_switch_diffusion.py \
    --dataset-dir "$DATASET_DIR" --structure-checkpoint "$STRUCTURE_CKPT" \
    --output-dir "$train_dir" --seed "$seed" --device cuda \
    --conditioning-mode state_only --training-objective "$objective" \
    --inverse-loss-weight 1 --utility-loss-weight 1 \
    --utility-temperature 1 --utility-schedule-power 1 \
    --trajectories 4 --forward-steps 20000 --record-every 100 \
    --train-candidates 16 --epochs 100 --batch-size 64 \
    --hidden-dim 256 --layers 3 --dropout 0.1 --train-only \
    > "$LOG_ROOT/${tag}_s${seed}_train.log" 2>&1
  sample "$gpu" "$train_dir/switch_diffusion.pt" \
    "${tag}_s${seed}_sample20k" "$seed" "$tag" model 20000 64 0.1
}

train_and_sample_target_upper_bound() {
  local gpu="$1" seed="$2"
  local train_dir="$OUT_ROOT/target_residual_s${seed}_train"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_switch_diffusion.py \
    --dataset-dir "$DATASET_DIR" --structure-checkpoint "$STRUCTURE_CKPT" \
    --output-dir "$train_dir" --seed "$seed" --device cuda \
    --conditioning-mode target_residual --training-objective inverse_ce \
    --trajectories 4 --forward-steps 20000 --record-every 100 \
    --train-candidates 16 --epochs 100 --batch-size 64 \
    --hidden-dim 256 --layers 3 --dropout 0.1 --train-only \
    > "$LOG_ROOT/target_residual_s${seed}_train.log" 2>&1
  sample "$gpu" "$train_dir/switch_diffusion.pt" \
    "target_residual_model_s${seed}" "$seed" \
    target_residual_model model 20000 64 0
  sample "$gpu" "$train_dir/switch_diffusion.pt" \
    "oracle_upper_bound_s${seed}" "$seed" \
    oracle_upper_bound oracle 20000 64 0
}

wait_all() {
  local status=0 pid
  for pid in "$@"; do
    if ! wait "$pid"; then status=1; fi
  done
  return "$status"
}

echo "[PAPER-MATRIX] stage A: ${GPU_COUNT}-seed learned-condition main result"
pids=()
for worker in "${!GPUS[@]}"; do
  gpu="${GPUS[$worker]}"
  seed=$((SEED_OFFSET + worker))
  (
    train_learned "$gpu" "$seed" learned
    sample "$gpu" "$OUT_ROOT/learned_s${seed}_train/switch_diffusion.pt" \
      "learned_s${seed}_sample20k" "$seed" learned_condition_main \
      model 20000 64 0 \
      --learned-condition-noop --learned-condition-noop-patience 1000
  ) &
  pids+=("$!")
done
if ! wait_all "${pids[@]}"; then exit 1; fi

echo "[PAPER-MATRIX] stage B: mechanism ablations"
seed="$SEED_OFFSET"
learned_ckpt="$OUT_ROOT/learned_s${seed}_train/switch_diffusion.pt"
pids=()
sample "${GPUS[0]}" "$learned_ckpt" learned_no_guard_s${seed} "$seed" \
  learned_no_guard model 20000 64 0 &
pids+=("$!")
sample "${GPUS[1]}" "$learned_ckpt" learned_no_early_stop_s${seed} "$seed" \
  learned_no_early_stop model 20000 64 0 \
  --learned-condition-noop &
pids+=("$!")
sample "${GPUS[2]}" "$learned_ckpt" random_null_s${seed} "$seed" \
  random_switch_null random 20000 64 0 &
pids+=("$!")
sample "${GPUS[3]}" "$learned_ckpt" learned_c16_s${seed} "$seed" \
  candidate_c16 model 20000 16 0 \
  --learned-condition-noop --learned-condition-noop-patience 1000 &
pids+=("$!")
sample "${GPUS[4]}" "$learned_ckpt" learned_c32_s${seed} "$seed" \
  candidate_c32 model 20000 32 0 \
  --learned-condition-noop --learned-condition-noop-patience 1000 &
pids+=("$!")
sample "${GPUS[5]}" "$learned_ckpt" learned_c128_s${seed} "$seed" \
  candidate_c128 model 20000 128 0 \
  --learned-condition-noop --learned-condition-noop-patience 1000 &
pids+=("$!")
train_and_sample_state_only "${GPUS[6]}" "$seed" scheduled_hybrid target_free_scheduled &
pids+=("$!")
if (( GPU_COUNT >= 8 )); then
  train_and_sample_target_upper_bound "${GPUS[7]}" "$seed" &
  pids+=("$!")
fi
if ! wait_all "${pids[@]}"; then exit 1; fi
if (( GPU_COUNT == 7 )); then
  train_and_sample_target_upper_bound "${GPUS[0]}" "$seed"
fi

echo "[PAPER-MATRIX] stage C: denoising-budget and kernel ablations"
pids=()
sample "${GPUS[0]}" "$learned_ckpt" learned_steps2k_s${seed} "$seed" \
  steps_2k model 2000 64 0 \
  --learned-condition-noop --learned-condition-noop-patience 1000 &
pids+=("$!")
sample "${GPUS[1]}" "$learned_ckpt" learned_steps5k_s${seed} "$seed" \
  steps_5k model 5000 64 0 \
  --learned-condition-noop --learned-condition-noop-patience 1000 &
pids+=("$!")
sample "${GPUS[2]}" "$learned_ckpt" learned_steps10k_s${seed} "$seed" \
  steps_10k model 10000 64 0 \
  --learned-condition-noop --learned-condition-noop-patience 1000 &
pids+=("$!")
sample "${GPUS[3]}" "$learned_ckpt" learned_steps30k_s${seed} "$seed" \
  steps_30k model 30000 64 0 \
  --learned-condition-noop --learned-condition-noop-patience 1000 &
pids+=("$!")
sample "${GPUS[4]}" "$learned_ckpt" learned_temp01_s${seed} "$seed" \
  temperature_01 model 20000 64 0.1 \
  --learned-condition-noop --learned-condition-noop-patience 1000 &
pids+=("$!")
sample "${GPUS[5]}" "$learned_ckpt" learned_patience250_s${seed} "$seed" \
  patience_250 model 20000 64 0 \
  --learned-condition-noop --learned-condition-noop-patience 250 &
pids+=("$!")
CUDA_VISIBLE_DEVICES="${GPUS[6]}" python scripts/run_switch_diffusion.py \
  --dataset-dir "$DATASET_DIR" --structure-checkpoint "$STRUCTURE_CKPT" \
  --output-dir "$OUT_ROOT/python_kernel_5k_s${seed}" \
  --run-label python_kernel_5k --seed "$seed" --device cuda \
  --test-only "$learned_ckpt" --randomization-steps 20000 \
  --sample-steps 5000 --sample-candidates 64 --sample-temperature 0 \
  --sampling-mode model --metrics-every 2000 --switch-kernel python \
  --learned-condition-noop --learned-condition-noop-patience 1000 \
  > "$LOG_ROOT/python_kernel_5k_s${seed}.log" 2>&1 &
pids+=("$!")
if (( GPU_COUNT >= 8 )); then
  sample "${GPUS[7]}" "$learned_ckpt" tensor_kernel_5k_s${seed} "$seed" \
    tensor_kernel_5k model 5000 64 0 \
    --learned-condition-noop --learned-condition-noop-patience 1000 &
  pids+=("$!")
fi
if ! wait_all "${pids[@]}"; then exit 1; fi
if (( GPU_COUNT == 7 )); then
  sample "${GPUS[0]}" "$learned_ckpt" tensor_kernel_5k_s${seed} "$seed" \
    tensor_kernel_5k model 5000 64 0 \
    --learned-condition-noop --learned-condition-noop-patience 1000
fi

echo "[PAPER-MATRIX] stage D: exact GPU-kernel benchmarks"
pids=()
for worker in "${!GPUS[@]}"; do
  gpu="${GPUS[$worker]}"
  candidates=$((16 * (2 ** (worker % 5))))
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/benchmark_switch_kernel.py \
    --dataset-dir "$DATASET_DIR" --structure-checkpoint "$STRUCTURE_CKPT" \
    --device cuda --candidates "$candidates" --repetitions 10 \
    --randomization-steps 2000 --seed "$worker" \
    --output "$OUT_ROOT/kernel_benchmark_c${candidates}_s${worker}.json" \
    > "$LOG_ROOT/kernel_benchmark_c${candidates}_s${worker}.log" 2>&1 &
  pids+=("$!")
done
if ! wait_all "${pids[@]}"; then exit 1; fi

python scripts/summarize_switch_paper_matrix.py \
  --dataset "$DATASET_NAME" --dataset-dir "$DATASET_DIR" \
  --root "$OUT_ROOT" \
  --raw-output "$OUT_ROOT/paper_raw.tsv" \
  --aggregate-output "$OUT_ROOT/paper_aggregate.tsv" \
  | tee "$LOG_ROOT/summary.log"
echo "[PAPER-MATRIX] all stages completed"

#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/lyh/gnn_project/DiHug}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/gqer_v2_strict_local_0712}"
ENV_NAME="${ENV_NAME:-sparse_block}"

source /data2/lyh/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
mkdir -p "$LOG_DIR"

TRAIN_COMMON="general.wandb=disabled general.gpus=1 \
  general.run_test_after_train=false \
  general.enable_test_sampling=false \
  general.enable_val_sampling=false \
  train.n_epochs=100 train.batch_size=1 train.save_model=true"

SAMPLE_COMMON="general.wandb=disabled general.gpus=1 \
  general.run_test_after_train=false \
  general.enable_test_sampling=true \
  general.enable_val_sampling=false \
  general.test_variance=1 \
  general.final_model_samples_to_generate=1 \
  general.test_sampling_full_steps=true \
  general.test_sampling_metrics_every=0 \
  general.edge_score_structure_diag=false"

run_one_gpu() {
  local gpu="$1" experiment="$2" base_name="$3" seed="$4"

  local train_name="${base_name}_train"
  local modelonly_name="${base_name}_modelonly_full100"
  local projected_name="${base_name}_projected_full100"

  echo "[$(date '+%F %T')] GPU $gpu training $train_name (seed=$seed)" \
    | tee -a "$LOG_DIR/launcher.log"

  CUDA_VISIBLE_DEVICES="$gpu" python -m dihug.main \
    "+experiment=$experiment" \
    $TRAIN_COMMON \
    "train.seed=$seed" \
    "general.name=$train_name" \
    >"$LOG_DIR/$train_name.log" 2>&1

  local ckpt
  ckpt=$(find "$ROOT/output" -path "*${train_name}*" -name last.ckpt | sort | tail -1)
  if [[ -z "$ckpt" ]]; then
    echo "[$(date '+%F %T')] FAILED: no checkpoint for $train_name" \
      | tee -a "$LOG_DIR/launcher.log"
    return 1
  fi

  echo "[$(date '+%F %T')] GPU $gpu model-only $modelonly_name" \
    | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m dihug.main \
    "+experiment=$experiment" \
    $SAMPLE_COMMON \
    "general.test_only=$ckpt" \
    "general.test_sampling_seeds=[$seed]" \
    "general.name=$modelonly_name" \
    model.sampling_structure_swap_limiter=false \
    >"$LOG_DIR/$modelonly_name.log" 2>&1

  echo "[$(date '+%F %T')] GPU $gpu projected $projected_name" \
    | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m dihug.main \
    "+experiment=$experiment" \
    $SAMPLE_COMMON \
    "general.test_only=$ckpt" \
    "general.test_sampling_seeds=[$seed]" \
    "general.name=$projected_name" \
    model.sampling_structure_swap_limiter=true \
    model.sampling_structure_swap_limiter_exact_sequential=true \
    >"$LOG_DIR/$projected_name.log" 2>&1

  echo "[$(date '+%F %T')] GPU $gpu DONE $base_name" \
    | tee -a "$LOG_DIR/launcher.log"
}

PIDS=()
gpu_idx=0
# GPU 0-3: IMDB seeds 0,1,3,4
for seed in 0 1 3 4; do
  run_one_gpu "$gpu_idx" imdb_movie_hgt_l3h4_gqer_v2 "imdb_strict_s${seed}" "$seed" &
  PIDS+=("$!")
  gpu_idx=$((gpu_idx + 1))
done

# GPU 4-7: PubMed seeds 0,1,3,4
for seed in 0 1 3 4; do
  run_one_gpu "$gpu_idx" pubmed_hetero_xey_hgt_gqer_v2_pw20 "pubmed_strict_s${seed}" "$seed" &
  PIDS+=("$!")
  gpu_idx=$((gpu_idx + 1))
done

echo "[$(date '+%F %T')] all 8 GPUs launched: ${PIDS[*]}" | tee -a "$LOG_DIR/launcher.log"

failed=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || failed=1
done

echo "[$(date '+%F %T')] all done, failed=$failed" | tee -a "$LOG_DIR/launcher.log"
exit "$failed"

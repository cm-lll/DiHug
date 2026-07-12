#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/jdt/gnn_project/DiHug}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/imdb_family_set_remote_0713}"
ENV_NAME="${ENV_NAME:-dihug}"
GPU_CONTROL="${GPU_CONTROL:-3}"
GPU_MAIN="${GPU_MAIN:-4}"
SEED="${SEED:-2}"

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
cd "$ROOT"
mkdir -p "$LOG_DIR"

TRAIN_COMMON=(
  +experiment=imdb_movie_hgt_l3h4_family_set
  general.wandb=disabled general.gpus=1
  general.run_test_after_train=false
  general.enable_test_sampling=false
  general.enable_val_sampling=false
  train.n_epochs=100 train.batch_size=1 train.save_model=true
)
SAMPLE_COMMON=(
  +experiment=imdb_movie_hgt_l3h4_family_set
  general.wandb=disabled general.gpus=1
  general.run_test_after_train=false
  general.enable_test_sampling=true
  general.enable_val_sampling=false
  general.test_variance=1
  general.final_model_samples_to_generate=1
  general.test_sampling_full_steps=true
  general.test_sampling_metrics_every=0
  model.sampling_structure_swap_limiter=false
  model.sampling_exact_k_final_only=true
)

run_train_and_sample() {
  local gpu="$1" variant="$2" set_weight="$3" bce_weight="$4"
  local train_name="imdb_fset_${variant}_s${SEED}_train100"
  local sample_name="imdb_fset_${variant}_s${SEED}_full100"
  echo "[$(date '+%F %T')] GPU=$gpu TRAIN $train_name" \
    | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m dihug.main \
    "${TRAIN_COMMON[@]}" "train.seed=$SEED" "general.name=$train_name" \
    "model.family_set_ranking_loss_weight=$set_weight" \
    "model.edge_exist_weight=$bce_weight" \
    >"$LOG_DIR/$train_name.log" 2>&1

  local ckpt
  ckpt=$(find "$ROOT/output" -path "*${train_name}*" -name last.ckpt \
    | sort | tail -1)
  [[ -n "$ckpt" ]] || { echo "missing checkpoint for $train_name"; return 1; }

  echo "[$(date '+%F %T')] GPU=$gpu SAMPLE $sample_name" \
    | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m dihug.main \
    "${SAMPLE_COMMON[@]}" "general.test_only=$ckpt" \
    "general.test_sampling_seeds=[$SEED]" "general.name=$sample_name" \
    "model.family_set_ranking_loss_weight=$set_weight" \
    "model.edge_exist_weight=$bce_weight" \
    >"$LOG_DIR/$sample_name.log" 2>&1
  echo "[$(date '+%F %T')] GPU=$gpu DONE $variant" \
    | tee -a "$LOG_DIR/launcher.log"
}

run_train_and_sample "$GPU_CONTROL" bce_control 0 1.0 &
run_train_and_sample "$GPU_MAIN" main 1 0.2 &
wait
echo "[$(date '+%F %T')] ALL DONE" | tee -a "$LOG_DIR/launcher.log"


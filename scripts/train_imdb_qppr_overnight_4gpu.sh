#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data2/lyh/miniconda3/envs/sparse_block/bin/python}"
LOG_DIR="${LOG_DIR:-logs/qppr_overnight_0711}"
GPUS_CSV="${GPUS_CSV:-2,3,4,5}"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if (( ${#GPUS[@]} != 4 )); then
  echo "[error] GPUS_CSV must contain exactly four GPUs" >&2
  exit 2
fi

export LD_LIBRARY_PATH="$(dirname "$(dirname "$PYTHON_BIN")")/lib:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

COMMON=(
  +experiment=imdb_movie_hgt_l3h4_population_role_quota
  general.wandb=disabled
  general.gpus=1
  general.run_test_after_train=true
  general.enable_test_sampling=true
  general.enable_val_sampling=false
  general.enable_val_pred_metrics=false
  general.test_variance=1
  general.final_model_samples_to_generate=1
  general.final_model_samples_to_save=1
  general.final_model_chains_to_save=0
  general.test_sampling_full_steps=true
  general.test_sampling_metrics_every=0
  general.log_sampling_triangle_trace=false
  train.n_epochs=100
  train.batch_size=1
  train.save_model=true
  model.population_role_loss_weight=0.03
  model.population_role_loss_t_power=1.0
)

declare -a PIDS=()

launch() {
  local gpu="$1"
  local name="$2"
  shift 2
  echo "[launch] $(date '+%F %T') gpu=$gpu name=$name" | tee -a "$LOG_DIR/launcher.log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    exec "$PYTHON_BIN" -m dihug.main \
      "${COMMON[@]}" \
      "general.name=$name" \
      "$@"
  ) >"$LOG_DIR/$name.log" 2>&1 &
  PIDS+=("$!")
}

# Third seed for the full QPPR objective.
launch "${GPUS[0]}" imdb_qppr_w003_t1_s3 \
  train.seed=3

# Component ablations at the paired seed-1 setting.
launch "${GPUS[1]}" imdb_qppr_degree_only_w003_t1_s1 \
  train.seed=1 \
  model.population_role_wedge_weight=0.0 \
  model.population_role_closure_weight=0.0
launch "${GPUS[2]}" imdb_qppr_motif_only_w003_t1_s1 \
  train.seed=1 \
  model.population_role_degree_weight=0.0

# Gradient-estimator ablation; soft_quota remains the primary method.
launch "${GPUS[3]}" imdb_qppr_st_w003_t1_s1 \
  train.seed=1 \
  model.population_role_projection_mode=straight_through

printf '%s\n' "${PIDS[@]}" >"$LOG_DIR/pids.txt"
wait
echo "[done] $(date '+%F %T') all IMDB QPPR overnight jobs finished" | tee -a "$LOG_DIR/launcher.log"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data2/lyh/miniconda3/envs/sparse_block/bin/python}"
LOG_DIR="${LOG_DIR:-logs/pubmed_raw_seed2_queued_0711}"
BASE_WAIT_PATTERN="${BASE_WAIT_PATTERN:-pubmed_popdegree_base_full100_s1}"
PR_WAIT_PATTERN="${PR_WAIT_PATTERN:-pubmed_popdegree_w0002_full100_s1}"
mkdir -p "$LOG_DIR"

export LD_LIBRARY_PATH="$(dirname "$(dirname "$PYTHON_BIN")")/lib:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

run_after() {
  local wait_pattern="$1"
  local gpu="$2"
  local name="$3"
  local weight="$4"
  while pgrep -f "$wait_pattern" >/dev/null; do
    sleep 60
  done
  echo "[launch] $(date '+%F %T') gpu=$gpu name=$name" | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m dihug.main \
    +experiment=pubmed_hetero_xey_hgt_population_degree_pw20 \
    general.wandb=disabled general.gpus=1 \
    general.run_test_after_train=true \
    general.enable_test_sampling=true \
    general.enable_val_sampling=false \
    general.enable_val_pred_metrics=false \
    general.test_variance=1 \
    general.final_model_samples_to_generate=1 \
    general.test_sampling_full_steps=true \
    general.test_sampling_metrics_every=0 \
    train.n_epochs=100 train.batch_size=1 train.save_model=true train.seed=2 \
    "model.population_role_loss_weight=$weight" \
    "general.name=$name" \
    >"$LOG_DIR/$name.log" 2>&1
}

run_after "$BASE_WAIT_PATTERN" 6 pubmed_popdegree_base_full100_s2 0.0 &
PID_BASE=$!
run_after "$PR_WAIT_PATTERN" 7 pubmed_popdegree_w0002_full100_s2 0.002 &
PID_PR=$!
printf '%s\n' "$PID_BASE" "$PID_PR" >"$LOG_DIR/pids.txt"
wait
echo "[done] $(date '+%F %T') queued PubMed seed-2 pair finished" | tee -a "$LOG_DIR/launcher.log"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
LOG_DIR="${LOG_DIR:-logs/pubmed_qppr_remote_0711}"
GPUS_CSV="${GPUS_CSV:-6,7,4}"
WAIT_PATTERN="${WAIT_PATTERN:-imdb_root_neighbor_full100_seedrobust}"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if (( ${#GPUS[@]} != 3 )); then
  echo "[error] GPUS_CSV must contain exactly three GPUs" >&2
  exit 2
fi

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

COMMON=(
  +experiment=pubmed_hetero_xey_hgt_population_degree_quota_pw20
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
)

launch() {
  local gpu="$1"
  local name="$2"
  local seed="$3"
  local weight="$4"
  echo "[launch] $(date '+%F %T') gpu=$gpu name=$name" | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m dihug.main \
    "${COMMON[@]}" \
    "train.seed=$seed" \
    "model.population_role_loss_weight=$weight" \
    "general.name=$name" \
    >"$LOG_DIR/$name.log" 2>&1
}

# Paired seed-1 sensitivity runs start immediately on the two idle GPUs.
launch "${GPUS[0]}" pubmed_qppr_w0002_full100_s1 1 0.002 &
PID_W002_S1=$!
launch "${GPUS[1]}" pubmed_qppr_w0005_full100_s1 1 0.005 &
PID_W005_S1=$!

# Preserve the running root-neighbor robustness job, then reuse its GPU.
(
  while pgrep -f "$WAIT_PATTERN" >/dev/null; do
    sleep 60
  done
  launch "${GPUS[2]}" pubmed_qppr_w0002_full100_s2 2 0.002
) &
PID_W002_S2=$!

printf '%s\n' "$PID_W002_S1" "$PID_W005_S1" "$PID_W002_S2" >"$LOG_DIR/pids.txt"
wait
echo "[done] $(date '+%F %T') all remote PubMed QPPR jobs finished" | tee -a "$LOG_DIR/launcher.log"

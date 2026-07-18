#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_twohop_time_gate_100ep}"
EPOCHS="${EPOCHS:-100}"
mkdir -p "$LOG_DIR"

launch() {
  local gpu="$1"
  local schedule="$2"
  local seed="$3"
  local name="twohop_${schedule}_a0p5_seed${seed}_ep${EPOCHS}"

  echo "[launch] gpu=${gpu} schedule=${schedule} seed=${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  CONDA_ENV="${CONDA_ENV:-sparse_block}" \
  EXPERIMENT=pubmed_query_gate_twohop_pw20 \
    bash scripts/run_pubmed_baseline_single.sh \
      general.gpus=1 \
      general.name="$name" \
      general.resume=null \
      general.resume_full=false \
      general.run_test_after_train=true \
      general.enable_test_sampling=true \
      general.test_variance=3 \
      'general.test_sampling_seeds=[0,1,2]' \
      general.test_sampling_full_steps=false \
      general.sampling_skip=25 \
      general.check_val_every_n_epochs=50 \
      train.n_epochs="$EPOCHS" \
      train.seed="$seed" \
      model.edge_fraction=0.1 \
      model.use_query_context_gate=false \
      model.use_two_hop_structure=true \
      model.two_hop_structure_hidden_dim=64 \
      model.two_hop_structure_scale=0.5 \
      model.two_hop_structure_schedule="$schedule" \
      >"$LOG_DIR/${name}.out" 2>&1 &
}

launch 0 fixed 0
launch 1 fixed 1
launch 2 linear_t 0
launch 3 linear_t 1
launch 4 quadratic_t 0
launch 5 quadratic_t 1
launch 6 alpha_bar_squared 0
launch 7 alpha_bar_squared 1

wait
echo "[done] all time-gated two-hop runs finished; logs: $LOG_DIR"

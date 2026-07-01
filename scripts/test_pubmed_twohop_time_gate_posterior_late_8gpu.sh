#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_twohop_time_gate_posterior_late}"
mkdir -p "$LOG_DIR"

launch() {
  local gpu="$1"
  local schedule="$2"
  local seed="$3"
  local run_name="twohop_${schedule}_a0p5_seed${seed}_ep100"
  local ckpt

  ckpt="$(find output/2026-06-23 \
    -path "*/output/sparse_diffusion/checkpoints/${run_name}/last.ckpt" \
    -type f -print -quit)"
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    echo "Missing checkpoint for ${run_name}" >&2
    return 1
  fi

  echo "[test] gpu=${gpu} schedule=${schedule} train_seed=${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  CONDA_ENV="${CONDA_ENV:-sparse_block}" \
  EXPERIMENT=pubmed_query_gate_twohop_pw20 \
    bash scripts/run_pubmed_baseline_single.sh \
      general.gpus=1 \
      "general.test_only=$ckpt" \
      "general.name=test_${run_name}_posterior_late" \
      general.run_test_after_train=false \
      general.enable_test_sampling=true \
      general.test_variance=3 \
      'general.test_sampling_seeds=[0,1,2]' \
      general.test_sampling_full_steps=false \
      general.sampling_skip=25 \
      model.edge_fraction=0.1 \
      model.use_query_context_gate=false \
      model.use_two_hop_structure=true \
      model.two_hop_structure_hidden_dim=64 \
      model.two_hop_structure_scale=0.5 \
      model.two_hop_structure_schedule="$schedule" \
      model.sampling_use_reverse_posterior=true \
      'model.sampling_reverse_posterior_mix_weights=[0,0,0.25,0.75]' \
      >"$LOG_DIR/${run_name}_posterior_late.out" 2>&1 &
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
echo "[done] late-posterior tests finished; logs: $LOG_DIR"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_alpha_bar_posterior_4_7step}"
RUN_NAME="${RUN_NAME:-twohop_fixed_a0p5_seed1_ep100}"
mkdir -p "$LOG_DIR"

CKPT="$(find output/2026-06-23 \
  -path "*/output/sparse_diffusion/checkpoints/${RUN_NAME}/last.ckpt" \
  -type f -print -quit)"
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "Missing checkpoint for ${RUN_NAME}" >&2
  exit 1
fi

run_case() {
  local gpu="$1"
  local steps="$2"
  local mode="$3"
  local scale="$4"
  local profile="$5"
  local schedule

  if [[ "$steps" == "4" ]]; then
    schedule='[100,75,50,25,0]'
  elif [[ "$steps" == "7" ]]; then
    schedule='[100,86,71,57,43,29,14,0]'
  else
    echo "Unsupported step count: $steps" >&2
    return 1
  fi

  echo "[test] gpu=${gpu} profile=${profile}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  CONDA_ENV="${CONDA_ENV:-sparse_block}" \
  EXPERIMENT=pubmed_query_gate_twohop_pw20 \
    bash scripts/run_pubmed_baseline_single.sh \
      general.gpus=1 \
      "general.test_only=$CKPT" \
      "general.name=test_${RUN_NAME}_${profile}" \
      general.run_test_after_train=false \
      general.enable_test_sampling=true \
      general.test_variance=1 \
      'general.test_sampling_seeds=[0]' \
      general.test_sampling_full_steps=false \
      "general.sampling_time_schedule=${schedule}" \
      model.edge_fraction=0.1 \
      model.use_query_context_gate=false \
      model.use_two_hop_structure=true \
      model.two_hop_structure_hidden_dim=64 \
      model.two_hop_structure_scale=0.5 \
      model.two_hop_structure_schedule=fixed \
      model.sampling_use_reverse_posterior=true \
      model.sampling_reverse_posterior_mix_weights=null \
      "model.sampling_reverse_posterior_mix_mode=${mode}" \
      "model.sampling_reverse_posterior_mix_scale=${scale}" \
      model.sampling_ranking_intervention_diag=true \
      >"$LOG_DIR/${profile}.out" 2>&1 &
}

run_case 0 4 alpha_bar_s 0.25 steps4_c0p25_alpha
run_case 1 4 alpha_bar_s 0.5  steps4_c0p5_alpha
run_case 2 4 alpha_bar_s 1.0  steps4_c1p0_alpha
run_case 3 7 alpha_bar_s 0.1  steps7_c0p1_alpha
run_case 4 7 alpha_bar_s 0.25 steps7_c0p25_alpha
run_case 5 7 alpha_bar_s 0.5  steps7_c0p5_alpha
run_case 6 7 alpha_bar_s 1.0  steps7_c1p0_alpha
run_case 7 7 alpha_bar_s_squared 0.5 steps7_c0p5_alpha2

wait
echo "[done] alpha-bar posterior sweep finished; logs: $LOG_DIR"

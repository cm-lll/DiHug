#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_alpha_bar_pareto_seeds}"
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
  local scale="$2"
  local sample_seed="$3"
  local scale_tag="${scale//./p}"
  local profile="steps7_c${scale_tag}_alpha_sseed${sample_seed}"

  echo "[test] gpu=${gpu} scale=${scale} sample_seed=${sample_seed}"
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
      "general.test_sampling_seeds=[${sample_seed}]" \
      general.test_sampling_full_steps=false \
      'general.sampling_time_schedule=[100,86,71,57,43,29,14,0]' \
      model.edge_fraction=0.1 \
      model.use_query_context_gate=false \
      model.use_two_hop_structure=true \
      model.two_hop_structure_hidden_dim=64 \
      model.two_hop_structure_scale=0.5 \
      model.two_hop_structure_schedule=fixed \
      model.sampling_use_reverse_posterior=true \
      model.sampling_reverse_posterior_mix_weights=null \
      model.sampling_reverse_posterior_mix_mode=alpha_bar_s \
      "model.sampling_reverse_posterior_mix_scale=${scale}" \
      model.sampling_ranking_intervention_diag=true \
      >"$LOG_DIR/${profile}.out" 2>&1 &
}

# New Pareto points, each with three paired sample seeds.
run_case 0 0.65 0
run_case 1 0.65 1
run_case 2 0.65 2
run_case 3 0.75 0
run_case 4 0.75 1
run_case 5 0.75 2

# Existing endpoints already have seed 0; add seed 1 as an initial robustness
# check while the remaining two GPUs are available.
run_case 6 0.5 1
run_case 7 1.0 1

wait
echo "[done] alpha-bar Pareto seed sweep finished; logs: $LOG_DIR"

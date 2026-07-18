#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_posterior_mix_sweep_fixed_seed1}"
RUN_NAME="${RUN_NAME:-twohop_fixed_a0p5_seed1_ep100}"
mkdir -p "$LOG_DIR"

CKPT="$(find output/2026-06-23 \
  -path "*/output/sparse_diffusion/checkpoints/${RUN_NAME}/last.ckpt" \
  -type f -print -quit)"
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "Missing checkpoint for ${RUN_NAME}" >&2
  exit 1
fi

launch() {
  local gpu="$1"
  local profile="$2"
  local weights="$3"

  echo "[test] gpu=${gpu} profile=${profile} weights=${weights}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  CONDA_ENV="${CONDA_ENV:-sparse_block}" \
  EXPERIMENT=pubmed_query_gate_twohop_pw20 \
    bash scripts/run_pubmed_baseline_single.sh \
      general.gpus=1 \
      "general.test_only=$CKPT" \
      "general.name=test_${RUN_NAME}_${profile}" \
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
      model.two_hop_structure_schedule=fixed \
      model.sampling_use_reverse_posterior=true \
      "model.sampling_reverse_posterior_mix_weights=${weights}" \
      >"$LOG_DIR/${profile}.out" 2>&1 &
}

# Reverse transitions: 100->75, 75->50, 50->25, 25->0.
# Full-on [1,1,1,1] and full-off [0,0,0,0] already have paired results.
launch 0 q1_off_hard_on       '[0,1,1,1]'
launch 1 q1_off_ramp_full     '[0,0.33,0.67,1]'
launch 2 q1_off_ramp_soft     '[0,0.25,0.5,0.75]'
launch 3 half_off_hard_on     '[0,0,1,1]'
launch 4 half_off_ramp_full   '[0,0,0.5,1]'
launch 5 half_off_ramp_soft   '[0,0,0.25,0.75]'
launch 6 q3_off_final_full    '[0,0,0,1]'
launch 7 q3_off_final_half    '[0,0,0,0.5]'

wait
echo "[done] posterior mix sweep finished; logs: $LOG_DIR"

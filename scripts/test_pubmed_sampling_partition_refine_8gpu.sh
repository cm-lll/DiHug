#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_sampling_partition_refine_7step_c0p5}"
RUN_NAME="${RUN_NAME:-twohop_fixed_a0p5_seed1_ep100}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
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
  local tag="$2"
  local rel_power="$3"
  local refine_balance="$4"
  local refine_iter="$5"
  shift 5

  echo "[test] gpu=${gpu} tag=${tag} rel_power=${rel_power} refine=${refine_balance} iter=${refine_iter}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  CONDA_ENV="${CONDA_ENV:-sparse_block}" \
  EXPERIMENT=pubmed_query_gate_twohop_pw20 \
    bash scripts/run_pubmed_baseline_single.sh \
      general.gpus=1 \
      "general.test_only=$CKPT" \
      "general.name=test_${RUN_NAME}_partition_${tag}" \
      general.run_test_after_train=false \
      general.enable_test_sampling=true \
      general.test_variance=1 \
      "general.test_sampling_seeds=[${SAMPLE_SEED}]" \
      general.test_sampling_full_steps=false \
      'general.sampling_time_schedule=[100,86,71,57,43,29,14,0]' \
      model.edge_fraction=0.1 \
      model.hetero_metis_relation_balance_power="$rel_power" \
      model.hetero_metis_refine_degree_balance="$refine_balance" \
      model.hetero_metis_refine_max_iter="$refine_iter" \
      model.use_query_context_gate=false \
      model.use_two_hop_structure=true \
      model.two_hop_structure_hidden_dim=64 \
      model.two_hop_structure_scale=0.5 \
      model.two_hop_structure_schedule=fixed \
      model.sampling_use_reverse_posterior=true \
      model.sampling_reverse_posterior_mix_weights=null \
      model.sampling_reverse_posterior_mix_mode=alpha_bar_s \
      model.sampling_reverse_posterior_mix_scale=0.5 \
      model.sampling_edge_selection=gumbel_exact_k \
      model.sampling_gumbel_temperature=0.01 \
      model.sampling_exact_k_connectivity_repair=true \
      model.sampling_exact_k_repair_max_swaps=0 \
      model.sampling_ranking_intervention_diag=true \
      "$@" \
      >"$LOG_DIR/${tag}.out" 2>&1 &
}

# Same checkpoint and denoising policy as the current best single-seed setting:
# two-hop fixed a=0.5 + 7 steps + 0.5*alpha_bar_s posterior + exact-K + repair.
# Only the sampling type-template partition is changed here.
run_case 0 current_r200_p05 0.5 true 200
run_case 1 r100_p05        0.5 true 100
run_case 2 r50_p05         0.5 true 50
run_case 3 r20_p05         0.5 true 20
run_case 4 r10_p05         0.5 true 10
run_case 5 no_refine_p05   0.5 false 0
run_case 6 no_refine_p00   0.0 false 0
run_case 7 r20_p00         0.0 true 20

wait
echo "[done] sampling partition-refine sweep finished; logs: $LOG_DIR"

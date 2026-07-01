#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_simultaneous_sampling_8gpu}"
mkdir -p "$LOG_DIR"

CKPT_SEED0="${CKPT_SEED0:-output/2026-06-28/01-03-13-family_adapter_queryfree_chunk131072_seed0/output/sparse_diffusion/checkpoints/family_adapter_queryfree_chunk131072_seed0/last.ckpt}"
CKPT_SEED1="${CKPT_SEED1:-output/2026-06-28/01-03-13-family_adapter_queryfree_chunk131072_seed1/output/sparse_diffusion/checkpoints/family_adapter_queryfree_chunk131072_seed1/last.ckpt}"
CKPT_SEED2="${CKPT_SEED2:-output/2026-06-28/01-03-13-family_adapter_queryfree_chunk131072_seed2/output/sparse_diffusion/checkpoints/family_adapter_queryfree_chunk131072_seed2/last.ckpt}"
CKPT_SEED3="${CKPT_SEED3:-output/2026-06-28/01-03-13-family_adapter_queryfree_chunk131072_seed3/output/sparse_diffusion/checkpoints/family_adapter_queryfree_chunk131072_seed3/last.ckpt}"

for ckpt in "$CKPT_SEED0" "$CKPT_SEED1" "$CKPT_SEED2" "$CKPT_SEED3"; do
  if [[ ! -f "$ckpt" ]]; then
    echo "Missing checkpoint: $ckpt" >&2
    exit 1
  fi
done

schedule_for_steps() {
  case "$1" in
    1)  echo '[100,0]' ;;
    2)  echo '[100,50,0]' ;;
    4)  echo '[100,75,50,25,0]' ;;
    7)  echo '[100,86,71,57,43,29,14,0]' ;;
    10) echo '[100,90,80,70,60,50,40,30,20,10,0]' ;;
    20) echo '[100,95,90,85,80,75,70,65,60,55,50,45,40,35,30,25,20,15,10,5,0]' ;;
    full) echo full ;;
    *)
      echo "Unsupported transition count: $1" >&2
      return 1
      ;;
  esac
}

run_case() {
  local gpu="$1"
  local ckpt="$2"
  local profile="$3"
  local steps="$4"
  shift 4

  local schedule
  schedule="$(schedule_for_steps "$steps")"
  echo "[test] gpu=${gpu} profile=${profile} steps=${steps}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export CONDA_ENV="${CONDA_ENV:-sparse_block}"
    local schedule_args=()
    if [[ "$schedule" == "full" ]]; then
      schedule_args=(general.test_sampling_full_steps=true general.sampling_time_schedule=null general.sampling_skip=null)
    else
      schedule_args=(general.test_sampling_full_steps=false "general.sampling_time_schedule=${schedule}")
    fi
    EXPERIMENT=pubmed_family_adapter_queryfree_pw20 bash scripts/run_pubmed_baseline_single.sh \
      general.gpus=1 \
      "general.test_only=$ckpt" \
      "general.name=test_${profile}" \
      general.run_test_after_train=false \
      general.enable_test_sampling=true \
      general.test_variance=1 \
      'general.test_sampling_seeds=[0]' \
      general.test_sampling_metrics_every=1 \
      general.verbose_sampling=true \
      general.log_sampling_triangle_trace=true \
      general.log_sampling_posterior_triangle_gap=true \
      general.edge_score_structure_diag=true \
      general.edge_score_structure_diag_max_negatives=100000 \
      "${schedule_args[@]}" \
      "$@"
  ) >"$LOG_DIR/${profile}.out" 2>&1
}

# SD-style: independent categorical/Bernoulli posterior sampling.
run_case 0 "$CKPT_SEED0" sim_seed0_bernoulli_s7 7 \
  model.sampling_edge_selection=bernoulli \
  model.sampling_use_reverse_posterior=true \
  model.sampling_reverse_posterior_mix_weights=null \
  model.sampling_reverse_posterior_mix_mode=full \
  model.sampling_reverse_posterior_mix_scale=1.0 &

# Bernoulli with a global density calibration, still stochastic like SD but less
# exposed to PubMed's extreme negative class ratio.
run_case 1 "$CKPT_SEED1" sim_seed1_bernoulli_density_s7 7 \
  model.sampling_edge_selection=bernoulli_expected_density \
  model.sampling_use_reverse_posterior=true \
  model.sampling_reverse_posterior_mix_weights=null \
  model.sampling_reverse_posterior_mix_mode=full \
  model.sampling_reverse_posterior_mix_scale=1.0 &

# Current exact-K, posterior off: old-log lesson that removing posterior inertia
# often recovers much more triangle structure.
run_case 2 "$CKPT_SEED2" sim_seed2_exactk_poff_s7 7 \
  model.sampling_edge_selection=gumbel_exact_k \
  model.sampling_use_reverse_posterior=false \
  model.sampling_queryfree_decode=true \
  model.sampling_queryfree_query_state=no_edge \
  model.sampling_edge_input_residual_scale=0.0 \
  model.sampling_ranking_intervention_diag=true \
  model.sampling_exact_k_connectivity_repair=true \
  model.sampling_exact_k_repair_max_swaps=0 &

# More denoising steps for the same old-log posterior-off recipe.
run_case 3 "$CKPT_SEED3" sim_seed3_exactk_poff_s10 10 \
  model.sampling_edge_selection=gumbel_exact_k \
  model.sampling_use_reverse_posterior=false \
  model.sampling_queryfree_decode=true \
  model.sampling_queryfree_query_state=no_edge \
  model.sampling_edge_input_residual_scale=0.0 \
  model.sampling_ranking_intervention_diag=true \
  model.sampling_exact_k_connectivity_repair=true \
  model.sampling_exact_k_repair_max_swaps=0 &

run_case 4 "$CKPT_SEED0" sim_seed0_exactk_poff_s20 20 \
  model.sampling_edge_selection=gumbel_exact_k \
  model.sampling_use_reverse_posterior=false \
  model.sampling_queryfree_decode=true \
  model.sampling_queryfree_query_state=no_edge \
  model.sampling_edge_input_residual_scale=0.0 \
  model.sampling_ranking_intervention_diag=true \
  model.sampling_exact_k_connectivity_repair=true \
  model.sampling_exact_k_repair_max_swaps=0 &

# Late posterior: use model ranking early, add only a final stabilizing posterior
# mix, matching the older successful ablation idea.
run_case 5 "$CKPT_SEED1" sim_seed1_exactk_latepost_s7 7 \
  model.sampling_edge_selection=gumbel_exact_k \
  model.sampling_use_reverse_posterior=true \
  'model.sampling_reverse_posterior_mix_weights=[0,0,0,0,0,0,0.5]' \
  model.sampling_queryfree_decode=true \
  model.sampling_queryfree_query_state=no_edge \
  model.sampling_edge_input_residual_scale=0.0 \
  model.sampling_ranking_intervention_diag=true \
  model.sampling_exact_k_connectivity_repair=true \
  model.sampling_exact_k_repair_max_swaps=0 &

# Full-step long runs. These are intentionally slow, but they answer whether
# a true SD-length chain helps or accumulates errors under the current sampler.
run_case 6 "$CKPT_SEED2" sim_seed2_bernoulli_density_full full \
  model.sampling_edge_selection=bernoulli_expected_density \
  model.sampling_use_reverse_posterior=true \
  model.sampling_reverse_posterior_mix_weights=null \
  model.sampling_reverse_posterior_mix_mode=full \
  model.sampling_reverse_posterior_mix_scale=1.0 &

run_case 7 "$CKPT_SEED3" sim_seed3_exactk_poff_full full \
  model.sampling_edge_selection=gumbel_exact_k \
  model.sampling_use_reverse_posterior=false \
  model.sampling_queryfree_decode=true \
  model.sampling_queryfree_query_state=no_edge \
  model.sampling_edge_input_residual_scale=0.0 \
  model.sampling_ranking_intervention_diag=true \
  model.sampling_exact_k_connectivity_repair=true \
  model.sampling_exact_k_repair_max_swaps=0 &

wait
echo "[done] simultaneous sampling sweep finished; logs: $LOG_DIR"

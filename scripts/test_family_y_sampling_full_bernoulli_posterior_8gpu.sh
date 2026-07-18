#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_family_y_sampling_full_bernoulli_posterior_8gpu}"
mkdir -p "$LOG_DIR"

CKPT_EDGEFILM="${CKPT_EDGEFILM:-output/2026-06-29/05-39-38-hetero_xey_family_y_edgefilm_seed0/output/sparse_diffusion/checkpoints/hetero_xey_family_y_edgefilm_seed0/last.ckpt}"
CKPT_BOTH="${CKPT_BOTH:-output/2026-06-29/05-39-38-hetero_xey_family_y_both_seed0/output/sparse_diffusion/checkpoints/hetero_xey_family_y_both_seed0/last.ckpt}"

for ckpt in "$CKPT_EDGEFILM" "$CKPT_BOTH"; do
  if [[ ! -f "$ckpt" ]]; then
    echo "Missing checkpoint: $ckpt" >&2
    exit 1
  fi
done

run_case() {
  local gpu="$1"
  local ckpt="$2"
  local name="$3"
  local mode="$4"
  shift 4

  local step_args=()
  case "$mode" in
    full)
      step_args=(
        general.test_sampling_full_steps=true
        general.sampling_time_schedule=null
        general.sampling_skip=null
      )
      ;;
    skip5)
      step_args=(
        general.test_sampling_full_steps=false
        general.sampling_skip=5
        general.sampling_time_schedule=null
      )
      ;;
    skip10)
      step_args=(
        general.test_sampling_full_steps=false
        general.sampling_skip=10
        general.sampling_time_schedule=null
      )
      ;;
    *)
      echo "Unsupported mode: $mode" >&2
      return 1
      ;;
  esac

  echo "[test] gpu=${gpu} name=${name} mode=${mode}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export CONDA_ENV="${CONDA_ENV:-sparse_block}"
    EXPERIMENT=pubmed_hetero_xey_family_pw20 bash scripts/run_pubmed_baseline_single.sh \
      general.gpus=1 \
      "general.test_only=${ckpt}" \
      "general.name=${name}" \
      general.wandb=disabled \
      general.run_test_after_train=false \
      general.enable_test_sampling=true \
      "general.test_variance=${TEST_VARIANCE:-3}" \
      "general.final_model_samples_to_generate=${SAMPLES_TO_GENERATE:-1}" \
      general.final_model_chains_to_save=0 \
      general.test_sampling_metrics_every=1 \
      general.verbose_sampling=true \
      general.log_sampling_triangle_trace=true \
      general.log_sampling_posterior_triangle_gap=true \
      model.use_sparse_hetero_y=true \
      model.use_sparse_family_y=true \
      model.use_family_y_film=true \
      model.sparse_family_y_degree_bins=5 \
      "${step_args[@]}" \
      "$@"
  ) >"$LOG_DIR/${name}.out" 2>&1 &
  echo "$!" >"$LOG_DIR/${name}.pid"
}

# 1-2. Full-step exact-K tests: does a full reverse chain accumulate more repair?
run_case 0 "$CKPT_EDGEFILM" family_y_edgefilm_full_exactk_alpha02 full \
  'general.test_sampling_seeds=[0,1,2]' \
  model.use_family_y_in_edge_film=true \
  model.use_family_y_in_attention=false \
  model.sampling_edge_selection=gumbel_exact_k \
  model.sampling_use_reverse_posterior=true \
  model.sampling_reverse_posterior_mix_weights=null \
  model.sampling_reverse_posterior_mix_mode=alpha_bar_s \
  model.sampling_reverse_posterior_mix_scale=0.2 \
  model.sampling_gumbel_temperature=0.01

run_case 1 "$CKPT_BOTH" family_y_both_full_exactk_alpha02 full \
  'general.test_sampling_seeds=[0,1,2]' \
  model.use_family_y_in_edge_film=true \
  model.use_family_y_in_attention=true \
  model.sampling_edge_selection=gumbel_exact_k \
  model.sampling_use_reverse_posterior=true \
  model.sampling_reverse_posterior_mix_weights=null \
  model.sampling_reverse_posterior_mix_mode=alpha_bar_s \
  model.sampling_reverse_posterior_mix_scale=0.2 \
  model.sampling_gumbel_temperature=0.01

# 3-4. Bernoulli-style sampling: pure Bernoulli and density-calibrated Bernoulli.
run_case 2 "$CKPT_EDGEFILM" family_y_edgefilm_bernoulli_skip10_fullpost skip10 \
  'general.test_sampling_seeds=[0,1,2]' \
  model.use_family_y_in_edge_film=true \
  model.use_family_y_in_attention=false \
  model.sampling_edge_selection=bernoulli \
  model.sampling_use_reverse_posterior=true \
  model.sampling_reverse_posterior_mix_weights=null \
  model.sampling_reverse_posterior_mix_mode=full \
  model.sampling_reverse_posterior_mix_scale=1.0

run_case 3 "$CKPT_EDGEFILM" family_y_edgefilm_bernoulli_density_skip10_fullpost skip10 \
  'general.test_sampling_seeds=[1,2,3]' \
  model.use_family_y_in_edge_film=true \
  model.use_family_y_in_attention=false \
  model.sampling_edge_selection=bernoulli_expected_density \
  model.sampling_use_reverse_posterior=true \
  model.sampling_reverse_posterior_mix_weights=null \
  model.sampling_reverse_posterior_mix_mode=full \
  model.sampling_reverse_posterior_mix_scale=1.0

# 5. More intermediate steps without full cost.
run_case 4 "$CKPT_EDGEFILM" family_y_edgefilm_skip5_exactk_alpha02 skip5 \
  'general.test_sampling_seeds=[0,1,2]' \
  model.use_family_y_in_edge_film=true \
  model.use_family_y_in_attention=false \
  model.sampling_edge_selection=gumbel_exact_k \
  model.sampling_use_reverse_posterior=true \
  model.sampling_reverse_posterior_mix_weights=null \
  model.sampling_reverse_posterior_mix_mode=alpha_bar_s \
  model.sampling_reverse_posterior_mix_scale=0.2 \
  model.sampling_gumbel_temperature=0.01

# 6. No posterior: isolates pure model accumulation.
run_case 5 "$CKPT_EDGEFILM" family_y_edgefilm_skip10_exactk_posterior_off skip10 \
  'general.test_sampling_seeds=[1,2,3]' \
  model.use_family_y_in_edge_film=true \
  model.use_family_y_in_attention=false \
  model.sampling_edge_selection=gumbel_exact_k \
  model.sampling_use_reverse_posterior=false \
  model.sampling_gumbel_temperature=0.01

# 7. Late posterior only: model first, posterior stabilizes at the end.
run_case 6 "$CKPT_EDGEFILM" family_y_edgefilm_skip10_exactk_latepost skip10 \
  'general.test_sampling_seeds=[2,3,4]' \
  model.use_family_y_in_edge_film=true \
  model.use_family_y_in_attention=false \
  model.sampling_edge_selection=gumbel_exact_k \
  model.sampling_use_reverse_posterior=true \
  'model.sampling_reverse_posterior_mix_weights=[0,0,0,0,0,0,0,0,0,0.5]' \
  model.sampling_gumbel_temperature=0.01

# 8. Weaker alpha-bar posterior. Together with alpha=0.2, posterior-off, and
# late-posterior, this checks whether posterior inertia is suppressing repair.
run_case 7 "$CKPT_EDGEFILM" family_y_edgefilm_skip10_exactk_alpha01 skip10 \
  'general.test_sampling_seeds=[3,4,5]' \
  model.use_family_y_in_edge_film=true \
  model.use_family_y_in_attention=false \
  model.sampling_edge_selection=gumbel_exact_k \
  model.sampling_use_reverse_posterior=true \
  model.sampling_reverse_posterior_mix_weights=null \
  model.sampling_reverse_posterior_mix_mode=alpha_bar_s \
  model.sampling_reverse_posterior_mix_scale=0.1 \
  model.sampling_gumbel_temperature=0.01

wait
echo "[done] family-y sampling sweep finished; logs: $LOG_DIR"

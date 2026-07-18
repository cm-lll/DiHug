#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_pubmed_structure_guided_sampling_8gpu}"
mkdir -p "$LOG_DIR"

CKPT_DEFAULT="output/2026-06-29/18-12-41-pubmed_edge_struct_famy_edgefilm_scale2_seed0/output/sparse_diffusion/checkpoints/pubmed_edge_struct_famy_edgefilm_scale2_seed0/last.ckpt"
CKPT="${CKPT:-$CKPT_DEFAULT}"
SAMPLING_SKIP="${SAMPLING_SKIP:-10}"
TEST_VARIANCE="${TEST_VARIANCE:-3}"
SAMPLES_TO_GENERATE="${SAMPLES_TO_GENERATE:-3}"

COMMON_OVERRIDES=(
  general.wandb=disabled
  general.gpus=1
  general.test_only="${CKPT}"
  general.enable_test_sampling=true
  general.test_variance="${TEST_VARIANCE}"
  general.final_model_samples_to_generate="${SAMPLES_TO_GENERATE}"
  general.final_model_chains_to_save=0
  general.test_sampling_full_steps=false
  general.sampling_skip="${SAMPLING_SKIP}"
  general.test_sampling_metrics_every=1
  general.verbose_sampling=true
  general.log_sampling_triangle_trace=true
  model.denoiser=graph_transformer
  model.use_sparse_hetero_y=true
  model.use_edge_struct_features=true
  model.edge_struct_feature_dim=8
  model.edge_struct_hidden_dim=64
  model.edge_struct_residual_scale=2.0
  model.use_sparse_family_y=true
  model.use_family_y_film=true
  model.use_family_y_in_edge_film=true
  model.edge_struct_use_family_y=true
  model.sampling_edge_selection=gumbel_exact_k
  model.sampling_use_reverse_posterior=true
  model.sampling_reverse_posterior_mix_mode=alpha_bar_s
  model.sampling_reverse_posterior_mix_scale=0.2
  model.sampling_gumbel_temperature=0.01
  model.sampling_structure_guidance_power=1.0
  model.sampling_structure_guidance_min_step_frac=0.0
  model.sampling_structure_guidance_max_step_frac=1.0
  model.sampling_structure_guidance_closure_weight=0.0
  model.sampling_structure_guidance_connect_weight=0.0
  model.sampling_structure_guidance_degree_pair_weight=0.0
  model.sampling_structure_guidance_assort_weight=0.0
)

launch() {
  local gpu="$1"
  local name="$2"
  shift 2
  local log_file="$LOG_DIR/${name}.out"
  echo "[launch] gpu=${gpu} name=${name}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export CONDA_ENV="${CONDA_ENV:-sparse_block}"
    EXPERIMENT=pubmed_hetero_xey_family_pw20 bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON_OVERRIDES[@]}" \
      "general.name=${name}" \
      "$@"
  ) >"$log_file" 2>&1 &
  echo "$!" >"$LOG_DIR/${name}.pid"
}

# Baseline retest for the same checkpoint and sampling code.
launch 0 structguide_base_skip${SAMPLING_SKIP}

# Closure-only: high-noise common-neighbor preference, decays toward low noise.
launch 1 structguide_closure05_skip${SAMPLING_SKIP} \
  model.sampling_structure_guidance_closure_weight=0.5

launch 2 structguide_closure10_skip${SAMPLING_SKIP} \
  model.sampling_structure_guidance_closure_weight=1.0

# Closure + weak disassortative endpoint correction.
launch 3 structguide_closure05_assort02_skip${SAMPLING_SKIP} \
  model.sampling_structure_guidance_closure_weight=0.5 \
  model.sampling_structure_guidance_assort_weight=0.2 \
  model.sampling_structure_guidance_assort_target=-0.18

# Degree-pair distribution as a soft ranking prior, not hard quota sampling.
launch 4 structguide_closure05_dp02_skip${SAMPLING_SKIP} \
  model.sampling_structure_guidance_closure_weight=0.5 \
  model.sampling_structure_guidance_degree_pair_weight=0.2

launch 5 structguide_closure05_dp02_assort02_skip${SAMPLING_SKIP} \
  model.sampling_structure_guidance_closure_weight=0.5 \
  model.sampling_structure_guidance_degree_pair_weight=0.2 \
  model.sampling_structure_guidance_assort_weight=0.2 \
  model.sampling_structure_guidance_assort_target=-0.18

# Connectivity bridge preference is kept weak; LCC is already mostly solved.
launch 6 structguide_closure05_connect02_skip${SAMPLING_SKIP} \
  model.sampling_structure_guidance_closure_weight=0.5 \
  model.sampling_structure_guidance_connect_weight=0.2

# Strong early-only guidance: use it mainly for high-noise anchor selection.
launch 7 structguide_early_closure10_assort03_skip${SAMPLING_SKIP} \
  model.sampling_structure_guidance_closure_weight=1.0 \
  model.sampling_structure_guidance_assort_weight=0.3 \
  model.sampling_structure_guidance_assort_target=-0.18 \
  model.sampling_structure_guidance_min_step_frac=0.4 \
  model.sampling_structure_guidance_power=1.0

wait
echo "[done] logs: ${LOG_DIR}"

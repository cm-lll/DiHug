#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_pubmed_structure_guided_edgefilm_pair_8gpu}"
mkdir -p "$LOG_DIR"

CKPT_EDGEFILM_DEFAULT="output/2026-06-29/18-12-41-pubmed_edge_struct_famy_edgefilm_seed0/output/sparse_diffusion/checkpoints/pubmed_edge_struct_famy_edgefilm_seed0/last.ckpt"
CKPT_EDGEFILM_SCALE2_DEFAULT="output/2026-06-29/18-12-41-pubmed_edge_struct_famy_edgefilm_scale2_seed0/output/sparse_diffusion/checkpoints/pubmed_edge_struct_famy_edgefilm_scale2_seed0/last.ckpt"
CKPT_EDGEFILM="${CKPT_EDGEFILM:-$CKPT_EDGEFILM_DEFAULT}"
CKPT_EDGEFILM_SCALE2="${CKPT_EDGEFILM_SCALE2:-$CKPT_EDGEFILM_SCALE2_DEFAULT}"

SAMPLING_SKIP="${SAMPLING_SKIP:-10}"
TEST_VARIANCE="${TEST_VARIANCE:-3}"
SAMPLES_TO_GENERATE="${SAMPLES_TO_GENERATE:-3}"

BASE_OVERRIDES=(
  general.wandb=disabled
  general.gpus=1
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
  model.use_sparse_family_y=true
  model.use_family_y_film=true
  model.use_family_y_in_attention=false
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
  local ckpt="$2"
  local scale="$3"
  local name="$4"
  shift 4
  local log_file="$LOG_DIR/${name}.out"
  echo "[launch] gpu=${gpu} name=${name} ckpt=${ckpt} scale=${scale}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export CONDA_ENV="${CONDA_ENV:-sparse_block}"
    EXPERIMENT=pubmed_hetero_xey_family_pw20 bash scripts/run_pubmed_baseline_single.sh \
      "${BASE_OVERRIDES[@]}" \
      "general.name=${name}" \
      "general.test_only=${ckpt}" \
      "model.edge_struct_residual_scale=${scale}" \
      "$@"
  ) >"$log_file" 2>&1 &
  echo "$!" >"$LOG_DIR/${name}.pid"
}

# Same four sampling settings for both backend-fusion checkpoints:
# base, closure, closure+assort, closure+degree-pair+assort.
launch 0 "$CKPT_EDGEFILM" 1.0 edgefilm_base_skip${SAMPLING_SKIP}
launch 1 "$CKPT_EDGEFILM" 1.0 edgefilm_closure05_skip${SAMPLING_SKIP} \
  model.sampling_structure_guidance_closure_weight=0.5
launch 2 "$CKPT_EDGEFILM" 1.0 edgefilm_closure05_assort02_skip${SAMPLING_SKIP} \
  model.sampling_structure_guidance_closure_weight=0.5 \
  model.sampling_structure_guidance_assort_weight=0.2 \
  model.sampling_structure_guidance_assort_target=-0.18
launch 3 "$CKPT_EDGEFILM" 1.0 edgefilm_closure05_dp02_assort02_skip${SAMPLING_SKIP} \
  model.sampling_structure_guidance_closure_weight=0.5 \
  model.sampling_structure_guidance_degree_pair_weight=0.2 \
  model.sampling_structure_guidance_assort_weight=0.2 \
  model.sampling_structure_guidance_assort_target=-0.18

launch 4 "$CKPT_EDGEFILM_SCALE2" 2.0 edgefilm_scale2_base_skip${SAMPLING_SKIP}
launch 5 "$CKPT_EDGEFILM_SCALE2" 2.0 edgefilm_scale2_closure05_skip${SAMPLING_SKIP} \
  model.sampling_structure_guidance_closure_weight=0.5
launch 6 "$CKPT_EDGEFILM_SCALE2" 2.0 edgefilm_scale2_closure05_assort02_skip${SAMPLING_SKIP} \
  model.sampling_structure_guidance_closure_weight=0.5 \
  model.sampling_structure_guidance_assort_weight=0.2 \
  model.sampling_structure_guidance_assort_target=-0.18
launch 7 "$CKPT_EDGEFILM_SCALE2" 2.0 edgefilm_scale2_closure05_dp02_assort02_skip${SAMPLING_SKIP} \
  model.sampling_structure_guidance_closure_weight=0.5 \
  model.sampling_structure_guidance_degree_pair_weight=0.2 \
  model.sampling_structure_guidance_assort_weight=0.2 \
  model.sampling_structure_guidance_assort_target=-0.18

wait
echo "[done] logs: ${LOG_DIR}"

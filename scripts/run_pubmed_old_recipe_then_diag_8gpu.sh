#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data2/lyh/miniconda3/envs/sparse_block/bin/python}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})

PHASE1_LOG_DIR="${PHASE1_LOG_DIR:-logs/logs_pubmed_edgefilm_old_recipe_isolation_8gpu}"
PHASE2_LOG_DIR="${PHASE2_LOG_DIR:-logs/logs_pubmed_edge_score_diag_compare_8gpu}"
RUN_PHASE1="${RUN_PHASE1:-1}"
RUN_PHASE2="${RUN_PHASE2:-1}"
RUN_PHASE2_ON_PHASE1_FAILURE="${RUN_PHASE2_ON_PHASE1_FAILURE:-0}"

TEST_VARIANCE="${TEST_VARIANCE:-3}"
SAMPLES_TO_GENERATE="${SAMPLES_TO_GENERATE:-3}"
TEST_METRICS_EVERY="${TEST_METRICS_EVERY:-1}"

CKPT_EDGEFILM_DEFAULT="output/2026-06-29/18-12-41-pubmed_edge_struct_famy_edgefilm_seed0/output/sparse_diffusion/checkpoints/pubmed_edge_struct_famy_edgefilm_seed0/last.ckpt"
CKPT_EDGEFILM_SCALE2_DEFAULT="output/2026-06-29/18-12-41-pubmed_edge_struct_famy_edgefilm_scale2_seed0/output/sparse_diffusion/checkpoints/pubmed_edge_struct_famy_edgefilm_scale2_seed0/last.ckpt"
CKPT_EDGEFILM="${CKPT_EDGEFILM:-$CKPT_EDGEFILM_DEFAULT}"
CKPT_EDGEFILM_SCALE2="${CKPT_EDGEFILM_SCALE2:-$CKPT_EDGEFILM_SCALE2_DEFAULT}"

mkdir -p "$PHASE1_LOG_DIR" "$PHASE2_LOG_DIR"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PYTHON_BIN")")/lib:${LD_LIBRARY_PATH:-}"

require_file() {
  local label="$1"
  local path="$2"
  [[ -f "$path" ]] || {
    echo "[error] missing $label: $path" >&2
    exit 2
  }
}

require_file "edgefilm checkpoint" "$CKPT_EDGEFILM"
require_file "edgefilm scale2 checkpoint" "$CKPT_EDGEFILM_SCALE2"
require_file "diagnostic script" "$ROOT/scripts/test_pubmed_edge_score_diag_compare_8gpu.sh"

BASE_OVERRIDES=(
  general.wandb=disabled
  general.gpus=1
  general.enable_test_sampling=true
  general.test_variance="${TEST_VARIANCE}"
  general.final_model_samples_to_generate="${SAMPLES_TO_GENERATE}"
  general.final_model_chains_to_save=0
  general.test_sampling_full_steps=false
  general.test_sampling_metrics_every="${TEST_METRICS_EVERY}"
  general.verbose_sampling=true
  general.log_sampling_triangle_trace=true
  model.denoiser=graph_transformer
  model.edge_fraction=0.1
  model.block_query=true
  model.block_partition_mode=hetero_metis
  model.block_query_full_block=true
  model.block_query_inter_fill=true
  model.block_query_include_uniform=false
  model.query_include_all_positive_edges=false
  model.sampling_block_mode=type_template
  model.sampling_block_template_init=false
  model.sampling_block_marginal_init=true
  model.sampling_block_family_budget_projection=false
  model.sampling_calibrate_exist_pos_weight=true
  model.sampling_use_reverse_posterior=true
  model.sampling_reverse_posterior_mix_weights=null
  model.sampling_reverse_posterior_mix_mode=alpha_bar_s
  model.sampling_gumbel_temperature=0.01
  model.use_sparse_hetero_y=true
  model.use_edge_struct_features=true
  model.edge_struct_feature_dim=8
  model.edge_struct_hidden_dim=64
  model.use_sparse_family_y=true
  model.use_family_y_film=true
  model.use_family_y_in_attention=false
  model.use_family_y_in_edge_film=true
  model.edge_struct_use_family_y=true
  model.sampling_structure_guidance_closure_weight=0.0
  model.sampling_structure_guidance_connect_weight=0.0
  model.sampling_structure_guidance_degree_pair_weight=0.0
  model.sampling_structure_guidance_assort_weight=0.0
)

declare -a PHASE1_PIDS=()
declare -a PHASE1_NAMES=()

launch_phase1() {
  local gpu="$1"
  local ckpt="$2"
  local scale="$3"
  local name="$4"
  shift 4
  local log_file="$PHASE1_LOG_DIR/${name}.out"
  echo "[phase1-launch] $(date '+%F %T') gpu=${gpu} name=${name}" | tee -a "$PHASE1_LOG_DIR/launcher.log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export CONDA_ENV="${CONDA_ENV:-sparse_block}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
    EXPERIMENT=pubmed_hetero_xey_family_pw20 bash scripts/run_pubmed_baseline_single.sh \
      "${BASE_OVERRIDES[@]}" \
      "general.name=${name}" \
      "general.test_only=${ckpt}" \
      "model.edge_struct_residual_scale=${scale}" \
      "$@"
  ) >"$log_file" 2>&1 &
  local pid="$!"
  echo "$pid" >"$PHASE1_LOG_DIR/${name}.pid"
  PHASE1_PIDS+=("$pid")
  PHASE1_NAMES+=("$name")
}

wait_phase1() {
  local failed=0
  local idx pid name
  for idx in "${!PHASE1_PIDS[@]}"; do
    pid="${PHASE1_PIDS[$idx]}"
    name="${PHASE1_NAMES[$idx]}"
    if wait "$pid"; then
      echo "[phase1-ok] $(date '+%F %T') name=${name}" | tee -a "$PHASE1_LOG_DIR/launcher.log"
      touch "$PHASE1_LOG_DIR/${name}.done"
    else
      echo "[phase1-fail] $(date '+%F %T') name=${name}" | tee -a "$PHASE1_LOG_DIR/launcher.log"
      touch "$PHASE1_LOG_DIR/${name}.failed"
      failed=1
    fi
  done
  return "$failed"
}

run_phase1() {
  # Two current family-y/edge-structure checkpoints, each under four old
  # successful sampling recipes:
  # 1) exact-K + connectivity repair, old posterior strength 0.5, 7-step schedule.
  # 2) exact-K + connectivity repair, posterior strength 0.75, 7-step schedule.
  # 3) connectivity_topk, pair_topm=64, old skip-25 recipe.
  # 4) connectivity_topk, pair_topm=64, with posterior strength 0.5.
  launch_phase1 "${GPUS[0]}" "$CKPT_EDGEFILM" 1.0 edgefilm_old_exactk_repair_a05_steps7 \
    'general.sampling_time_schedule=[100,86,71,57,43,29,14,0]' \
    general.sampling_skip=10 \
    model.sampling_edge_selection=gumbel_exact_k \
    model.sampling_reverse_posterior_mix_scale=0.5 \
    model.sampling_exact_k_connectivity_repair=true \
    model.sampling_exact_k_repair_max_swaps=0

  launch_phase1 "${GPUS[1]}" "$CKPT_EDGEFILM" 1.0 edgefilm_old_exactk_repair_a075_steps7 \
    'general.sampling_time_schedule=[100,86,71,57,43,29,14,0]' \
    general.sampling_skip=10 \
    model.sampling_edge_selection=gumbel_exact_k \
    model.sampling_reverse_posterior_mix_scale=0.75 \
    model.sampling_exact_k_connectivity_repair=true \
    model.sampling_exact_k_repair_max_swaps=0

  launch_phase1 "${GPUS[2]}" "$CKPT_EDGEFILM" 1.0 edgefilm_old_connect_topk_m64_skip25 \
    general.sampling_time_schedule=null \
    general.sampling_skip=25 \
    model.sampling_edge_selection=connectivity_topk \
    model.sampling_connectivity_pair_topm=64 \
    model.sampling_reverse_posterior_mix_scale=0.2 \
    model.sampling_exact_k_connectivity_repair=false

  launch_phase1 "${GPUS[3]}" "$CKPT_EDGEFILM" 1.0 edgefilm_old_connect_topk_m64_a05_skip25 \
    general.sampling_time_schedule=null \
    general.sampling_skip=25 \
    model.sampling_edge_selection=connectivity_topk \
    model.sampling_connectivity_pair_topm=64 \
    model.sampling_reverse_posterior_mix_scale=0.5 \
    model.sampling_exact_k_connectivity_repair=false

  launch_phase1 "${GPUS[4]}" "$CKPT_EDGEFILM_SCALE2" 2.0 edgefilm_scale2_old_exactk_repair_a05_steps7 \
    'general.sampling_time_schedule=[100,86,71,57,43,29,14,0]' \
    general.sampling_skip=10 \
    model.sampling_edge_selection=gumbel_exact_k \
    model.sampling_reverse_posterior_mix_scale=0.5 \
    model.sampling_exact_k_connectivity_repair=true \
    model.sampling_exact_k_repair_max_swaps=0

  launch_phase1 "${GPUS[5]}" "$CKPT_EDGEFILM_SCALE2" 2.0 edgefilm_scale2_old_exactk_repair_a075_steps7 \
    'general.sampling_time_schedule=[100,86,71,57,43,29,14,0]' \
    general.sampling_skip=10 \
    model.sampling_edge_selection=gumbel_exact_k \
    model.sampling_reverse_posterior_mix_scale=0.75 \
    model.sampling_exact_k_connectivity_repair=true \
    model.sampling_exact_k_repair_max_swaps=0

  launch_phase1 "${GPUS[6]}" "$CKPT_EDGEFILM_SCALE2" 2.0 edgefilm_scale2_old_connect_topk_m64_skip25 \
    general.sampling_time_schedule=null \
    general.sampling_skip=25 \
    model.sampling_edge_selection=connectivity_topk \
    model.sampling_connectivity_pair_topm=64 \
    model.sampling_reverse_posterior_mix_scale=0.2 \
    model.sampling_exact_k_connectivity_repair=false

  launch_phase1 "${GPUS[7]}" "$CKPT_EDGEFILM_SCALE2" 2.0 edgefilm_scale2_old_connect_topk_m64_a05_skip25 \
    general.sampling_time_schedule=null \
    general.sampling_skip=25 \
    model.sampling_edge_selection=connectivity_topk \
    model.sampling_connectivity_pair_topm=64 \
    model.sampling_reverse_posterior_mix_scale=0.5 \
    model.sampling_exact_k_connectivity_repair=false

  wait_phase1
}

run_phase2() {
  echo "[phase2-launch] $(date '+%F %T') script=test_pubmed_edge_score_diag_compare_8gpu.sh" | tee -a "$PHASE2_LOG_DIR/launcher_chain.log"
  LOG_DIR="$PHASE2_LOG_DIR" \
  TEST_VARIANCE="${PHASE2_TEST_VARIANCE:-1}" \
  SAMPLES_TO_GENERATE="${PHASE2_SAMPLES_TO_GENERATE:-1}" \
  bash scripts/test_pubmed_edge_score_diag_compare_8gpu.sh
  echo "[phase2-done] $(date '+%F %T') logs=${PHASE2_LOG_DIR}" | tee -a "$PHASE2_LOG_DIR/launcher_chain.log"
}

phase1_rc=0
if [[ "$RUN_PHASE1" == "1" ]]; then
  run_phase1 || phase1_rc=$?
  echo "[phase1-done] $(date '+%F %T') rc=${phase1_rc} logs=${PHASE1_LOG_DIR}" | tee -a "$PHASE1_LOG_DIR/launcher.log"
else
  echo "[phase1-skip] RUN_PHASE1=${RUN_PHASE1}"
fi

if [[ "$RUN_PHASE2" == "1" ]]; then
  if [[ "$phase1_rc" -eq 0 || "$RUN_PHASE2_ON_PHASE1_FAILURE" == "1" ]]; then
    run_phase2
  else
    echo "[phase2-skip] phase1 failed; set RUN_PHASE2_ON_PHASE1_FAILURE=1 to continue" >&2
  fi
fi

exit "$phase1_rc"

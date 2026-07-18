#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_pubmed_edge_score_diag_compare_8gpu}"
mkdir -p "$LOG_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data2/lyh/miniconda3/envs/sparse_block/bin/python}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
SAMPLING_SKIP="${SAMPLING_SKIP:-10}"
TEST_SEEDS="${TEST_SEEDS:-[0]}"
TEST_VARIANCE="${TEST_VARIANCE:-1}"
SAMPLES_TO_GENERATE="${SAMPLES_TO_GENERATE:-1}"

export LD_LIBRARY_PATH="$(dirname "$(dirname "$PYTHON_BIN")")/lib:${LD_LIBRARY_PATH:-}"

latest_ckpt_for_name() {
  local run_name="$1"
  find "$ROOT/output" -path "*/checkpoints/${run_name}/last.ckpt" -type f \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

OLD_SEQBASE_CKPT="${OLD_SEQBASE_CKPT:-$ROOT/output/2026-06-28/14-49-55-hetero_xey_sequential_base_seed0_ep100_skip10/output/sparse_diffusion/checkpoints/hetero_xey_sequential_base_seed0_ep100_skip10/last.ckpt}"
SEQ_EDGE_STRUCT_CKPT="$(latest_ckpt_for_name pubmed_seq_edge_struct_seed0)"
SEQ_EDGE_STRUCT_SCALE2_CKPT="$(latest_ckpt_for_name pubmed_seq_edge_struct_scale2_seed0)"
FAMY_EDGEFILM_SCALE2_CKPT="$(latest_ckpt_for_name pubmed_edge_struct_famy_edgefilm_scale2_seed0)"
FAMY_DPJS005_CKPT="$(latest_ckpt_for_name pubmed_edge_struct_famy_dpjs005_seed0)"

require_ckpt() {
  local label="$1"
  local path="$2"
  [[ -n "$path" && -f "$path" ]] || {
    echo "[error] missing checkpoint for $label: $path" >&2
    exit 2
  }
}

require_ckpt old_seqbase "$OLD_SEQBASE_CKPT"
require_ckpt seq_edge_struct "$SEQ_EDGE_STRUCT_CKPT"
require_ckpt seq_edge_struct_scale2 "$SEQ_EDGE_STRUCT_SCALE2_CKPT"
require_ckpt famy_edgefilm_scale2 "$FAMY_EDGEFILM_SCALE2_CKPT"
require_ckpt famy_dpjs005 "$FAMY_DPJS005_CKPT"

COMMON=(
  +experiment=pubmed_hetero_xey_family_pw20
  general.wandb=disabled
  general.gpus=1
  general.run_test_after_train=false
  general.enable_test_sampling=true
  general.enable_val_sampling=false
  general.enable_val_pred_metrics=false
  general.test_variance="${TEST_VARIANCE}"
  general.final_model_samples_to_generate="${SAMPLES_TO_GENERATE}"
  general.final_model_chains_to_save=0
  general.verbose_sampling=true
  general.log_sampling_triangle_trace=true
  general.edge_score_structure_diag=true
  general.edge_score_structure_diag_max_negatives=100000
  general.test_sampling_full_steps=false
  general.test_sampling_metrics_every=1
  general.test_sampling_seeds="${TEST_SEEDS}"
  general.sampling_skip="${SAMPLING_SKIP}"
  model.edge_fraction=0.1
  model.block_query=true
  model.block_partition_mode=hetero_metis
  model.block_query_full_block=true
  model.block_query_inter_fill=true
  model.block_query_include_uniform=false
  model.query_include_all_positive_edges=false
  model.sampling_block_mode=type_template
  model.sampling_block_template_init=false
  model.sampling_block_marginal_init=false
  model.train_all_blocks_per_noise=false
  model.sampling_calibrate_exist_pos_weight=true
  model.sampling_use_reverse_posterior=true
  model.sampling_reverse_posterior_mix_weights=null
  model.sampling_reverse_posterior_mix_mode=alpha_bar_s
  model.sampling_reverse_posterior_mix_scale=0.2
  model.sampling_gumbel_temperature=0.01
  model.sampling_degree_pair_bins=5
  model.sampling_degree_pair_bias_clip=4.0
)

ARCH_OLD=(
  model.use_sparse_hetero_y=false
  model.use_sparse_family_y=false
  model.use_edge_struct_features=false
)

ARCH_EDGE_STRUCT=(
  model.use_sparse_hetero_y=false
  model.use_sparse_family_y=false
  model.use_family_y_film=false
  model.use_family_y_in_attention=false
  model.use_family_y_in_edge_film=false
  model.use_edge_struct_features=true
  model.edge_struct_feature_dim=8
  model.edge_struct_hidden_dim=64
  model.edge_struct_residual_scale=1.0
  model.edge_struct_use_family_y=false
)

ARCH_EDGE_STRUCT_SCALE2=(
  "${ARCH_EDGE_STRUCT[@]}"
  model.edge_struct_residual_scale=2.0
)

ARCH_FAMY_EDGEFILM_SCALE2=(
  model.use_sparse_hetero_y=true
  model.use_sparse_family_y=true
  model.use_family_y_film=true
  model.use_family_y_in_attention=false
  model.use_family_y_in_edge_film=true
  model.use_edge_struct_features=true
  model.edge_struct_feature_dim=8
  model.edge_struct_hidden_dim=64
  model.edge_struct_residual_scale=2.0
  model.edge_struct_use_family_y=true
)

launch() {
  local gpu="$1"
  local name="$2"
  local ckpt="$3"
  local arch_array_name="$4"
  shift 4
  local -n arch_ref="$arch_array_name"
  local log_file="$LOG_DIR/${name}.out"
  echo "[launch] gpu=${gpu} name=${name}" | tee -a "$LOG_DIR/launcher.log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
    "$PYTHON_BIN" -m dihug.main \
      "${COMMON[@]}" \
      "${arch_ref[@]}" \
      general.test_only="$ckpt" \
      general.name="$name" \
      "$@"
  ) >"$log_file" 2>&1 &
  echo "$!" >"$LOG_DIR/${name}.pid"
}

launch "${GPUS[0]}" diag_old_seqbase_degpair_s015 "$OLD_SEQBASE_CKPT" ARCH_OLD \
  model.sampling_edge_selection=gumbel_exact_k_degree_pair \
  model.sampling_degree_pair_strength=0.15 \
  model.sampling_exact_k_connectivity_repair=false

launch "${GPUS[1]}" diag_old_seqbase_quota_s100 "$OLD_SEQBASE_CKPT" ARCH_OLD \
  model.sampling_edge_selection=gumbel_exact_k_degree_pair_quota \
  model.sampling_degree_pair_strength=1.00 \
  model.sampling_exact_k_connectivity_repair=false

launch "${GPUS[2]}" diag_seq_edge_struct_exactk "$SEQ_EDGE_STRUCT_CKPT" ARCH_EDGE_STRUCT \
  model.sampling_edge_selection=gumbel_exact_k \
  model.sampling_exact_k_connectivity_repair=false

launch "${GPUS[3]}" diag_seq_edge_struct_quota_s100 "$SEQ_EDGE_STRUCT_CKPT" ARCH_EDGE_STRUCT \
  model.sampling_edge_selection=gumbel_exact_k_degree_pair_quota \
  model.sampling_degree_pair_strength=1.00 \
  model.sampling_exact_k_connectivity_repair=false

launch "${GPUS[4]}" diag_seq_edge_struct_scale2_exactk "$SEQ_EDGE_STRUCT_SCALE2_CKPT" ARCH_EDGE_STRUCT_SCALE2 \
  model.sampling_edge_selection=gumbel_exact_k \
  model.sampling_exact_k_connectivity_repair=false

launch "${GPUS[5]}" diag_seq_edge_struct_scale2_quota_s100 "$SEQ_EDGE_STRUCT_SCALE2_CKPT" ARCH_EDGE_STRUCT_SCALE2 \
  model.sampling_edge_selection=gumbel_exact_k_degree_pair_quota \
  model.sampling_degree_pair_strength=1.00 \
  model.sampling_exact_k_connectivity_repair=false

launch "${GPUS[6]}" diag_famy_edgefilm_scale2_exactk "$FAMY_EDGEFILM_SCALE2_CKPT" ARCH_FAMY_EDGEFILM_SCALE2 \
  model.sampling_edge_selection=gumbel_exact_k \
  model.sampling_exact_k_connectivity_repair=false

launch "${GPUS[7]}" diag_famy_edgefilm_scale2_quota_s100 "$FAMY_EDGEFILM_SCALE2_CKPT" ARCH_FAMY_EDGEFILM_SCALE2 \
  model.sampling_edge_selection=gumbel_exact_k_degree_pair_quota \
  model.sampling_degree_pair_strength=1.00 \
  model.sampling_exact_k_connectivity_repair=false

wait
echo "[done] logs: ${LOG_DIR}" | tee -a "$LOG_DIR/launcher.log"

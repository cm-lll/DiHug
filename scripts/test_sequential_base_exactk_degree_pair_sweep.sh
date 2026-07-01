#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_sequential_base_exactk_degree_pair_sweep}"
CKPT="${CKPT:-$ROOT/output/2026-06-28/14-49-55-hetero_xey_sequential_base_seed0_ep100_skip10/output/sparse_diffusion/checkpoints/hetero_xey_sequential_base_seed0_ep100_skip10/last.ckpt}"
PYTHON_BIN="${PYTHON_BIN:-/data2/lyh/miniconda3/envs/sparse_block/bin/python}"
SAMPLE_SEEDS="${SAMPLE_SEEDS:-[0]}"
SKIP="${SKIP:-10}"
METRICS_EVERY="${METRICS_EVERY:-1}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
mkdir -p "$LOG_DIR"

if [[ ! -f "$CKPT" ]]; then
  echo "[error] checkpoint not found: $CKPT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[error] python not executable: $PYTHON_BIN" >&2
  exit 2
fi

export LD_LIBRARY_PATH="$(dirname "$(dirname "$PYTHON_BIN")")/lib:${LD_LIBRARY_PATH:-}"

COMMON_OVERRIDES=(
  general.wandb=disabled
  general.gpus=1
  "general.test_only=${CKPT}"
  general.run_test_after_train=false
  general.enable_test_sampling=true
  general.enable_val_sampling=false
  general.enable_val_pred_metrics=false
  general.test_variance=1
  general.verbose_sampling=true
  general.log_sampling_triangle_trace=true
  general.test_sampling_full_steps=false
  general.test_sampling_metrics_every="${METRICS_EVERY}"
  "general.test_sampling_seeds=${SAMPLE_SEEDS}"
  general.sampling_skip="${SKIP}"
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
  model.use_sparse_hetero_y=false
  model.use_family_y_film=false
  model.use_family_edge_update=false
  model.sampling_calibrate_exist_pos_weight=true
  model.sampling_use_reverse_posterior=true
  model.sampling_reverse_posterior_mix_weights=null
  model.sampling_reverse_posterior_mix_mode=alpha_bar_s
  model.sampling_reverse_posterior_mix_scale=0.2
  model.sampling_gumbel_temperature=0.01
  model.sampling_degree_pair_bins=5
  model.sampling_degree_pair_bias_clip=4.0
)

launch() {
  local idx="$1"
  local tag="$2"
  shift 2
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local name="seqbase_${tag}_skip${SKIP}_seed0"
  local log_file="$LOG_DIR/${name}.out"
  echo "[launch] gpu=${gpu} name=${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}" \
    nohup "$PYTHON_BIN" -m dihug.main \
      +experiment=pubmed_hetero_xey_family_pw20 \
      "${COMMON_OVERRIDES[@]}" \
      "$@" \
      "general.name=${name}" \
      >"$log_file" 2>&1 &
  echo "$!" >"$LOG_DIR/${name}.pid"
}

# Keep one ordinary exact-K baseline in the same code/log setting.
launch 0 exactk_weakpost \
  model.sampling_edge_selection=gumbel_exact_k \
  model.sampling_exact_k_connectivity_repair=false

# Exact-K is still responsible for family quotas and total edge count; the
# degree-pair term only nudges ranking, so sweep conservatively.
launch 1 degpair_s003 \
  model.sampling_edge_selection=gumbel_exact_k_degree_pair \
  model.sampling_degree_pair_strength=0.03 \
  model.sampling_exact_k_connectivity_repair=false

launch 2 degpair_s006 \
  model.sampling_edge_selection=gumbel_exact_k_degree_pair \
  model.sampling_degree_pair_strength=0.06 \
  model.sampling_exact_k_connectivity_repair=false

launch 3 degpair_s010 \
  model.sampling_edge_selection=gumbel_exact_k_degree_pair \
  model.sampling_degree_pair_strength=0.10 \
  model.sampling_exact_k_connectivity_repair=false

launch 4 degpair_s015 \
  model.sampling_edge_selection=gumbel_exact_k_degree_pair \
  model.sampling_degree_pair_strength=0.15 \
  model.sampling_exact_k_connectivity_repair=false

launch 5 degpair_s020 \
  model.sampling_edge_selection=gumbel_exact_k_degree_pair \
  model.sampling_degree_pair_strength=0.20 \
  model.sampling_exact_k_connectivity_repair=false

# Deterministic counterpart: tells us whether the bias itself fixes assortativity
# or whether Gumbel noise is still doing important exploration.
launch 6 degpair_det_s010 \
  model.sampling_edge_selection=deterministic_exact_k_degree_pair \
  model.sampling_degree_pair_strength=0.10 \
  model.sampling_exact_k_connectivity_repair=false

# Connectivity repair can improve LCC, but may hurt triangles; test only one
# middle-strength point first.
launch 7 degpair_s006_repair \
  model.sampling_edge_selection=gumbel_exact_k_degree_pair \
  model.sampling_degree_pair_strength=0.06 \
  model.sampling_exact_k_connectivity_repair=true \
  model.sampling_exact_k_repair_max_swaps=0

echo "[launch] submitted jobs. Logs: $ROOT/$LOG_DIR"
echo "[hint] monitor: tail -f $LOG_DIR/*.out"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_degree_pair_dist_loss_8gpu}"
PYTHON_BIN="${PYTHON_BIN:-/data2/lyh/miniconda3/envs/sparse_block/bin/python}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-0}"
SKIP="${SKIP:-10}"
SAMPLE_SEEDS="${SAMPLE_SEEDS:-[0]}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[error] python not executable: $PYTHON_BIN" >&2
  exit 2
fi
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PYTHON_BIN")")/lib:${LD_LIBRARY_PATH:-}"

COMMON_OVERRIDES=(
  general.wandb=disabled
  general.gpus=1
  general.log_every_n_steps=1
  train.n_epochs="${EPOCHS}"
  train.batch_size=1
  train.save_model=true
  train.seed="${SEED}"
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
  model.train_all_blocks_per_noise=true
  model.train_all_blocks_step_mode=sequential
  model.train_all_blocks_shuffle=true
  model.train_all_blocks_count=0
  model.use_sparse_hetero_y=false
  model.use_family_y_film=false
  model.use_family_edge_update=false
  model.family_role_loss_weight=0.0
  model.degree_pair_dist_loss_warmup_epochs=10
  model.degree_pair_dist_min_query_edges=32
  model.degree_pair_dist_target_smoothing=0.0
  general.run_test_after_train=true
  general.enable_test_sampling=true
  general.enable_val_sampling=false
  general.enable_val_pred_metrics=false
  general.test_variance=1
  "general.test_sampling_seeds=${SAMPLE_SEEDS}"
  general.test_sampling_full_steps=false
  general.sampling_skip="${SKIP}"
  general.verbose_sampling=true
  general.log_sampling_triangle_trace=true
  general.test_sampling_metrics_every=1
  model.sampling_edge_selection=gumbel_exact_k
  model.sampling_gumbel_temperature=0.01
  model.sampling_calibrate_exist_pos_weight=true
  model.sampling_use_reverse_posterior=true
  model.sampling_reverse_posterior_mix_weights=null
  model.sampling_reverse_posterior_mix_mode=alpha_bar_s
  model.sampling_reverse_posterior_mix_scale=0.2
  model.sampling_exact_k_connectivity_repair=false
)

launch() {
  local idx="$1"
  local tag="$2"
  shift 2
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  local name="pubmed_seqbase_${tag}_seed${SEED}_ep${EPOCHS}_skip${SKIP}"
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

# Baseline reproduced under the same launcher/logging setup.
launch 0 baseline \
  model.degree_pair_dist_loss_weight=0.0

# JS is bounded and usually less brittle than KL for sparse bins.
launch 1 dp_js_w002 \
  model.degree_pair_dist_loss_type=js \
  model.degree_pair_dist_loss_weight=0.02

launch 2 dp_js_w005 \
  model.degree_pair_dist_loss_type=js \
  model.degree_pair_dist_loss_weight=0.05

launch 3 dp_js_w010 \
  model.degree_pair_dist_loss_type=js \
  model.degree_pair_dist_loss_weight=0.10

launch 4 dp_js_w020 \
  model.degree_pair_dist_loss_type=js \
  model.degree_pair_dist_loss_weight=0.20

# L1 is more direct and may preserve ranking better than distributional KL/JS.
launch 5 dp_l1_w005 \
  model.degree_pair_dist_loss_type=l1 \
  model.degree_pair_dist_loss_weight=0.05

launch 6 dp_l1_w010 \
  model.degree_pair_dist_loss_type=l1 \
  model.degree_pair_dist_loss_weight=0.10

# A smoothed target checks whether strict zero-mass bins are too harsh.
launch 7 dp_js_w010_smooth005 \
  model.degree_pair_dist_loss_type=js \
  model.degree_pair_dist_loss_weight=0.10 \
  model.degree_pair_dist_target_smoothing=0.05

echo "[launch] submitted 8 jobs. Logs: $ROOT/$LOG_DIR"
echo "[hint] monitor: tail -f $LOG_DIR/*.out"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_hetero_xey_y_ablation_8gpu}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-0}"
TEST_VARIANCE="${TEST_VARIANCE:-1}"
SAMPLE_SEEDS="${SAMPLE_SEEDS:-[0]}"
mkdir -p "$LOG_DIR"

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
  general.run_test_after_train=true
  general.enable_test_sampling=true
  general.enable_val_sampling=false
  general.enable_val_pred_metrics=false
  general.test_variance="${TEST_VARIANCE}"
  "general.test_sampling_seeds=${SAMPLE_SEEDS}"
  general.test_sampling_full_steps=false
  general.sampling_skip=10
)

mode_overrides() {
  local mode="$1"
  case "$mode" in
    accumulate)
      printf '%s\n' \
        model.train_all_blocks_per_noise=true \
        model.train_all_blocks_step_mode=accumulate \
        model.train_all_blocks_shuffle=true \
        model.train_all_blocks_count=0
      ;;
    sequential)
      printf '%s\n' \
        model.train_all_blocks_per_noise=true \
        model.train_all_blocks_step_mode=sequential \
        model.train_all_blocks_shuffle=true \
        model.train_all_blocks_count=0
      ;;
    single)
      printf '%s\n' \
        model.train_all_blocks_per_noise=false \
        model.train_all_blocks_step_mode=sequential \
        model.train_all_blocks_count=0
      ;;
    *)
      echo "Unknown mode: ${mode}" >&2
      return 2
      ;;
  esac
}

launch() {
  local gpu="$1"
  local mode="$2"
  local tag="$3"
  shift 3
  local name="hetero_xey_${mode}_${tag}_seed${SEED}_ep${EPOCHS}_skip10"
  local log_file="$LOG_DIR/${name}.out"
  local mode_args=()
  mapfile -t mode_args < <(mode_overrides "$mode")

  echo "[launch] gpu=${gpu} mode=${mode} tag=${tag} name=${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  CONDA_ENV="${CONDA_ENV:-sparse_block}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}" \
  EXPERIMENT=pubmed_hetero_xey_family_pw20 \
    nohup bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON_OVERRIDES[@]}" \
      "${mode_args[@]}" \
      "$@" \
      "general.name=${name}" \
      >"$log_file" 2>&1 &
  echo "$!" >"$LOG_DIR/${name}.pid"
}

# Optimization-mode controls without extra hetero-y.
launch 0 accumulate base \
  model.use_sparse_hetero_y=false model.use_family_y_film=false model.use_family_edge_update=false
launch 1 sequential base \
  model.use_sparse_hetero_y=false model.use_family_y_film=false model.use_family_edge_update=false
launch 2 single base \
  model.use_sparse_hetero_y=false model.use_family_y_film=false model.use_family_edge_update=false

# Does sparse hetero graph-level y help? Does per-family y->E FiLM help further?
launch 3 accumulate hy_global \
  model.use_sparse_hetero_y=true model.use_family_y_film=false model.use_family_edge_update=false
launch 4 accumulate hy_family \
  model.use_sparse_hetero_y=true model.use_family_y_film=true model.use_family_edge_update=false

# If y helps, check whether the training-block schedule still matters under y.
launch 5 sequential hy_family \
  model.use_sparse_hetero_y=true model.use_family_y_film=true model.use_family_edge_update=false
launch 6 single hy_family \
  model.use_sparse_hetero_y=true model.use_family_y_film=true model.use_family_edge_update=false

# Orthogonal architecture ablation: explicit EdgeUpdate_f([A,E]) on top of family-y.
launch 7 accumulate hy_family_edgeupd \
  model.use_sparse_hetero_y=true model.use_family_y_film=true model.use_family_edge_update=true

echo "[launch] submitted 8 independent nohup jobs. Logs: $ROOT/$LOG_DIR"
echo "[hint] monitor: tail -f $LOG_DIR/*.out"

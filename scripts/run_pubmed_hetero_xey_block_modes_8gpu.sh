#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_hetero_xey_block_modes_8gpu}"
EPOCHS="${EPOCHS:-100}"
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

launch() {
  local gpu="$1"
  local mode="$2"
  local seed="$3"
  local extra_tag="${4:-base}"
  shift 4 || true
  local name="hetero_xey_${mode}_${extra_tag}_seed${seed}_ep${EPOCHS}_skip10"
  local log_file="$LOG_DIR/${name}.out"
  local mode_overrides=()

  case "$mode" in
    accumulate)
      mode_overrides=(
        model.train_all_blocks_per_noise=true
        model.train_all_blocks_step_mode=accumulate
        model.train_all_blocks_shuffle=true
        model.train_all_blocks_count=0
      )
      ;;
    sequential)
      mode_overrides=(
        model.train_all_blocks_per_noise=true
        model.train_all_blocks_step_mode=sequential
        model.train_all_blocks_shuffle=true
        model.train_all_blocks_count=0
      )
      ;;
    single)
      mode_overrides=(
        model.train_all_blocks_per_noise=false
        model.train_all_blocks_step_mode=sequential
        model.train_all_blocks_count=0
      )
      ;;
    *)
      echo "Unknown mode: ${mode}" >&2
      return 2
      ;;
  esac

  echo "[launch] gpu=${gpu} mode=${mode} tag=${extra_tag} seed=${seed} name=${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  CONDA_ENV="${CONDA_ENV:-sparse_block}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}" \
  EXPERIMENT=pubmed_hetero_xey_family_pw20 \
    nohup bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON_OVERRIDES[@]}" \
      "${mode_overrides[@]}" \
      "$@" \
      "general.name=${name}" \
      "train.seed=${seed}" \
      >"$log_file" 2>&1 &
  echo "$!" >"$LOG_DIR/${name}.pid"
}

# Main comparison: optimization schedule under the same X/E/y family-FiLM model.
launch 0 accumulate 0 base model.use_family_edge_update=false
launch 1 accumulate 1 base model.use_family_edge_update=false
launch 2 sequential 0 base model.use_family_edge_update=false
launch 3 sequential 1 base model.use_family_edge_update=false
launch 4 single 0 base model.use_family_edge_update=false
launch 5 single 1 base model.use_family_edge_update=false

# Extra two cards: does explicit EdgeUpdate_f([A,E]) help beyond DiGress-style
# family FiLM? Use accumulate because it is the cleanest full-coverage objective.
launch 6 accumulate 0 edgeupd model.use_family_edge_update=true
launch 7 accumulate 1 edgeupd model.use_family_edge_update=true

echo "[launch] submitted 8 independent nohup jobs. Logs: $ROOT/$LOG_DIR"
echo "[hint] monitor: tail -f $LOG_DIR/*.out"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data2/lyh/miniconda3/envs/sparse_block/bin/python}"
LOG_DIR="${LOG_DIR:-logs/logs_pubmed_family_y_retry_tests}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
SKIP="${SKIP:-10}"
TEST_SEEDS="${TEST_SEEDS:-[0]}"
TEST_VARIANCE="${TEST_VARIANCE:-1}"
SAMPLES_TO_GENERATE="${SAMPLES_TO_GENERATE:-1}"
mkdir -p "$LOG_DIR"

export LD_LIBRARY_PATH="$(dirname "$(dirname "$PYTHON_BIN")")/lib:${LD_LIBRARY_PATH:-}"

latest_ckpt_for_name() {
  local run_name="$1"
  find "$ROOT/output" -path "*/checkpoints/${run_name}/last.ckpt" -type f \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

is_pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1
}

wait_for_slot() {
  while true; do
    local running=0 pid_file pid
    for pid_file in "$LOG_DIR"/*.pid; do
      [[ -e "$pid_file" ]] || continue
      pid="$(cat "$pid_file")"
      if is_pid_alive "$pid"; then
        running=$((running + 1))
      fi
    done
    [[ "$running" -lt "${#GPUS[@]}" ]] && return 0
    sleep 30
  done
}

wait_for_gpu() {
  local slot
  while true; do
    for slot in "${!GPUS[@]}"; do
      local pid_file="$LOG_DIR/gpu_${GPUS[$slot]}.pid"
      if [[ ! -f "$pid_file" ]] || ! is_pid_alive "$(cat "$pid_file")"; then
        echo "${GPUS[$slot]}"
        return 0
      fi
    done
    sleep 30
  done
}

COMMON="$PYTHON_BIN -m dihug.main +experiment=pubmed_hetero_xey_family_pw20 general.wandb=disabled general.gpus=1 general.run_test_after_train=false general.enable_test_sampling=true general.enable_val_sampling=false general.enable_val_pred_metrics=false general.test_variance=$TEST_VARIANCE general.final_model_samples_to_generate=$SAMPLES_TO_GENERATE general.final_model_chains_to_save=0 general.verbose_sampling=true general.log_sampling_triangle_trace=true general.test_sampling_full_steps=false general.test_sampling_metrics_every=1 general.test_sampling_seeds='$TEST_SEEDS' general.sampling_skip=$SKIP model.edge_fraction=0.1 model.block_query=true model.block_partition_mode=hetero_metis model.block_query_full_block=true model.block_query_inter_fill=true model.block_query_include_uniform=false model.query_include_all_positive_edges=false model.sampling_block_mode=type_template model.sampling_block_template_init=false model.sampling_block_marginal_init=false model.train_all_blocks_per_noise=false model.sampling_calibrate_exist_pos_weight=true model.sampling_use_reverse_posterior=true model.sampling_reverse_posterior_mix_weights=null model.sampling_reverse_posterior_mix_mode=alpha_bar_s model.sampling_reverse_posterior_mix_scale=0.2 model.sampling_gumbel_temperature=0.01 model.sampling_degree_pair_bins=5 model.sampling_degree_pair_bias_clip=4.0 model.use_edge_struct_features=true model.edge_struct_feature_dim=8 model.edge_struct_hidden_dim=64 model.edge_struct_residual_scale=2.0 model.use_sparse_hetero_y=true model.use_sparse_family_y=true model.use_family_y_film=true model.use_family_y_in_attention=false model.use_family_y_in_edge_film=true model.edge_struct_use_family_y=true"

launch_one() {
  local gpu="$1"
  local run="$2"
  local suffix="$3"
  local extra="$4"
  local ckpt
  ckpt="$(latest_ckpt_for_name "$run")"
  [[ -n "$ckpt" && -f "$ckpt" ]] || { echo "[error] checkpoint not found for $run" >&2; exit 2; }
  local name="${run}_${suffix}_test"
  local log="$LOG_DIR/$name.out"
  echo "[launch] $(date '+%F %T') gpu=$gpu name=$name ckpt=$ckpt" | tee -a "$LOG_DIR/launcher.log"
  (
    set +e
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
    eval "$COMMON general.test_only=$ckpt general.name=$name $extra"
    rc=$?
    echo "TASK_EXIT=$rc"
    exit "$rc"
  ) > "$log" 2>&1 &
  echo "$!" > "$LOG_DIR/gpu_${gpu}.pid"
  echo "$!" > "$LOG_DIR/$name.pid"
}

declare -a RUNS=(
  pubmed_seq_edge_struct_famy_edgefilm_scale2_seed0
  pubmed_seq_edge_struct_famy_dpjs005_seed0
)
declare -a SUFFIXES=(exactk degpair_s010 degpair_s015 quota_s050 quota_s100)

for run in "${RUNS[@]}"; do
  for suffix in "${SUFFIXES[@]}"; do
    case "$suffix" in
      exactk) extra="model.sampling_edge_selection=gumbel_exact_k model.sampling_exact_k_connectivity_repair=false" ;;
      degpair_s010) extra="model.sampling_edge_selection=gumbel_exact_k_degree_pair model.sampling_degree_pair_strength=0.10 model.sampling_exact_k_connectivity_repair=false" ;;
      degpair_s015) extra="model.sampling_edge_selection=gumbel_exact_k_degree_pair model.sampling_degree_pair_strength=0.15 model.sampling_exact_k_connectivity_repair=false" ;;
      quota_s050) extra="model.sampling_edge_selection=gumbel_exact_k_degree_pair_quota model.sampling_degree_pair_strength=0.50 model.sampling_exact_k_connectivity_repair=false" ;;
      quota_s100) extra="model.sampling_edge_selection=gumbel_exact_k_degree_pair_quota model.sampling_degree_pair_strength=1.00 model.sampling_exact_k_connectivity_repair=false" ;;
    esac
    wait_for_slot
    gpu="$(wait_for_gpu)"
    launch_one "$gpu" "$run" "$suffix" "$extra"
    sleep 5
  done
done

wait
echo "[done] $(date '+%F %T')" | tee -a "$LOG_DIR/launcher.log"

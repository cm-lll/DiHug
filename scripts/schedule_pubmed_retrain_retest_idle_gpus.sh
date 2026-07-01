#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_pubmed_retrain_retest_idle_scheduler}"
mkdir -p "$LOG_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data2/lyh/miniconda3/envs/sparse_block/bin/python}"
CONDA_ENV="${CONDA_ENV:-sparse_block}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
POLL_SECONDS="${POLL_SECONDS:-600}"
IDLE_MEM_MB="${IDLE_MEM_MB:-700}"
IDLE_UTIL_PCT="${IDLE_UTIL_PCT:-5}"
N_EPOCHS="${N_EPOCHS:-100}"
TRAIN_SEED="${TRAIN_SEED:-0}"
SKIP="${SKIP:-10}"
TEST_SEEDS="${TEST_SEEDS:-[0]}"
TEST_VARIANCE="${TEST_VARIANCE:-1}"
SAMPLES_TO_GENERATE="${SAMPLES_TO_GENERATE:-1}"
METRICS_EVERY="${METRICS_EVERY:-1}"
OLD_SEQBASE_CKPT="${OLD_SEQBASE_CKPT:-$ROOT/output/2026-06-28/14-49-55-hetero_xey_sequential_base_seed0_ep100_skip10/output/sparse_diffusion/checkpoints/hetero_xey_sequential_base_seed0_ep100_skip10/last.ckpt}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[error] python not executable: $PYTHON_BIN" >&2
  exit 2
fi

export LD_LIBRARY_PATH="$(dirname "$(dirname "$PYTHON_BIN")")/lib:${LD_LIBRARY_PATH:-}"

declare -a TASK_KIND=()
declare -a TASK_NAME=()
declare -a TASK_DEP=()
declare -a TASK_CMD=()
declare -a TASK_STATUS=()

add_task() {
  TASK_KIND+=("$1")
  TASK_NAME+=("$2")
  TASK_DEP+=("$3")
  TASK_CMD+=("$4")
  TASK_STATUS+=("pending")
}

latest_ckpt_for_name() {
  local run_name="$1"
  find "$ROOT/output" -path "*/checkpoints/${run_name}/last.ckpt" -type f \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

is_pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1
}

is_gpu_idle() {
  local gpu="$1"
  local line mem util
  line="$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F, -v g="$gpu" '$1+0==g {gsub(/ /,""); print $2" "$3}')"
  [[ -n "$line" ]] || return 1
  util="$(awk '{print $1}' <<<"$line")"
  mem="$(awk '{print $2}' <<<"$line")"
  [[ "${util:-999}" -le "$IDLE_UTIL_PCT" && "${mem:-999999}" -le "$IDLE_MEM_MB" ]]
}

task_done_file() {
  echo "$LOG_DIR/$1.done"
}

task_pid_file() {
  echo "$LOG_DIR/$1.pid"
}

task_log_file() {
  echo "$LOG_DIR/$1.out"
}

mark_finished_tasks() {
  local i name pid_file done_file pid
  for i in "${!TASK_NAME[@]}"; do
    name="${TASK_NAME[$i]}"
    pid_file="$(task_pid_file "$name")"
    done_file="$(task_done_file "$name")"
    [[ "${TASK_STATUS[$i]}" == "running" ]] || continue
    if [[ -f "$done_file" ]]; then
      TASK_STATUS[$i]="done"
      continue
    fi
    if [[ -f "$pid_file" ]]; then
      pid="$(cat "$pid_file")"
      if ! is_pid_alive "$pid"; then
        if grep -q "TASK_EXIT=0" "$(task_log_file "$name")" 2>/dev/null; then
          TASK_STATUS[$i]="done"
          touch "$done_file"
        else
          TASK_STATUS[$i]="failed"
          touch "$LOG_DIR/$name.failed"
          echo "[warn] task failed: $name"
        fi
      fi
    fi
  done
}

dep_ready() {
  local dep="$1"
  [[ -z "$dep" || "$dep" == "-" ]] && return 0
  [[ -f "$(task_done_file "$dep")" ]]
}

command_for_task() {
  local name="$1"
  local raw="$2"
  local dep="$3"
  local ckpt=""
  if [[ "$raw" == *"__TEST_DEP__"* ]]; then
    ckpt="$(latest_ckpt_for_name "$dep")"
    [[ -n "$ckpt" && -f "$ckpt" ]] || return 1
    raw="${raw//__TEST_DEP__/$ckpt}"
  elif [[ "$raw" == *"__TEST_OLD__"* ]]; then
    ckpt="$OLD_SEQBASE_CKPT"
    [[ -n "$ckpt" && -f "$ckpt" ]] || return 1
    raw="${raw//__TEST_OLD__/$ckpt}"
  fi
  printf '%s' "$raw"
}

initialize_task_status_from_files() {
  local i name
  for i in "${!TASK_NAME[@]}"; do
    name="${TASK_NAME[$i]}"
    if [[ -f "$(task_done_file "$name")" ]]; then
      TASK_STATUS[$i]="done"
    elif [[ -f "$LOG_DIR/$name.failed" ]]; then
      TASK_STATUS[$i]="failed"
    fi
  done
}

launch_task_on_gpu() {
  local idx="$1"
  local gpu="$2"
  local name="${TASK_NAME[$idx]}"
  local dep="${TASK_DEP[$idx]}"
  local raw="${TASK_CMD[$idx]}"
  local cmd log pid_file
  cmd="$(command_for_task "$name" "$raw" "$dep")" || return 1
  log="$(task_log_file "$name")"
  pid_file="$(task_pid_file "$name")"
  echo "[launch] $(date '+%F %T') gpu=$gpu kind=${TASK_KIND[$idx]} name=$name" | tee -a "$LOG_DIR/scheduler.log"
  (
    set +e
    export CUDA_VISIBLE_DEVICES="$gpu"
    export CONDA_ENV="$CONDA_ENV"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
    eval "$cmd"
    rc=$?
    echo "TASK_EXIT=$rc"
    exit "$rc"
  ) >"$log" 2>&1 &
  echo "$!" >"$pid_file"
  TASK_STATUS[$idx]="running"
  return 0
}

next_pending_task_index() {
  local kind pass i dep
  for kind in train test; do
    for i in "${!TASK_NAME[@]}"; do
      [[ "${TASK_STATUS[$i]}" == "pending" ]] || continue
      [[ "${TASK_KIND[$i]}" == "$kind" ]] || continue
      dep="${TASK_DEP[$i]}"
      dep_ready "$dep" || continue
      echo "$i"
      return 0
    done
  done
  return 1
}

remaining_tasks() {
  local i count=0
  for i in "${!TASK_NAME[@]}"; do
    case "${TASK_STATUS[$i]}" in
      pending|running) count=$((count + 1));;
    esac
  done
  echo "$count"
}

TRAIN_COMMON="EXPERIMENT=pubmed_hetero_xey_family_pw20 bash scripts/run_pubmed_baseline_single.sh general.wandb=disabled general.gpus=1 train.n_epochs=$N_EPOCHS train.batch_size=1 train.save_model=true train.seed=$TRAIN_SEED general.run_test_after_train=false general.enable_test_sampling=false general.enable_val_sampling=false general.enable_val_pred_metrics=false model.edge_fraction=0.1 model.block_query=true model.block_partition_mode=hetero_metis model.block_query_full_block=true model.block_query_inter_fill=true model.block_query_include_uniform=false model.query_include_all_positive_edges=false model.sampling_block_mode=type_template model.sampling_block_template_init=false model.sampling_block_marginal_init=false model.train_all_blocks_per_noise=true model.train_all_blocks_step_mode=sequential model.train_all_blocks_shuffle=true model.train_all_blocks_count=0 model.use_sparse_hetero_y=false model.use_sparse_family_y=false model.use_family_y_film=false model.use_family_y_in_attention=false model.use_family_y_in_edge_film=false model.use_family_edge_update=false model.use_edge_struct_features=true model.edge_struct_feature_dim=8 model.edge_struct_hidden_dim=64 model.edge_struct_residual_scale=1.0 model.edge_struct_use_family_y=false model.degree_pair_dist_loss_weight=0.0 model.closure_pos_loss_weight=0.0"

TEST_COMMON="$PYTHON_BIN -m dihug.main +experiment=pubmed_hetero_xey_family_pw20 general.wandb=disabled general.gpus=1 general.run_test_after_train=false general.enable_test_sampling=true general.enable_val_sampling=false general.enable_val_pred_metrics=false general.test_variance=$TEST_VARIANCE general.final_model_samples_to_generate=$SAMPLES_TO_GENERATE general.final_model_chains_to_save=0 general.verbose_sampling=true general.log_sampling_triangle_trace=true general.test_sampling_full_steps=false general.test_sampling_metrics_every=$METRICS_EVERY general.test_sampling_seeds='$TEST_SEEDS' general.sampling_skip=$SKIP model.edge_fraction=0.1 model.block_query=true model.block_partition_mode=hetero_metis model.block_query_full_block=true model.block_query_inter_fill=true model.block_query_include_uniform=false model.query_include_all_positive_edges=false model.sampling_block_mode=type_template model.sampling_block_template_init=false model.sampling_block_marginal_init=false model.train_all_blocks_per_noise=false model.sampling_calibrate_exist_pos_weight=true model.sampling_use_reverse_posterior=true model.sampling_reverse_posterior_mix_weights=null model.sampling_reverse_posterior_mix_mode=alpha_bar_s model.sampling_reverse_posterior_mix_scale=0.2 model.sampling_gumbel_temperature=0.01 model.sampling_degree_pair_bins=5 model.sampling_degree_pair_bias_clip=4.0"

test_arch_overrides_for_dep() {
  local dep="$1"
  local overrides="model.use_edge_struct_features=true model.edge_struct_feature_dim=8 model.edge_struct_hidden_dim=64 model.use_sparse_hetero_y=false model.use_sparse_family_y=false model.use_family_y_film=false model.use_family_y_in_attention=false model.use_family_y_in_edge_film=false model.edge_struct_use_family_y=false model.degree_pair_dist_loss_weight=0.0"
  if [[ "$dep" == *"scale2"* || "$dep" == *"famy"* ]]; then
    overrides="$overrides model.edge_struct_residual_scale=2.0"
  else
    overrides="$overrides model.edge_struct_residual_scale=1.0"
  fi
  if [[ "$dep" == *"famy"* ]]; then
    overrides="$overrides model.use_sparse_hetero_y=true model.use_sparse_family_y=true model.use_family_y_film=true model.use_family_y_in_edge_film=true model.edge_struct_use_family_y=true"
  fi
  if [[ "$dep" == *"dpjs005"* ]]; then
    overrides="$overrides model.degree_pair_dist_loss_weight=0.05 model.degree_pair_dist_loss_type=js"
  fi
  printf '%s' "$overrides"
}

# Training priority: reproduce the old successful sequential update regime,
# then add only one axis at a time.
add_task train pubmed_seq_edge_struct_seed0 - \
  "$TRAIN_COMMON general.name=pubmed_seq_edge_struct_seed0"
add_task train pubmed_seq_edge_struct_scale2_seed0 - \
  "$TRAIN_COMMON general.name=pubmed_seq_edge_struct_scale2_seed0 model.edge_struct_residual_scale=2.0"
add_task train pubmed_seq_edge_struct_famy_edgefilm_scale2_seed0 - \
  "$TRAIN_COMMON general.name=pubmed_seq_edge_struct_famy_edgefilm_scale2_seed0 model.use_sparse_hetero_y=true model.use_sparse_family_y=true model.use_family_y_film=true model.use_family_y_in_edge_film=true model.edge_struct_use_family_y=true model.edge_struct_residual_scale=2.0"
add_task train pubmed_seq_edge_struct_famy_dpjs005_seed0 - \
  "$TRAIN_COMMON general.name=pubmed_seq_edge_struct_famy_dpjs005_seed0 model.use_sparse_hetero_y=true model.use_sparse_family_y=true model.use_family_y_film=true model.use_family_y_in_edge_film=true model.edge_struct_use_family_y=true model.edge_struct_residual_scale=2.0 model.degree_pair_dist_loss_weight=0.05 model.degree_pair_dist_loss_type=js"

# Retest the old sequential checkpoint to keep a direct reference in the same
# scheduler/log format.
add_task test retest_old_seqbase_degpair_s010 - \
  "$TEST_COMMON general.test_only=__TEST_OLD__ general.name=retest_old_seqbase_degpair_s010 model.use_sparse_hetero_y=false model.use_sparse_family_y=false model.use_edge_struct_features=false model.sampling_edge_selection=gumbel_exact_k_degree_pair model.sampling_degree_pair_strength=0.10 model.sampling_exact_k_connectivity_repair=false"
add_task test retest_old_seqbase_degpair_s015 - \
  "$TEST_COMMON general.test_only=__TEST_OLD__ general.name=retest_old_seqbase_degpair_s015 model.use_sparse_hetero_y=false model.use_sparse_family_y=false model.use_edge_struct_features=false model.sampling_edge_selection=gumbel_exact_k_degree_pair model.sampling_degree_pair_strength=0.15 model.sampling_exact_k_connectivity_repair=false"
add_task test retest_old_seqbase_quota_s050 - \
  "$TEST_COMMON general.test_only=__TEST_OLD__ general.name=retest_old_seqbase_quota_s050 model.use_sparse_hetero_y=false model.use_sparse_family_y=false model.use_edge_struct_features=false model.sampling_edge_selection=gumbel_exact_k_degree_pair_quota model.sampling_degree_pair_strength=0.50 model.sampling_exact_k_connectivity_repair=false"
add_task test retest_old_seqbase_quota_s100 - \
  "$TEST_COMMON general.test_only=__TEST_OLD__ general.name=retest_old_seqbase_quota_s100 model.use_sparse_hetero_y=false model.use_sparse_family_y=false model.use_edge_struct_features=false model.sampling_edge_selection=gumbel_exact_k_degree_pair_quota model.sampling_degree_pair_strength=1.00 model.sampling_exact_k_connectivity_repair=false"

# Test every retrained checkpoint under the old high-triangle samplers and one
# plain exact-K baseline.
for dep in \
  pubmed_seq_edge_struct_seed0 \
  pubmed_seq_edge_struct_scale2_seed0 \
  pubmed_seq_edge_struct_famy_edgefilm_scale2_seed0 \
  pubmed_seq_edge_struct_famy_dpjs005_seed0
do
  arch_overrides="$(test_arch_overrides_for_dep "$dep")"
  add_task test "${dep}_exactk" "$dep" \
    "$TEST_COMMON $arch_overrides general.test_only=__TEST_DEP__ general.name=${dep}_exactk_test model.sampling_edge_selection=gumbel_exact_k model.sampling_exact_k_connectivity_repair=false"
  add_task test "${dep}_degpair_s010" "$dep" \
    "$TEST_COMMON $arch_overrides general.test_only=__TEST_DEP__ general.name=${dep}_degpair_s010_test model.sampling_edge_selection=gumbel_exact_k_degree_pair model.sampling_degree_pair_strength=0.10 model.sampling_exact_k_connectivity_repair=false"
  add_task test "${dep}_degpair_s015" "$dep" \
    "$TEST_COMMON $arch_overrides general.test_only=__TEST_DEP__ general.name=${dep}_degpair_s015_test model.sampling_edge_selection=gumbel_exact_k_degree_pair model.sampling_degree_pair_strength=0.15 model.sampling_exact_k_connectivity_repair=false"
  add_task test "${dep}_quota_s050" "$dep" \
    "$TEST_COMMON $arch_overrides general.test_only=__TEST_DEP__ general.name=${dep}_quota_s050_test model.sampling_edge_selection=gumbel_exact_k_degree_pair_quota model.sampling_degree_pair_strength=0.50 model.sampling_exact_k_connectivity_repair=false"
  add_task test "${dep}_quota_s100" "$dep" \
    "$TEST_COMMON $arch_overrides general.test_only=__TEST_DEP__ general.name=${dep}_quota_s100_test model.sampling_edge_selection=gumbel_exact_k_degree_pair_quota model.sampling_degree_pair_strength=1.00 model.sampling_exact_k_connectivity_repair=false"
done

echo "[scheduler] tasks=${#TASK_NAME[@]} poll=${POLL_SECONDS}s idle_mem<=${IDLE_MEM_MB}MB idle_util<=${IDLE_UTIL_PCT}% logs=$LOG_DIR"
echo "[scheduler] tasks=${#TASK_NAME[@]} poll=${POLL_SECONDS}s idle_mem<=${IDLE_MEM_MB}MB idle_util<=${IDLE_UTIL_PCT}% logs=$LOG_DIR" >> "$LOG_DIR/scheduler.log"
printf '%s\n' "${TASK_NAME[@]}" > "$LOG_DIR/task_order.txt"
initialize_task_status_from_files

while [[ "$(remaining_tasks)" -gt 0 ]]; do
  mark_finished_tasks
  for gpu in "${GPUS[@]}"; do
    mark_finished_tasks
    is_gpu_idle "$gpu" || continue
    idx="$(next_pending_task_index || true)"
    [[ -n "${idx:-}" ]] || break
    launch_task_on_gpu "$idx" "$gpu" || true
    sleep 5
  done
  {
    echo "[$(date '+%F %T')] status"
    for i in "${!TASK_NAME[@]}"; do
      echo "  ${TASK_STATUS[$i]} ${TASK_KIND[$i]} ${TASK_NAME[$i]} dep=${TASK_DEP[$i]}"
    done
  } > "$LOG_DIR/status.txt"
  {
    echo "[$(date '+%F %T')] heartbeat remaining=$(remaining_tasks)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | sed 's/^/  gpu /'
  } >> "$LOG_DIR/scheduler.log"
  [[ "$(remaining_tasks)" -eq 0 ]] && break
  sleep "$POLL_SECONDS"
done

mark_finished_tasks
echo "[scheduler] all runnable tasks finished at $(date '+%F %T')" | tee -a "$LOG_DIR/scheduler.log"

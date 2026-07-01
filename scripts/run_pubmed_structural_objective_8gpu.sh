#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_pubmed_structural_objective_8gpu}"
mkdir -p "$LOG_DIR"

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
N_EPOCHS="${N_EPOCHS:-100}"
TRAIN_SEED="${TRAIN_SEED:-0}"
TEST_VARIANCE="${TEST_VARIANCE:-3}"
SAMPLES_TO_GENERATE="${SAMPLES_TO_GENERATE:-3}"
SAMPLING_SKIP="${SAMPLING_SKIP:-10}"
SAMPLING_STRENGTH="${SAMPLING_STRENGTH:-0.15}"

COMMON=(
  general.wandb=disabled
  general.gpus=1
  general.run_test_after_train=true
  general.enable_test_sampling=true
  general.enable_val_sampling=false
  general.enable_val_pred_metrics=false
  general.test_variance="${TEST_VARIANCE}"
  general.final_model_samples_to_generate="${SAMPLES_TO_GENERATE}"
  general.final_model_chains_to_save=0
  general.test_sampling_full_steps=false
  general.sampling_skip="${SAMPLING_SKIP}"
  general.test_sampling_metrics_every=1
  general.verbose_sampling=true
  general.log_sampling_triangle_trace=true
  general.edge_score_structure_diag=true
  general.edge_score_structure_diag_max_negatives=100000
  train.n_epochs="${N_EPOCHS}"
  train.batch_size=1
  train.save_model=true
  train.seed="${TRAIN_SEED}"
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
  model.sampling_block_marginal_init=false
  model.train_all_blocks_per_noise=true
  model.train_all_blocks_step_mode=sequential
  model.train_all_blocks_shuffle=true
  model.train_all_blocks_count=0
  model.use_sparse_hetero_y=false
  model.use_sparse_family_y=false
  model.use_family_y_film=false
  model.use_family_y_in_attention=false
  model.use_family_y_in_edge_film=false
  model.use_family_edge_update=false
  model.use_edge_struct_features=false
  model.edge_struct_use_family_y=false
  model.sampling_calibrate_exist_pos_weight=true
  model.sampling_use_reverse_posterior=true
  model.sampling_reverse_posterior_mix_weights=null
  model.sampling_reverse_posterior_mix_mode=alpha_bar_s
  model.sampling_reverse_posterior_mix_scale=0.2
  model.sampling_edge_selection=gumbel_exact_k_degree_pair
  model.sampling_degree_pair_strength="${SAMPLING_STRENGTH}"
  model.sampling_exact_k_connectivity_repair=false
  model.sampling_gumbel_temperature=0.01
  model.degree_pair_dist_loss_weight=0.0
  model.family_count_loss_weight=0.0
  model.closure_rank_loss_weight=0.0
  model.closure_pos_loss_weight=0.0
  model.edge_count_loss_weight=0.0
)

declare -a PIDS=()
declare -a NAMES=()

launch() {
  local gpu="$1"
  local name="$2"
  shift 2
  local log_file="$LOG_DIR/${name}.out"
  echo "[launch] $(date '+%F %T') gpu=${gpu} name=${name}" | tee -a "$LOG_DIR/launcher.log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export CONDA_ENV="${CONDA_ENV:-sparse_block}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
    EXPERIMENT=pubmed_hetero_xey_family_pw20 bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON[@]}" \
      "general.name=${name}" \
      "$@"
  ) >"$log_file" 2>&1 &
  local pid="$!"
  echo "$pid" >"$LOG_DIR/${name}.pid"
  PIDS+=("$pid")
  NAMES+=("$name")
}

launch "${GPUS[0]}" structobj_bce_anchor \
  model.train_t_schedule=random

launch "${GPUS[1]}" structobj_dpjs_w002_t1 \
  model.degree_pair_dist_loss_weight=0.02 \
  model.degree_pair_dist_loss_t_power=1.0 \
  model.degree_pair_dist_loss_warmup_epochs=5

launch "${GPUS[2]}" structobj_dpjs_w005_t1 \
  model.degree_pair_dist_loss_weight=0.05 \
  model.degree_pair_dist_loss_t_power=1.0 \
  model.degree_pair_dist_loss_warmup_epochs=5

launch "${GPUS[3]}" structobj_family_count_dpjs \
  model.family_count_loss_weight=0.02 \
  model.family_count_loss_t_power=1.0 \
  model.family_count_loss_warmup_epochs=5 \
  model.degree_pair_dist_loss_weight=0.02 \
  model.degree_pair_dist_loss_t_power=1.0 \
  model.degree_pair_dist_loss_warmup_epochs=5

launch "${GPUS[4]}" structobj_closure_rank_w002 \
  model.closure_rank_loss_weight=0.02 \
  model.closure_rank_loss_t_power=1.0 \
  model.closure_rank_loss_warmup_epochs=5 \
  model.closure_rank_pairs_per_family=2048

launch "${GPUS[5]}" structobj_closure_rank_w005 \
  model.closure_rank_loss_weight=0.05 \
  model.closure_rank_loss_t_power=1.0 \
  model.closure_rank_loss_warmup_epochs=5 \
  model.closure_rank_pairs_per_family=4096

launch "${GPUS[6]}" structobj_combo_mild \
  model.family_count_loss_weight=0.01 \
  model.family_count_loss_t_power=1.0 \
  model.family_count_loss_warmup_epochs=5 \
  model.degree_pair_dist_loss_weight=0.02 \
  model.degree_pair_dist_loss_t_power=1.0 \
  model.degree_pair_dist_loss_warmup_epochs=5 \
  model.closure_rank_loss_weight=0.02 \
  model.closure_rank_loss_t_power=1.0 \
  model.closure_rank_loss_warmup_epochs=5 \
  model.closure_rank_pairs_per_family=2048

launch "${GPUS[7]}" structobj_combo_strong_cycle \
  model.train_t_schedule=cycle \
  model.train_t_cycle_repeats=1 \
  model.family_count_loss_weight=0.02 \
  model.family_count_loss_t_power=1.0 \
  model.family_count_loss_warmup_epochs=5 \
  model.degree_pair_dist_loss_weight=0.05 \
  model.degree_pair_dist_loss_t_power=1.0 \
  model.degree_pair_dist_loss_warmup_epochs=5 \
  model.closure_rank_loss_weight=0.05 \
  model.closure_rank_loss_t_power=1.0 \
  model.closure_rank_loss_warmup_epochs=5 \
  model.closure_rank_pairs_per_family=4096

failed=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[ok] $(date '+%F %T') ${NAMES[$i]}" | tee -a "$LOG_DIR/launcher.log"
    touch "$LOG_DIR/${NAMES[$i]}.done"
  else
    echo "[fail] $(date '+%F %T') ${NAMES[$i]}" | tee -a "$LOG_DIR/launcher.log"
    touch "$LOG_DIR/${NAMES[$i]}.failed"
    failed=1
  fi
done

echo "[done] $(date '+%F %T') rc=${failed} logs=${LOG_DIR}" | tee -a "$LOG_DIR/launcher.log"
exit "$failed"

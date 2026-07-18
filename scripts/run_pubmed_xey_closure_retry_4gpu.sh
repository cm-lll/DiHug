#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_pubmed_xey_structural_block_vs_accumulate_8gpu}"
mkdir -p "$LOG_DIR"

GPUS=(${GPUS:-2 3 6 7})
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
  model.train_all_blocks_shuffle=true
  model.train_all_blocks_count=0
  model.use_sparse_hetero_y=true
  model.use_sparse_family_y=true
  model.sparse_family_y_degree_bins=5
  model.use_family_y_film=true
  model.use_family_y_in_attention=false
  model.use_family_y_in_edge_film=true
  model.use_edge_struct_features=true
  model.edge_struct_feature_dim=8
  model.edge_struct_hidden_dim=64
  model.edge_struct_residual_scale=2.0
  model.edge_struct_use_family_y=true
  model.sampling_calibrate_exist_pos_weight=true
  model.sampling_use_reverse_posterior=true
  model.sampling_reverse_posterior_mix_weights=null
  model.sampling_reverse_posterior_mix_mode=alpha_bar_s
  model.sampling_reverse_posterior_mix_scale=0.2
  model.sampling_edge_selection=gumbel_exact_k_degree_pair
  model.sampling_degree_pair_strength="${SAMPLING_STRENGTH}"
  model.sampling_exact_k_connectivity_repair=false
  model.sampling_gumbel_temperature=0.01
  model.degree_pair_dist_target=block
  model.train_structure_probe_every_epochs="${STRUCT_PROBE_EVERY:-1}"
  model.train_structure_probe_min_query_edges=256
  model.degree_pair_dist_loss_weight=0.0
  model.family_count_loss_weight=0.0
  model.closure_rank_loss_weight=0.0
  model.closure_pos_loss_weight=0.0
  model.edge_count_loss_weight=0.0
  model.closure_rank_common_neighbor_chunk_size="${CLOSURE_CHUNK:-65536}"
)

declare -a PIDS=()
declare -a NAMES=()

launch() {
  local gpu="$1"
  local name="$2"
  shift 2
  local log_file="$LOG_DIR/${name}.out"
  echo "[launch] $(date '+%F %T') gpu=${gpu} name=${name}" | tee -a "$LOG_DIR/closure_retry_launcher.log"
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

COMBO_MILD=(
  model.family_count_loss_weight=0.01
  model.family_count_loss_t_power=1.0
  model.family_count_loss_warmup_epochs=5
  model.degree_pair_dist_loss_weight=0.02
  model.degree_pair_dist_loss_t_power=1.0
  model.degree_pair_dist_loss_warmup_epochs=5
  model.closure_rank_loss_weight=0.02
  model.closure_rank_loss_t_power=1.0
  model.closure_rank_loss_warmup_epochs=5
  model.closure_rank_pairs_per_family=2048
)

launch "${GPUS[0]}" xey_struct_block_closure_retry \
  model.train_all_blocks_step_mode=sequential \
  model.closure_rank_loss_weight=0.05 \
  model.closure_rank_loss_t_power=1.0 \
  model.closure_rank_loss_warmup_epochs=5 \
  model.closure_rank_pairs_per_family=4096

launch "${GPUS[1]}" xey_struct_block_combo_mild_retry \
  model.train_all_blocks_step_mode=sequential \
  "${COMBO_MILD[@]}"

launch "${GPUS[2]}" xey_struct_global_closure_retry \
  model.train_all_blocks_step_mode=accumulate_streaming_global \
  model.closure_rank_loss_weight=0.05 \
  model.closure_rank_loss_t_power=1.0 \
  model.closure_rank_loss_warmup_epochs=5 \
  model.closure_rank_pairs_per_family=4096

launch "${GPUS[3]}" xey_struct_global_combo_mild_retry \
  model.train_all_blocks_step_mode=accumulate_streaming_global \
  "${COMBO_MILD[@]}"

failed=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[ok] $(date '+%F %T') ${NAMES[$i]}" | tee -a "$LOG_DIR/closure_retry_launcher.log"
    touch "$LOG_DIR/${NAMES[$i]}.done"
  else
    echo "[fail] $(date '+%F %T') ${NAMES[$i]}" | tee -a "$LOG_DIR/closure_retry_launcher.log"
    touch "$LOG_DIR/${NAMES[$i]}.failed"
    failed=1
  fi
done

echo "[done] $(date '+%F %T') rc=${failed} logs=${LOG_DIR}" | tee -a "$LOG_DIR/closure_retry_launcher.log"
exit "$failed"

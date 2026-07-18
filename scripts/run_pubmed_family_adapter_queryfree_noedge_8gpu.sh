#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-$ROOT/logs/logs_family_adapter_queryfree_noedge_8gpu}"
mkdir -p "$LOG_DIR"

COMMON_OVERRIDES=(
  general.wandb=disabled
  general.gpus=1
  train.n_epochs="${N_EPOCHS:-100}"
  train.batch_size=1
  train.save_model=true
  general.run_test_after_train=true
  general.enable_test_sampling=true
  general.enable_val_sampling=false
  general.test_variance=3
  general.test_sampling_metrics_every=1
  general.verbose_sampling=true
  general.log_sampling_triangle_trace=true
  general.log_sampling_posterior_triangle_gap=true
  general.edge_score_structure_diag=true
  general.edge_score_structure_diag_max_negatives=100000
  general.test_sampling_full_steps=false
  'general.sampling_time_schedule=[100,86,71,57,43,29,14,0]'
  general.check_val_every_n_epochs=1
  model.train_family_staged_query_chunk_size="${QUERY_CHUNK_SIZE:-262144}"
  model.train_queryfree_query_state=no_edge
  model.sampling_edge_selection=gumbel_exact_k
  model.sampling_use_reverse_posterior=false
  model.sampling_queryfree_decode=true
  model.sampling_queryfree_query_state=no_edge
  model.sampling_edge_input_residual_scale=0.0
  model.sampling_ranking_intervention_diag=true
  model.sampling_exact_k_connectivity_repair=false
)

launch() {
  local gpu="$1"
  local seed="$2"
  local name="family_adapter_queryfree_noedge_chunk${QUERY_CHUNK_SIZE:-262144}_seed${seed}"
  local log_file="$LOG_DIR/${name}.out"
  echo "[launch] gpu=${gpu} name=${name} seed=${seed}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export CONDA_ENV="${CONDA_ENV:-sparse_block}"
    export CONDA_BASE="${CONDA_BASE:-/data2/lyh/miniconda3}"
    EXPERIMENT=pubmed_family_adapter_queryfree_pw20 bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON_OVERRIDES[@]}" \
      "general.name=${name}" \
      "train.seed=${seed}"
  ) >"$log_file" 2>&1 &
  echo "$!" >"$LOG_DIR/${name}.pid"
}

for gpu in 0 1 2 3 4 5 6 7; do
  launch "$gpu" "$gpu"
done

echo "[launch] all jobs submitted. Logs: $LOG_DIR"
wait
echo "[done] query-free no-edge simultaneous sweep finished. Logs: $LOG_DIR"

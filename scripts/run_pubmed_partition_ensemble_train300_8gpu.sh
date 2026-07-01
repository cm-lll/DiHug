#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="logs/logs_partition_ensemble_train300"
mkdir -p "$LOG_DIR"

COMMON=(
  general.gpus=1
  train.n_epochs=300
  train.batch_size=1
  model.edge_fraction=0.1
  model.use_edge_state_update=false
  model.exist_pos_weight=20
  model.edge_count_loss_weight=0.0
  model.degree_loss_weight=0.0
  model.closure_pos_loss_weight=0.0
  model.train_all_blocks_per_noise=false
  model.train_query_repeats=1
  model.train_partition_ensemble=true
  model.train_partition_ensemble_num_blocks=10
  general.run_test_after_train=true
  general.enable_test_sampling=true
  general.test_variance=3
  general.test_sampling_full_steps=false
  "general.sampling_time_schedule=[100,75,50,25,0]"
  model.sampling_partition_ensemble_metis_weight=0.5
  model.sampling_calibrate_exist_pos_weight=true
  model.sampling_block_marginal_init=true
  model.sampling_block_template_init=false
  model.sampling_use_reverse_posterior=true
  model.sampling_block_family_budget_projection=false
  model.sampling_edge_selection=gumbel_exact_k
  model.sampling_gumbel_temperature=0.01
  model.sampling_exact_k_connectivity_repair=true
)

launch() {
  local gpu="$1"
  local train_weight="$2"
  local sample_mode="$3"
  local seed="$4"
  local tag="$5"
  local name="pubmed_ef010_parttrain_${tag}_seed${seed}_ep300"

  CUDA_VISIBLE_DEVICES="$gpu" CONDA_ENV=sparse_block \
    nohup bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON[@]}" \
      "train.seed=$seed" \
      "model.train_partition_ensemble_metis_weight=$train_weight" \
      "model.sampling_partition_ensemble=$sample_mode" \
      "general.name=$name" \
      > "$LOG_DIR/${name}.out" 2>&1 &
  echo "GPU${gpu}: ${name}, PID=$!"
}

# Four independent seeds for the proposed 0.5/0.5 mechanism.
launch 0 0.5 mean 0 mean
launch 1 0.5 mean 1 mean
launch 2 0.5 mean 2 mean
launch 3 0.5 mean 3 mean

# Two seeds for each single-view training control.
launch 4 1.0 metis_only 0 metis
launch 5 1.0 metis_only 1 metis
launch 6 0.0 random_only 0 random
launch 7 0.0 random_only 1 random

echo "Logs: $ROOT/$LOG_DIR"

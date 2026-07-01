#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="logs/logs_partition_ensemble_long"
mkdir -p "$LOG_DIR"

CKPT="output/2026-06-20/22-43-34-pubmed_ef010_noedge_posterior_pw20_ep300/output/sparse_diffusion/checkpoints/pubmed_ef010_noedge_posterior_pw20_ep300/last.ckpt"
if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
fi

COMMON=(
  general.gpus=1
  "general.test_only=$CKPT"
  general.test_sampling_full_steps=false
  "general.sampling_time_schedule=[100,75,50,25,0]"
  general.enable_test_sampling=true
  general.test_variance=1
  general.verbose_sampling=false
  model.edge_fraction=0.1
  model.use_edge_state_update=false
  model.exist_pos_weight=20
  model.sampling_calibrate_exist_pos_weight=true
  model.sampling_block_marginal_init=true
  model.sampling_block_template_init=false
  model.sampling_use_reverse_posterior=true
  model.sampling_block_family_budget_projection=false
  model.sampling_edge_selection=gumbel_exact_k
  model.sampling_gumbel_temperature=0.01
  model.sampling_exact_k_connectivity_repair=true
  model.sampling_partition_ensemble_metis_weight=0.5
)

run_one() {
  local gpu="$1"
  local mode="$2"
  local seed="$3"
  local name="partens_${mode}_seed${seed}"
  echo "[$(date '+%F %T')] GPU${gpu} start ${name}"
  CUDA_VISIBLE_DEVICES="$gpu" CONDA_ENV=sparse_block \
    bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON[@]}" \
      "train.seed=$seed" \
      "model.sampling_partition_ensemble=$mode" \
      "general.name=$name" \
      > "$LOG_DIR/${name}.out" 2>&1
  echo "[$(date '+%F %T')] GPU${gpu} done  ${name}"
}

worker() {
  local gpu="$1"
  shift
  local spec mode seed
  for spec in "$@"; do
    mode="${spec%%:*}"
    seed="${spec##*:}"
    run_one "$gpu" "$mode" "$seed"
  done
}

# Thirty paired jobs: ten seeds for each view. Jobs are queued per GPU, so the
# experiment continues unattended until every mode/seed has completed.
worker 0 metis_only:0 random_only:0 mean:0 metis_only:8 &
worker 1 metis_only:1 random_only:1 mean:1 metis_only:9 &
worker 2 metis_only:2 random_only:2 mean:2 random_only:8 &
worker 3 metis_only:3 random_only:3 mean:3 random_only:9 &
worker 4 metis_only:4 random_only:4 mean:4 mean:8 &
worker 5 metis_only:5 random_only:5 mean:5 mean:9 &
worker 6 metis_only:6 random_only:6 mean:6 &
worker 7 metis_only:7 random_only:7 mean:7 &

wait
echo "All partition-ensemble jobs finished. Logs: $ROOT/$LOG_DIR"

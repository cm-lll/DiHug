#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="logs/logs_permutation_equivariance"
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
  model.sampling_exact_k_connectivity_repair=false
)

launch() {
  local gpu="$1"
  local seed="$2"
  local permutation_seed="$3"
  local name="$4"

  CUDA_VISIBLE_DEVICES="$gpu" CONDA_ENV=sparse_block \
    nohup bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON[@]}" \
      "train.seed=$seed" \
      "general.fixed_node_type_permutation_seed=$permutation_seed" \
      "general.name=$name" \
      > "$LOG_DIR/${name}.out" 2>&1 &
  echo "GPU${gpu}: ${name}, PID=$!"
}

# Paired distribution test. GPU0/4, 1/5, 2/6, 3/7 share model/sampling seeds.
launch 0 0 null permtest_original_seed0
launch 1 1 null permtest_original_seed1
launch 2 2 null permtest_original_seed2
launch 3 3 null permtest_original_seed3
launch 4 0 1000 permtest_permuted_seed0
launch 5 1 1001 permtest_permuted_seed1
launch 6 2 1002 permtest_permuted_seed2
launch 7 3 1003 permtest_permuted_seed3

echo "Logs: $ROOT/$LOG_DIR"

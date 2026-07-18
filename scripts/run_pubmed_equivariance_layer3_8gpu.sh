#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="logs/logs_equivariance_layer3"
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
  general.equivariance_initial_noise_mode=mapped_reference
  model.edge_fraction=0.1
  model.use_edge_state_update=false
  model.exist_pos_weight=20
  model.sampling_calibrate_exist_pos_weight=true
  model.sampling_block_marginal_init=true
  model.sampling_block_template_init=false
  model.sampling_use_reverse_posterior=true
  model.sampling_block_family_budget_projection=false
  model.sampling_exact_k_connectivity_repair=false
)

launch() {
  local gpu="$1"
  local permutation_seed="$2"
  local partition_mode="$3"
  local selection="$4"
  local gumbel_mode="$5"
  local gumbel_offset="$6"
  local name="$7"

  CUDA_VISIBLE_DEVICES="$gpu" CONDA_ENV=sparse_block \
    nohup bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON[@]}" \
      train.seed=0 \
      "general.fixed_node_type_permutation_seed=$permutation_seed" \
      "general.equivariance_partition_mode=$partition_mode" \
      "general.equivariance_gumbel_mode=$gumbel_mode" \
      "general.equivariance_gumbel_seed_offset=$gumbel_offset" \
      "model.sampling_edge_selection=$selection" \
      model.sampling_gumbel_temperature=0.01 \
      "general.name=$name" \
      > "$LOG_DIR/${name}.out" 2>&1 &
  echo "GPU${gpu}: ${name}, PID=$!"
}

# A: mapped pseudo-blocks + mapped common random numbers. The pair should match.
launch 0 null mapped_reference gumbel_exact_k mapped_reference 0 l3_A_original_mappedblock_mappednoise
launch 1 3000 mapped_reference gumbel_exact_k mapped_reference 0 l3_A_permuted_mappedblock_mappednoise

# B: mapped pseudo-blocks + independent Gumbel streams. This measures sampling variance.
launch 2 null mapped_reference gumbel_exact_k independent 0 l3_B_original_mappedblock_indnoise
launch 3 3000 mapped_reference gumbel_exact_k independent 104729 l3_B_permuted_mappedblock_indnoise

# C: rebuilt/current pseudo-blocks + deterministic exact-K. This isolates partitioning.
launch 4 null current deterministic_exact_k position 0 l3_C_original_rebuilt_deterministic
launch 5 3000 current deterministic_exact_k position 0 l3_C_permuted_rebuilt_deterministic

# D: rebuilt/current pseudo-blocks + mapped common random numbers. This measures
# partition/randomness interaction without changing the semantic z_T.
launch 6 null current gumbel_exact_k mapped_reference 0 l3_D_original_rebuilt_mappednoise
launch 7 3000 current gumbel_exact_k mapped_reference 0 l3_D_permuted_rebuilt_mappednoise

echo "Logs: $ROOT/$LOG_DIR"

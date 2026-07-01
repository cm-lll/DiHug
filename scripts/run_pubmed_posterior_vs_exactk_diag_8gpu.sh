#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="logs/logs_posterior_vs_exactk_diag"
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
  general.sampling_skip=25
  general.enable_test_sampling=true
  general.test_variance=1
  general.verbose_sampling=true
  model.edge_fraction=0.1
  model.use_edge_state_update=false
  model.exist_pos_weight=20
  model.sampling_calibrate_exist_pos_weight=true
  model.sampling_block_marginal_init=true
  model.sampling_block_template_init=false
  model.sampling_use_reverse_posterior=true
  model.sampling_reverse_posterior_stochastic_steps=true
  model.sampling_block_family_budget_projection=false
  model.sampling_exact_k_connectivity_repair=false
)

launch() {
  local gpu="$1"
  local mode="$2"
  local seed="$3"
  local name="$4"

  CUDA_VISIBLE_DEVICES="$gpu" CONDA_ENV=sparse_block \
    nohup bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON[@]}" \
      "model.sampling_edge_selection=$mode" \
      model.sampling_gumbel_temperature=0.01 \
      "train.seed=$seed" \
      "general.name=$name" \
      > "$LOG_DIR/${name}.out" 2>&1 &

  echo "GPU${gpu}: ${name}, PID=$!"
}

# Paired seeds: GPU0/4, GPU1/5, GPU2/6, GPU3/7 use identical initialization
# and query randomness, differing only in the final selection mechanism.
launch 0 bernoulli      0 diag_posterior_bernoulli_seed0_4step
launch 1 bernoulli      1 diag_posterior_bernoulli_seed1_4step
launch 2 bernoulli      2 diag_posterior_bernoulli_seed2_4step
launch 3 bernoulli      3 diag_posterior_bernoulli_seed3_4step
launch 4 gumbel_exact_k 0 diag_gumbel_exactk_seed0_4step
launch 5 gumbel_exact_k 1 diag_gumbel_exactk_seed1_4step
launch 6 gumbel_exact_k 2 diag_gumbel_exactk_seed2_4step
launch 7 gumbel_exact_k 3 diag_gumbel_exactk_seed3_4step

echo "Logs: $ROOT/$LOG_DIR"

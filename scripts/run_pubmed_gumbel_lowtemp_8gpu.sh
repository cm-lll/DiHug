#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs_gumbel_crn

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
  general.verbose_sampling=false
  model.edge_fraction=0.1
  model.use_edge_state_update=false
  model.exist_pos_weight=20
  model.sampling_calibrate_exist_pos_weight=true
  model.sampling_block_marginal_init=true
  model.sampling_block_template_init=false
  model.sampling_use_reverse_posterior=true
  model.sampling_block_family_budget_projection=false
)

launch() {
  local gpu="$1"
  local mode="$2"
  local temperature="$3"
  local seed="$4"
  local name="$5"

  CUDA_VISIBLE_DEVICES="$gpu" CONDA_ENV=sparse_block \
    nohup bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON[@]}" \
      "model.sampling_edge_selection=$mode" \
      "model.sampling_gumbel_temperature=$temperature" \
      "train.seed=$seed" \
      "general.name=$name" \
      > "logs_gumbel_crn/${name}.out" 2>&1 &

  echo "GPU${gpu}: ${name}, PID=$!"
}

launch 0 gumbel_exact_k         0.10 0 crn_gumbel_t010_seed0_4step
launch 1 gumbel_exact_k         0.05 0 crn_gumbel_t005_seed0_4step
launch 2 gumbel_exact_k         0.02 0 crn_gumbel_t002_seed0_4step
launch 3 gumbel_exact_k         0.01 0 crn_gumbel_t001_seed0_4step
launch 4 deterministic_exact_k  1.00 0 crn_deterministic_seed0_4step
launch 5 gumbel_exact_k         0.05 1 crn_gumbel_t005_seed1_4step
launch 6 gumbel_exact_k         0.02 1 crn_gumbel_t002_seed1_4step
launch 7 deterministic_exact_k  1.00 1 crn_deterministic_seed1_4step

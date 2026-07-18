#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="logs/logs_exactk_repair"
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
  model.sampling_exact_k_repair_max_swaps=0
)

launch() {
  local gpu="$1"
  local skip="$2"
  local seed="$3"
  local repair="$4"
  local name="$5"

  CUDA_VISIBLE_DEVICES="$gpu" CONDA_ENV=sparse_block \
    nohup bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON[@]}" \
      "general.sampling_skip=$skip" \
      "train.seed=$seed" \
      "model.sampling_exact_k_connectivity_repair=$repair" \
      "general.name=$name" \
      > "$LOG_DIR/${name}.out" 2>&1 &

  echo "GPU${gpu}: ${name}, PID=$!"
}

# Three independent seeds for each repaired setting.
launch 0 25 0 true exactk_repair_4step_seed0
launch 1 25 1 true exactk_repair_4step_seed1
launch 2 25 2 true exactk_repair_4step_seed2
launch 3 15 0 true exactk_repair_7step_seed0
launch 4 15 1 true exactk_repair_7step_seed1
launch 5 15 2 true exactk_repair_7step_seed2

# Same-code baselines, useful for exact paired before/after comparisons.
launch 6 25 0 false exactk_norepair_4step_seed0
launch 7 15 0 false exactk_norepair_7step_seed0

echo "Logs: $ROOT/$LOG_DIR"

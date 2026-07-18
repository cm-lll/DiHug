#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="logs/logs_nonuniform_schedule"
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
  general.test_sampling_metrics_every=1
  general.verbose_sampling=true
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
  local schedule="$2"
  local seed="$3"
  local repair="$4"
  local name="$5"

  CUDA_VISIBLE_DEVICES="$gpu" CONDA_ENV=sparse_block \
    nohup bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON[@]}" \
      "general.sampling_time_schedule=$schedule" \
      "train.seed=$seed" \
      "model.sampling_exact_k_connectivity_repair=$repair" \
      "general.name=$name" \
      > "$LOG_DIR/${name}.out" 2>&1 &

  echo "GPU${gpu}: ${name}, PID=$!"
}

# B: strong 100->75 start followed by uniform 15-step refinements.
launch 0 '[100,75,60,45,30,15,0]' 0 true nonuniform_B_seed0
launch 1 '[100,75,60,45,30,15,0]' 1 true nonuniform_B_seed1
launch 2 '[100,75,60,45,30,15,0]' 2 true nonuniform_B_seed2

# C: preserve the original 4-step path and add one conservative 15->0 finish.
launch 3 '[100,75,50,25,15,0]' 0 true nonuniform_C_seed0
launch 4 '[100,75,50,25,15,0]' 1 true nonuniform_C_seed1
launch 5 '[100,75,50,25,15,0]' 2 true nonuniform_C_seed2

# A: original 4-step paired baseline, with and without final connectivity repair.
launch 6 '[100,75,50,25,0]' 0 true  baseline_A_repair_seed0
launch 7 '[100,75,50,25,0]' 0 false baseline_A_norepair_seed0

echo "Logs: $ROOT/$LOG_DIR"

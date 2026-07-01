#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="logs/logs_exactk_path_diag"
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
  model.sampling_exact_k_connectivity_repair=false
)

launch() {
  local gpu="$1"
  local skip="$2"
  local seed="$3"
  local name="$4"

  CUDA_VISIBLE_DEVICES="$gpu" CONDA_ENV=sparse_block \
    nohup bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON[@]}" \
      "general.sampling_skip=$skip" \
      "train.seed=$seed" \
      "general.name=$name" \
      > "$LOG_DIR/${name}.out" 2>&1 &

  echo "GPU${gpu}: ${name}, PID=$!"
}

# Paired seeds. GPU0/4, 1/5, 2/6, 3/7 differ only in path length.
launch 0 15 0 pathdiag_7step_seed0
launch 1 15 1 pathdiag_7step_seed1
launch 2 15 2 pathdiag_7step_seed2
launch 3 15 3 pathdiag_7step_seed3
launch 4 25 0 pathdiag_4step_seed0
launch 5 25 1 pathdiag_4step_seed1
launch 6 25 2 pathdiag_4step_seed2
launch 7 25 3 pathdiag_4step_seed3

echo "Logs: $ROOT/$LOG_DIR"

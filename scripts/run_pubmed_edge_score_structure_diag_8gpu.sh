#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="logs/logs_edge_score_structure_diag"
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
  general.test_sampling_metrics_every=0
  general.verbose_sampling=false
  general.edge_score_structure_diag=true
  general.edge_score_structure_diag_max_negatives=100000
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

for gpu in 0 1 2 3 4 5 6 7; do
  name="edge_score_structure_diag_seed${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" CONDA_ENV=sparse_block \
    nohup bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON[@]}" \
      "train.seed=$gpu" \
      "general.name=$name" \
      > "$LOG_DIR/${name}.out" 2>&1 &
  echo "GPU${gpu}: ${name}, PID=$!"
done

echo "Logs: $ROOT/$LOG_DIR"

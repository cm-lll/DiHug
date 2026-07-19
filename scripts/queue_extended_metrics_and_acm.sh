#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs/paper_extended_metrics_main16_0719 logs/acm_target_free_queue_0719

while tmux has-session -t dihug_extended_metrics_main16 2>/dev/null \
  || tmux has-session -t dihug_pubmed_extended_metrics 2>/dev/null; do
  echo "[QUEUE] waiting for canonical IMDB/PubMed regeneration: $(date -Is)"
  sleep 60
done

graph_count="$(find results/paper_extended_metrics_main16_0719 -name final_graph.json 2>/dev/null | wc -l)"
echo "[QUEUE] canonical final graphs found: $graph_count"
if [[ "$graph_count" -gt 0 ]]; then
  eval "$(conda shell.bash hook)"
  conda activate sparse_block
  export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  python scripts/generate_extended_structure_tables.py \
    --results-root results/paper_extended_metrics_main16_0719 \
    --output-dir results/paper_extended_metrics_main16_0719/extended_structure \
    --dataset imdb --dataset pubmed \
    > logs/paper_extended_metrics_main16_0719/extended_structure.log 2>&1
fi

echo "[QUEUE] starting ACM smoke"
SMOKE_ONLY=1 GPU=0 \
  OUT_ROOT=results/acm_target_free_smoke_0719 \
  LOG_ROOT=logs/acm_target_free_smoke_0719 \
  bash scripts/run_acm_target_free_main.sh

echo "[QUEUE] ACM smoke passed; starting the formal 8-seed protocol"
OUT_ROOT=results/acm_target_free_main_0719 \
LOG_ROOT=logs/acm_target_free_main_0719 \
GPU_LIST="0 1 2 3 4 5 6 7" \
  bash scripts/run_acm_target_free_8gpu.sh

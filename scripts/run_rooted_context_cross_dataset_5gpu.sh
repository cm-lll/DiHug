#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-results/rooted_context_cross_dataset}"
LOG_ROOT="${LOG_ROOT:-logs/rooted_context_cross_dataset}"
mkdir -p "$ROOT" "$LOG_ROOT"

DATASETS=(
  "imdb|data/IMDB_movie_subgraphs/subgraph_000|results/imdb_exchangeable_compact_s0_3/model/mixed_edit_diffusion.pt"
  "dblp|data/DBLP_four_area_subgraphs/subgraph_000|results/hgen_official_fullgraph_pilot_0716/dblp/seed0/model/mixed_edit_diffusion.pt"
  "syn100|data/HGEN_syn_100_subgraphs/subgraph_000|results/hgen_official_fullgraph_pilot_0716/syn100/seed0/model/mixed_edit_diffusion.pt"
  "syn200|data/HGEN_syn_200_subgraphs/subgraph_000|results/hgen_official_fullgraph_pilot_0716/syn200/seed0/model/mixed_edit_diffusion.pt"
  "syn500|data/HGEN_syn_500_subgraphs/subgraph_000|results/hgen_official_fullgraph_pilot_0716/syn500/seed0/model/mixed_edit_diffusion.pt"
)

pids=()
for gpu in "${!DATASETS[@]}"; do
  IFS='|' read -r name dataset checkpoint <<< "${DATASETS[$gpu]}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
    python scripts/run_rooted_context_residual_gate.py \
      --dataset-dir "$dataset" \
      --legacy-checkpoint "$checkpoint" \
      --output-dir "$ROOT/$name" \
      --device cuda \
      --seed 0 \
      --trajectories 4 \
      --examples-per-trajectory 16 \
      --switches-per-example 64 \
      --candidates 16 \
      --closure-fraction 1.0 \
      --epochs 15 \
      --batch-size 2 \
      --hidden-dim 64 \
      --heads 4 \
      --layers 2 \
      > "$LOG_ROOT/$name.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"

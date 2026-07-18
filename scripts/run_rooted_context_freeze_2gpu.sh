#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-results/rooted_context_freeze}"
LOG_ROOT="${LOG_ROOT:-logs/rooted_context_freeze}"
mkdir -p "$ROOT" "$LOG_ROOT"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python scripts/run_rooted_context_residual_gate.py \
    --dataset-dir data/PubMed_baseline_subgraphs/subgraph_000 \
    --legacy-checkpoint results/pubmed_role_corruption_gate_0716/f100_all/model/mixed_edit_diffusion.pt \
    --output-dir "$ROOT/pubmed" --device cuda --seed 0 \
    --trajectories 4 --examples-per-trajectory 16 --switches-per-example 64 \
    --candidates 16 --closure-fraction 1.0 --epochs 15 --batch-size 2 \
    --hidden-dim 64 --heads 4 --layers 2 \
    > "$LOG_ROOT/pubmed.log" 2>&1 &
pubmed_pid=$!

CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python scripts/run_rooted_context_residual_gate.py \
    --dataset-dir data/IMDB_movie_subgraphs/subgraph_000 \
    --legacy-checkpoint results/imdb_exchangeable_compact_s0_3/model/mixed_edit_diffusion.pt \
    --output-dir "$ROOT/imdb" --device cuda --seed 0 \
    --trajectories 4 --examples-per-trajectory 16 --switches-per-example 64 \
    --candidates 16 --closure-fraction 1.0 --epochs 15 --batch-size 2 \
    --hidden-dim 64 --heads 4 --layers 2 \
    > "$LOG_ROOT/imdb.log" 2>&1 &
imdb_pid=$!

wait "$pubmed_pid"
wait "$imdb_pid"

#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROOT
export DATASET_NAME=imdb
export DATASET_DIR="${DATASET_DIR:-$ROOT/data/IMDB_movie_subgraphs/subgraph_000}"
export STRUCTURE_CKPT="${STRUCTURE_CKPT:-$ROOT/results/imdb_switch_structure_seed0/structure_checkpoint.json}"
export OUT_ROOT="${OUT_ROOT:-$ROOT/results/imdb_switch_paper_8gpu}"
export LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imdb_switch_paper_8gpu}"

exec bash "$ROOT/scripts/run_switch_paper_matrix_8gpu.sh"

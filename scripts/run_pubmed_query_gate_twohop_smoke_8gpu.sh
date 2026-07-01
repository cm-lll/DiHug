#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_query_gate_twohop_smoke}"
mkdir -p "$LOG_DIR"
EXTRA_OVERRIDES=("$@")

run_one() {
  local gpu="$1"
  local name="$2"
  shift 2

  echo "[launch] gpu=${gpu} name=${name}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  CONDA_ENV="${CONDA_ENV:-sparse_block}" \
  EXPERIMENT=pubmed_query_gate_twohop_pw20 \
    bash scripts/run_pubmed_baseline_single.sh \
      general.gpus=1 \
      general.name="$name" \
      general.run_test_after_train=false \
      general.enable_test_sampling=false \
      general.check_val_every_n_epochs=100 \
      train.n_epochs=1 \
      model.edge_fraction="${EDGE_FRACTION:-0.1}" \
      "$@" \
      "${EXTRA_OVERRIDES[@]}" \
      >"$LOG_DIR/${name}.out" 2>&1 &
}

# Mechanism ablations.
run_one 0 smoke_qg2h_control \
  model.use_query_context_gate=false \
  model.use_two_hop_structure=false

run_one 1 smoke_qg2h_gate_only \
  model.use_query_context_gate=true \
  model.query_context_gate_init=0.2 \
  model.use_two_hop_structure=false

run_one 2 smoke_qg2h_twohop_only_h64 \
  model.use_query_context_gate=false \
  model.use_two_hop_structure=true \
  model.two_hop_structure_hidden_dim=64

run_one 3 smoke_qg2h_default_g020_h64 \
  model.use_query_context_gate=true \
  model.query_context_gate_init=0.2 \
  model.use_two_hop_structure=true \
  model.two_hop_structure_hidden_dim=64

# Gate initialization sweep.
run_one 4 smoke_qg2h_g005_h64 \
  model.use_query_context_gate=true \
  model.query_context_gate_init=0.05 \
  model.use_two_hop_structure=true \
  model.two_hop_structure_hidden_dim=64

run_one 5 smoke_qg2h_g050_h64 \
  model.use_query_context_gate=true \
  model.query_context_gate_init=0.5 \
  model.use_two_hop_structure=true \
  model.two_hop_structure_hidden_dim=64

# Structural residual capacity sweep.
run_one 6 smoke_qg2h_g020_h32 \
  model.use_query_context_gate=true \
  model.query_context_gate_init=0.2 \
  model.use_two_hop_structure=true \
  model.two_hop_structure_hidden_dim=32

run_one 7 smoke_qg2h_g020_h128 \
  model.use_query_context_gate=true \
  model.query_context_gate_init=0.2 \
  model.use_two_hop_structure=true \
  model.two_hop_structure_hidden_dim=128

wait
echo "[done] all smoke runs finished; logs: $LOG_DIR"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_ROOT="${RUN_ROOT:-output/2026-06-22/18-26-09}"
LOG_DIR="${LOG_DIR:-logs/logs_query_gate_twohop_checkpoint_test}"
mkdir -p "$LOG_DIR"

launch() {
  local gpu="$1"
  local name="$2"
  shift 2
  local ckpt="${RUN_ROOT}-${name}/output/sparse_diffusion/checkpoints/${name}/last.ckpt"

  if [[ ! -f "$ckpt" ]]; then
    echo "Missing checkpoint: $ckpt" >&2
    return 1
  fi

  echo "[test] gpu=${gpu} name=${name}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  CONDA_ENV="${CONDA_ENV:-sparse_block}" \
  EXPERIMENT=pubmed_query_gate_twohop_pw20 \
    bash scripts/run_pubmed_baseline_single.sh \
      general.gpus=1 \
      "general.test_only=$ckpt" \
      "general.name=test_${name}" \
      general.run_test_after_train=false \
      general.enable_test_sampling=true \
      general.test_variance=3 \
      general.test_sampling_full_steps=false \
      general.sampling_skip=25 \
      model.edge_fraction=0.1 \
      "$@" \
      >"$LOG_DIR/${name}.out" 2>&1 &
}

if [[ "${SKIP_CONTROL:-0}" != "1" ]]; then
  launch 0 smoke_qg2h_control \
    model.use_query_context_gate=false \
    model.use_two_hop_structure=false
fi

launch 1 smoke_qg2h_gate_only \
  model.use_query_context_gate=true \
  model.query_context_gate_init=0.2 \
  model.use_two_hop_structure=false

launch 2 smoke_qg2h_twohop_only_h64 \
  model.use_query_context_gate=false \
  model.use_two_hop_structure=true \
  model.two_hop_structure_hidden_dim=64

launch 3 smoke_qg2h_default_g020_h64 \
  model.use_query_context_gate=true \
  model.query_context_gate_init=0.2 \
  model.use_two_hop_structure=true \
  model.two_hop_structure_hidden_dim=64

launch 4 smoke_qg2h_g005_h64 \
  model.use_query_context_gate=true \
  model.query_context_gate_init=0.05 \
  model.use_two_hop_structure=true \
  model.two_hop_structure_hidden_dim=64

launch 5 smoke_qg2h_g050_h64 \
  model.use_query_context_gate=true \
  model.query_context_gate_init=0.5 \
  model.use_two_hop_structure=true \
  model.two_hop_structure_hidden_dim=64

launch 6 smoke_qg2h_g020_h32 \
  model.use_query_context_gate=true \
  model.query_context_gate_init=0.2 \
  model.use_two_hop_structure=true \
  model.two_hop_structure_hidden_dim=32

launch 7 smoke_qg2h_g020_h128 \
  model.use_query_context_gate=true \
  model.query_context_gate_init=0.2 \
  model.use_two_hop_structure=true \
  model.two_hop_structure_hidden_dim=128

wait
echo "Tests finished; logs: $LOG_DIR"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_sampling_steps_posterior_fixed_seed1}"
RUN_NAME="${RUN_NAME:-twohop_fixed_a0p5_seed1_ep100}"
mkdir -p "$LOG_DIR"

CKPT="$(find output/2026-06-23 \
  -path "*/output/sparse_diffusion/checkpoints/${RUN_NAME}/last.ckpt" \
  -type f -print -quit)"
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "Missing checkpoint for ${RUN_NAME}" >&2
  exit 1
fi

schedule_for_steps() {
  case "$1" in
    1)  echo '[100,0]' ;;
    2)  echo '[100,50,0]' ;;
    4)  echo '[100,75,50,25,0]' ;;
    7)  echo '[100,86,71,57,43,29,14,0]' ;;
    10) echo '[100,90,80,70,60,50,40,30,20,10,0]' ;;
    20) echo '[100,95,90,85,80,75,70,65,60,55,50,45,40,35,30,25,20,15,10,5,0]' ;;
    *)
      echo "Unsupported transition count: $1" >&2
      return 1
      ;;
  esac
}

run_case() {
  local gpu="$1"
  local steps="$2"
  local posterior="$3"
  local schedule
  local profile

  schedule="$(schedule_for_steps "$steps")"
  profile="steps${steps}_posterior_${posterior}"
  echo "[test] gpu=${gpu} profile=${profile} schedule=${schedule}"

  CUDA_VISIBLE_DEVICES="$gpu" \
  CONDA_ENV="${CONDA_ENV:-sparse_block}" \
  EXPERIMENT=pubmed_query_gate_twohop_pw20 \
    bash scripts/run_pubmed_baseline_single.sh \
      general.gpus=1 \
      "general.test_only=$CKPT" \
      "general.name=test_${RUN_NAME}_${profile}" \
      general.run_test_after_train=false \
      general.enable_test_sampling=true \
      general.test_variance=1 \
      'general.test_sampling_seeds=[0]' \
      general.test_sampling_full_steps=false \
      "general.sampling_time_schedule=${schedule}" \
      general.test_sampling_metrics_every=0 \
      model.edge_fraction=0.1 \
      model.use_query_context_gate=false \
      model.use_two_hop_structure=true \
      model.two_hop_structure_hidden_dim=64 \
      model.two_hop_structure_scale=0.5 \
      model.two_hop_structure_schedule=fixed \
      "model.sampling_use_reverse_posterior=${posterior}" \
      model.sampling_reverse_posterior_mix_weights=null \
      >"$LOG_DIR/${profile}.out" 2>&1
}

# Keep one process per GPU. Short cases are queued behind the 7/4-step runs;
# the two 20-step cases occupy dedicated GPUs because they dominate runtime.
worker_0() { run_case 0 20 false; }
worker_1() { run_case 1 20 true; }
worker_2() { run_case 2 10 false; }
worker_3() { run_case 3 10 true; }
worker_4() { run_case 4 7 false; run_case 4 1 false; }
worker_5() { run_case 5 7 true;  run_case 5 1 true; }
worker_6() { run_case 6 4 false; run_case 6 2 false; }
worker_7() { run_case 7 4 true;  run_case 7 2 true; }

worker_0 &
worker_1 &
worker_2 &
worker_3 &
worker_4 &
worker_5 &
worker_6 &
worker_7 &

wait
echo "[done] sampling-step posterior sweep finished; logs: $LOG_DIR"

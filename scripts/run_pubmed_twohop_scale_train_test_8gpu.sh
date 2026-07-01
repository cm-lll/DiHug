#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_twohop_scale_train_test_100ep}"
EPOCHS="${EPOCHS:-100}"
mkdir -p "$LOG_DIR"

launch() {
  local gpu="$1"
  local scale="$2"
  local seed="$3"
  local scale_tag="${scale/./p}"
  local name="twohop_only_a${scale_tag}_seed${seed}_ep${EPOCHS}"

  echo "[launch] gpu=${gpu} scale=${scale} seed=${seed} name=${name}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  CONDA_ENV="${CONDA_ENV:-sparse_block}" \
  EXPERIMENT=pubmed_query_gate_twohop_pw20 \
    bash scripts/run_pubmed_baseline_single.sh \
      general.gpus=1 \
      general.name="$name" \
      general.run_test_after_train=true \
      general.enable_test_sampling=true \
      general.test_variance=3 \
      general.test_sampling_full_steps=false \
      general.sampling_skip=25 \
      general.check_val_every_n_epochs=50 \
      train.n_epochs="$EPOCHS" \
      train.seed="$seed" \
      model.edge_fraction=0.1 \
      model.use_query_context_gate=false \
      model.use_two_hop_structure=true \
      model.two_hop_structure_hidden_dim=64 \
      model.two_hop_structure_scale="$scale" \
      >"$LOG_DIR/${name}.out" 2>&1 &
}

launch 0 0.10 0
launch 1 0.10 1
launch 2 0.25 0
launch 3 0.25 1
launch 4 0.50 0
launch 5 0.50 1
launch 6 1.00 0
launch 7 1.00 1

wait
echo "[done] all training and sampling tests finished; logs: $LOG_DIR"

#!/usr/bin/env bash
# From-scratch ACM legacy training followed by a paired rooted-context gate.
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

DATASET="${DATASET:-data/ACM_subgraphs/subgraph_000}"
OUT_ROOT="${OUT_ROOT:-results/acm_context_paired_gate}"
LOG_ROOT="${LOG_ROOT:-logs/acm_context_paired_gate}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
KERNEL="${KERNEL:-rs}"
TRAIN_TRAJECTORIES="${TRAIN_TRAJECTORIES:-2}"
TRAIN_FORWARD_STEPS="${TRAIN_FORWARD_STEPS:-2000}"
TRAIN_RECORD_EVERY="${TRAIN_RECORD_EVERY:-20}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-5}"
CONTEXT_TRAJECTORIES="${CONTEXT_TRAJECTORIES:-4}"
CONTEXT_EXAMPLES_PER_TRAJECTORY="${CONTEXT_EXAMPLES_PER_TRAJECTORY:-12}"
CONTEXT_SWITCHES_PER_EXAMPLE="${CONTEXT_SWITCHES_PER_EXAMPLE:-256}"
CONTEXT_STATE_PROTOCOL="${CONTEXT_STATE_PROTOCOL:-legacy_switch_chain}"
ROLE_RANDOMIZATION_STEPS_PER_EDGE="${ROLE_RANDOMIZATION_STEPS_PER_EDGE:-2}"
ROLE_CORRUPTION_FRACTION="${ROLE_CORRUPTION_FRACTION:-1.0}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

MODEL_DIR="${OUT_ROOT}/legacy_model"
CONTEXT_DIR="${OUT_ROOT}/context"
if [[ ! -f "${MODEL_DIR}/mixed_edit_diffusion.pt" ]]; then
  echo "[ACM-LEGACY-TRAIN] trajectories=${TRAIN_TRAJECTORIES} forward=${TRAIN_FORWARD_STEPS} epochs=${TRAIN_EPOCHS}" \
    | tee "${LOG_ROOT}/driver.log"
  CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/run_mixed_edit_diffusion.py \
    --mode train --denoiser listwise --kernel "${KERNEL}" \
    --dataset-dir "${DATASET}" --output-dir "${MODEL_DIR}" --device cuda \
    --seed "${SEED}" --trajectories "${TRAIN_TRAJECTORIES}" \
    --forward-steps "${TRAIN_FORWARD_STEPS}" --record-every "${TRAIN_RECORD_EVERY}" \
    --train-candidates 16 --feature-profile typed_closure_compact_v2 \
    --epochs "${TRAIN_EPOCHS}" --batch-size 64 --hidden-dim 256 --layers 3 \
    --training-objective inverse_ce --fixed-t-trials 16 \
    --role-corruption-fraction "${ROLE_CORRUPTION_FRACTION}" \
    --role-correlation-strengths "0,0.25,0.5,0.75,1" \
    --role-prior-components 64 \
    --role-prior-randomization-steps-per-edge "${ROLE_RANDOMIZATION_STEPS_PER_EDGE}" \
    > "${LOG_ROOT}/legacy_train.log" 2>&1
fi

echo "[ACM-CONTEXT-TRAIN] epochs=${CONTEXT_EPOCHS}" | tee -a "${LOG_ROOT}/driver.log"
CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/run_rooted_context_residual_gate.py \
  --dataset-dir "${DATASET}" \
  --legacy-checkpoint "${MODEL_DIR}/mixed_edit_diffusion.pt" \
  --output-dir "${CONTEXT_DIR}" --device cuda --seed "${SEED}" \
  --trajectories "${CONTEXT_TRAJECTORIES}" \
  --examples-per-trajectory "${CONTEXT_EXAMPLES_PER_TRAJECTORY}" \
  --switches-per-example "${CONTEXT_SWITCHES_PER_EXAMPLE}" \
  --candidates 16 --closure-fraction 1.0 \
  --epochs "${CONTEXT_EPOCHS}" --batch-size 1 \
  --hidden-dim 64 --heads 4 --layers 2 \
  --state-protocol "${CONTEXT_STATE_PROTOCOL}" \
  > "${LOG_ROOT}/context_train.log" 2>&1

python - "${CONTEXT_DIR}/gate_report.json" <<'PY' | tee -a "${LOG_ROOT}/driver.log"
import json, sys
r = json.load(open(sys.argv[1]))
print(
    "[ACM-CONTEXT-GATE] "
    f"base_utility={r['base_selected_utility']:.6f} "
    f"context_utility={r['context_selected_utility']:.6f} "
    f"gain={r['utility_gain']:.6f} non_regression={r['non_regression']:.6f}"
)
PY
echo "[ACM-CONTEXT-DONE] ${CONTEXT_DIR}/gate_report.json" | tee -a "${LOG_ROOT}/driver.log"

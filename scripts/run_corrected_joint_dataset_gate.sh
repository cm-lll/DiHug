#!/usr/bin/env bash
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

TAG="${TAG:-imdb_role_query_joint_0719}"
OUT_ROOT="${OUT_ROOT:-results/$TAG}"
LOG_ROOT="${LOG_ROOT:-logs/$TAG}"
DATA="${DATA:-data/IMDB_movie_subgraphs/subgraph_000}"
SEED="${SEED:-37}"
PREP_GPU="${PREP_GPU:-0}"
SAMPLE_BASE_GPU="${SAMPLE_BASE_GPU:-1}"
SAMPLE_JOINT_GPU="${SAMPLE_JOINT_GPU:-2}"
SERIAL_SAMPLING="${SERIAL_SAMPLING:-0}"
QUERY_EDGES_PER_FAMILY="${QUERY_EDGES_PER_FAMILY:-64}"
TOTAL_EDITS="${TOTAL_EDITS:-10000}"
OUTER_STEPS="${OUTER_STEPS:-100}"
CONTEXT_REFRESH_STEPS="${CONTEXT_REFRESH_STEPS:-20}"
ROLE_WEIGHT="${ROLE_WEIGHT:-0.05}"
ROLE_GAIN_SCALE="${ROLE_GAIN_SCALE:-10}"
ROLE_TEACHER_WEIGHT="${ROLE_TEACHER_WEIGHT:-0.5}"
RESIDUAL_MIN_UTILITY_GAIN="${RESIDUAL_MIN_UTILITY_GAIN:-0.05}"
TARGETED_FRACTION="${TARGETED_FRACTION:-0.5}"
CLOSURE_FRACTION="${CLOSURE_FRACTION:-0.5}"

if (( TOTAL_EDITS % OUTER_STEPS != 0 )); then
  echo "TOTAL_EDITS must be divisible by OUTER_STEPS" >&2
  exit 2
fi
EDITS_PER_OUTER=$((TOTAL_EDITS / OUTER_STEPS))
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/driver.log"

CACHE="$OUT_ROOT/aligned_role_query_dataset.pt"
BASE="$OUT_ROOT/paired_explicit_mlp.pt"
JOINT_DIR="$OUT_ROOT/role_query_joint"
JOINT="$JOINT_DIR/aligned_context_edit.pt"
PREP_SOURCE_ROOT="${PREP_SOURCE_ROOT:-}"

if [[ -n "$PREP_SOURCE_ROOT" ]]; then
  CACHE="$PREP_SOURCE_ROOT/aligned_role_query_dataset.pt"
  BASE="$PREP_SOURCE_ROOT/paired_explicit_mlp.pt"
  if [[ ! -f "$CACHE" || ! -f "$BASE" ]]; then
    echo "missing reusable prepare artifacts under $PREP_SOURCE_ROOT" >&2
    exit 2
  fi
  echo "[REUSE-PREPARE] cache=$CACHE checkpoint=$BASE" \
    | tee -a "$LOG_ROOT/driver.log"
else
  echo "[PREPARE] closed-form trajectories and paired explicit MLP" \
    | tee -a "$LOG_ROOT/driver.log"
  CUDA_VISIBLE_DEVICES="$PREP_GPU" python -u scripts/run_aligned_context_edit_gate.py \
    --mode prepare --dataset-dir "$DATA" --output-dir "$OUT_ROOT" \
    --dataset-cache "$CACHE" --seed "$SEED" --device cuda \
    --trajectories "${TRAJECTORIES:-16}" \
    --forward-steps "${FORWARD_STEPS:-1000}" \
    --record-every "${RECORD_EVERY:-10}" \
    --candidates "${CANDIDATES:-16}" \
    --feature-profile typed_closure_compact_v2 \
    --supervision rooted_utility --forward-mode closed_form \
    --candidate-mode unified \
    --targeted-role-candidate-fraction "$TARGETED_FRACTION" \
    --closure-candidate-fraction "$CLOSURE_FRACTION" \
    --base-epochs "${BASE_EPOCHS:-100}" \
    --role-conditioned-hgt --query-edges-per-family "$QUERY_EDGES_PER_FAMILY" \
    > "$LOG_ROOT/prepare.log" 2>&1
fi

echo "[JOINT-TRAIN] HGT and MLP unfrozen from epoch 0 under role-aware utility" \
  | tee -a "$LOG_ROOT/driver.log"
CUDA_VISIBLE_DEVICES="$PREP_GPU" python -u scripts/run_aligned_context_edit_gate.py \
  --mode train-context --dataset-dir "$DATA" --output-dir "$JOINT_DIR" \
  --dataset-cache "$CACHE" --base-checkpoint "$BASE" \
  --seed "$SEED" --device cuda --epochs "${JOINT_EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-8}" --learning-rate "${LEARNING_RATE:-3e-4}" \
  --legacy-learning-rate-scale "${LEGACY_LR_SCALE:-0.25}" \
  --legacy-init pretrained --legacy-unfreeze-epoch 0 \
  --training-objective utility_role_kl \
  --utility-temperature "${UTILITY_TEMPERATURE:-1.0}" \
  --trust-weight "${TRUST_WEIGHT:-0.02}" \
  --role-weight "$ROLE_WEIGHT" --role-gain-scale "$ROLE_GAIN_SCALE" \
  --role-schedule-power "${ROLE_SCHEDULE_POWER:-1}" \
  --role-schedule-floor "${ROLE_SCHEDULE_FLOOR:-1}" \
  --role-teacher-weight "$ROLE_TEACHER_WEIGHT" \
  --residual-min-utility-gain "$RESIDUAL_MIN_UTILITY_GAIN" \
  --context-mode current --explicit-feature-mode full \
  --hidden-dim "${HIDDEN_DIM:-64}" --heads "${HEADS:-4}" \
  --layers "${LAYERS:-2}" --dropout "${DROPOUT:-0.1}" \
  --role-conditioned-hgt --query-edges-per-family "$QUERY_EDGES_PER_FAMILY" \
  --query-context-gate-init "${QUERY_GATE_INIT:-0.2}" \
  > "$LOG_ROOT/joint_train.log" 2>&1

COMMON=(
  --dataset-dir "$DATA" --seed "$SEED" --device cuda
  --sampling-backend cpu --steps "$TOTAL_EDITS" --metrics-every "$EDITS_PER_OUTER"
  --model-time-horizon-steps "$OUTER_STEPS" --edits-per-outer-step "$EDITS_PER_OUTER"
  --family-count-mode binomial --candidates "${CANDIDATES:-16}"
  --role-weight "$ROLE_WEIGHT" --gain-scale "$ROLE_GAIN_SCALE"
  --schedule-power "${ROLE_SCHEDULE_POWER:-1}" --schedule-floor "${ROLE_SCHEDULE_FLOOR:-1}"
  --closure-candidate-fraction "$CLOSURE_FRACTION" --auto-role-balance 0.5
  --targeted-role-candidate-fraction "$TARGETED_FRACTION"
  --context-refresh-steps "$CONTEXT_REFRESH_STEPS"
  --phase-policy unified
)

echo "[PAIRED-SAMPLE] identical seed, prior, candidates and edit budget" \
  | tee -a "$LOG_ROOT/driver.log"
if [[ "$SERIAL_SAMPLING" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="$SAMPLE_BASE_GPU" python -u scripts/run_role_guided_replacement_gate.py \
    "${COMMON[@]}" --model-checkpoint "$BASE" --output-dir "$OUT_ROOT/explicit" \
    > "$LOG_ROOT/explicit_sample.log" 2>&1
  CUDA_VISIBLE_DEVICES="$SAMPLE_JOINT_GPU" python -u scripts/run_role_guided_replacement_gate.py \
    "${COMMON[@]}" --context-checkpoint "$JOINT" --context-scale -1 \
    --output-dir "$OUT_ROOT/role_query_joint/sample" \
    > "$LOG_ROOT/joint_sample.log" 2>&1
else
  CUDA_VISIBLE_DEVICES="$SAMPLE_BASE_GPU" python -u scripts/run_role_guided_replacement_gate.py \
    "${COMMON[@]}" --model-checkpoint "$BASE" --output-dir "$OUT_ROOT/explicit" \
    > "$LOG_ROOT/explicit_sample.log" 2>&1 &
  BASE_PID=$!
  CUDA_VISIBLE_DEVICES="$SAMPLE_JOINT_GPU" python -u scripts/run_role_guided_replacement_gate.py \
    "${COMMON[@]}" --context-checkpoint "$JOINT" --context-scale -1 \
    --output-dir "$OUT_ROOT/role_query_joint/sample" \
    > "$LOG_ROOT/joint_sample.log" 2>&1 &
  JOINT_PID=$!
  wait "$BASE_PID"
  wait "$JOINT_PID"
fi

python - "$OUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
base = json.loads((root / "explicit/gate_report.json").read_text())
joint = json.loads((root / "role_query_joint/sample/gate_report.json").read_text())
train = json.loads((root / "role_query_joint/train_report.json").read_text())
if base["initial"] != joint["initial"]:
    raise RuntimeError("paired role-query gate did not share the initial graph")
payload = {
    "format": "dihug-role-query-joint-gate-v1",
    "initial": base["initial"],
    "real": base["real"],
    "explicit": base["final"],
    "role_query_joint": joint["final"],
    "train_report": train["train_report"],
    "total_edits": base["steps"],
}
(root / "paired_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
)
print("[ROLE-QUERY-JOINT-DONE] " + json.dumps(payload, sort_keys=True))
PY

echo "[ALL-DONE] $OUT_ROOT" | tee -a "$LOG_ROOT/driver.log"

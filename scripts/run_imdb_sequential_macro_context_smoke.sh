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

TAG="${TAG:-imdb_sequential_macro_context_smoke_0719}"
OUT_ROOT="${OUT_ROOT:-results/$TAG}"
LOG_ROOT="${LOG_ROOT:-logs/$TAG}"
DATA="data/IMDB_movie_subgraphs/subgraph_000"
LEGACY="results/imdb_exchangeable_compact_s0_3/model/mixed_edit_diffusion.pt"
SEED="${SEED:-19}"
TRAIN_GPU="${TRAIN_GPU:-0}"
LEGACY_GPU="${LEGACY_GPU:-0}"
CONTEXT_GPU="${CONTEXT_GPU:-1}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python -u scripts/run_sequential_macro_context_gate.py \
  --dataset-dir "$DATA" \
  --legacy-checkpoint "$LEGACY" \
  --output-dir "$OUT_ROOT/train" \
  --device cuda \
  --seed "$SEED" \
  --trajectories "${TRAJECTORIES:-4}" \
  --macro-steps "${TRAIN_MACRO_STEPS:-20}" \
  --micro-steps 20 \
  --record-every "${RECORD_EVERY:-2}" \
  --candidates 16 \
  --epochs "${EPOCHS:-3}" \
  --supervision rooted_utility --freeze-legacy \
  --hidden-dim 64 --layers 2 --heads 4 \
  > "$LOG_ROOT/train.log" 2>&1

CONTEXT="$OUT_ROOT/train/sequential_macro_context.pt"
COMMON=(
  --dataset-dir "$DATA"
  --context-checkpoint "$CONTEXT"
  --seed "$SEED"
  --device cuda
  --steps 2000
  --metrics-every 200
  --model-time-horizon-steps 4000
  --family-count-mode binomial
  --candidates 16
  --role-weight 0.05
  --gain-scale 10
  --schedule-power 1
  --schedule-floor 1
  --role-error-per-node-threshold 5.5
  --closure-candidate-fraction 1.0
  --auto-role-balance 0.5
  --targeted-role-candidate-fraction 0.5
  --context-refresh-steps 20
)

CUDA_VISIBLE_DEVICES="$LEGACY_GPU" python -u scripts/run_role_guided_replacement_gate.py \
  "${COMMON[@]}" --context-scale 0 --output-dir "$OUT_ROOT/legacy" \
  > "$LOG_ROOT/legacy.log" 2>&1 &
PID_LEGACY=$!

CUDA_VISIBLE_DEVICES="$CONTEXT_GPU" python -u scripts/run_role_guided_replacement_gate.py \
  "${COMMON[@]}" --context-scale -1 --output-dir "$OUT_ROOT/context" \
  > "$LOG_ROOT/context.log" 2>&1 &
PID_CONTEXT=$!

wait "$PID_LEGACY"
wait "$PID_CONTEXT"

python - "$OUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
legacy = json.loads((root / "legacy/gate_report.json").read_text())
context = json.loads((root / "context/gate_report.json").read_text())
if legacy["initial"] != context["initial"]:
    raise RuntimeError("paired macro smoke did not share the same initial graph")
payload = {
    "format": "dihug-sequential-macro-context-smoke-v1",
    "initial": legacy["initial"],
    "real": legacy["real"],
    "legacy": legacy["final"],
    "context": context["final"],
    "legacy_actions": legacy["action_counts"],
    "context_actions": context["action_counts"],
    "context_scale": context["context_scale"],
    "micro_steps": 2000,
    "macro_steps": 100,
    "micro_steps_per_macro": 20,
}
(root / "paired_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
)
print("[SEQUENTIAL-MACRO-SMOKE] " + json.dumps(payload, sort_keys=True))
PY

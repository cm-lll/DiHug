#!/usr/bin/env bash
# ACM under the frozen transductive, target-free Binomial edit protocol.
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

DATASET="${DATASET:-data/ACM_subgraphs/subgraph_000}"
OUT_ROOT="${OUT_ROOT:-results/acm_target_free_main}"
LOG_ROOT="${LOG_ROOT:-logs/acm_target_free_main}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

PROFILE="$OUT_ROOT/dataset_profile.json"
ROLE_PRIOR="$OUT_ROOT/anonymous_role_prior.json"
MODEL_DIR="$OUT_ROOT/model_seed${SEED}"
SAMPLE_DIR="$OUT_ROOT/seed${SEED}/legacy"

python scripts/select_mixed_edit_dataset_profile.py \
  --dataset-dir "$DATASET" --output "$PROFILE" \
  > "$LOG_ROOT/profile.log" 2>&1
python scripts/freeze_anonymous_role_prior.py \
  --dataset-dir "$DATASET" --output "$ROLE_PRIOR" \
  > "$LOG_ROOT/role_prior.log" 2>&1

read -r TRAJECTORIES FORWARD_STEPS RECORD_EVERY EPOCHS SAMPLE_STEPS ROLE_COMPONENTS \
  TRAIN_CANDIDATES SAMPLE_CANDIDATES KERNEL < <(
  python - "$PROFILE" "$SMOKE_ONLY" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
if int(sys.argv[2]):
    print(2, 1000, 10, 1, 100, min(int(p["role_components"]), 64),
          min(int(p["train_candidates"]), 16),
          min(int(p["sample_candidates"]), 16), p["kernel"])
else:
    print(p["trajectories"], p["forward_steps"], p["record_every"],
          p["epochs"], p["sample_steps"], p["role_components"],
          p["train_candidates"], p["sample_candidates"], p["kernel"])
PY
)

echo "[ACM-TRAIN] smoke=$SMOKE_ONLY seed=$SEED kernel=$KERNEL trajectories=$TRAJECTORIES forward=$FORWARD_STEPS epochs=$EPOCHS candidates=$TRAIN_CANDIDATES" \
  | tee "$LOG_ROOT/driver.log"
CUDA_VISIBLE_DEVICES="${GPU:-0}" python -u scripts/run_mixed_edit_diffusion.py \
  --mode train --denoiser listwise --kernel "$KERNEL" \
  --dataset-dir "$DATASET" --output-dir "$MODEL_DIR" --device "$DEVICE" \
  --seed "$SEED" --trajectories "$TRAJECTORIES" \
  --forward-steps "$FORWARD_STEPS" --record-every "$RECORD_EVERY" \
  --train-candidates "$TRAIN_CANDIDATES" \
  --feature-profile typed_closure_compact_v2 \
  --epochs "$EPOCHS" --batch-size 64 --hidden-dim 256 --layers 3 \
  --training-objective inverse_ce --fixed-t-trials 16 \
  --role-corruption-fraction 1.0 \
  --role-correlation-strengths "0,0.25,0.5,0.75,1" \
  --role-prior-components "$ROLE_COMPONENTS" \
  --role-prior-randomization-steps-per-edge 2 \
  > "$LOG_ROOT/train_seed${SEED}.log" 2>&1

echo "[ACM-SAMPLE] target-free Binomial initialization, steps=$SAMPLE_STEPS candidates=$SAMPLE_CANDIDATES" \
  | tee -a "$LOG_ROOT/driver.log"
CUDA_VISIBLE_DEVICES="${GPU:-0}" python -u scripts/run_role_guided_replacement_gate.py \
  --dataset-dir "$DATASET" \
  --model-checkpoint "$MODEL_DIR/mixed_edit_diffusion.pt" \
  --role-prior "$ROLE_PRIOR" --strict-target-free \
  --output-dir "$SAMPLE_DIR" --seed "$SEED" --device "$DEVICE" \
  --sampling-backend gpu --family-count-mode binomial \
  --steps "$SAMPLE_STEPS" --candidates "$SAMPLE_CANDIDATES" \
  --metrics-every "$(( SAMPLE_STEPS < 10 ? 1 : SAMPLE_STEPS / 10 ))" \
  --model-time-horizon-steps "$(( SAMPLE_STEPS * 2 ))" \
  --role-weight 0.05 --gain-scale 10 --schedule-power 1 --schedule-floor 1 \
  --role-error-per-node-threshold 5.5 --closure-candidate-fraction 1 \
  --auto-role-balance 0.5 --targeted-role-candidate-fraction 0.5 \
  > "$LOG_ROOT/sample_seed${SEED}.log" 2>&1

python scripts/evaluate_heterogeneous_generation.py \
  --target "$DATASET" --generated "$SAMPLE_DIR/final_graph.json" \
  --output "$SAMPLE_DIR/extended_metrics.json" \
  > "$LOG_ROOT/evaluate_seed${SEED}.log" 2>&1
echo "[ACM-DONE] $SAMPLE_DIR/extended_metrics.json" | tee -a "$LOG_ROOT/driver.log"

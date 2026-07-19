#!/usr/bin/env bash
# One independently trained ACM model, followed by multi-seed target-free generation.
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
OUT_ROOT="${OUT_ROOT:-results/acm_target_free_8gpu}"
LOG_ROOT="${LOG_ROOT:-logs/acm_target_free_8gpu}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
TRAIN_SEED="${TRAIN_SEED:-0}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-7}"
read -r -a GPUS <<< "$GPU_LIST"
[[ "${#GPUS[@]}" -gt 0 ]] || { echo "GPU_LIST is empty" >&2; exit 2; }
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/driver.log"

PROFILE="$OUT_ROOT/dataset_profile.json"
ROLE_PRIOR="$OUT_ROOT/anonymous_role_prior.json"
MODEL_DIR="$OUT_ROOT/model_seed${TRAIN_SEED}"
python scripts/select_mixed_edit_dataset_profile.py \
  --dataset-dir "$DATASET" --output "$PROFILE" > "$LOG_ROOT/profile.log" 2>&1
python scripts/freeze_anonymous_role_prior.py \
  --dataset-dir "$DATASET" --output "$ROLE_PRIOR" > "$LOG_ROOT/role_prior.log" 2>&1

read -r TRAJECTORIES FORWARD_STEPS RECORD_EVERY EPOCHS SAMPLE_STEPS ROLE_COMPONENTS < <(
  python - "$PROFILE" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print(p["trajectories"], p["forward_steps"], p["record_every"], p["epochs"],
      p["sample_steps"], p["role_components"])
PY
)
TRAJECTORIES="${TRAIN_TRAJECTORIES:-$TRAJECTORIES}"
FORWARD_STEPS="${TRAIN_FORWARD_STEPS:-$FORWARD_STEPS}"
RECORD_EVERY="${TRAIN_RECORD_EVERY:-$RECORD_EVERY}"
EPOCHS="${TRAIN_EPOCHS:-$EPOCHS}"
SAMPLE_STEPS="${SAMPLE_STEPS:-$SAMPLE_STEPS}"

if [[ ! -f "$MODEL_DIR/mixed_edit_diffusion.pt" ]]; then
  echo "[ACM-TRAIN] gpu=${GPUS[0]} seed=$TRAIN_SEED trajectories=$TRAJECTORIES forward=$FORWARD_STEPS epochs=$EPOCHS" \
    | tee -a "$LOG_ROOT/driver.log"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -u scripts/run_mixed_edit_diffusion.py \
    --mode train --denoiser listwise --kernel bdrs \
    --dataset-dir "$DATASET" --output-dir "$MODEL_DIR" --device cuda \
    --seed "$TRAIN_SEED" --trajectories "$TRAJECTORIES" \
    --forward-steps "$FORWARD_STEPS" --record-every "$RECORD_EVERY" \
    --train-candidates 16 --feature-profile typed_closure_compact_v2 \
    --epochs "$EPOCHS" --batch-size 64 --hidden-dim 256 --layers 3 \
    --training-objective inverse_ce --fixed-t-trials 16 \
    --role-corruption-fraction 1.0 \
    --role-correlation-strengths "0,0.25,0.5,0.75,1" \
    --role-prior-components "$ROLE_COMPONENTS" \
    --role-prior-randomization-steps-per-edge 2 \
    > "$LOG_ROOT/train_seed${TRAIN_SEED}.log" 2>&1
fi

run_seed() {
  local gpu="$1" seed="$2" output="$OUT_ROOT/acm/seed${seed}/legacy"
  mkdir -p "$output"
  echo "[ACM-SAMPLE-START] gpu=$gpu seed=$seed steps=$SAMPLE_STEPS" \
    | tee -a "$LOG_ROOT/driver.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_role_guided_replacement_gate.py \
    --dataset-dir "$DATASET" --model-checkpoint "$MODEL_DIR/mixed_edit_diffusion.pt" \
    --role-prior "$ROLE_PRIOR" --strict-target-free \
    --output-dir "$output" --seed "$seed" --device cuda \
    --sampling-backend gpu --family-count-mode binomial \
    --steps "$SAMPLE_STEPS" --candidates 16 \
    --metrics-every "$(( SAMPLE_STEPS / 10 ))" \
    --model-time-horizon-steps "$(( SAMPLE_STEPS * 2 ))" \
    --role-weight 0.05 --gain-scale 10 --schedule-power 1 --schedule-floor 1 \
    --role-error-per-node-threshold 5.5 --closure-candidate-fraction 1 \
    --auto-role-balance 0.5 --targeted-role-candidate-fraction 0.5 \
    > "$LOG_ROOT/sample_seed${seed}.log" 2>&1
  echo "[ACM-SAMPLE-DONE] gpu=$gpu seed=$seed" | tee -a "$LOG_ROOT/driver.log"
}

pids=()
for seed in $(seq "$SEED_START" "$SEED_END"); do
  slot=$(( (seed - SEED_START) % ${#GPUS[@]} ))
  run_seed "${GPUS[$slot]}" "$seed" & pids+=("$!")
  sleep 5
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "ACM sampling failed" >&2; exit 1; }

python scripts/generate_extended_structure_tables.py \
  --results-root "$OUT_ROOT" --output-dir "$OUT_ROOT/extended_structure" \
  --dataset acm --mode legacy
echo "[ACM-COMPLETE] $OUT_ROOT/extended_structure/extended_structure.tsv" \
  | tee -a "$LOG_ROOT/driver.log"

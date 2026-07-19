#!/usr/bin/env bash
# PubMed-only continuation for saved-graph extended metrics.
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

OUT_ROOT="${OUT_ROOT:-results/paper_extended_metrics_main16_0719}"
LOG_ROOT="${LOG_ROOT:-logs/paper_extended_metrics_main16_0719}"
CONTEXT="${PUBMED_CONTEXT:-/data2/lyh/gnn_project/DiHug/results/rooted_context_freeze/pubmed/rooted_context_residual.pt}"
DATASET="data/PubMed_baseline_subgraphs/subgraph_000"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-15}"
STEPS="${PUBMED_STEPS:-10000}"
read -r -a GPUS <<< "$GPU_LIST"
[[ -f "$CONTEXT" ]] || { echo "missing PubMed context checkpoint: $CONTEXT" >&2; exit 2; }
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_pair() {
  local gpu="$1" seed="$2" base="$OUT_ROOT/pubmed/seed${seed}"
  mkdir -p "$base"
  local common=(
    --dataset-dir "$DATASET" --context-checkpoint "$CONTEXT"
    --strict-target-free --seed "$seed" --device cuda --steps "$STEPS"
    --metrics-every "$((STEPS / 10))" --model-time-horizon-steps "$((STEPS * 2))"
    --family-count-mode binomial --candidates 16 --role-weight 0.05
    --gain-scale 10 --schedule-power 1 --schedule-floor 1
    --role-error-per-node-threshold 5.5 --closure-candidate-fraction 1
    --auto-role-balance 0.5 --targeted-role-candidate-fraction 0.5
    --context-refresh-steps 20
  )
  echo "[PUBMED-PAIR-START] gpu=$gpu seed=$seed" | tee -a "$LOG_ROOT/driver.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_role_guided_replacement_gate.py \
    "${common[@]}" --context-scale 0 --output-dir "$base/legacy" \
    > "$LOG_ROOT/pubmed_s${seed}_legacy.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_role_guided_replacement_gate.py \
    "${common[@]}" --context-scale -1 --output-dir "$base/context" \
    > "$LOG_ROOT/pubmed_s${seed}_context.log" 2>&1
  echo "[PUBMED-PAIR-DONE] gpu=$gpu seed=$seed" | tee -a "$LOG_ROOT/driver.log"
}

worker() {
  local slot="$1" gpu="${GPUS[$slot]}" seed
  for seed in $(seq "$((SEED_START + slot))" "${#GPUS[@]}" "$SEED_END"); do
    run_pair "$gpu" "$seed"
  done
}

pids=()
for slot in "${!GPUS[@]}"; do worker "$slot" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "PubMed continuation failed" >&2; exit 1; }
echo "[PUBMED-EXTENDED-DONE]" | tee -a "$LOG_ROOT/driver.log"

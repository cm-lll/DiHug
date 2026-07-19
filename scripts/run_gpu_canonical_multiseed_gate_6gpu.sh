#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/results/gpu_canonical_multiseed_gate}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/gpu_canonical_multiseed_gate}"
IMDB_CONTEXT="${IMDB_CONTEXT:?set IMDB_CONTEXT to rooted_context_residual.pt}"
PUBMED_CONTEXT="${PUBMED_CONTEXT:?set PUBMED_CONTEXT to rooted_context_residual.pt}"
SEEDS=(${SEEDS:-1 2 3})
GPU_LIST=(${GPU_LIST:-0 1 2 3 4 5})

if (( ${#SEEDS[@]} * 2 > ${#GPU_LIST[@]} )); then
  echo "Need at least $(( ${#SEEDS[@]} * 2 )) GPU ids" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

pids=()
launch() {
  local gpu="$1" dataset="$2" seed="$3" candidates="$4" context_scale="$5"
  local dataset_dir context
  if [[ "${dataset}" == "imdb" ]]; then
    dataset_dir="data/IMDB_movie_subgraphs/subgraph_000"
    context="${IMDB_CONTEXT}"
  else
    dataset_dir="data/PubMed_baseline_subgraphs/subgraph_000"
    context="${PUBMED_CONTEXT}"
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" python -u scripts/run_role_guided_replacement_gate.py \
    --dataset-dir "${dataset_dir}" \
    --context-checkpoint "${context}" \
    --strict-target-free \
    --family-count-mode binomial \
    --seed "${seed}" \
    --device cuda \
    --sampling-backend gpu \
    --gpu-candidate-mode canonical_ordered \
    --steps 10000 \
    --candidates "${candidates}" \
    --metrics-every 1000 \
    --model-time-horizon-steps 20000 \
    --role-weight 0.05 \
    --gain-scale 10 \
    --schedule-power 1 \
    --schedule-floor 1 \
    --role-error-per-node-threshold 5.5 \
    --closure-candidate-fraction 1 \
    --auto-role-balance 0.5 \
    --targeted-role-candidate-fraction 0.5 \
    --context-refresh-steps 20 \
    --context-scale "${context_scale}" \
    --output-dir "${OUT_ROOT}/${dataset}_seed${seed}" \
    > "${LOG_ROOT}/${dataset}_seed${seed}.log" 2>&1 &
  pids+=("$!")
}

slot=0
for seed in "${SEEDS[@]}"; do
  launch "${GPU_LIST[slot]}" imdb "${seed}" 64 0
  ((slot+=1))
done
for seed in "${SEEDS[@]}"; do
  launch "${GPU_LIST[slot]}" pubmed "${seed}" 16 -1
  ((slot+=1))
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if (( status != 0 )); then
  echo "At least one canonical GPU gate failed; inspect ${LOG_ROOT}" >&2
  exit "${status}"
fi
echo "GPU canonical multiseed gate complete: ${OUT_ROOT}"

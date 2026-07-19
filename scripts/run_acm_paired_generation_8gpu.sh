#!/usr/bin/env bash
# Paired ACM generation from identical target-free Binomial initial graphs.
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

DATASET="${DATASET:-data/ACM_subgraphs/subgraph_000}"
CONTEXT_CHECKPOINT="${CONTEXT_CHECKPOINT:-results/acm_rs_context_paired_gate_0720/context/rooted_context_residual.pt}"
OUT_ROOT="${OUT_ROOT:-results/acm_rs_relative028_paired_generation_0720}"
LOG_ROOT="${LOG_ROOT:-logs/acm_rs_relative028_paired_generation_0720}"
SEEDS="${SEEDS:-0 1 2 3}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
STEPS="${STEPS:-50000}"
CANDIDATES="${CANDIDATES:-64}"
METRICS_EVERY="${METRICS_EVERY:-5000}"
MODEL_TIME_HORIZON_STEPS="${MODEL_TIME_HORIZON_STEPS:-100000}"
RELATIVE_ROLE_ERROR_THRESHOLD="${RELATIVE_ROLE_ERROR_THRESHOLD:-0.28}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

read -r -a seeds <<< "${SEEDS}"
read -r -a gpus <<< "${GPU_LIST}"
needed=$(( ${#seeds[@]} * 2 ))
if (( ${#gpus[@]} < needed )); then
  echo "need ${needed} GPUs for ${#seeds[@]} paired seeds" >&2
  exit 2
fi
[[ -f "${CONTEXT_CHECKPOINT}" ]] || {
  echo "missing context checkpoint: ${CONTEXT_CHECKPOINT}" >&2
  exit 2
}

pids=()
job=0
for seed in "${seeds[@]}"; do
  for mode in legacy context; do
    gpu="${gpus[$job]}"
    output="${OUT_ROOT}/seed${seed}/${mode}"
    log="${LOG_ROOT}/seed${seed}_${mode}.log"
    echo "[ACM-PAIR-LAUNCH] seed=${seed} mode=${mode} gpu=${gpu} steps=${STEPS} C=${CANDIDATES} relative_threshold=${RELATIVE_ROLE_ERROR_THRESHOLD}"
    CUDA_VISIBLE_DEVICES="${gpu}" python -u scripts/run_batched_context_generation.py \
      --context-checkpoint "${CONTEXT_CHECKPOINT}" \
      --evaluation-dataset-dir "${DATASET}" \
      --output-dir "${output}" --seeds "${seed}" --mode "${mode}" \
      --device cuda --family-count-mode binomial \
      --steps "${STEPS}" --candidates "${CANDIDATES}" \
      --metrics-every "${METRICS_EVERY}" \
      --model-time-horizon-steps "${MODEL_TIME_HORIZON_STEPS}" \
      --relative-role-error-threshold "${RELATIVE_ROLE_ERROR_THRESHOLD}" \
      --context-refresh-steps 20 \
      > "${log}" 2>&1 &
    pids+=("$!")
    job=$((job + 1))
  done
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
(( status == 0 )) || {
  echo "at least one ACM paired generation failed" >&2
  exit 1
}

python - "${OUT_ROOT}" "${SEEDS}" "${RELATIVE_ROLE_ERROR_THRESHOLD}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
seeds = [int(value) for value in sys.argv[2].split()]
relative_role_error_threshold = float(sys.argv[3])
rows = []
for seed in seeds:
    pair = {}
    for mode in ("legacy", "context"):
        path = root / f"seed{seed}" / mode / f"seed{seed}" / mode / "gate_report.json"
        report = json.loads(path.read_text())
        pair[mode] = report
        rows.append(
            {
                "seed": seed,
                "mode": mode,
                "initial": report["initial"],
                "final": report["final"],
                "real": report.get("evaluation_real"),
                "action_counts": report["action_counts"],
                "trace": report["trace"],
            }
        )
    if pair["legacy"]["initial"] != pair["context"]["initial"]:
        raise RuntimeError(f"seed {seed} does not share an identical initial graph")

aggregates = []
for mode in ("legacy", "context"):
    selected = [row for row in rows if row["mode"] == mode]
    item = {"mode": mode, "n": len(selected)}
    for metric in ("triangles", "clustering", "lcc", "assortativity", "sum_degree_sq"):
        values = [float(row["final"][metric]) for row in selected]
        item[f"{metric}_mean"] = statistics.fmean(values)
        item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    aggregates.append(item)

summary = {
    "format": "dihug-acm-paired-generation-v1",
    "seeds": seeds,
    "family_count_mode": "binomial",
    "relative_role_error_threshold": relative_role_error_threshold,
    "steps": int(next(row["trace"][-1]["step"] for row in rows)),
    "rows": rows,
    "aggregates": aggregates,
}
(root / "aggregate.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps({"aggregates": aggregates}, indent=2), flush=True)
PY

echo "[ACM-PAIR-DONE] ${OUT_ROOT}/aggregate.json"

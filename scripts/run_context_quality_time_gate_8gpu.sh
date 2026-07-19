#!/usr/bin/env bash
# Test whether the frozen graph-context residual reaches structural quality in
# fewer edits than the legacy listwise scorer. This is an efficiency gate, not
# a new generation protocol.
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
BUDGETS="${BUDGETS:-2500 5000 7500 10000}"
GEN_SEED="${GEN_SEED:-0}"
OUT_ROOT="${OUT_ROOT:-results/context_quality_time_gate}"
LOG_ROOT="${LOG_ROOT:-logs/context_quality_time_gate}"
IMDB_DATA="${IMDB_DATA:-data/IMDB_movie_subgraphs/subgraph_000}"
PUBMED_DATA="${PUBMED_DATA:-data/PubMed_baseline_subgraphs/subgraph_000}"
IMDB_CONTEXT="${IMDB_CONTEXT:-/data2/lyh/gnn_project/DiHug/results/rooted_context_freeze/imdb/rooted_context_residual.pt}"
PUBMED_CONTEXT="${PUBMED_CONTEXT:-/data2/lyh/gnn_project/DiHug/results/rooted_context_freeze/pubmed/rooted_context_residual.pt}"
read -r -a GPUS <<< "$GPU_LIST"
read -r -a STEPS <<< "$BUDGETS"
[[ "${#GPUS[@]}" -eq 8 ]] || { echo "GPU_LIST must contain eight GPUs" >&2; exit 2; }
[[ -f "$IMDB_CONTEXT" ]] || { echo "missing IMDB context: $IMDB_CONTEXT" >&2; exit 2; }
[[ -f "$PUBMED_CONTEXT" ]] || { echo "missing PubMed context: $PUBMED_CONTEXT" >&2; exit 2; }
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/driver.log"

dataset_path() {
  case "$1" in
    imdb) echo "$IMDB_DATA" ;;
    pubmed) echo "$PUBMED_DATA" ;;
    *) return 2 ;;
  esac
}

context_path() {
  case "$1" in
    imdb) echo "$IMDB_CONTEXT" ;;
    pubmed) echo "$PUBMED_CONTEXT" ;;
    *) return 2 ;;
  esac
}

run_one() {
  local gpu="$1" dataset="$2" mode="$3" budget="$4"
  local output="$OUT_ROOT/$dataset/${mode}_${budget}"
  local log="$LOG_ROOT/${dataset}_${mode}_${budget}.log"
  local scale=0
  [[ "$mode" == context ]] && scale=-1
  if [[ -f "$output/gate_report.json" ]]; then
    echo "[SKIP] dataset=$dataset mode=$mode budget=$budget" | tee -a "$LOG_ROOT/driver.log"
    return
  fi
  mkdir -p "$output"
  echo "[START] gpu=$gpu dataset=$dataset mode=$mode budget=$budget" | tee -a "$LOG_ROOT/driver.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_role_guided_replacement_gate.py \
    --dataset-dir "$(dataset_path "$dataset")" \
    --context-checkpoint "$(context_path "$dataset")" \
    --strict-target-free --output-dir "$output" --seed "$GEN_SEED" \
    --device cuda --sampling-backend gpu --gpu-proposal-pool-size 0 \
    --family-count-mode binomial --steps "$budget" --candidates 16 \
    --metrics-every "$budget" --model-time-horizon-steps "$((budget * 2))" \
    --role-weight 0.05 --gain-scale 10 --schedule-power 1 --schedule-floor 1 \
    --role-error-per-node-threshold 5.5 --closure-candidate-fraction 1 \
    --auto-role-balance 0.5 --targeted-role-candidate-fraction 0.5 \
    --context-refresh-steps 20 --context-scale "$scale" > "$log" 2>&1
  echo "[DONE] gpu=$gpu dataset=$dataset mode=$mode budget=$budget" | tee -a "$LOG_ROOT/driver.log"
}

tasks=()
for dataset in imdb pubmed; do
  for mode in legacy context; do
    for budget in "${STEPS[@]}"; do
      tasks+=("$dataset $mode $budget")
    done
  done
done

worker() {
  local slot="$1" index dataset mode budget
  for ((index=slot; index<${#tasks[@]}; index+=${#GPUS[@]})); do
    read -r dataset mode budget <<< "${tasks[$index]}"
    run_one "${GPUS[$slot]}" "$dataset" "$mode" "$budget"
  done
}

pids=()
for slot in "${!GPUS[@]}"; do worker "$slot" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "context quality-time gate failed" >&2; exit 1; }

python - "$OUT_ROOT" "$IMDB_DATA" "$PUBMED_DATA" <<'PY'
import csv
import json
import pathlib
import sys

from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics

root = pathlib.Path(sys.argv[1])
targets = {
    "imdb": graph_metrics(load_typed_graph(pathlib.Path(sys.argv[2]), active_only=True)),
    "pubmed": graph_metrics(load_typed_graph(pathlib.Path(sys.argv[3]), active_only=True)),
}
rows = []
for report_path in sorted(root.glob("*/*/gate_report.json")):
    dataset = report_path.parts[-3]
    mode, budget_text = report_path.parent.name.rsplit("_", 1)
    report = json.loads(report_path.read_text())
    final = report["final"]
    target = targets[dataset]
    errors = {}
    for metric in ("triangles", "clustering", "assortativity", "sum_degree_sq"):
        denominator = max(abs(float(target[metric])), 1.0e-12)
        errors[metric] = abs(float(final[metric]) - float(target[metric])) / denominator
    rows.append({
        "dataset": dataset,
        "mode": mode,
        "budget": int(budget_text),
        "sampling_seconds": float(report["sampling_seconds"]),
        "steps_per_second": float(report["sampling_steps_per_second"]),
        "triangles": float(final["triangles"]),
        "clustering": float(final["clustering"]),
        "assortativity": float(final["assortativity"]),
        "sum_degree_sq": float(final["sum_degree_sq"]),
        "triangle_error": errors["triangles"],
        "clustering_error": errors["clustering"],
        "assortativity_error": errors["assortativity"],
        "sum_degree_sq_error": errors["sum_degree_sq"],
        "mean_structure_error": sum(errors.values()) / len(errors),
        "lcc": float(final["lcc"]),
        "edges": float(final["edges"]),
    })

columns = list(rows[0])
with (root / "quality_time.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
    writer.writeheader(); writer.writerows(rows)
(root / "quality_time.json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True))
print(json.dumps(rows, indent=2, sort_keys=True))
PY

echo "[CONTEXT-QUALITY-TIME-DONE] $OUT_ROOT/quality_time.tsv" | tee -a "$LOG_ROOT/driver.log"

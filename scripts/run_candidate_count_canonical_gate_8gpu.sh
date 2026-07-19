#!/usr/bin/env bash
# Paired candidate-count gate under the frozen target-free paper protocol.
# Every task uses the same generation seed and therefore the same Binomial
# family-count draw and initial graph; only the candidate-list width changes.
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
CANDIDATE_COUNTS="${CANDIDATE_COUNTS:-8 16 32 64}"
GEN_SEED="${GEN_SEED:-0}"
STEPS="${STEPS:-10000}"
OUT_ROOT="${OUT_ROOT:-results/candidate_count_canonical_gate}"
LOG_ROOT="${LOG_ROOT:-logs/candidate_count_canonical_gate}"
IMDB_DATA="${IMDB_DATA:-data/IMDB_movie_subgraphs/subgraph_000}"
PUBMED_DATA="${PUBMED_DATA:-data/PubMed_baseline_subgraphs/subgraph_000}"
IMDB_CONTEXT="${IMDB_CONTEXT:-/data2/lyh/gnn_project/DiHug/results/rooted_context_freeze/imdb/rooted_context_residual.pt}"
PUBMED_CONTEXT="${PUBMED_CONTEXT:-/data2/lyh/gnn_project/DiHug/results/rooted_context_freeze/pubmed/rooted_context_residual.pt}"
read -r -a GPUS <<< "$GPU_LIST"
read -r -a WIDTHS <<< "$CANDIDATE_COUNTS"

[[ "${#GPUS[@]}" -ge 1 ]] || { echo "GPU_LIST is empty" >&2; exit 2; }
[[ -f "$IMDB_CONTEXT" ]] || { echo "missing IMDB checkpoint: $IMDB_CONTEXT" >&2; exit 2; }
[[ -f "$PUBMED_CONTEXT" ]] || { echo "missing PubMed checkpoint: $PUBMED_CONTEXT" >&2; exit 2; }
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/driver.log"

tasks=()
for dataset in imdb pubmed; do
  for candidates in "${WIDTHS[@]}"; do
    tasks+=("$dataset $candidates")
  done
done

run_one() {
  local gpu="$1" dataset="$2" candidates="$3"
  local data checkpoint context_scale
  case "$dataset" in
    imdb)
      data="$IMDB_DATA"; checkpoint="$IMDB_CONTEXT"; context_scale=0 ;;
    pubmed)
      data="$PUBMED_DATA"; checkpoint="$PUBMED_CONTEXT"; context_scale=-1 ;;
    *) return 2 ;;
  esac
  local output="$OUT_ROOT/$dataset/c${candidates}_seed${GEN_SEED}"
  local log="$LOG_ROOT/${dataset}_c${candidates}_seed${GEN_SEED}.log"
  mkdir -p "$output"
  echo "[START] gpu=$gpu dataset=$dataset candidates=$candidates" | tee -a "$LOG_ROOT/driver.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_role_guided_replacement_gate.py \
    --dataset-dir "$data" --context-checkpoint "$checkpoint" \
    --strict-target-free --family-count-mode binomial \
    --seed "$GEN_SEED" --device cuda --sampling-backend cpu \
    --steps "$STEPS" --candidates "$candidates" --metrics-every "$STEPS" \
    --model-time-horizon-steps "$((STEPS * 2))" \
    --role-weight 0.05 --gain-scale 10 --schedule-power 1 --schedule-floor 1 \
    --role-error-per-node-threshold 5.5 --closure-candidate-fraction 1 \
    --auto-role-balance 0.5 --targeted-role-candidate-fraction 0.5 \
    --context-refresh-steps 20 --context-scale "$context_scale" \
    --output-dir "$output" > "$log" 2>&1
  echo "[DONE] gpu=$gpu dataset=$dataset candidates=$candidates" | tee -a "$LOG_ROOT/driver.log"
}

pids=()
for index in "${!tasks[@]}"; do
  read -r dataset candidates <<< "${tasks[$index]}"
  run_one "${GPUS[$((index % ${#GPUS[@]}))]}" "$dataset" "$candidates" &
  pids+=("$!")
  sleep 2
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "candidate-count gate failed" >&2; exit 1; }

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
for path in sorted(root.glob("*/c*_seed*/gate_report.json")):
    dataset = path.parts[-3]
    candidates = int(path.parent.name.split("_", 1)[0][1:])
    report = json.loads(path.read_text())
    initial = report["initial"]
    final = report["final"]
    target = targets[dataset]
    errors = {}
    for metric in ("triangles", "clustering", "assortativity", "sum_degree_sq"):
        errors[metric] = abs(float(final[metric]) - float(target[metric])) / max(
            abs(float(target[metric])), 1.0e-12
        )
    rows.append({
        "dataset": dataset,
        "candidates": candidates,
        "initial_triangles": initial["triangles"],
        "triangles": final["triangles"],
        "clustering": final["clustering"],
        "assortativity": final["assortativity"],
        "sum_degree_sq": final["sum_degree_sq"],
        "triangle_error": errors["triangles"],
        "clustering_error": errors["clustering"],
        "assortativity_error": errors["assortativity"],
        "sum_degree_sq_error": errors["sum_degree_sq"],
        "mean_structure_error": sum(errors.values()) / len(errors),
        "sampling_seconds": report["sampling_seconds"],
        "steps_per_second": report["sampling_steps_per_second"],
        "lcc": final["lcc"],
        "edges": final["edges"],
    })
columns = list(rows[0])
with (root / "candidate_count.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
    writer.writeheader(); writer.writerows(rows)
(root / "candidate_count.json").write_text(json.dumps({"rows": rows}, indent=2))
print(json.dumps(rows, indent=2))
PY

echo "[CANDIDATE-COUNT-DONE] $OUT_ROOT/candidate_count.tsv" | tee -a "$LOG_ROOT/driver.log"

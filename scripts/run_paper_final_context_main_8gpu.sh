#!/usr/bin/env bash
# Paired final-method evaluation for the frozen rooted-context edit sampler.
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-7}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
MACHINE_TAG="${MACHINE_TAG:-local}"
FAMILY_COUNT_MODE="${FAMILY_COUNT_MODE:-binomial}"
OUT_ROOT="${OUT_ROOT:-results/paper_final_context_${FAMILY_COUNT_MODE}_${MACHINE_TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/paper_final_context_${FAMILY_COUNT_MODE}_${MACHINE_TAG}}"
PYTHON="${PYTHON:-python}"
read -r -a GPUS <<< "$GPU_LIST"
NGPU="${#GPUS[@]}"

if [[ "$NGPU" -eq 0 ]]; then
  echo "GPU_LIST must contain at least one GPU" >&2
  exit 2
fi
if [[ "$FAMILY_COUNT_MODE" != "fixed" && "$FAMILY_COUNT_MODE" != "binomial" ]]; then
  echo "FAMILY_COUNT_MODE must be fixed or binomial" >&2
  exit 2
fi

IMDB_DATA="data/IMDB_movie_subgraphs/subgraph_000"
PUBMED_DATA="data/PubMed_baseline_subgraphs/subgraph_000"
IMDB_CONTEXT="${IMDB_CONTEXT:-results/rooted_context_freeze/imdb/rooted_context_residual.pt}"
PUBMED_CONTEXT="${PUBMED_CONTEXT:-results/rooted_context_freeze/pubmed/rooted_context_residual.pt}"

for required in "$IMDB_CONTEXT" "$PUBMED_CONTEXT"; do
  if [[ ! -f "$required" ]]; then
    echo "missing frozen context checkpoint: $required" >&2
    exit 2
  fi
done

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/driver.log"

COMMON_ARGS=(
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

run_pair() {
  local gpu="$1" dataset="$2" data_dir="$3" checkpoint="$4" seed="$5" steps="$6"
  local base="$OUT_ROOT/$dataset/seed${seed}"
  local log_prefix="$LOG_ROOT/${dataset}_s${seed}"
  mkdir -p "$base"

  local run_args=(
    --dataset-dir "$data_dir"
    --context-checkpoint "$checkpoint"
    --seed "$seed"
    --device cuda
    --steps "$steps"
    --metrics-every "$((steps / 10))"
    --model-time-horizon-steps "$((steps * 2))"
    --family-count-mode "$FAMILY_COUNT_MODE"
    --strict-target-free
    "${COMMON_ARGS[@]}"
  )

  echo "[PAIR-START] gpu=$gpu dataset=$dataset seed=$seed steps=$steps" \
    | tee -a "$LOG_ROOT/driver.log"

  # scale=0 is the exact legacy scorer under the same self-contained prior,
  # initial graph, candidate generator, and sampling budget.
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_role_guided_replacement_gate.py \
    "${run_args[@]}" --context-scale 0 --output-dir "$base/legacy" \
    > "${log_prefix}_legacy.log" 2>&1

  # Negative scale keeps the validated scale stored in the frozen checkpoint.
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_role_guided_replacement_gate.py \
    "${run_args[@]}" --context-scale -1 --output-dir "$base/context" \
    > "${log_prefix}_context.log" 2>&1

  echo "[PAIR-DONE] gpu=$gpu dataset=$dataset seed=$seed" \
    | tee -a "$LOG_ROOT/driver.log"
}

run_dataset() {
  local dataset="$1" data_dir="$2" checkpoint="$3" steps="$4"
  local pids=()
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    local slot=$(( (seed - SEED_START) % NGPU ))
    run_pair "${GPUS[$slot]}" "$dataset" "$data_dir" "$checkpoint" "$seed" "$steps" &
    pids+=("$!")
    sleep 3
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "$dataset paired evaluation failed" >&2
    exit 1
  fi
}

echo "[PAPER-FINAL] commit=$(git rev-parse HEAD) machine=$MACHINE_TAG family_count_mode=$FAMILY_COUNT_MODE seeds=$SEED_START-$SEED_END" \
  | tee -a "$LOG_ROOT/driver.log"
run_dataset imdb "$IMDB_DATA" "$IMDB_CONTEXT" "${IMDB_STEPS:-10000}"
run_dataset pubmed "$PUBMED_DATA" "$PUBMED_CONTEXT" "${PUBMED_STEPS:-10000}"

"$PYTHON" - "$OUT_ROOT" "$MACHINE_TAG" "$(git rev-parse HEAD)" "$FAMILY_COUNT_MODE" <<'PY'
import csv
import hashlib
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
machine = sys.argv[2]
commit = sys.argv[3]
family_count_mode = sys.argv[4]
rows = []

for context_path in sorted(root.glob("*/seed*/context/gate_report.json")):
    dataset = context_path.parts[-4]
    seed = int(context_path.parts[-3].removeprefix("seed"))
    legacy_path = context_path.parents[1] / "legacy" / "gate_report.json"
    context = json.loads(context_path.read_text())
    legacy = json.loads(legacy_path.read_text())
    if context["initial"] != legacy["initial"]:
        raise RuntimeError(f"paired initial mismatch: {dataset}/seed{seed}")
    expected_source = f"{family_count_mode}_family_count_prior"
    for report in (legacy, context):
        if report.get("initial_graph_source") != expected_source:
            raise RuntimeError(
                f"initialization protocol mismatch: {dataset}/seed{seed}: "
                f"{report.get('initial_graph_source')} != {expected_source}"
            )
    for mode, report in (("legacy", legacy), ("context", context)):
        final = report["final"]
        initial = report["initial"]
        real = report["real"]
        row = {
            "machine": machine,
            "dataset": dataset,
            "seed": seed,
            "mode": mode,
            "steps": int(report["steps"]),
            "context_scale": float(report.get("context_scale") or 0.0),
            "family_count_mode": family_count_mode,
            "initial_graph_source": report["initial_graph_source"],
        }
        for prefix, metrics in (("initial", initial), ("final", final), ("real", real)):
            for key in (
                "edges", "triangles", "clustering", "transitivity", "lcc",
                "assortativity", "sum_degree_sq", "max_degree",
            ):
                row[f"{prefix}_{key}"] = metrics[key]
        rows.append(row)

columns = list(rows[0])
with (root / "paper_main.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

aggregates = []
for dataset in sorted({row["dataset"] for row in rows}):
    for mode in ("legacy", "context"):
        group = [row for row in rows if row["dataset"] == dataset and row["mode"] == mode]
        item = {"dataset": dataset, "mode": mode, "n": len(group)}
        for metric in (
            "edges", "triangles", "clustering", "transitivity", "lcc",
            "assortativity", "sum_degree_sq", "max_degree",
        ):
            values = [float(row[f"final_{metric}"]) for row in group]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = statistics.pstdev(values)
        aggregates.append(item)

(root / "aggregate.json").write_text(json.dumps({
    "format": "dihug-paper-final-context-v1",
    "commit": commit,
    "machine": machine,
    "family_count_mode": family_count_mode,
    "initial_graph_source": f"{family_count_mode}_family_count_prior",
    "generation_scope": (
        "unconditional-family-count" if family_count_mode == "binomial"
        else "conditional-fixed-family-count"
    ),
    "paired_initials_verified": True,
    "aggregates": aggregates,
}, indent=2, sort_keys=True))
print(json.dumps(aggregates, indent=2, sort_keys=True))
PY

echo "[PAPER-FINAL] complete: $OUT_ROOT/aggregate.json" | tee -a "$LOG_ROOT/driver.log"
python scripts/generate_extended_structure_tables.py \
  --results-root "$OUT_ROOT" --output-dir "$OUT_ROOT/extended_structure" \
  --dataset imdb --dataset pubmed
echo "[PAPER-FINAL] extended metrics: $OUT_ROOT/extended_structure/extended_structure.tsv" \
  | tee -a "$LOG_ROOT/driver.log"

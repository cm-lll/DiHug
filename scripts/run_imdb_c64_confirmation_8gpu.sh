#!/usr/bin/env bash
# Confirm the IMDB C64 quality gain with paired C16/C64 runs on seeds 4-7.
# The pair for each seed shares the Binomial family-count draw, initial graph,
# frozen checkpoint, edit budget, and all sampling controls.
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
SEEDS="${SEEDS:-4 5 6 7}"
STEPS="${STEPS:-10000}"
OUT_ROOT="${OUT_ROOT:-results/imdb_c64_confirmation}"
LOG_ROOT="${LOG_ROOT:-logs/imdb_c64_confirmation}"
IMDB_DATA="${IMDB_DATA:-data/IMDB_movie_subgraphs/subgraph_000}"
IMDB_CONTEXT="${IMDB_CONTEXT:-/data2/lyh/gnn_project/DiHug/results/rooted_context_freeze/imdb/rooted_context_residual.pt}"
read -r -a GPUS <<< "$GPU_LIST"
read -r -a SEED_VALUES <<< "$SEEDS"

[[ "${#GPUS[@]}" -eq 8 ]] || { echo "GPU_LIST must contain eight GPUs" >&2; exit 2; }
[[ "${#SEED_VALUES[@]}" -eq 4 ]] || { echo "SEEDS must contain four seeds" >&2; exit 2; }
[[ -d "$IMDB_DATA" ]] || { echo "missing IMDB data: $IMDB_DATA" >&2; exit 2; }
[[ -f "$IMDB_CONTEXT" ]] || { echo "missing IMDB checkpoint: $IMDB_CONTEXT" >&2; exit 2; }
if [[ "${VERIFY_ONLY:-0}" == "1" ]]; then
  echo "[VERIFY-OK] commit=$(git rev-parse HEAD) seeds=$SEEDS steps=$STEPS"
  echo "[VERIFY-OK] data=$IMDB_DATA"
  echo "[VERIFY-OK] checkpoint=$IMDB_CONTEXT"
  exit 0
fi
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/driver.log"

run_one() {
  local gpu="$1" seed="$2" candidates="$3"
  local output="$OUT_ROOT/seed${seed}/c${candidates}"
  local log="$LOG_ROOT/seed${seed}_c${candidates}.log"
  mkdir -p "$output"
  echo "[START] gpu=$gpu seed=$seed candidates=$candidates" | tee -a "$LOG_ROOT/driver.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_role_guided_replacement_gate.py \
    --dataset-dir "$IMDB_DATA" --context-checkpoint "$IMDB_CONTEXT" \
    --strict-target-free --family-count-mode binomial \
    --seed "$seed" --device cuda --sampling-backend cpu \
    --steps "$STEPS" --candidates "$candidates" --metrics-every "$STEPS" \
    --model-time-horizon-steps "$((STEPS * 2))" \
    --role-weight 0.05 --gain-scale 10 --schedule-power 1 --schedule-floor 1 \
    --role-error-per-node-threshold 5.5 --closure-candidate-fraction 1 \
    --auto-role-balance 0.5 --targeted-role-candidate-fraction 0.5 \
    --context-refresh-steps 20 --context-scale 0 \
    --output-dir "$output" > "$log" 2>&1
  echo "[DONE] gpu=$gpu seed=$seed candidates=$candidates" | tee -a "$LOG_ROOT/driver.log"
}

pids=()
for index in "${!SEED_VALUES[@]}"; do
  seed="${SEED_VALUES[$index]}"
  run_one "${GPUS[$index]}" "$seed" 16 & pids+=("$!")
  run_one "${GPUS[$((index + 4))]}" "$seed" 64 & pids+=("$!")
  sleep 2
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "IMDB C64 confirmation failed" >&2; exit 1; }

python - "$OUT_ROOT" "$IMDB_DATA" "$IMDB_CONTEXT" <<'PY'
import csv
import hashlib
import json
import pathlib
import statistics
import sys

from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics

root = pathlib.Path(sys.argv[1])
target = graph_metrics(load_typed_graph(pathlib.Path(sys.argv[2]), active_only=True))
checkpoint = pathlib.Path(sys.argv[3])
rows = []
for seed_dir in sorted(root.glob("seed*")):
    seed = int(seed_dir.name.removeprefix("seed"))
    reports = {}
    for candidates in (16, 64):
        report = json.loads((seed_dir / f"c{candidates}" / "gate_report.json").read_text())
        reports[candidates] = report
        final = report["final"]
        errors = []
        for metric in ("triangles", "clustering", "assortativity", "sum_degree_sq"):
            errors.append(
                abs(float(final[metric]) - float(target[metric]))
                / max(abs(float(target[metric])), 1.0e-12)
            )
        rows.append({
            "seed": seed,
            "candidates": candidates,
            "initial_triangles": report["initial"]["triangles"],
            "triangles": final["triangles"],
            "clustering": final["clustering"],
            "assortativity": final["assortativity"],
            "sum_degree_sq": final["sum_degree_sq"],
            "mean_structure_error": sum(errors) / len(errors),
            "sampling_seconds": report["sampling_seconds"],
            "steps_per_second": report["sampling_steps_per_second"],
            "lcc": final["lcc"],
            "edges": final["edges"],
        })
    if reports[16]["initial"] != reports[64]["initial"]:
        raise RuntimeError(f"paired initial graph mismatch for seed {seed}")

columns = list(rows[0])
with (root / "paired_results.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
    writer.writeheader(); writer.writerows(rows)

summary = {}
for candidates in (16, 64):
    subset = [row for row in rows if row["candidates"] == candidates]
    summary[f"c{candidates}"] = {
        key: {
            "mean": statistics.mean(float(row[key]) for row in subset),
            "std": statistics.stdev(float(row[key]) for row in subset),
        }
        for key in ("triangles", "clustering", "mean_structure_error", "sampling_seconds")
    }
payload = {
    "commit": __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "rows": rows,
    "summary": summary,
}
(root / "paired_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "[IMDB-C64-CONFIRMATION-DONE] $OUT_ROOT/paired_results.tsv" | tee -a "$LOG_ROOT/driver.log"

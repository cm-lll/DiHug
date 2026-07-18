#!/usr/bin/env bash
# Continue the target-free DBLP trajectories until role alignment can finish,
# leaving enough reverse time for degree-preserving switch refinement.
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

SOURCE_ROOT="${SOURCE_ROOT:-results/paper_cross_dataset_binomial_0717/dblp}"
OUT_ROOT="${OUT_ROOT:-results/dblp_role_completion_150k_0717}"
LOG_ROOT="${LOG_ROOT:-logs/dblp_role_completion_150k_0717}"
CHECKPOINT="${CHECKPOINT:-results/hgen_official_fullgraph_pilot_0716/dblp/seed0/model/mixed_edit_diffusion.pt}"
CONTINUATION_STEPS="${CONTINUATION_STEPS:-100000}"
START_STEP="${START_STEP:-50000}"
TIME_HORIZON="${TIME_HORIZON:-100000}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/driver.log"

run_seed() {
  local gpu="$1" seed="$2"
  local source="$SOURCE_ROOT/seed${seed}/legacy/final_graph.json"
  local state="$SOURCE_ROOT/seed${seed}/legacy/sampling_state.json"
  local output="$OUT_ROOT/seed${seed}"
  [[ -f "$source" && -f "$state" ]] || {
    echo "missing continuation state for seed $seed" >&2
    return 2
  }
  echo "[START] gpu=$gpu seed=$seed start=$START_STEP add=$CONTINUATION_STEPS" \
    | tee -a "$LOG_ROOT/driver.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_role_guided_replacement_gate.py \
    --dataset-dir data/DBLP_four_area_subgraphs/subgraph_000 \
    --seed "$seed" --device cuda --sampling-backend gpu \
    --gpu-proposal-pool-size 0 --gpu-proposal-chunk-size 4096 \
    --steps "$CONTINUATION_STEPS" --candidates 16 \
    --metrics-every 5000 --family-count-mode binomial \
    --role-weight 0.05 --gain-scale 10 \
    --schedule-power 1 --schedule-floor 1 \
    --role-error-per-node-threshold 5.5 \
    --model-time-horizon-steps "$TIME_HORIZON" \
    --model-time-start-step "$START_STEP" \
    --closure-candidate-fraction 1.0 \
    --auto-role-balance 0.5 \
    --targeted-role-candidate-fraction 0.5 \
    --context-refresh-steps 20 \
    --initial-graph "$source" \
    --model-checkpoint "$CHECKPOINT" \
    --output-dir "$output" > "$LOG_ROOT/seed${seed}.log" 2>&1
  echo "[DONE] gpu=$gpu seed=$seed" | tee -a "$LOG_ROOT/driver.log"
}

pids=()
for seed in $(seq 0 7); do
  run_seed "$seed" "$seed" &
  pids+=("$!")
  sleep 2
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "one or more DBLP continuations failed" >&2; exit 1; }

python - "$OUT_ROOT" <<'PY'
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
reports = [json.loads(path.read_text()) for path in sorted(root.glob("seed*/gate_report.json"))]
if len(reports) != 8:
    raise RuntimeError(f"expected 8 reports, found {len(reports)}")

metrics = (
    "edges", "triangles", "clustering", "transitivity", "lcc",
    "assortativity", "sum_degree_sq", "max_degree",
)
summary = {"format": "dihug-dblp-role-completion-v1", "n": len(reports)}
for phase in ("initial", "final", "real"):
    for metric in metrics:
        values = [float(report[phase][metric]) for report in reports]
        summary[f"{phase}_{metric}_mean"] = statistics.fmean(values)
        summary[f"{phase}_{metric}_std"] = statistics.pstdev(values)
summary["phase_transition_steps"] = [
    report["trace"][-1]["role_phase_transition_step"] for report in reports
]
summary["final_role_error_per_node"] = [
    report["trace"][-1]["relation_degree_error_per_node"] for report in reports
]
(root / "aggregate.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[COMPLETE] $OUT_ROOT/aggregate.json" | tee -a "$LOG_ROOT/driver.log"

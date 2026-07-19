#!/usr/bin/env bash
# Diagnose ACM role-recovery budget before applying closure refinement.
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

DATASET="${DATASET:-data/ACM_subgraphs/subgraph_000}"
CHECKPOINT="${CHECKPOINT:-results/role_stop_context_gate_0720/acm_no_role/rooted_context_residual.pt}"
OUT_ROOT="${OUT_ROOT:-results/acm_role_budget_parameter_gate_0720}"
LOG_ROOT="${LOG_ROOT:-logs/acm_role_budget_parameter_gate_0720}"
SEED="${SEED:-0}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

read -r -a gpus <<< "$GPU_LIST"
(( ${#gpus[@]} >= 8 )) || { echo "eight GPUs are required" >&2; exit 2; }
[[ -d "$DATASET" ]] || { echo "missing dataset: $DATASET" >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 2; }

# name steps candidates relative_threshold targeted_fraction auto_balance phase
configs=(
  "baseline_r28_c64 50000 64 0.28 0.50 0.50 auto"
  "r20_c64 60000 64 0.20 0.50 0.50 auto"
  "r15_c64 70000 64 0.15 0.50 0.50 auto"
  "r10_c64 85000 64 0.10 0.50 0.50 auto"
  "r15_c128 70000 128 0.15 0.50 0.50 auto"
  "r10_c128 85000 128 0.10 0.50 0.50 auto"
  "r15_c128_target1 70000 128 0.15 1.00 1.00 auto"
  "reloc_only_c128 70000 128 0.00 1.00 1.00 auto"
)

pids=()
for index in "${!configs[@]}"; do
  read -r name steps candidates threshold targeted balance phase <<< "${configs[$index]}"
  gpu="${gpus[$index]}"
  output="$OUT_ROOT/$name"
  log="$LOG_ROOT/$name.log"
  echo "[ACM-PARAM-START] gpu=$gpu name=$name steps=$steps C=$candidates rho=$threshold targeted=$targeted balance=$balance phase=$phase"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_role_guided_replacement_gate.py \
    --dataset-dir "$DATASET" \
    --context-checkpoint "$CHECKPOINT" --context-scale 0 \
    --output-dir "$output" --seed "$SEED" --device cuda \
    --sampling-backend gpu --gpu-candidate-mode canonical_ordered \
    --gpu-proposal-pool-size 0 --gpu-proposal-chunk-size 4096 \
    --family-count-mode binomial --strict-target-free \
    --steps "$steps" --candidates "$candidates" \
    --metrics-every 2500 --model-time-horizon-steps 170000 \
    --role-weight 0.05 --gain-scale 10 --schedule-power 1 --schedule-floor 1 \
    --role-error-per-node-threshold 5.5 \
    --relative-role-error-threshold "$threshold" \
    --closure-candidate-fraction 1.0 \
    --targeted-role-candidate-fraction "$targeted" \
    --auto-role-balance "$balance" \
    --phase-policy "$phase" \
    > "$log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "one or more ACM parameter jobs failed" >&2; exit 1; }

python - "$OUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/gate_report.json")):
    report = json.loads(path.read_text())
    final = report["final"]
    real = report["real"]
    trace = report["trace"]
    transition = next(
        (
            int(row["role_phase_transition_step"])
            for row in trace
            if row.get("role_phase_transition_step") not in (None, -1)
        ),
        None,
    )
    item = {
        "name": path.parent.name,
        "steps": int(report["steps"]),
        "candidates": int(report["candidates"]),
        "transition_step": transition,
        "action_counts": report["action_counts"],
    }
    for metric in (
        "triangles", "clustering", "transitivity", "assortativity", "lcc",
        "sum_degree_sq", "max_degree",
    ):
        generated = float(final[metric])
        target = float(real[metric])
        item[metric] = generated
        item[f"{metric}_relative_error"] = abs(generated - target) / max(abs(target), 1e-12)
    rows.append(item)

summary = {"format": "dihug-acm-role-budget-parameter-gate-v1", "rows": rows}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
for row in sorted(rows, key=lambda value: value["triangles_relative_error"]):
    print(
        row["name"],
        f"tri={row['triangles']:.0f}",
        f"tri_err={row['triangles_relative_error']:.3f}",
        f"clust={row['clustering']:.5f}",
        f"clust_err={row['clustering_relative_error']:.3f}",
        f"degree2_err={row['sum_degree_sq_relative_error']:.3f}",
        f"maxdeg_err={row['max_degree_relative_error']:.3f}",
        f"transition={row['transition_step']}",
    )
PY

echo "[ACM-PARAM-DONE] $OUT_ROOT/summary.json"

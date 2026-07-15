#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="${DATASET:-data/IMDB_movie_subgraphs/subgraph_000}"
OUT_ROOT="${OUT_ROOT:-results/imdb_exchangeable_compact_switch_main}"
LOG_ROOT="${LOG_ROOT:-logs/imdb_exchangeable_compact_switch_main}"
PYTHON="${PYTHON:-python}"
PRIOR="$OUT_ROOT/prior/exchangeable_role_prior.json"
CHECKPOINT="$OUT_ROOT/model/mixed_edit_diffusion.pt"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

# The prior stores only type-level mixture parameters and family marginals.
# Systematic component sampling is exchangeable and reduces finite-graph role noise.
if [[ ! -f "$PRIOR" ]]; then
  "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
    --mode fit-gate --dataset-dir "$DATASET" --output-dir "$OUT_ROOT/prior" \
    --components 64 --role-model categorical \
    --role-component-sampling systematic --prior-samples 32 --seed 0 \
    > "$LOG_ROOT/prior.log" 2>&1
fi

# Train one shared reverse kernel. The compact profile keeps local edit features
# and exact typed/degree-normalized closure deltas; it does not use target metrics.
if [[ ! -f "$CHECKPOINT" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/run_mixed_edit_diffusion.py \
    --mode train --denoiser listwise --kernel bdrs --dataset-dir "$DATASET" \
    --output-dir "$OUT_ROOT/model" --device cuda --seed 0 \
    --trajectories 32 --forward-steps 1000 --record-every 10 \
    --train-candidates 16 --feature-profile typed_closure_compact_v2 \
    --epochs 100 --batch-size 64 --hidden-dim 256 --layers 3 \
    --training-objective inverse_ce --fixed-t-trials 32 \
    > "$LOG_ROOT/train.log" 2>&1
fi

pids=()
for seed in {0..7}; do
  phase_a="$OUT_ROOT/seed${seed}_reverse20k"
  phase_b="$OUT_ROOT/seed${seed}_refine10k"
  phase_c="$OUT_ROOT/seed${seed}_refine20k"
  (
    CUDA_VISIBLE_DEVICES="$seed" "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
      --mode sample --output-dir "$phase_a" --prior "$PRIOR" \
      --checkpoint "$CHECKPOINT" --device cuda --seed "$seed" \
      --sample-steps 20000 --sample-candidates 64 --sample-actions switch \
      --motif-refinement-stage --metrics-every 1000 \
      > "$LOG_ROOT/seed${seed}_reverse20k.log" 2>&1

    CUDA_VISIBLE_DEVICES="$seed" "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
      --mode sample --output-dir "$phase_b" --prior "$PRIOR" \
      --checkpoint "$CHECKPOINT" --device cuda --seed "$((seed + 200))" \
      --sample-steps 10000 --sample-candidates 64 --sample-actions switch \
      --motif-refinement-stage --initial-graph "$phase_a/generated_graph.json" \
      --fixed-time-fraction 0.0 --metrics-every 1000 \
      > "$LOG_ROOT/seed${seed}_refine10k.log" 2>&1

    CUDA_VISIBLE_DEVICES="$seed" "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
      --mode sample --output-dir "$phase_c" --prior "$PRIOR" \
      --checkpoint "$CHECKPOINT" --device cuda --seed "$((seed + 400))" \
      --sample-steps 10000 --sample-candidates 64 --sample-actions switch \
      --motif-refinement-stage --initial-graph "$phase_b/generated_graph.json" \
      --fixed-time-fraction 0.0 --metrics-every 1000 \
      > "$LOG_ROOT/seed${seed}_refine20k.log" 2>&1
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

# Evaluation reads the real graph only after generation. No real metric or edge
# enters either sampling phase.
"$PYTHON" - "$DATASET" "$OUT_ROOT" <<'PY'
import json
import pathlib
import statistics
import sys

from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics

dataset = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
real = graph_metrics(load_typed_graph(dataset, active_only=True))
rows = []
for seed in range(8):
    report = json.loads(
        (root / f"seed{seed}_refine20k" / "sampling_report.json").read_text()
    )
    final = report["final"]
    rows.append(
        {
            "seed": seed,
            **final,
            "triangle_relative_error": abs(final["triangles"] - real["triangles"])
            / max(real["triangles"], 1.0),
            "clustering_relative_error": abs(final["clustering"] - real["clustering"])
            / max(real["clustering"], 1.0e-12),
        }
    )
keys = (
    "edges",
    "triangles",
    "clustering",
    "transitivity",
    "lcc",
    "assortativity",
    "sum_degree_sq",
    "max_degree",
)
summary = {
    "format": "dihug-imdb-exchangeable-compact-switch-main-v1",
    "inference_uses_clean_graph": False,
    "real_metrics_used_for_posthoc_evaluation_only": True,
    "real": real,
    "rows": rows,
    "mean": {key: statistics.fmean(row[key] for row in rows) for key in keys},
    "std": {key: statistics.pstdev(row[key] for row in rows) for key in keys},
    "within_10pct_both": sum(
        row["triangle_relative_error"] <= 0.1
        and row["clustering_relative_error"] <= 0.1
        for row in rows
    ),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[IMDB-EXCHANGEABLE-COMPACT-SWITCH] complete: $OUT_ROOT/summary.json"

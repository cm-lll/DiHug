#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="${DATASET:-data/PubMed_baseline_subgraphs/subgraph_000}"
OUT_ROOT="${OUT_ROOT:-results/pubmed_exchangeable_compact_switch_main}"
LOG_ROOT="${LOG_ROOT:-logs/pubmed_exchangeable_compact_switch_main}"
PYTHON="${PYTHON:-python}"
PRIOR="$OUT_ROOT/prior/exchangeable_role_prior.json"
CHECKPOINT="$OUT_ROOT/model/mixed_edit_diffusion.pt"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

# The unbounded empirical codebook is still type-exchangeable: it stores only
# anonymous role vectors and their frequencies, never node assignments or edges.
if [[ ! -f "$PRIOR" ]]; then
  "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
    --mode fit-gate --dataset-dir "$DATASET" --output-dir "$OUT_ROOT/prior" \
    --components 0 --role-model categorical \
    --role-component-sampling systematic --prior-samples 16 --seed 0 \
    > "$LOG_ROOT/prior.log" 2>&1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/run_mixed_edit_diffusion.py \
    --mode train --denoiser listwise --kernel bdrs --dataset-dir "$DATASET" \
    --output-dir "$OUT_ROOT/model" --device cuda --seed 0 \
    --trajectories 32 --forward-steps 3000 --record-every 30 \
    --train-candidates 16 --feature-profile typed_closure_compact_v2 \
    --epochs 100 --batch-size 64 --hidden-dim 256 --layers 3 \
    --training-objective inverse_ce --fixed-t-trials 32 \
    > "$LOG_ROOT/train.log" 2>&1
fi

pids=()
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES="$seed" "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
    --mode sample --output-dir "$OUT_ROOT/seed${seed}_sample5k" \
    --prior "$PRIOR" --checkpoint "$CHECKPOINT" --device cuda --seed "$seed" \
    --sample-steps 5000 --sample-candidates 64 --sample-actions switch \
    --motif-refinement-stage --metrics-every 500 \
    > "$LOG_ROOT/seed${seed}_sample5k.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

# The clean graph is loaded only for post-generation evaluation.
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
        (root / f"seed{seed}_sample5k" / "sampling_report.json").read_text()
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
            "transitivity_relative_error": abs(
                final["transitivity"] - real["transitivity"]
            )
            / max(real["transitivity"], 1.0e-12),
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
    "format": "dihug-pubmed-exchangeable-compact-switch-main-v1",
    "inference_uses_clean_graph": False,
    "real_metrics_used_for_posthoc_evaluation_only": True,
    "real": real,
    "rows": rows,
    "mean": {key: statistics.fmean(row[key] for row in rows) for key in keys},
    "std": {key: statistics.pstdev(row[key] for row in rows) for key in keys},
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[PUBMED-EXCHANGEABLE-COMPACT-SWITCH] complete: $OUT_ROOT/summary.json"

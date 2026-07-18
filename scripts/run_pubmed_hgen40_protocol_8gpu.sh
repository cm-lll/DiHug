#!/usr/bin/env bash
set -euo pipefail

# Canonical fair-comparison runner for HGEN's materialized 40% held-out edge
# split. The held-out adjacency is used only to build the evaluation target.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="${DATASET:-data/PubMed_baseline_subgraphs/subgraph_000}"
HGEN_TEST_GRAPH="${HGEN_TEST_GRAPH:-/data2/lyh/gnn_project/HGEN/PubMed_baseline/test_graph.p}"
PRIOR="${PRIOR:-results/pubmed_exchangeable_compact_switch_gate_0716/prior_c512/exchangeable_role_prior.json}"
CHECKPOINT="${CHECKPOINT:-results/pubmed_exchangeable_compact_switch_gate_0716/model/mixed_edit_diffusion.pt}"
OUT_ROOT="${OUT_ROOT:-results/pubmed_hgen40_protocol}"
LOG_ROOT="${LOG_ROOT:-logs/pubmed_hgen40_protocol}"
# The full-graph terminal refinement uses 5,000 transitions.  A triangle
# survives an independent p-edge view with probability p^3, so the held-out
# protocol uses round(5000 * 0.4^3) = 320 transitions.  This schedule is fixed
# without reading held-out graph metrics.
SAMPLE_STEPS="${SAMPLE_STEPS:-320}"
SAMPLE_CANDIDATES="${SAMPLE_CANDIDATES:-64}"
SAMPLE_ACTIONS="${SAMPLE_ACTIONS:-switch}"
EDGE_VIEW_SOURCE="${EDGE_VIEW_SOURCE:-hgen_split}"
PYTHON="${PYTHON:-python}"
HGEN_PYTHON="${HGEN_PYTHON:-/data2/lyh/miniconda3/envs/hgen_env/bin/python}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

"$PYTHON" scripts/build_hgen_edge_split_target.py \
  --dataset-dir "$DATASET" --hgen-test-graph "$HGEN_TEST_GRAPH" \
  --output-dir "$OUT_ROOT" > "$LOG_ROOT/build_target.log" 2>&1

EDGE_COUNT="$($PYTHON -c "import json; print(json.load(open('$OUT_ROOT/split_report.json'))['generation_budget']['num_edges'])")"
pids=()
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES="$seed" "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
    --mode sample --output-dir "$OUT_ROOT/seed${seed}_sample${SAMPLE_STEPS}" \
    --prior "$PRIOR" --checkpoint "$CHECKPOINT" --device cuda --seed "$seed" \
    --edge-view-count "$EDGE_COUNT" \
    --edge-view-source "$EDGE_VIEW_SOURCE" \
    --sample-steps "$SAMPLE_STEPS" --sample-candidates "$SAMPLE_CANDIDATES" \
    --sample-actions "$SAMPLE_ACTIONS" --motif-refinement-stage \
    --metrics-every 250 > "$LOG_ROOT/seed${seed}.log" 2>&1 &
  pids+=("$!")
  sleep 2
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PYTHON" - "$OUT_ROOT" "$SAMPLE_STEPS" "$SAMPLE_ACTIONS" "$EDGE_VIEW_SOURCE" <<'PY'
import json
import pathlib
import statistics
import sys

from sparse_diffusion.structure_energy.graph import TypedGraph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics

root = pathlib.Path(sys.argv[1])
steps = int(sys.argv[2])
actions = sys.argv[3]
edge_view_source = sys.argv[4]
split = json.loads((root / "split_report.json").read_text())
target = TypedGraph.from_json_dict(json.loads((root / "target_graph.json").read_text()))
target_metrics = graph_metrics(target)
rows = []
for seed in range(8):
    sample_dir = root / f"seed{seed}_sample{steps}"
    report = json.loads((sample_dir / "sampling_report.json").read_text())
    generated = TypedGraph.from_json_dict(
        json.loads((sample_dir / "generated_graph.json").read_text())
    )
    final = graph_metrics(generated)
    target_edges = set(target.edge_family)
    generated_edges = set(generated.edge_family)
    row = {"seed": seed, **final}
    for key in ("triangles", "clustering", "transitivity", "lcc", "sum_degree_sq"):
        row[f"{key}_relative_error"] = abs(final[key] - target_metrics[key]) / max(
            abs(target_metrics[key]), 1.0e-12
        )
    row["edge_overlap"] = len(target_edges & generated_edges) / max(len(target_edges), 1)
    rows.append(row)

metric_keys = tuple(graph_metrics(target))
summary = {
    "format": "dihug-hgen40-heldout-edge-evaluation-v1",
    "protocol": {
        "nodes": "full_node_set",
        "target": "hgen_40_percent_held_out_edges",
        "split_seed": split["split_seed"],
        "generation_condition": "node_types_schema_and_exact_total_edge_budget_only",
        "inference_uses_target_adjacency": False,
        "inference_uses_target_family_counts": False,
        "inference_uses_target_degrees": False,
        "initialization": f"{edge_view_source}_view_of_learned_full_graph_prior_sample",
        "steps": steps,
        "actions": actions,
    },
    "target": target_metrics,
    "rows": rows,
    "mean": {key: statistics.fmean(row[key] for row in rows) for key in metric_keys},
    "std": {key: statistics.pstdev(row[key] for row in rows) for key in metric_keys},
    "relative_error_mean": {
        key: statistics.fmean(row[f"{key}_relative_error"] for row in rows)
        for key in ("triangles", "clustering", "transitivity", "lcc", "sum_degree_sq")
    },
    "edge_overlap_mean": statistics.fmean(row["edge_overlap"] for row in rows),
}
(root / f"summary_{steps}.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

if [[ -x "$HGEN_PYTHON" ]]; then
  generated=()
  for seed in {0..7}; do
    generated+=("$OUT_ROOT/seed${seed}_sample${SAMPLE_STEPS}/generated_graph.json")
  done
  "$HGEN_PYTHON" scripts/evaluate_hgen_edge_split_metrics.py \
    --target "$OUT_ROOT/target_graph.json" \
    --generated "${generated[@]}" \
    --output "$OUT_ROOT/hgen_metric_summary_${SAMPLE_STEPS}.json" \
    > "$LOG_ROOT/hgen_metrics_${SAMPLE_STEPS}.log" 2>&1
fi

echo "[PUBMED-HGEN40] complete: $OUT_ROOT/summary_${SAMPLE_STEPS}.json"

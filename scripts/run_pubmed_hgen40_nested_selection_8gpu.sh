#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="${DATASET:-data/PubMed_baseline_subgraphs/subgraph_000}"
PRIOR="${PRIOR:-results/pubmed_exchangeable_compact_switch_gate_0716/prior_c512/exchangeable_role_prior.json}"
CHECKPOINT="${CHECKPOINT:-results/pubmed_exchangeable_compact_switch_gate_0716/model/mixed_edit_diffusion.pt}"
HGEN_TEST_GRAPH="${HGEN_TEST_GRAPH:-/data2/lyh/gnn_project/HGEN/PubMed_baseline/test_graph.p}"
OUT_ROOT="${OUT_ROOT:-results/pubmed_hgen40_nested_hgen_view}"
LOG_ROOT="${LOG_ROOT:-logs/pubmed_hgen40_nested_hgen_view}"
PYTHON="${PYTHON:-python}"
VALIDATION_SEEDS="${VALIDATION_SEEDS:-1701 1702 1703}"
mkdir -p "$OUT_ROOT/validation_targets" "$OUT_ROOT/validation_runs" "$LOG_ROOT"

for required in "$PRIOR" "$CHECKPOINT" "$HGEN_TEST_GRAPH"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 2; }
done

# Hyperparameters are selected only on independent HGEN-style 40% splits of
# the training graph. The official held-out graph is not opened until the
# selected configuration has been frozen.
for seed in $VALIDATION_SEEDS; do
  target_dir="$OUT_ROOT/validation_targets/seed${seed}"
  if [[ ! -f "$target_dir/target_graph.json" ]]; then
    "$PYTHON" scripts/build_hgen_edge_split_target.py \
      --dataset-dir "$DATASET" --generate-split --output-dir "$target_dir" \
      --test-share 0.4 --split-seed "$seed" \
      > "$LOG_ROOT/build_validation_seed${seed}.log" 2>&1
  fi
done
"$PYTHON" - "$OUT_ROOT/validation_targets" $VALIDATION_SEEDS <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
seeds = [int(value) for value in sys.argv[2:]]
rows = []
edge_counts = set()
for seed in seeds:
    report = json.loads((root / f"seed{seed}" / "split_report.json").read_text())
    edge_counts.add(int(report["generation_budget"]["num_edges"]))
    rows.append({"seed": seed, "metrics": report["target_metrics"]})
if len(edge_counts) != 1:
    raise AssertionError("HGEN-style validation splits disagree on edge count")
(root / "manifest.json").write_text(json.dumps({
    "format": "dihug-hgen40-nested-validation-v2",
    "split_algorithm": "hgen_every_node_edge_cover",
    "edge_share": 0.4,
    "edge_count": next(iter(edge_counts)),
    "uses_official_test_graph": False,
    "rows": rows,
}, indent=2, sort_keys=True))
PY

EDGE_COUNT="$($PYTHON -c "import json; print(json.load(open('$OUT_ROOT/validation_targets/manifest.json'))['edge_count'])")"
CONFIGS=(
  "switch 16 320"
  "switch 16 640"
  "switch 64 320"
  "switch 64 640"
  "relocation_switch 16 320"
  "relocation_switch 16 640"
  "relocation_switch 64 320"
  "relocation_switch 64 640"
)

pids=()
for worker in "${!CONFIGS[@]}"; do
  read -r actions candidates steps <<< "${CONFIGS[$worker]}"
  tag="${actions}_c${candidates}_s${steps}"
  (
    for seed in $VALIDATION_SEEDS; do
      run_dir="$OUT_ROOT/validation_runs/$tag/seed${seed}"
      mkdir -p "$run_dir"
      if [[ ! -f "$run_dir/sampling_report.json" ]]; then
        CUDA_VISIBLE_DEVICES="$worker" "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
          --mode sample --output-dir "$run_dir" --prior "$PRIOR" \
          --checkpoint "$CHECKPOINT" --device cuda --seed "$seed" \
          --edge-view-count "$EDGE_COUNT" --edge-view-source hgen_split \
          --sample-steps "$steps" \
          --sample-candidates "$candidates" --sample-actions "$actions" \
          --motif-refinement-stage --metrics-every "$steps" \
          > "$LOG_ROOT/${tag}_seed${seed}.log" 2>&1
      fi
    done
  ) &
  pids+=("$!")
  sleep 2
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PYTHON" - "$OUT_ROOT" $VALIDATION_SEEDS <<'PY'
import json
import pathlib
import statistics
import sys

from sparse_diffusion.structure_energy.graph import TypedGraph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics

root = pathlib.Path(sys.argv[1])
seeds = [int(value) for value in sys.argv[2:]]
weights = {
    "triangles": 0.25,
    "clustering": 0.20,
    "transitivity": 0.15,
    "lcc": 0.20,
    "sum_degree_sq": 0.10,
    "assortativity": 0.10,
}
rows = []
for config_dir in sorted((root / "validation_runs").iterdir()):
    action_part, candidate_part, step_part = config_dir.name.rsplit("_", 2)
    per_seed = []
    for seed in seeds:
        target = TypedGraph.from_json_dict(json.loads(
            (root / "validation_targets" / f"seed{seed}" / "target_graph.json").read_text()
        ))
        target_metrics = graph_metrics(target)
        report = json.loads(
            (config_dir / f"seed{seed}" / "sampling_report.json").read_text()
        )
        generated = report["final"]
        errors = {
            key: abs(generated[key] - target_metrics[key])
            / max(abs(target_metrics[key]), 1.0e-12)
            for key in weights
        }
        per_seed.append({
            "seed": seed,
            "score": sum(weights[key] * errors[key] for key in weights),
            "errors": errors,
            "generated": generated,
            "target": target_metrics,
        })
    rows.append({
        "tag": config_dir.name,
        "actions": action_part,
        "candidates": int(candidate_part[1:]),
        "steps": int(step_part[1:]),
        "score_mean": statistics.fmean(item["score"] for item in per_seed),
        "score_std": statistics.pstdev(item["score"] for item in per_seed),
        "per_seed": per_seed,
    })
rows.sort(key=lambda item: (item["score_mean"], item["score_std"], item["tag"]))
summary = {
    "format": "dihug-hgen40-nested-selection-v1",
    "selection_uses_official_test_graph": False,
    "weights": weights,
    "best": rows[0],
    "rows": rows,
}
(root / "validation_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True)
)
print(json.dumps(summary["best"], sort_keys=True))
PY

read -r BEST_ACTIONS BEST_CANDIDATES BEST_STEPS < <(
  "$PYTHON" - "$OUT_ROOT/validation_summary.json" <<'PY'
import json, sys
best = json.load(open(sys.argv[1]))["best"]
print(best["actions"], best["candidates"], best["steps"])
PY
)

echo "[HGEN40-NESTED] selected actions=$BEST_ACTIONS candidates=$BEST_CANDIDATES steps=$BEST_STEPS"
OUT_ROOT="$OUT_ROOT/official" LOG_ROOT="$LOG_ROOT/official" \
  DATASET="$DATASET" HGEN_TEST_GRAPH="$HGEN_TEST_GRAPH" \
  PRIOR="$PRIOR" CHECKPOINT="$CHECKPOINT" \
  SAMPLE_ACTIONS="$BEST_ACTIONS" SAMPLE_CANDIDATES="$BEST_CANDIDATES" \
  SAMPLE_STEPS="$BEST_STEPS" \
  bash scripts/run_pubmed_hgen40_protocol_8gpu.sh

echo "[HGEN40-NESTED] complete: $OUT_ROOT/official/summary_${BEST_STEPS}.json"

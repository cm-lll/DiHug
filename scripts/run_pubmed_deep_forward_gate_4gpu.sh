#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-data/PubMed_baseline_subgraphs/subgraph_000}"
OUT_ROOT="${OUT_ROOT:-results/pubmed_deep_forward_gate_0716}"
LOG_ROOT="${LOG_ROOT:-logs/pubmed_deep_forward_gate_0716}"
PRIOR="${PRIOR:-results/pubmed_role_corruption_gate_0716/shared_prior_rho05/exchangeable_role_prior.json}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

if [[ ! -f "$PRIOR" ]]; then
  prior_dir="$OUT_ROOT/shared_prior_rho05"
  "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
    --mode fit-gate --dataset-dir "$DATASET" --output-dir "$prior_dir" \
    --components 512 --role-model categorical \
    --role-component-sampling systematic --role-correlation-strength 0.5 \
    --randomization-steps-per-edge 20 --prior-samples 2 --seed 0 \
    > "$LOG_ROOT/prior_rho05.log" 2>&1
  PRIOR="$prior_dir/exchangeable_role_prior.json"
fi

run_depth() {
  local gpu="$1" name="$2" seed="$3" forward_steps="$4"
  local record_every=$((forward_steps / 100))
  local out="$OUT_ROOT/$name"
  local log="$LOG_ROOT/$name.log"
  mkdir -p "$out"
  echo "[START] gpu=$gpu name=$name seed=$seed forward=$forward_steps" \
    | tee -a "$LOG_ROOT/launcher.log"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_mixed_edit_diffusion.py \
    --mode train --denoiser listwise --kernel bdrs \
    --dataset-dir "$DATASET" --output-dir "$out/model" --device cuda \
    --seed "$seed" --trajectories 32 --forward-steps "$forward_steps" \
    --record-every "$record_every" --train-candidates 16 \
    --feature-profile typed_closure_compact_v2 \
    --epochs 20 --batch-size 64 --hidden-dim 256 --layers 3 \
    --training-objective inverse_ce --fixed-t-trials 16 \
    > "$log" 2>&1

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
    --mode sample --output-dir "$out/rho05_sample5k" --prior "$PRIOR" \
    --checkpoint "$out/model/mixed_edit_diffusion.pt" --device cuda \
    --seed "$seed" --sample-steps 5000 --sample-candidates 64 \
    --sample-actions relocation_switch --motif-refinement-stage \
    --metrics-every 500 >> "$log" 2>&1
  echo "[DONE] gpu=$gpu name=$name" | tee -a "$LOG_ROOT/launcher.log"
}

pids=()
run_depth 0 depth3k_s0 0 3000 & pids+=("$!")
sleep 15
run_depth 1 depth10k_s0 0 10000 & pids+=("$!")
sleep 15
run_depth 2 depth20k_s0 0 20000 & pids+=("$!")
sleep 15
run_depth 3 depth20k_s1 1 20000 & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "deep forward gate failed" >&2; exit 1; }

"$PYTHON" - "$DATASET" "$OUT_ROOT" <<'PY'
import json
import pathlib
import sys

from scripts.run_exchangeable_role_prior_gate import degree_distances
from sparse_diffusion.structure_energy.graph import TypedGraph, load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics

dataset = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
clean = load_typed_graph(dataset, active_only=True)
real = graph_metrics(clean)
specs = (
    ("depth3k_s0", 0, 3000),
    ("depth10k_s0", 0, 10000),
    ("depth20k_s0", 0, 20000),
    ("depth20k_s1", 1, 20000),
)
rows = []
for name, seed, forward_steps in specs:
    report = json.loads((root / name / "rho05_sample5k" / "sampling_report.json").read_text())
    train = json.loads((root / name / "model" / "train_report.json").read_text())
    generated = TypedGraph.from_json_dict(
        json.loads((root / name / "rho05_sample5k" / "generated_graph.json").read_text())
    )
    final = report["final"]
    action_counts = report["action_counts"]
    applied = action_counts["relocation"] + action_counts["switch"]
    rows.append({
        "name": name,
        "seed": seed,
        "forward_steps": forward_steps,
        "record_every": forward_steps // 100,
        "train_examples": train["train"]["train_examples"] + train["train"]["validation_examples"],
        "validation_top1": train["train"]["validation_top1"],
        "initial": report["initial"],
        "final": final,
        "triangle_relative_error": abs(final["triangles"] - real["triangles"]) / real["triangles"],
        "clustering_relative_error": abs(final["clustering"] - real["clustering"]) / real["clustering"],
        "final_degree_distances": degree_distances(clean, generated),
        "action_counts": action_counts,
        "switch_fraction_among_applied": action_counts["switch"] / max(applied, 1),
    })

by_name = {row["name"]: row for row in rows}
control = by_name["depth3k_s0"]
deep = by_name["depth20k_s0"]
payload = {
    "format": "dihug-pubmed-deep-forward-gate-v1",
    "inference_uses_clean_graph": False,
    "real_metrics_used_for_posthoc_evaluation_only": True,
    "real": real,
    "rows": rows,
    "gate": {
        "deep20k_beats_3k_triangles": deep["final"]["triangles"] > control["final"]["triangles"],
        "deep20k_beats_3k_clustering": deep["final"]["clustering"] > control["final"]["clustering"],
        "deep20k_improves_role_correlation": deep["final_degree_distances"]["relation_degree_correlation_error"] < control["final_degree_distances"]["relation_degree_correlation_error"],
        "deep20k_keeps_switch_share": deep["switch_fraction_among_applied"] >= 0.15,
    },
}
(root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "[PUBMED-DEEP-FORWARD-GATE] complete: $OUT_ROOT/summary.json"

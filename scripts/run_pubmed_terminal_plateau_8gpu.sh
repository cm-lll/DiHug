#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-data/PubMed_baseline_subgraphs/subgraph_000}"
PRIOR="${PRIOR:-results/pubmed_role_corruption_gate_0716/shared_prior_rho05/exchangeable_role_prior.json}"
OUT_ROOT="${OUT_ROOT:-results/pubmed_terminal_plateau_0716}"
LOG_ROOT="${LOG_ROOT:-logs/pubmed_terminal_plateau_0716}"
SAMPLE_STEPS="${SAMPLE_STEPS:-15000}"
TIME_SCHEDULE_STEPS="${TIME_SCHEDULE_STEPS:-5000}"
METRICS_EVERY="${METRICS_EVERY:-500}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/launcher.log"

declare -a MODELS=(
  "depth10k depth10k_s0/model/mixed_edit_diffusion.pt"
  "depth10k depth10k_s0/model/mixed_edit_diffusion.pt"
  "depth10k depth10k_s0/model/mixed_edit_diffusion.pt"
  "depth10k depth10k_s0/model/mixed_edit_diffusion.pt"
  "depth3k depth3k_s0/model/mixed_edit_diffusion.pt"
  "depth3k depth3k_s0/model/mixed_edit_diffusion.pt"
  "depth3k depth3k_s0/model/mixed_edit_diffusion.pt"
  "depth3k depth3k_s0/model/mixed_edit_diffusion.pt"
)
declare -a SEEDS=(0 1 2 3 0 1 2 3)

run_one() {
  local gpu="$1" model_name="$2" checkpoint_rel="$3" seed="$4"
  local name="${model_name}_sample_s${seed}"
  local out="$OUT_ROOT/$name"
  local log="$LOG_ROOT/$name.log"
  mkdir -p "$out"
  echo "[START] gpu=$gpu model=$model_name seed=$seed" | tee -a "$LOG_ROOT/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
    --mode sample --output-dir "$out" --prior "$PRIOR" \
    --checkpoint "results/pubmed_deep_forward_gate_0716/$checkpoint_rel" \
    --device cuda --seed "$seed" \
    --sample-steps "$SAMPLE_STEPS" \
    --time-schedule-steps "$TIME_SCHEDULE_STEPS" \
    --sample-candidates 64 --sample-actions relocation_switch \
    --motif-refinement-stage --metrics-every "$METRICS_EVERY" \
    > "$log" 2>&1
  echo "[DONE] gpu=$gpu model=$model_name seed=$seed" | tee -a "$LOG_ROOT/launcher.log"
}

pids=()
for gpu in $(seq 0 7); do
  read -r model checkpoint_rel <<< "${MODELS[$gpu]}"
  run_one "$gpu" "$model" "$checkpoint_rel" "${SEEDS[$gpu]}" &
  pids+=("$!")
  sleep 5
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "terminal plateau matrix failed" >&2; exit 1; }

"$PYTHON" - "$DATASET" "$OUT_ROOT" "$TIME_SCHEDULE_STEPS" <<'PY'
import json
import pathlib
import statistics
import sys

from scripts.run_exchangeable_role_prior_gate import degree_distances
from sparse_diffusion.structure_energy.graph import TypedGraph, load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics

dataset = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
schedule_steps = int(sys.argv[3])
clean = load_typed_graph(dataset, active_only=True)
real = graph_metrics(clean)


def relative_span(values):
    values = [float(value) for value in values]
    return (max(values) - min(values)) / max(abs(statistics.mean(values)), 1.0e-12)


rows = []
for model in ("depth3k", "depth10k"):
    for seed in range(4):
        run_dir = root / f"{model}_sample_s{seed}"
        report = json.loads((run_dir / "sampling_report.json").read_text())
        generated = TypedGraph.from_json_dict(
            json.loads((run_dir / "generated_graph.json").read_text())
        )
        trace = report["trace"]
        at_schedule_end = next(
            row for row in trace if int(row["step"]) == schedule_steps
        )
        final = report["final"]
        tail = trace[-5:]
        spans = {
            metric: relative_span([row[metric] for row in tail])
            for metric in ("triangles", "clustering", "lcc", "sum_degree_sq")
        }
        plateau = all(value <= 0.01 for value in spans.values())
        rows.append({
            "model": model,
            "seed": seed,
            "schedule_end": at_schedule_end,
            "final": final,
            "terminal_gain": {
                metric: float(final[metric]) - float(at_schedule_end[metric])
                for metric in ("triangles", "clustering", "lcc", "sum_degree_sq")
            },
            "tail_relative_span": spans,
            "plateau_last_2k": plateau,
            "final_degree_distances": degree_distances(clean, generated),
            "triangle_relative_error": abs(final["triangles"] - real["triangles"])
            / real["triangles"],
            "clustering_relative_error": abs(final["clustering"] - real["clustering"])
            / real["clustering"],
            "action_counts": report["action_counts"],
        })

groups = {}
for model in ("depth3k", "depth10k"):
    selected = [row for row in rows if row["model"] == model]
    groups[model] = {
        "seeds": len(selected),
        "plateau_count": sum(row["plateau_last_2k"] for row in selected),
    }
    for stage in ("schedule_end", "final"):
        groups[model][stage] = {}
        for metric in ("triangles", "clustering", "lcc", "sum_degree_sq"):
            values = [float(row[stage][metric]) for row in selected]
            groups[model][stage][metric] = {
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values),
            }

payload = {
    "format": "dihug-pubmed-terminal-plateau-v1",
    "sampling_uses_clean_graph": False,
    "real_metrics_used_for_posthoc_evaluation_only": True,
    "plateau_definition": (
        "last five checkpoints (2k steps) each have <=1% relative span in "
        "triangles, clustering, LCC, and sum_degree_sq"
    ),
    "real": real,
    "groups": groups,
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "[PUBMED-TERMINAL-PLATEAU] complete: $OUT_ROOT/summary.json"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-data/PubMed_baseline_subgraphs/subgraph_000}"
OUT_ROOT="${OUT_ROOT:-results/pubmed_role_corruption_gate_0716}"
LOG_ROOT="${LOG_ROOT:-logs/pubmed_role_corruption_gate_0716}"
PRIOR="$OUT_ROOT/shared_prior_rho05/exchangeable_role_prior.json"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

if [[ ! -f "$PRIOR" ]]; then
  "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
    --mode fit-gate --dataset-dir "$DATASET" \
    --output-dir "$OUT_ROOT/shared_prior_rho05" \
    --components 512 --role-model categorical \
    --role-component-sampling systematic --role-correlation-strength 0.5 \
    --randomization-steps-per-edge 20 --prior-samples 2 --seed 0 \
    > "$LOG_ROOT/prior_rho05.log" 2>&1
fi

run_gate() {
  local gpu="$1" name="$2" fraction="$3" strengths="$4"
  local out="$OUT_ROOT/$name"
  local log="$LOG_ROOT/$name.log"
  mkdir -p "$out"
  echo "[START] gpu=$gpu name=$name fraction=$fraction strengths=$strengths" \
    | tee -a "$LOG_ROOT/launcher.log"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_mixed_edit_diffusion.py \
    --mode train --denoiser listwise --kernel bdrs \
    --dataset-dir "$DATASET" --output-dir "$out/model" --device cuda --seed 0 \
    --trajectories 16 --forward-steps 3000 --record-every 30 \
    --train-candidates 16 --feature-profile typed_closure_compact_v2 \
    --epochs 20 --batch-size 64 --hidden-dim 256 --layers 3 \
    --training-objective inverse_ce --fixed-t-trials 16 \
    --role-corruption-fraction "$fraction" \
    --role-correlation-strengths "$strengths" \
    --role-prior-components 512 \
    --role-prior-randomization-steps-per-edge 2 \
    > "$log" 2>&1

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
    --mode sample --output-dir "$out/rho05_sample5k" --prior "$PRIOR" \
    --checkpoint "$out/model/mixed_edit_diffusion.pt" --device cuda --seed 0 \
    --sample-steps 5000 --sample-candidates 64 \
    --sample-actions relocation_switch --motif-refinement-stage \
    --metrics-every 500 >> "$log" 2>&1
  echo "[DONE] gpu=$gpu name=$name" | tee -a "$LOG_ROOT/launcher.log"
}

pids=()
run_gate 0 control 0.0 "1" & pids+=("$!")
sleep 15
run_gate 1 f50_all 0.5 "0,0.25,0.5,0.75,1" & pids+=("$!")
sleep 15
run_gate 2 f50_weak 0.5 "0,0.25,0.5" & pids+=("$!")
sleep 15
run_gate 3 f100_all 1.0 "0,0.25,0.5,0.75,1" & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "role corruption gate failed" >&2; exit 1; }

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
rows = []
for name in ("control", "f50_all", "f50_weak", "f100_all"):
    report = json.loads((root / name / "rho05_sample5k" / "sampling_report.json").read_text())
    train = json.loads((root / name / "model" / "train_report.json").read_text())
    generated = TypedGraph.from_json_dict(
        json.loads((root / name / "rho05_sample5k" / "generated_graph.json").read_text())
    )
    final = report["final"]
    rows.append({
        "name": name,
        "training_noise": train["training_noise"],
        "validation_top1": train["train"].get("validation_top1"),
        "initial": report["initial"],
        "final": final,
        "triangle_relative_error": abs(final["triangles"] - real["triangles"]) / real["triangles"],
        "clustering_relative_error": abs(final["clustering"] - real["clustering"]) / real["clustering"],
        "final_degree_distances": degree_distances(clean, generated),
        "action_counts": report["action_counts"],
    })

baseline_path = pathlib.Path(
    "results/role_prior_strength_ablation_0716_s0/strength/"
    "pubmed_rho0p50/sample/sampling_report.json"
)
baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
payload = {
    "format": "dihug-pubmed-role-corruption-gate-v1",
    "inference_uses_clean_graph": False,
    "real_metrics_used_for_posthoc_evaluation_only": True,
    "real": real,
    "old_checkpoint_rho05_reference": baseline,
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "[PUBMED-ROLE-CORRUPTION-GATE] complete: $OUT_ROOT/summary.json"

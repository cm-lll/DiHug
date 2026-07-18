#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-results/hgen_official_fullgraph_pilot_0716}"
LOG_ROOT="${LOG_ROOT:-logs/hgen_official_fullgraph_pilot_0716}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/launcher.log"

dataset_dir() {
  case "$1" in
    syn100) echo "data/HGEN_syn_100_subgraphs/subgraph_000" ;;
    syn200) echo "data/HGEN_syn_200_subgraphs/subgraph_000" ;;
    syn500) echo "data/HGEN_syn_500_subgraphs/subgraph_000" ;;
    dblp) echo "data/DBLP_four_area_subgraphs/subgraph_000" ;;
    *) echo "unknown dataset key: $1" >&2; return 1 ;;
  esac
}

fit_prior() {
  local key="$1" randomization="$2"
  local out="$OUT_ROOT/$key/shared_prior"
  mkdir -p "$out"
  if [[ -f "$out/exchangeable_role_prior.json" ]]; then return; fi
  "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
    --mode fit-gate --dataset-dir "$(dataset_dir "$key")" --output-dir "$out" \
    --components 64 --role-model categorical \
    --role-component-sampling systematic --role-correlation-strength 1.0 \
    --randomization-steps-per-edge "$randomization" --prior-samples 2 --seed 0 \
    > "$LOG_ROOT/${key}_prior.log" 2>&1
}

fit_prior syn100 10 & prior_pids=("$!")
fit_prior syn200 10 & prior_pids+=("$!")
fit_prior syn500 10 & prior_pids+=("$!")
fit_prior dblp 2 & prior_pids+=("$!")
for pid in "${prior_pids[@]}"; do wait "$pid"; done

run_one() {
  local gpu="$1" key="$2" seed="$3" forward_steps="$4" sample_steps="$5"
  local record_every=$((forward_steps / 100))
  local out="$OUT_ROOT/$key/seed${seed}"
  local log="$LOG_ROOT/${key}_s${seed}.log"
  mkdir -p "$out"
  echo "[START] gpu=$gpu dataset=$key seed=$seed forward=$forward_steps sample=$sample_steps" \
    | tee -a "$LOG_ROOT/launcher.log"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_mixed_edit_diffusion.py \
    --mode train --denoiser listwise --kernel bdrs \
    --dataset-dir "$(dataset_dir "$key")" --output-dir "$out/model" \
    --device cuda --seed "$seed" --trajectories 32 \
    --forward-steps "$forward_steps" --record-every "$record_every" \
    --train-candidates 16 --feature-profile typed_closure_compact_v2 \
    --epochs 20 --batch-size 64 --hidden-dim 256 --layers 3 \
    --training-objective inverse_ce --fixed-t-trials 16 \
    > "$log" 2>&1

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
    --mode sample --output-dir "$out/sample" \
    --prior "$OUT_ROOT/$key/shared_prior/exchangeable_role_prior.json" \
    --checkpoint "$out/model/mixed_edit_diffusion.pt" --device cuda \
    --seed "$seed" --sample-steps "$sample_steps" --sample-candidates 64 \
    --sample-actions relocation_switch --motif-refinement-stage \
    --metrics-every "$((sample_steps / 10))" \
    >> "$log" 2>&1
  echo "[DONE] gpu=$gpu dataset=$key seed=$seed" | tee -a "$LOG_ROOT/launcher.log"
}

declare -a KEYS=(syn100 syn100 syn200 syn200 syn500 syn500 dblp dblp)
declare -a SEEDS=(0 1 0 1 0 1 0 1)
declare -a FORWARD=(1000 1000 2000 2000 5000 5000 20000 20000)
declare -a SAMPLE=(500 500 1000 1000 2500 2500 10000 10000)

pids=()
for gpu in $(seq 0 7); do
  run_one "$gpu" "${KEYS[$gpu]}" "${SEEDS[$gpu]}" \
    "${FORWARD[$gpu]}" "${SAMPLE[$gpu]}" &
  pids+=("$!")
  sleep 5
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "HGEN full-graph pilot failed" >&2; exit 1; }

"$PYTHON" - "$OUT_ROOT" <<'PY'
import json
import pathlib
import statistics
import sys

from scripts.run_exchangeable_role_prior_gate import degree_distances
from sparse_diffusion.structure_energy.graph import TypedGraph, load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics

root = pathlib.Path(sys.argv[1])
dataset_dirs = {
    "syn100": pathlib.Path("data/HGEN_syn_100_subgraphs/subgraph_000"),
    "syn200": pathlib.Path("data/HGEN_syn_200_subgraphs/subgraph_000"),
    "syn500": pathlib.Path("data/HGEN_syn_500_subgraphs/subgraph_000"),
    "dblp": pathlib.Path("data/DBLP_four_area_subgraphs/subgraph_000"),
}
rows = []
real_by_dataset = {}
for key, dataset_dir in dataset_dirs.items():
    clean = load_typed_graph(dataset_dir, active_only=True)
    real = graph_metrics(clean)
    real_by_dataset[key] = real
    for seed in (0, 1):
        run_dir = root / key / f"seed{seed}"
        report = json.loads((run_dir / "sample" / "sampling_report.json").read_text())
        train = json.loads((run_dir / "model" / "train_report.json").read_text())
        generated = TypedGraph.from_json_dict(
            json.loads((run_dir / "sample" / "generated_graph.json").read_text())
        )
        final = report["final"]
        rows.append({
            "dataset": key,
            "seed": seed,
            "validation_top1": train["train"]["validation_top1"],
            "initial": report["initial"],
            "final": final,
            "action_counts": report["action_counts"],
            "degree_distances": degree_distances(clean, generated),
            "relative_errors": {
                metric: abs(float(final[metric]) - float(real[metric]))
                / max(abs(float(real[metric])), 1.0)
                for metric in ("triangles", "clustering", "lcc", "sum_degree_sq")
            },
        })

groups = {}
for key in dataset_dirs:
    selected = [row for row in rows if row["dataset"] == key]
    groups[key] = {}
    for stage in ("initial", "final"):
        groups[key][stage] = {}
        for metric in ("triangles", "clustering", "lcc", "sum_degree_sq"):
            values = [float(row[stage][metric]) for row in selected]
            groups[key][stage][metric] = {
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values),
            }

payload = {
    "format": "dihug-hgen-official-fullgraph-pilot-v1",
    "sampling_uses_clean_graph": False,
    "real_metrics_used_for_posthoc_evaluation_only": True,
    "real": real_by_dataset,
    "groups": groups,
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "[HGEN-OFFICIAL-FULLGRAPH] complete: $OUT_ROOT/summary.json"

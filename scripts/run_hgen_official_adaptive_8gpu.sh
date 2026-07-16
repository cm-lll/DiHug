#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-results/hgen_official_adaptive_0716}"
LOG_ROOT="${LOG_ROOT:-logs/hgen_official_adaptive_0716}"
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

for key in syn100 syn200 syn500 dblp; do
  "$PYTHON" scripts/select_mixed_edit_dataset_profile.py \
    --dataset-dir "$(dataset_dir "$key")" \
    --output "$OUT_ROOT/$key/adaptive_profile.json" \
    > "$LOG_ROOT/${key}_profile.log" 2>&1
done

load_profile() {
  local path="$1"
  eval "$("$PYTHON" - "$path" <<'PY'
import json
import shlex
import sys

profile = json.load(open(sys.argv[1], encoding="utf-8"))
mapping = {
    "ROLE_COMPONENTS": profile["role_components"],
    "RANDOMIZATION": profile["randomization_steps_per_edge"],
    "TRAJECTORIES": profile["trajectories"],
    "FORWARD_STEPS": profile["forward_steps"],
    "RECORD_EVERY": profile["record_every"],
    "EPOCHS": profile["epochs"],
    "TRAIN_CANDIDATES": profile["train_candidates"],
    "SAMPLE_STEPS": profile["sample_steps"],
    "SAMPLE_CANDIDATES": profile["sample_candidates"],
    "KERNEL": profile["kernel"],
    "FEATURE_PROFILE": profile["feature_profile"],
}
for key, value in mapping.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"
}

fit_prior() {
  local key="$1"
  local profile="$OUT_ROOT/$key/adaptive_profile.json"
  local out="$OUT_ROOT/$key/shared_prior"
  load_profile "$profile"
  mkdir -p "$out"
  "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
    --mode fit-gate --dataset-dir "$(dataset_dir "$key")" --output-dir "$out" \
    --components "$ROLE_COMPONENTS" --role-model categorical \
    --role-component-sampling systematic --role-correlation-strength 1.0 \
    --randomization-steps-per-edge "$RANDOMIZATION" \
    --prior-samples 8 --seed 0 > "$LOG_ROOT/${key}_prior.log" 2>&1
}

fit_prior syn100 & prior_pids=("$!")
fit_prior syn200 & prior_pids+=("$!")
fit_prior syn500 & prior_pids+=("$!")
fit_prior dblp & prior_pids+=("$!")
for pid in "${prior_pids[@]}"; do wait "$pid"; done

run_one() {
  local gpu="$1" key="$2" seed="$3"
  local profile="$OUT_ROOT/$key/adaptive_profile.json"
  local out="$OUT_ROOT/$key/seed${seed}"
  local log="$LOG_ROOT/${key}_s${seed}.log"
  load_profile "$profile"
  mkdir -p "$out"
  echo "[START] gpu=$gpu dataset=$key seed=$seed kernel=$KERNEL forward=$FORWARD_STEPS sample=$SAMPLE_STEPS epochs=$EPOCHS" \
    | tee -a "$LOG_ROOT/launcher.log"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_mixed_edit_diffusion.py \
    --mode train --denoiser listwise --kernel "$KERNEL" \
    --dataset-dir "$(dataset_dir "$key")" --output-dir "$out/model" \
    --device cuda --seed "$seed" --trajectories "$TRAJECTORIES" \
    --forward-steps "$FORWARD_STEPS" --record-every "$RECORD_EVERY" \
    --train-candidates "$TRAIN_CANDIDATES" --feature-profile "$FEATURE_PROFILE" \
    --epochs "$EPOCHS" --batch-size 64 --hidden-dim 256 --layers 3 \
    --training-objective inverse_ce --fixed-t-trials 32 > "$log" 2>&1

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
    --mode sample --output-dir "$out/sample" \
    --prior "$OUT_ROOT/$key/shared_prior/exchangeable_role_prior.json" \
    --checkpoint "$out/model/mixed_edit_diffusion.pt" --device cuda \
    --seed "$seed" --sample-steps "$SAMPLE_STEPS" \
    --sample-candidates "$SAMPLE_CANDIDATES" --sample-actions checkpoint \
    --metrics-every "$((SAMPLE_STEPS / 10))" >> "$log" 2>&1
  echo "[DONE] gpu=$gpu dataset=$key seed=$seed" | tee -a "$LOG_ROOT/launcher.log"
}

declare -a KEYS=(syn100 syn100 syn200 syn200 syn500 syn500 dblp dblp)
declare -a SEEDS=(0 1 0 1 0 1 0 1)
pids=()
for gpu in $(seq 0 7); do
  run_one "$gpu" "${KEYS[$gpu]}" "${SEEDS[$gpu]}" &
  pids+=("$!")
  sleep 5
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "adaptive HGEN experiment failed" >&2; exit 1; }

"$PYTHON" - "$OUT_ROOT" <<'PY'
import json
import pathlib
import statistics
import sys

from scripts.run_exchangeable_role_prior_gate import degree_distances
from sparse_diffusion.structure_energy.graph import TypedGraph, load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics

root = pathlib.Path(sys.argv[1])
datasets = {
    "syn100": pathlib.Path("data/HGEN_syn_100_subgraphs/subgraph_000"),
    "syn200": pathlib.Path("data/HGEN_syn_200_subgraphs/subgraph_000"),
    "syn500": pathlib.Path("data/HGEN_syn_500_subgraphs/subgraph_000"),
    "dblp": pathlib.Path("data/DBLP_four_area_subgraphs/subgraph_000"),
}
rows = []
real_by_dataset = {}
profiles = {}
for key, dataset in datasets.items():
    clean = load_typed_graph(dataset, active_only=True)
    real = graph_metrics(clean)
    real_by_dataset[key] = real
    profiles[key] = json.loads((root / key / "adaptive_profile.json").read_text())
    for seed in (0, 1):
        run = root / key / f"seed{seed}"
        report = json.loads((run / "sample/sampling_report.json").read_text())
        train = json.loads((run / "model/train_report.json").read_text())
        generated = TypedGraph.from_json_dict(
            json.loads((run / "sample/generated_graph.json").read_text())
        )
        errors = {}
        for metric in ("triangles", "clustering", "lcc", "sum_degree_sq"):
            absolute = abs(float(report["final"][metric]) - float(real[metric]))
            errors[metric] = {
                "absolute": absolute,
                "relative": (
                    absolute / abs(float(real[metric]))
                    if abs(float(real[metric])) > 1.0e-12
                    else None
                ),
            }
        rows.append({
            "dataset": key,
            "seed": seed,
            "validation_top1": train["train"]["validation_top1"],
            "initial": report["initial"],
            "final": report["final"],
            "action_counts": report["action_counts"],
            "errors": errors,
            "degree_distances": degree_distances(clean, generated),
        })

groups = {}
for key in datasets:
    selected = [row for row in rows if row["dataset"] == key]
    groups[key] = {}
    for stage in ("initial", "final"):
        groups[key][stage] = {}
        for metric in ("edges", "triangles", "clustering", "lcc", "sum_degree_sq"):
            values = [float(row[stage][metric]) for row in selected]
            groups[key][stage][metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values),
            }

payload = {
    "format": "dihug-hgen-official-adaptive-v1",
    "sampling_uses_clean_graph": False,
    "real_metrics_used_for_posthoc_evaluation_only": True,
    "selection_uses_final_graph_metrics": False,
    "profiles": profiles,
    "real": real_by_dataset,
    "groups": groups,
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "[HGEN-OFFICIAL-ADAPTIVE] complete: $OUT_ROOT/summary.json"

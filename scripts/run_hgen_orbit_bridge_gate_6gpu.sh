#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-results/hgen_orbit_bridge_gate_0716}"
LOG_ROOT="${LOG_ROOT:-logs/hgen_orbit_bridge_gate_0716}"
EPOCHS="${EPOCHS:-20}"
TRAJECTORIES="${TRAJECTORIES:-16}"
NOOP_EXAMPLES="${NOOP_EXAMPLES:-8}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/launcher.log"

dataset_dir() {
  case "$1" in
    syn100) echo "data/HGEN_syn_100_subgraphs/subgraph_000" ;;
    syn200) echo "data/HGEN_syn_200_subgraphs/subgraph_000" ;;
    syn500) echo "data/HGEN_syn_500_subgraphs/subgraph_000" ;;
    *) return 1 ;;
  esac
}

load_profile() {
  local key="$1"
  eval "$("$PYTHON" - "$OUT_ROOT" "$key" <<'PY'
import json
import pathlib
import shlex
import sys

root = pathlib.Path(sys.argv[1])
key = sys.argv[2]
path = pathlib.Path("results/hgen_official_adaptive_0716") / key / "adaptive_profile.json"
profile = json.loads(path.read_text(encoding="utf-8"))
for name, value in {
    "FORWARD_STEPS": profile["forward_steps"],
    "RECORD_EVERY": profile["record_every"],
    "TRAIN_CANDIDATES": profile["train_candidates"],
    "SAMPLE_STEPS": profile["sample_steps"],
    "SAMPLE_CANDIDATES": profile["sample_candidates"],
    "ROLE_COMPONENTS": profile["role_components"],
    "RANDOMIZATION": profile["randomization_steps_per_edge"],
}.items():
    print(f"{name}={shlex.quote(str(value))}")
PY
)"
}

run_variant() {
  local gpu="$1" key="$2" seed="$3" variant="$4"
  local out="$OUT_ROOT/$key/seed${seed}/$variant"
  local log="$LOG_ROOT/${key}_s${seed}_${variant}.log"
  local orbit_args=()
  if [[ "$variant" == "orbit" ]]; then
    orbit_args+=(--type-orbit-augmentation)
  fi
  mkdir -p "$out"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_mixed_edit_diffusion.py \
    --mode train --denoiser listwise --kernel bdrs \
    --dataset-dir "$(dataset_dir "$key")" --output-dir "$out/model" \
    --device cuda --seed "$seed" --trajectories "$TRAJECTORIES" \
    --forward-steps "$FORWARD_STEPS" --record-every "$RECORD_EVERY" \
    --train-candidates "$TRAIN_CANDIDATES" \
    --feature-profile typed_closure_compact_v2 \
    --epochs "$EPOCHS" --batch-size 64 --hidden-dim 256 --layers 3 \
    --training-objective inverse_ce --fixed-t-trials 16 \
    --role-corruption-fraction 1 --role-correlation-strengths 1 \
    --role-prior-components "$ROLE_COMPONENTS" \
    --role-prior-randomization-steps-per-edge "$RANDOMIZATION" \
    --clean-noop-examples-per-trajectory "$NOOP_EXAMPLES" \
    "${orbit_args[@]}" > "$log" 2>&1

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
    --mode sample --output-dir "$out/sample" \
    --prior "results/hgen_official_adaptive_0716/$key/shared_prior/exchangeable_role_prior.json" \
    --checkpoint "$out/model/mixed_edit_diffusion.pt" --device cuda \
    --seed "$seed" --sample-steps "$SAMPLE_STEPS" \
    --sample-candidates "$SAMPLE_CANDIDATES" --sample-actions checkpoint \
    --metrics-every "$((SAMPLE_STEPS / 10))" >> "$log" 2>&1
}

run_pair() {
  local gpu="$1" key="$2" seed="$3"
  load_profile "$key"
  echo "[START] gpu=$gpu dataset=$key seed=$seed" | tee -a "$LOG_ROOT/launcher.log"
  run_variant "$gpu" "$key" "$seed" original
  run_variant "$gpu" "$key" "$seed" orbit
  echo "[DONE] gpu=$gpu dataset=$key seed=$seed" | tee -a "$LOG_ROOT/launcher.log"
}

keys=(syn100 syn100 syn200 syn200 syn500 syn500)
seeds=(0 1 0 1 0 1)
pids=()
for gpu in $(seq 0 5); do
  run_pair "$gpu" "${keys[$gpu]}" "${seeds[$gpu]}" &
  pids+=("$!")
  sleep 3
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "orbit bridge gate failed" >&2; exit 1; }

"$PYTHON" - "$OUT_ROOT" <<'PY'
import json
import pathlib
import statistics
import sys

from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics

root = pathlib.Path(sys.argv[1])
datasets = {
    "syn100": pathlib.Path("data/HGEN_syn_100_subgraphs/subgraph_000"),
    "syn200": pathlib.Path("data/HGEN_syn_200_subgraphs/subgraph_000"),
    "syn500": pathlib.Path("data/HGEN_syn_500_subgraphs/subgraph_000"),
}
rows = []
for key, path in datasets.items():
    real = graph_metrics(load_typed_graph(path, active_only=True))
    for seed in (0, 1):
        for variant in ("original", "orbit"):
            run = root / key / f"seed{seed}" / variant
            sample = json.loads((run / "sample/sampling_report.json").read_text())
            train = json.loads((run / "model/train_report.json").read_text())
            final = sample["final"]
            rows.append({
                "dataset": key,
                "seed": seed,
                "variant": variant,
                "real": real,
                "initial": sample["initial"],
                "final": final,
                "triangle_relative_error": abs(final["triangles"] - real["triangles"]) / max(real["triangles"], 1.0),
                "clustering_relative_error": abs(final["clustering"] - real["clustering"]) / max(real["clustering"], 1.0e-12),
                "validation_top1": train["train"]["validation_top1"],
                "action_counts": sample["action_counts"],
            })

groups = {}
for key in datasets:
    groups[key] = {}
    for variant in ("original", "orbit"):
        selected = [r for r in rows if r["dataset"] == key and r["variant"] == variant]
        groups[key][variant] = {
            metric: {
                "mean": statistics.fmean(float(r[metric]) for r in selected),
                "std": statistics.pstdev(float(r[metric]) for r in selected),
            }
            for metric in ("triangle_relative_error", "clustering_relative_error", "validation_top1")
        }
payload = {
    "format": "dihug-type-orbit-bridge-gate-v1",
    "paired_prior_and_sampling_seed": True,
    "selection_uses_final_metrics": False,
    "groups": groups,
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "[ORBIT-BRIDGE-GATE] complete: $OUT_ROOT/summary.json"

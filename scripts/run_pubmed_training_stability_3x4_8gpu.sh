#!/usr/bin/env bash
# PubMed training-seed stability under the frozen unconditional protocol.
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

DATASET="${DATASET:-data/PubMed_baseline_subgraphs/subgraph_000}"
OUT_ROOT="${OUT_ROOT:-results/pubmed_training_stability_3x4}"
LOG_ROOT="${LOG_ROOT:-logs/pubmed_training_stability_3x4}"
GPU_LIST="${GPU_LIST:-0 1 2}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2}"
GEN_SEEDS="${GEN_SEEDS:-0 1 2 3}"
SAMPLE_STEPS="${SAMPLE_STEPS:-10000}"
PYTHON="${PYTHON:-python}"
read -r -a GPUS <<< "$GPU_LIST"
read -r -a TS <<< "$TRAIN_SEEDS"

if [[ "${#GPUS[@]}" -lt "${#TS[@]}" ]]; then
  echo "GPU_LIST must provide at least one GPU per training seed" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/driver.log"

run_training_seed() {
  local gpu="$1" train_seed="$2"
  local base="$OUT_ROOT/train_seed${train_seed}"
  local log="$LOG_ROOT/train_seed${train_seed}.log"
  local legacy_dir="$base/legacy_model"
  local context_dir="$base/context_model"
  local legacy_ckpt="$legacy_dir/mixed_edit_diffusion.pt"
  local context_ckpt="$context_dir/rooted_context_residual.pt"
  mkdir -p "$base"

  echo "[TRAIN-START] gpu=$gpu train_seed=$train_seed" | tee -a "$LOG_ROOT/driver.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_mixed_edit_diffusion.py \
    --mode train --denoiser listwise --kernel bdrs \
    --dataset-dir "$DATASET" --output-dir "$legacy_dir" --device cuda \
    --seed "$train_seed" --trajectories 16 --forward-steps 3000 \
    --record-every 30 --train-candidates 16 \
    --feature-profile typed_closure_compact_v2 \
    --epochs 20 --batch-size 64 --hidden-dim 256 --layers 3 \
    --training-objective inverse_ce --fixed-t-trials 16 \
    --role-corruption-fraction 1.0 \
    --role-correlation-strengths "0,0.25,0.5,0.75,1" \
    --role-prior-components 512 \
    --role-prior-randomization-steps-per-edge 2 \
    > "$log" 2>&1

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_rooted_context_residual_gate.py \
    --dataset-dir "$DATASET" --legacy-checkpoint "$legacy_ckpt" \
    --output-dir "$context_dir" --device cuda --seed "$train_seed" \
    --trajectories 4 --examples-per-trajectory 16 --switches-per-example 64 \
    --candidates 16 --closure-fraction 1.0 --epochs 15 --batch-size 2 \
    --hidden-dim 64 --heads 4 --layers 2 \
    >> "$log" 2>&1

  for gen_seed in $GEN_SEEDS; do
    local pair="$base/gen_seed${gen_seed}"
    local common=(
      --dataset-dir "$DATASET" --context-checkpoint "$context_ckpt"
      --seed "$gen_seed" --device cuda --steps "$SAMPLE_STEPS"
      --metrics-every "$((SAMPLE_STEPS / 10))"
      --model-time-horizon-steps "$((SAMPLE_STEPS * 2))"
      --family-count-mode binomial --candidates 16 --role-weight 0.05
      --gain-scale 10 --schedule-power 1 --schedule-floor 1
      --role-error-per-node-threshold 5.5 --closure-candidate-fraction 1.0
      --auto-role-balance 0.5 --targeted-role-candidate-fraction 0.5
      --context-refresh-steps 20
    )
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_role_guided_replacement_gate.py \
      "${common[@]}" --context-scale 0 --output-dir "$pair/legacy" \
      >> "$log" 2>&1
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_role_guided_replacement_gate.py \
      "${common[@]}" --context-scale -1 --output-dir "$pair/context" \
      >> "$log" 2>&1
  done
  echo "[TRAIN-DONE] gpu=$gpu train_seed=$train_seed" | tee -a "$LOG_ROOT/driver.log"
}

pids=()
for idx in "${!TS[@]}"; do
  run_training_seed "${GPUS[$idx]}" "${TS[$idx]}" &
  pids+=("$!")
  sleep 15
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "training stability matrix failed" >&2; exit 1; }

"$PYTHON" - "$OUT_ROOT" "$(git rev-parse HEAD)" <<'PY'
import csv
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
commit = sys.argv[2]
rows = []
for context_path in sorted(root.glob("train_seed*/gen_seed*/context/gate_report.json")):
    train_seed = int(context_path.parts[-4].removeprefix("train_seed"))
    gen_seed = int(context_path.parts[-3].removeprefix("gen_seed"))
    legacy_path = context_path.parents[1] / "legacy" / "gate_report.json"
    context = json.loads(context_path.read_text())
    legacy = json.loads(legacy_path.read_text())
    if context["initial"] != legacy["initial"]:
        raise RuntimeError(f"paired initial mismatch: ts={train_seed}, gs={gen_seed}")
    if context.get("initial_graph_source") != "binomial_family_count_prior":
        raise RuntimeError("stability run did not use the Binomial protocol")
    train_report = json.loads(
        (root / f"train_seed{train_seed}/legacy_model/train_report.json").read_text()
    )["train"]
    for mode, report in (("legacy", legacy), ("context", context)):
        final = report["final"]
        rows.append({
            "train_seed": train_seed,
            "generation_seed": gen_seed,
            "mode": mode,
            "validation_top1": train_report.get("validation_top1"),
            "validation_selected_utility": train_report.get("validation_selected_utility"),
            **{f"final_{key}": final[key] for key in (
                "edges", "triangles", "clustering", "transitivity", "lcc",
                "assortativity", "sum_degree_sq", "max_degree",
            )},
        })

with (root / "training_stability.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

summary = {"format": "dihug-pubmed-training-stability-3x4-v1", "commit": commit}
for mode in ("legacy", "context"):
    mode_rows = [row for row in rows if row["mode"] == mode]
    per_training = []
    for train_seed in sorted({row["train_seed"] for row in mode_rows}):
        group = [row for row in mode_rows if row["train_seed"] == train_seed]
        per_training.append({
            "train_seed": train_seed,
            "n": len(group),
            "triangles_mean": statistics.fmean(row["final_triangles"] for row in group),
            "clustering_mean": statistics.fmean(row["final_clustering"] for row in group),
            "validation_top1": group[0]["validation_top1"],
            "validation_selected_utility": group[0]["validation_selected_utility"],
        })
    summary[mode] = {
        "per_training_seed": per_training,
        "between_training_triangles_mean": statistics.fmean(
            row["triangles_mean"] for row in per_training
        ),
        "between_training_triangles_std": statistics.pstdev(
            row["triangles_mean"] for row in per_training
        ),
        "between_training_clustering_mean": statistics.fmean(
            row["clustering_mean"] for row in per_training
        ),
        "between_training_clustering_std": statistics.pstdev(
            row["clustering_mean"] for row in per_training
        ),
    }
(root / "training_stability.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[TRAINING-STABILITY] complete: $OUT_ROOT/training_stability.json"

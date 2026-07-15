#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/lyh/gnn_project/DiHug}"
DATASET_DIR="${DATASET_DIR:-data/IMDB_movie_subgraphs/subgraph_000}"
TAG="${TAG:-imdb_mixed_edit_gates}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/$TAG}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/$TAG}"
CONDA_SH="${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
STARTUP_STAGGER_SECONDS="${STARTUP_STAGGER_SECONDS:-15}"

source "$CONDA_SH"
conda activate sparse_block
cd "$ROOT"
mkdir -p "$OUT_ROOT/oracle" "$OUT_ROOT/model" "$LOG_ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

run_oracle() {
  local name="$1" kernel="$2" seed="$3" candidates="$4"
  python scripts/run_mixed_edit_diffusion.py \
    --mode oracle --dataset-dir "$DATASET_DIR" \
    --output-dir "$OUT_ROOT/oracle/$name" \
    --kernel "$kernel" --seed "$seed" --device cpu \
    --sample-steps 5000 --sample-candidates "$candidates" \
    --metrics-every 500 > "$LOG_ROOT/oracle_${name}.log" 2>&1
}

oracle_names=(bd_s0 bdr_s0 bdrs_s0 bdrs_s1 bdrs_c16 bdrs_c32 bdrs_c128 bdrs_s2)
oracle_kernels=(bd bdr bdrs bdrs bdrs bdrs bdrs bdrs)
oracle_seeds=(0 0 0 1 0 0 0 2)
oracle_candidates=(64 64 64 64 16 32 128 64)
pids=()
for index in "${!oracle_names[@]}"; do
  run_oracle \
    "${oracle_names[$index]}" "${oracle_kernels[$index]}" \
    "${oracle_seeds[$index]}" "${oracle_candidates[$index]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

python - "$OUT_ROOT/oracle" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/oracle_report.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    initial = float(payload["initial"]["edge_edit_distance"])
    final = float(payload["final"]["edge_edit_distance"])
    rows.append({
        "name": path.parent.name,
        "kernel": payload["kernel"],
        "initial_distance": initial,
        "final_distance": final,
        "relative_reduction": (initial - final) / max(initial, 1.0),
        "action_counts": payload["action_counts"],
    })
if not rows or not all(row["final_distance"] < row["initial_distance"] for row in rows):
    raise SystemExit("oracle candidate-coverage gate failed")
(root / "summary.json").write_text(
    json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
)
PY

names=(bd_s0 bdr_s0 bdrs_s0_c16 bdrs_s0_c32 bdrs_s1_c32 bdrs_s2_c32 bdrs_s0_c64 bdrs_s1_c64)
kernels=(bd bdr bdrs bdrs bdrs bdrs bdrs bdrs)
seeds=(0 0 0 0 1 2 0 1)
train_candidates=(32 32 16 32 32 32 64 64)
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  name="${names[$gpu]}"
  kernel="${kernels[$gpu]}"
  seed="${seeds[$gpu]}"
  candidates="${train_candidates[$gpu]}"
  (
    sleep "$((gpu * STARTUP_STAGGER_SECONDS))"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_mixed_edit_diffusion.py \
      --mode train --dataset-dir "$DATASET_DIR" \
      --output-dir "$OUT_ROOT/model/$name" \
      --kernel "$kernel" --seed "$seed" --device cuda \
      --trajectories 32 --forward-steps 2000 --record-every 20 \
      --train-candidates "$candidates" --epochs 100 --batch-size 64 \
      --hidden-dim 256 --layers 3 --fixed-t-trials 128 \
      > "$LOG_ROOT/train_${name}.log" 2>&1
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_mixed_edit_diffusion.py \
      --mode sample --output-dir "$OUT_ROOT/model/${name}_sample" \
      --checkpoint "$OUT_ROOT/model/$name/mixed_edit_diffusion.pt" \
      --kernel "$kernel" --seed "$seed" --device cuda \
      --sample-steps 20000 --sample-candidates 64 \
      --sample-temperature 0 --metrics-every 1000 \
      > "$LOG_ROOT/sample_${name}.log" 2>&1
  ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"

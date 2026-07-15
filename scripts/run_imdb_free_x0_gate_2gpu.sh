#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT_ROOT="${OUT_ROOT:-results/imdb_free_x0_gate}"
LOG_ROOT="${LOG_ROOT:-logs/imdb_free_x0_gate}"
GPU_C16="${GPU_C16:-0}"
GPU_C64="${GPU_C64:-1}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_train() {
  local gpu="$1" candidates="$2" tag="c${2}"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_hgt_listwise_edit_gate.py \
    --mode paired-train --supervision free_x0 --teacher-temperature 1.0 \
    --dataset-dir data/IMDB_movie_subgraphs/subgraph_000 \
    --output-dir "$OUT_ROOT/$tag" \
    --seed 0 --device cuda --trajectories 8 --forward-steps 1000 --record-every 20 \
    --candidates "$candidates" --epochs 20 --batch-size 4 \
    --hidden-dim 128 --heads 4 --hgt-layers 2 --scorer-layers 2 \
    --fixed-t-trials 64 > "$LOG_ROOT/${tag}_train.log" 2>&1
}

run_train "$GPU_C16" 16 & pid16=$!
run_train "$GPU_C64" 64 & pid64=$!
wait "$pid16"
wait "$pid64"

best="$({ python - "$OUT_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
passed = []
for count in (16, 64):
    report = json.loads((root / f"c{count}" / "paired_gate_report.json").read_text())
    hgt_rows, mlp_rows = report["hgt"]["fixed_t"], report["mlp"]["fixed_t"]
    hgt = sum(row["selected_improvement"] for row in hgt_rows.values()) / 3.0
    mlp = sum(row["selected_improvement"] for row in mlp_rows.values()) / 3.0
    delta = hgt - mlp
    validation_delta = (
        report["hgt"]["train"]["validation_selected_utility"]
        - report["mlp"]["train"]["validation_selected_utility"]
    )
    nonnegative = sum(
        hgt_rows[t]["selected_improvement"]
        >= mlp_rows[t]["selected_improvement"]
        for t in hgt_rows
    )
    print(
        f"[FREE-X0-GATE] C={count} hgt_utility={hgt:.6f} "
        f"mlp_utility={mlp:.6f} "
        f"delta={delta:.6f} validation_delta={validation_delta:.6f} "
        f"nonnegative_times={nonnegative}/3",
        file=sys.stderr,
    )
    if delta >= 0.05 and validation_delta > 0.0 and nonnegative >= 2:
        passed.append((delta, count))
print(max(passed)[1] if passed else "STOP")
PY
} 2> >(tee "$LOG_ROOT/gate_decision.log" >&2))"

if [[ "$best" == "STOP" ]]; then
  echo "[FREE-X0-GATE] stopped before free trajectory" | tee -a "$LOG_ROOT/gate_decision.log"
  exit 0
fi

gpu="$GPU_C16"; [[ "$best" == "64" ]] && gpu="$GPU_C64"
for steps in 2000 5000; do
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_hgt_listwise_edit_gate.py \
    --mode sample --output-dir "$OUT_ROOT/c${best}_sample_${steps}" \
    --checkpoint "$OUT_ROOT/c${best}/hgt_listwise_edit.pt" \
    --seed 0 --device cuda --candidates "$best" \
    --sample-steps "$steps" --metrics-every 250 \
    > "$LOG_ROOT/c${best}_sample_${steps}.log" 2>&1
done
echo "[FREE-X0-GATE] complete"

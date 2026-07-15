#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-results/imdb_hgt_listwise_gate}"
LOG_ROOT="${LOG_ROOT:-logs/imdb_hgt_listwise_gate}"
GPU_C16="${GPU_C16:-0}"
GPU_C64="${GPU_C64:-1}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_train() {
  local gpu="$1" candidates="$2" tag="c${2}"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_hgt_listwise_edit_gate.py \
    --mode paired-train \
    --dataset-dir data/IMDB_movie_subgraphs/subgraph_000 \
    --output-dir "$OUT_ROOT/$tag" \
    --seed 0 --device cuda \
    --trajectories 8 --forward-steps 1000 --record-every 20 \
    --candidates "$candidates" --epochs 20 --batch-size 4 \
    --hidden-dim 128 --heads 4 --hgt-layers 2 --scorer-layers 2 \
    --fixed-t-trials 64 \
    > "$LOG_ROOT/${tag}_train.log" 2>&1
}

run_train "$GPU_C16" 16 & pid16=$!
run_train "$GPU_C64" 64 & pid64=$!
wait "$pid16"
wait "$pid64"

best="$({ python - "$OUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
passed = []
for count in (16, 64):
    report = json.loads((root / f"c{count}" / "paired_gate_report.json").read_text())
    hgt_rows = report["hgt"]["fixed_t"]
    mlp_rows = report["mlp"]["fixed_t"]
    hgt = sum(x["top1"] for x in hgt_rows.values()) / 3.0
    mlp = sum(x["top1"] for x in mlp_rows.values()) / 3.0
    delta = hgt - mlp
    validation_delta = (
        report["hgt"]["train"]["top1"]
        - report["mlp"]["train"]["validation_top1"]
    )
    nonnegative_times = sum(
        hgt_rows[key]["top1"] >= mlp_rows[key]["top1"] for key in hgt_rows
    )
    print(
        f"[HGT-GATE] C={count} hgt_fixed_top1={hgt:.6f} "
        f"mlp_fixed_top1={mlp:.6f} delta={delta:.6f} "
        f"validation_delta={validation_delta:.6f} "
        f"nonnegative_times={nonnegative_times}/3",
        file=sys.stderr,
    )
    # A one-bin fluctuation is not enough to justify a costly free trajectory.
    if delta >= 0.05 and validation_delta > 0.0 and nonnegative_times >= 2:
        passed.append((delta, count))
if not passed:
    print("STOP")
else:
    print(max(passed)[1])
PY
} 2> >(tee "$LOG_ROOT/gate_decision.log" >&2))"

if [[ "$best" == "STOP" ]]; then
  echo "[HGT-GATE] stopped: HGT did not beat paired MLP on fixed-t recall" | tee -a "$LOG_ROOT/gate_decision.log"
  exit 0
fi

gpu="$GPU_C16"
[[ "$best" == "64" ]] && gpu="$GPU_C64"
for steps in 2000 5000; do
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_hgt_listwise_edit_gate.py \
    --mode sample \
    --output-dir "$OUT_ROOT/c${best}_sample_${steps}" \
    --checkpoint "$OUT_ROOT/c${best}/hgt_listwise_edit.pt" \
    --seed 0 --device cuda --candidates "$best" \
    --sample-steps "$steps" --metrics-every 250 \
    > "$LOG_ROOT/c${best}_sample_${steps}.log" 2>&1
done

python - "$OUT_ROOT" "$best" > "$LOG_ROOT/trajectory_gate.log" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
count = int(sys.argv[2])
for steps in (2000, 5000):
    report = json.loads((root / f"c{count}_sample_{steps}" / "sampling_report.json").read_text())
    before, after = report["initial"], report["final"]
    print(
        f"[HGT-TRAJECTORY] C={count} steps={steps} "
        f"tri={before['triangles']:.0f}->{after['triangles']:.0f} "
        f"clustering={before['clustering']:.6f}->{after['clustering']:.6f} "
        f"sumdeg2={before['sum_degree_sq']:.0f}->{after['sum_degree_sq']:.0f} "
        f"maxdeg={before['max_degree']:.0f}->{after['max_degree']:.0f} "
        f"lcc={before['lcc']:.0f}->{after['lcc']:.0f}"
    )
PY

echo "[HGT-GATE] all requested stages complete"

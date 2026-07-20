#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/lyh/gnn_project/DiHug_clean_joint}"
DATA_ROOT="${DATA_ROOT:-/data2/lyh/gnn_project/DiHug/data}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/classic_baseline_smoke}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/classic_baseline_smoke}"
GPU_LIST=(${GPU_LIST:-1 2 3 4})
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

DATASET="$DATA_ROOT/HGEN_syn_100_subgraphs/subgraph_000"
METHODS=(vgae netgan graphrnn graphvae)
for index in "${!METHODS[@]}"; do
  method="${METHODS[$index]}"
  gpu="${GPU_LIST[$index]}"
  extra=(--epochs 40 --samples 2)
  if [[ "$method" == netgan ]]; then
    extra=(--iterations 200 --walk-samples 10000 --batch-size 32 --samples 2)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python "$ROOT/scripts/run_classic_heterogeneous_baseline.py" \
      --method "$method" --dataset-dir "$DATASET" \
      --output-dir "$OUT_ROOT/$method" --device cuda:0 --seed 0 \
      "${extra[@]}" > "$LOG_ROOT/$method.log" 2>&1 &
  echo "$! $method GPU=$gpu"
done
wait

python - "$OUT_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = {}
for path in sorted(root.glob("*/report.json")):
    report = json.loads(path.read_text())
    summary[path.parent.name] = report["aggregate"]
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

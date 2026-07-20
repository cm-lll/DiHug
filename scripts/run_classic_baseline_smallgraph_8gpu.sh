#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/lyh/gnn_project/DiHug_clean_joint}"
DATA_ROOT="${DATA_ROOT:-/data2/lyh/gnn_project/DiHug/data}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/classic_baseline_smallgraph}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/classic_baseline_smallgraph}"
GPU_LIST=(${GPU_LIST:-0 1 2 3 4 5 6 7})
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

jobs=(
  "syn100 vgae"
  "syn100 netgan"
  "syn100 graphrnn"
  "syn100 graphvae"
  "syn200 vgae"
  "syn200 netgan"
  "syn200 graphrnn"
  "syn200 graphvae"
)

declare -A datasets=(
  [syn100]="$DATA_ROOT/HGEN_syn_100_subgraphs/subgraph_000"
  [syn200]="$DATA_ROOT/HGEN_syn_200_subgraphs/subgraph_000"
)

pids=()
for index in "${!jobs[@]}"; do
  read -r dataset method <<<"${jobs[$index]}"
  gpu="${GPU_LIST[$index]}"
  extra=(--epochs 300 --samples 8 --hidden 128 --latent 32 --max-prev-node 256)
  if [[ "$method" == netgan ]]; then
    extra=(--iterations 2000 --walk-samples 50000 --batch-size 64 --samples 8)
  fi
  name="${dataset}_${method}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python "$ROOT/scripts/run_classic_heterogeneous_baseline.py" \
      --method "$method" --dataset-dir "${datasets[$dataset]}" \
      --output-dir "$OUT_ROOT/$name" --device cuda:0 --seed 0 \
      "${extra[@]}" > "$LOG_ROOT/$name.log" 2>&1 &
  pids+=("$!")
  echo "$! $name GPU=$gpu"
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one baseline failed; inspect $LOG_ROOT" >&2
  exit 1
fi

python - "$OUT_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/report.json")):
    report = json.loads(path.read_text())
    dataset, method = path.parent.name.split("_", 1)
    aggregate = report["aggregate"]
    global_metrics = aggregate["global"]
    rows.append({
        "dataset": dataset,
        "method": method,
        "edges": global_metrics["edges"]["generated"]["mean"],
        "lcc": global_metrics["lcc"]["generated"]["mean"],
        "triangles": global_metrics["triangles"]["generated"]["mean"],
        "clustering": global_metrics["clustering"]["generated"]["mean"],
        "assortativity": global_metrics["assortativity"]["generated"]["mean"],
        "degree_mmd": aggregate["degree_mmd"]["mean"],
    })
(root / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
columns = list(rows[0]) if rows else []
with (root / "summary.tsv").open("w") as handle:
    handle.write("\t".join(columns) + "\n")
    for row in rows:
        handle.write("\t".join(str(row[column]) for column in columns) + "\n")
print(json.dumps(rows, indent=2, sort_keys=True))
PY

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TAG="${TAG:-latent_representation_gate_0719}"
OUT_ROOT="${OUT_ROOT:-results/$TAG}"
LOG_ROOT="${LOG_ROOT:-logs/$TAG}"
SEED="${SEED:-71}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/driver.log"

launch() {
  local gpu="$1" name="$2" data="$3"
  shift 3
  echo "[LAUNCH] dataset=$name gpu=$gpu" | tee -a "$LOG_ROOT/driver.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_latent_representation_gate.py \
    --dataset-dir "$data" --output-dir "$OUT_ROOT/$name" --seed "$SEED" \
    --device cuda "$@" > "$LOG_ROOT/$name.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
}

PIDS=()
NAMES=()

launch 0 imdb data/IMDB_movie_subgraphs/subgraph_000 \
  --trajectories 6 --queries-per-family 128 --epochs 12 \
  --hidden-dim 64 --heads 4 --layers 2

launch 1 pubmed data/PubMed_baseline_subgraphs/subgraph_000 \
  --trajectories 6 --queries-per-family 192 --epochs 12 \
  --hidden-dim 64 --heads 4 --layers 2

launch 2 dblp data/DBLP_four_area_subgraphs/subgraph_000 \
  --trajectories 4 --queries-per-family 128 --epochs 8 \
  --hidden-dim 48 --heads 4 --layers 1

launch 3 syn200 "${SYN200_DATA:-/data2/lyh/gnn_project/DiHug/data/HGEN_syn_200_subgraphs/subgraph_000}" \
  --trajectories 8 --queries-per-family 96 --epochs 15 \
  --hidden-dim 32 --heads 4 --layers 2

failed=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "[DONE] dataset=${NAMES[$index]}" | tee -a "$LOG_ROOT/driver.log"
  else
    echo "[FAILED] dataset=${NAMES[$index]}" | tee -a "$LOG_ROOT/driver.log"
    failed=1
  fi
done

python - "$OUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for name in ("imdb", "pubmed", "dblp", "syn200"):
    path = root / name / "representation_report.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"dataset": name, **payload})
    else:
        rows.append({"dataset": name, "status": "missing", "passed": False})
(root / "summary.json").write_text(
    json.dumps({"format": "dihug-latent-representation-4dataset-v1", "rows": rows}, indent=2),
    encoding="utf-8",
)
print("[SUMMARY] " + json.dumps(rows, sort_keys=True))
PY

exit "$failed"

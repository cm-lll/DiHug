#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TAG="${TAG:-corrected_joint_cross_dataset_gate_0719}"
OUT_ROOT="${OUT_ROOT:-results/$TAG}"
LOG_ROOT="${LOG_ROOT:-logs/$TAG}"
SEED="${SEED:-61}"
IMDB_DATA="${IMDB_DATA:-data/IMDB_movie_subgraphs/subgraph_000}"
PUBMED_DATA="${PUBMED_DATA:-data/PubMed_baseline_subgraphs/subgraph_000}"
DBLP_DATA="${DBLP_DATA:-data/DBLP_four_area_subgraphs/subgraph_000}"
SYN200_DATA="${SYN200_DATA:-data/HGEN_syn_200_subgraphs/subgraph_000}"
REUSE_PREP_ROOT="${REUSE_PREP_ROOT:-}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/driver.log"

launch() {
  local gpu="$1"
  local name="$2"
  local data="$3"
  shift 3
  echo "[LAUNCH] dataset=$name gpu=$gpu" | tee -a "$LOG_ROOT/driver.log"
  env \
    TAG="${TAG}_${name}" \
    OUT_ROOT="$OUT_ROOT/$name" \
    LOG_ROOT="$LOG_ROOT/$name" \
    DATA="$data" \
    SEED="$SEED" \
    PREP_GPU="$gpu" SAMPLE_BASE_GPU="$gpu" SAMPLE_JOINT_GPU="$gpu" \
    SERIAL_SAMPLING=1 \
    PREP_SOURCE_ROOT="${REUSE_PREP_ROOT:+$REUSE_PREP_ROOT/$name}" \
    "$@" \
    bash scripts/run_corrected_joint_dataset_gate.sh \
    > "$LOG_ROOT/${name}_worker.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
}

PIDS=()
NAMES=()

# IMDB: balanced role/closure pool; moderate context refresh.
launch 0 imdb "$IMDB_DATA" \
  TRAJECTORIES=4 FORWARD_STEPS=100 RECORD_EVERY=10 CANDIDATES=16 \
  BASE_EPOCHS=20 JOINT_EPOCHS=10 BATCH_SIZE=4 \
  QUERY_EDGES_PER_FAMILY=16 HIDDEN_DIM=64 HEADS=4 LAYERS=2 \
  TOTAL_EDITS=2000 OUTER_STEPS=50 CONTEXT_REFRESH_STEPS=20 \
  ROLE_WEIGHT=0.05 ROLE_TEACHER_WEIGHT=0.5 \
  TARGETED_FRACTION=0.5 CLOSURE_FRACTION=0.5

# PubMed: refresh context more often because typed closure changes faster.
launch 1 pubmed "$PUBMED_DATA" \
  TRAJECTORIES=4 FORWARD_STEPS=100 RECORD_EVERY=10 CANDIDATES=16 \
  BASE_EPOCHS=20 JOINT_EPOCHS=10 BATCH_SIZE=2 \
  QUERY_EDGES_PER_FAMILY=16 HIDDEN_DIM=64 HEADS=4 LAYERS=2 \
  TOTAL_EDITS=3000 OUTER_STEPS=50 CONTEXT_REFRESH_STEPS=10 \
  ROLE_WEIGHT=0.05 ROLE_TEACHER_WEIGHT=0.5 \
  TARGETED_FRACTION=0.5 CLOSURE_FRACTION=0.5

# DBLP has no triangle-valid typed schema. Spend the budget on role alignment,
# use fewer expensive 11k-node snapshots, and keep the encoder inside 11 GB.
launch 2 dblp "$DBLP_DATA" \
  TRAJECTORIES=2 FORWARD_STEPS=40 RECORD_EVERY=10 CANDIDATES=8 \
  BASE_EPOCHS=10 JOINT_EPOCHS=5 BATCH_SIZE=1 \
  QUERY_EDGES_PER_FAMILY=4 HIDDEN_DIM=32 HEADS=4 LAYERS=1 \
  TOTAL_EDITS=1000 OUTER_STEPS=50 CONTEXT_REFRESH_STEPS=20 \
  ROLE_WEIGHT=0.10 ROLE_TEACHER_WEIGHT=1.0 \
  TARGETED_FRACTION=1.0 CLOSURE_FRACTION=0.0

# syn200 begins close to its Bernoulli null; use more trajectories but a small
# edit budget so noop/stability behavior is tested instead of forced editing.
launch 3 syn200 "$SYN200_DATA" \
  TRAJECTORIES=8 FORWARD_STEPS=100 RECORD_EVERY=10 CANDIDATES=16 \
  BASE_EPOCHS=30 JOINT_EPOCHS=15 BATCH_SIZE=8 \
  QUERY_EDGES_PER_FAMILY=8 HIDDEN_DIM=32 HEADS=4 LAYERS=2 \
  TOTAL_EDITS=500 OUTER_STEPS=50 CONTEXT_REFRESH_STEPS=5 \
  ROLE_WEIGHT=0.03 ROLE_TEACHER_WEIGHT=0.25 \
  TARGETED_FRACTION=0.5 CLOSURE_FRACTION=0.5

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
for dataset in ("imdb", "pubmed", "dblp", "syn200"):
    path = root / dataset / "paired_summary.json"
    if not path.exists():
        rows.append({"dataset": dataset, "status": "missing"})
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        {
            "dataset": dataset,
            "status": "complete",
            "initial": payload["initial"],
            "real": payload["real"],
            "explicit": payload["explicit"],
            "joint": payload["role_query_joint"],
            "train": payload["train_report"],
        }
    )
(root / "cross_dataset_summary.json").write_text(
    json.dumps({"format": "dihug-corrected-joint-cross-dataset-v1", "rows": rows}, indent=2),
    encoding="utf-8",
)
print("[CROSS-DATASET-SUMMARY] " + json.dumps(rows, sort_keys=True))
PY

exit "$failed"

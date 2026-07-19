#!/usr/bin/env bash
# Cross-machine paired generation for the stop-calibrated context checkpoints.
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

BUNDLE="${BUNDLE:-paper_tables/new_context_artifacts_732649a.tar.gz}"
EXPECTED_SHA256="dbc7416c00b5d71f95810884824109de518d8ee5e9dccc217d49c9cb1d31bc58"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-results/new_context_artifacts_732649a}"
OUT_ROOT="${OUT_ROOT:-results/new_context_cross_dataset_xiaod}"
LOG_ROOT="${LOG_ROOT:-logs/new_context_cross_dataset_xiaod}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-7}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
DATASETS="${DATASETS:-imdb pubmed syn200}"
PYTHON="${PYTHON:-python}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-gpu}"
GPU_CANDIDATE_MODE="${GPU_CANDIDATE_MODE:-canonical_ordered}"
GPU_PROPOSAL_POOL_SIZE="${GPU_PROPOSAL_POOL_SIZE:-0}"
GPU_PROPOSAL_CHUNK_SIZE="${GPU_PROPOSAL_CHUNK_SIZE:-4096}"
IMDB_DATA="${IMDB_DATA:-data/IMDB_movie_subgraphs/subgraph_000}"
PUBMED_DATA="${PUBMED_DATA:-data/PubMed_baseline_subgraphs/subgraph_000}"
SYN200_DATA="${SYN200_DATA:-data/HGEN_syn_200_subgraphs/subgraph_000}"

actual_sha256="$(sha256sum "$BUNDLE" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "artifact SHA256 mismatch: $actual_sha256 != $EXPECTED_SHA256" >&2
  exit 2
fi

rm -rf "$ARTIFACT_ROOT"
mkdir -p "$ARTIFACT_ROOT"
tar -xzf "$BUNDLE" -C "$ARTIFACT_ROOT"

dataset_dir() {
  case "$1" in
    imdb) echo "$IMDB_DATA" ;;
    pubmed) echo "$PUBMED_DATA" ;;
    syn200) echo "$SYN200_DATA" ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}

context_checkpoint() {
  case "$1" in
    imdb) echo "$ARTIFACT_ROOT/imdb_no_role/rooted_context_residual.pt" ;;
    pubmed) echo "$ARTIFACT_ROOT/pubmed_no_role/rooted_context_residual.pt" ;;
    syn200) echo "$ARTIFACT_ROOT/syn200_no_role/rooted_context_residual.pt" ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}

sample_steps() {
  case "$1" in
    imdb) echo "${IMDB_STEPS:-10000}" ;;
    pubmed) echo "${PUBMED_STEPS:-10000}" ;;
    syn200) echo "${SYN200_STEPS:-1800}" ;;
  esac
}

for dataset in $DATASETS; do
  [[ -d "$(dataset_dir "$dataset")" ]] || {
    echo "missing dataset: $(dataset_dir "$dataset")" >&2
    exit 2
  }
  [[ -f "$(context_checkpoint "$dataset")" ]] || {
    echo "missing context artifact: $(context_checkpoint "$dataset")" >&2
    exit 2
  }
done

"$PYTHON" - "$ARTIFACT_ROOT" <<'PY'
import pathlib
import sys
import torch

root = pathlib.Path(sys.argv[1])
expected = {
    "imdb_no_role": 0,
    "pubmed_no_role": 0,
    "syn200_no_role": 0,
}
for name, role_dim in expected.items():
    path = root / name / "rooted_context_residual.pt"
    payload = torch.load(path, map_location="cpu")
    actual = int(payload["model_config"]["role_condition_dim"])
    if actual != role_dim:
        raise RuntimeError(f"{path}: role_condition_dim={actual}, expected {role_dim}")
    report = payload.get("report", {})
    if float(report.get("stop_loss_weight", 0.0)) <= 0.0:
        raise RuntimeError(f"{path}: checkpoint lacks stop calibration")
print("[NEW-CONTEXT-ARTIFACTS] verified", flush=True)
PY

if [[ "${VERIFY_ONLY:-0}" == "1" ]]; then
  echo "[VERIFY-ONLY] artifact and datasets are ready"
  exit 0
fi

read -r -a gpus <<< "$GPU_LIST"
ngpu="${#gpus[@]}"
(( ngpu > 0 )) || { echo "GPU_LIST is empty" >&2; exit 2; }
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/driver.log"

run_pair() {
  local gpu="$1" dataset="$2" seed="$3"
  local data checkpoint steps base prefix
  data="$(dataset_dir "$dataset")"
  checkpoint="$(context_checkpoint "$dataset")"
  steps="$(sample_steps "$dataset")"
  base="$OUT_ROOT/$dataset/seed${seed}"
  prefix="$LOG_ROOT/${dataset}_s${seed}"
  mkdir -p "$base"

  local common=(
    --dataset-dir "$data"
    --context-checkpoint "$checkpoint"
    --seed "$seed"
    --device cuda
    --sampling-backend "$SAMPLING_BACKEND"
    --gpu-candidate-mode "$GPU_CANDIDATE_MODE"
    --gpu-proposal-pool-size "$GPU_PROPOSAL_POOL_SIZE"
    --gpu-proposal-chunk-size "$GPU_PROPOSAL_CHUNK_SIZE"
    --steps "$steps"
    --candidates 16
    --metrics-every "$(( steps / 10 ))"
    --model-time-horizon-steps "$(( steps * 2 ))"
    --family-count-mode binomial
    --strict-target-free
    --role-weight 0.05
    --gain-scale 10
    --schedule-power 1
    --schedule-floor 1
    --role-error-per-node-threshold 5.5
    --closure-candidate-fraction 1.0
    --auto-role-balance 0.5
    --targeted-role-candidate-fraction 0.5
    --context-refresh-steps 20
  )

  echo "[PAIR-START] dataset=$dataset seed=$seed gpu=$gpu steps=$steps" \
    | tee -a "$LOG_ROOT/driver.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_role_guided_replacement_gate.py \
    "${common[@]}" --context-scale 0 --output-dir "$base/base" \
    > "${prefix}_base.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_role_guided_replacement_gate.py \
    "${common[@]}" --context-scale -1 --output-dir "$base/context" \
    > "${prefix}_context.log" 2>&1
  echo "[PAIR-DONE] dataset=$dataset seed=$seed gpu=$gpu" \
    | tee -a "$LOG_ROOT/driver.log"
}

for dataset in $DATASETS; do
  echo "[DATASET-START] $dataset" | tee -a "$LOG_ROOT/driver.log"
  pids=()
  index=0
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    gpu="${gpus[$(( index % ngpu ))]}"
    run_pair "$gpu" "$dataset" "$seed" &
    pids+=("$!")
    index=$((index + 1))
    sleep 3
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { echo "$dataset paired generation failed" >&2; exit 1; }
  echo "[DATASET-DONE] $dataset" | tee -a "$LOG_ROOT/driver.log"
done

"$PYTHON" - "$OUT_ROOT" "$EXPECTED_SHA256" <<'PY'
import csv
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
artifact_sha256 = sys.argv[2]
rows = []
for context_path in sorted(root.glob("*/seed*/context/gate_report.json")):
    dataset = context_path.parts[-4]
    seed = int(context_path.parts[-3].removeprefix("seed"))
    base_path = context_path.parents[1] / "base" / "gate_report.json"
    context = json.loads(context_path.read_text())
    base = json.loads(base_path.read_text())
    if context["initial"] != base["initial"]:
        raise RuntimeError(f"paired initial mismatch: {dataset}/seed{seed}")
    for mode, report in (("base", base), ("context", context)):
        row = {"dataset": dataset, "seed": seed, "mode": mode}
        for prefix in ("initial", "final", "real"):
            for key, value in report[prefix].items():
                if isinstance(value, (int, float)):
                    row[f"{prefix}_{key}"] = value
        rows.append(row)

if not rows:
    raise RuntimeError("no paired reports found")
columns = sorted({key for row in rows for key in row})
with (root / "paired_results.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

aggregates = []
for dataset in sorted({row["dataset"] for row in rows}):
    for mode in ("base", "context"):
        selected = [r for r in rows if r["dataset"] == dataset and r["mode"] == mode]
        item = {"dataset": dataset, "mode": mode, "n": len(selected)}
        for metric in (
            "edges", "triangles", "clustering", "transitivity", "lcc",
            "assortativity", "sum_degree_sq", "max_degree",
        ):
            key = f"final_{metric}"
            values = [float(row[key]) for row in selected]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregates.append(item)

summary = {
    "format": "dihug-new-context-cross-dataset-v1",
    "artifact_sha256": artifact_sha256,
    "rows": rows,
    "aggregates": aggregates,
}
(root / "aggregate.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps({"aggregates": aggregates}, indent=2), flush=True)
PY

echo "[NEW-CONTEXT-CROSS-DATASET-DONE] $OUT_ROOT/aggregate.json"

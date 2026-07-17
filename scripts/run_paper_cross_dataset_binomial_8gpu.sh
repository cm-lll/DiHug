#!/usr/bin/env bash
# Target-free cross-dataset evaluation from the learned Binomial-M top-noise prior.
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

OUT_ROOT="${OUT_ROOT:-results/paper_cross_dataset_binomial_0717}"
LOG_ROOT="${LOG_ROOT:-logs/paper_cross_dataset_binomial_0717}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-7}"
PYTHON="${PYTHON:-python}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-gpu}"
GPU_PROPOSAL_POOL_SIZE="${GPU_PROPOSAL_POOL_SIZE:-0}"
GPU_PROPOSAL_CHUNK_SIZE="${GPU_PROPOSAL_CHUNK_SIZE:-4096}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/driver.log"

dataset_dir() {
  case "$1" in
    dblp) echo data/DBLP_four_area_subgraphs/subgraph_000 ;;
    syn100) echo data/HGEN_syn_100_subgraphs/subgraph_000 ;;
    syn200) echo data/HGEN_syn_200_subgraphs/subgraph_000 ;;
    syn500) echo data/HGEN_syn_500_subgraphs/subgraph_000 ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}

legacy_checkpoint() {
  echo "results/hgen_official_fullgraph_pilot_0716/$1/seed0/model/mixed_edit_diffusion.pt"
}

context_checkpoint() {
  echo "results/rooted_context_end_to_end_0717/context/$1/rooted_context_residual.pt"
}

sample_steps() {
  case "$1" in
    dblp) echo "${DBLP_STEPS:-50000}" ;;
    syn100) echo "${SYN100_STEPS:-1000}" ;;
    syn200) echo "${SYN200_STEPS:-1800}" ;;
    syn500) echo "${SYN500_STEPS:-5000}" ;;
  esac
}

common_args() {
  local key="$1" seed="$2" steps="$3"
  printf '%s\0' \
    --dataset-dir "$(dataset_dir "$key")" \
    --seed "$seed" --device cuda --sampling-backend "$SAMPLING_BACKEND" \
    --gpu-proposal-pool-size "$GPU_PROPOSAL_POOL_SIZE" \
    --gpu-proposal-chunk-size "$GPU_PROPOSAL_CHUNK_SIZE" \
    --steps "$steps" --candidates 16 \
    --metrics-every "$(( steps / 10 ))" --family-count-mode binomial \
    --role-weight 0.05 --gain-scale 10 --schedule-power 1 --schedule-floor 1 \
    --role-error-per-node-threshold 5.5 \
    --model-time-horizon-steps "$(( steps * 2 ))" \
    --closure-candidate-fraction 1.0 --auto-role-balance 0.5 \
    --targeted-role-candidate-fraction 0.5 --context-refresh-steps 20
}

run_seed() {
  local gpu="$1" key="$2" seed="$3" steps
  steps="$(sample_steps "$key")"
  local base="$OUT_ROOT/$key/seed${seed}"
  local log_prefix="$LOG_ROOT/${key}_s${seed}"
  local args=()
  while IFS= read -r -d '' value; do args+=("$value"); done < <(common_args "$key" "$seed" "$steps")
  mkdir -p "$base"

  if [[ "$key" == dblp ]]; then
    if [[ -f "$base/legacy/gate_report.json" ]]; then
      echo "[SKIP] gpu=$gpu dataset=$key seed=$seed existing=legacy" \
        | tee -a "$LOG_ROOT/driver.log"
      return
    fi
  elif [[ -f "$base/legacy/gate_report.json" && -f "$base/context/gate_report.json" ]]; then
    echo "[SKIP] gpu=$gpu dataset=$key seed=$seed existing=paired" \
      | tee -a "$LOG_ROOT/driver.log"
    return
  fi

  echo "[START] gpu=$gpu dataset=$key seed=$seed steps=$steps prior=binomial" \
    | tee -a "$LOG_ROOT/driver.log"
  if [[ "$key" == dblp ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_role_guided_replacement_gate.py \
      "${args[@]}" --model-checkpoint "$(legacy_checkpoint "$key")" \
      --output-dir "$base/legacy" > "${log_prefix}_legacy.log" 2>&1
  else
    local context
    context="$(context_checkpoint "$key")"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_role_guided_replacement_gate.py \
      "${args[@]}" --context-checkpoint "$context" --context-scale 0 \
      --output-dir "$base/legacy" > "${log_prefix}_legacy.log" 2>&1
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_role_guided_replacement_gate.py \
      "${args[@]}" --context-checkpoint "$context" --context-scale -1 \
      --output-dir "$base/context" > "${log_prefix}_context.log" 2>&1
  fi
  echo "[DONE] gpu=$gpu dataset=$key seed=$seed" | tee -a "$LOG_ROOT/driver.log"
}

required=(
  "$(legacy_checkpoint dblp)"
  "$(context_checkpoint syn100)"
  "$(context_checkpoint syn200)"
  "$(context_checkpoint syn500)"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "missing checkpoint: $path" >&2; exit 2; }
done

# One worker owns one seed. Small synthetic evaluations run first, then every
# worker reaches one DBLP job so the expensive 50k-step stage uses all GPUs.
worker() {
  local gpu="$1" seed="$2"
  local key
  for key in syn100 syn200 syn500 dblp; do
    run_seed "$gpu" "$key" "$seed"
  done
}

echo "[CROSS-DATASET] commit=$(git rev-parse HEAD) seeds=$SEED_START-$SEED_END" \
  | tee -a "$LOG_ROOT/driver.log"
pids=()
gpu=0
for seed in $(seq "$SEED_START" "$SEED_END"); do
  worker "$gpu" "$seed" &
  pids+=("$!")
  gpu=$((gpu + 1))
  if (( gpu == 8 )); then gpu=0; fi
  sleep 3
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "one or more workers failed" >&2; exit 1; }

"$PYTHON" - "$OUT_ROOT" <<'PY'
import csv
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for report_path in sorted(root.glob("*/seed*/*/gate_report.json")):
    dataset = report_path.parts[-4]
    seed = int(report_path.parts[-3].removeprefix("seed"))
    mode = report_path.parts[-2]
    report = json.loads(report_path.read_text())
    if report.get("family_count_mode") != "binomial":
        raise RuntimeError(f"non-Binomial report in final set: {report_path}")
    row = {
        "dataset": dataset,
        "seed": seed,
        "mode": mode,
        "steps": report["steps"],
        "family_count_mode": report["family_count_mode"],
        "sampling_uses_clean_adjacency": report["sampling_uses_clean_adjacency"],
    }
    for prefix in ("initial", "final", "real"):
        for metric, value in report[prefix].items():
            row[f"{prefix}_{metric}"] = value
    rows.append(row)

if not rows:
    raise RuntimeError("no cross-dataset reports found")
columns = sorted({key for row in rows for key in row})
with (root / "paper_cross_dataset.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, columns, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

metrics = (
    "edges", "triangles", "clustering", "transitivity", "lcc",
    "assortativity", "sum_degree_sq", "max_degree",
)
aggregates = []
for dataset in sorted({row["dataset"] for row in rows}):
    for mode in sorted({row["mode"] for row in rows if row["dataset"] == dataset}):
        group = [row for row in rows if row["dataset"] == dataset and row["mode"] == mode]
        item = {"dataset": dataset, "mode": mode, "n": len(group)}
        for phase in ("initial", "final"):
            for metric in metrics:
                values = [float(row[f"{phase}_{metric}"]) for row in group]
                item[f"{phase}_{metric}_mean"] = statistics.fmean(values)
                item[f"{phase}_{metric}_std"] = statistics.pstdev(values)
        for metric in metrics:
            item[f"real_{metric}"] = float(group[0][f"real_{metric}"])
        aggregates.append(item)

(root / "aggregate.json").write_text(json.dumps({
    "format": "dihug-cross-dataset-binomial-v1",
    "initialization": "independent family Bernoulli; M_f is Binomial",
    "exact_training_family_counts_used": False,
    "clean_adjacency_used_at_sampling": False,
    "aggregates": aggregates,
}, indent=2, sort_keys=True))
print(json.dumps(aggregates, indent=2, sort_keys=True))
PY

echo "[CROSS-DATASET] complete: $OUT_ROOT/aggregate.json" | tee -a "$LOG_ROOT/driver.log"

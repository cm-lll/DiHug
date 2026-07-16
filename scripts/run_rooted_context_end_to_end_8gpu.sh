#!/usr/bin/env bash
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"

ROOT="${ROOT:-results/rooted_context_end_to_end_0717}"
LOG_ROOT="${LOG_ROOT:-logs/rooted_context_end_to_end_0717}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
mkdir -p "$ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/driver.log"

dataset_dir() {
  case "$1" in
    syn100) echo data/HGEN_syn_100_subgraphs/subgraph_000 ;;
    syn200) echo data/HGEN_syn_200_subgraphs/subgraph_000 ;;
    syn500) echo data/HGEN_syn_500_subgraphs/subgraph_000 ;;
    imdb) echo data/IMDB_movie_subgraphs/subgraph_000 ;;
    pubmed) echo data/PubMed_baseline_subgraphs/subgraph_000 ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}

legacy_checkpoint() {
  case "$1" in
    syn100|syn200|syn500)
      echo "results/hgen_official_fullgraph_pilot_0716/$1/seed0/model/mixed_edit_diffusion.pt" ;;
    imdb) echo results/imdb_exchangeable_compact_s0_3/model/mixed_edit_diffusion.pt ;;
    pubmed) echo results/pubmed_role_corruption_gate_0716/f100_all/model/mixed_edit_diffusion.pt ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}

context_checkpoint() {
  case "$1" in
    syn100|syn200|syn500) echo "$ROOT/context/$1/rooted_context_residual.pt" ;;
    imdb) echo results/rooted_context_freeze/imdb/rooted_context_residual.pt ;;
    pubmed) echo results/rooted_context_freeze/pubmed/rooted_context_residual.pt ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}

train_context() {
  local gpu="$1" key="$2"
  local output="$ROOT/context/$key"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_rooted_context_residual_gate.py \
    --dataset-dir "$(dataset_dir "$key")" \
    --legacy-checkpoint "$(legacy_checkpoint "$key")" \
    --output-dir "$output" --device cuda --seed 0 \
    --trajectories 4 --examples-per-trajectory 16 --switches-per-example 64 \
    --candidates 16 --closure-fraction 1.0 --epochs 15 --batch-size 2 \
    --hidden-dim 64 --heads 4 --layers 2 \
    > "$LOG_ROOT/train_${key}.log" 2>&1
}

echo "[STAGE-A] train self-contained synthetic context bundles" | tee -a "$LOG_ROOT/driver.log"
pids=()
for item in "0 syn100" "1 syn200" "2 syn500"; do
  read -r gpu key <<< "$item"
  train_context "$gpu" "$key" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

sample_pair() {
  local gpu="$1" key="$2" seed="$3" steps="$4"
  local checkpoint
  checkpoint="$(context_checkpoint "$key")"
  local base="$ROOT/sample/$key/seed${seed}"
  mkdir -p "$base"
  local common=(
    --dataset-dir "$(dataset_dir "$key")"
    --context-checkpoint "$checkpoint"
    --seed "$seed" --device cuda --steps "$steps" --candidates 16
    --metrics-every "$(( steps / 10 ))"
    --role-weight 0.05 --gain-scale 10 --schedule-power 1 --schedule-floor 1
    --role-error-per-node-threshold 5.5
    --model-time-horizon-steps "$(( steps * 2 ))"
    --closure-candidate-fraction 1.0
    --auto-role-balance 0.5
    --targeted-role-candidate-fraction 0.5
    --context-refresh-steps 20
  )
  echo "[PAIR-START] gpu=$gpu dataset=$key seed=$seed steps=$steps" \
    | tee -a "$LOG_ROOT/driver.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_role_guided_replacement_gate.py \
    "${common[@]}" --context-scale 0 --output-dir "$base/legacy" \
    > "$LOG_ROOT/${key}_s${seed}_legacy.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_role_guided_replacement_gate.py \
    "${common[@]}" --output-dir "$base/context" \
    > "$LOG_ROOT/${key}_s${seed}_context.log" 2>&1
  echo "[PAIR-DONE] gpu=$gpu dataset=$key seed=$seed" | tee -a "$LOG_ROOT/driver.log"
}

echo "[STAGE-B] paired end-to-end sampling" | tee -a "$LOG_ROOT/driver.log"
TASKS=(
  "0 syn100 0 1000"
  "1 syn100 1 1000"
  "2 syn200 0 2000"
  "3 syn200 1 2000"
  "4 syn500 0 5000"
  "5 syn500 1 5000"
  "6 imdb 0 10000"
  "7 pubmed 0 10000"
)
pids=()
for task in "${TASKS[@]}"; do
  read -r gpu key seed steps <<< "$task"
  sample_pair "$gpu" "$key" "$seed" "$steps" &
  pids+=("$!")
  sleep 3
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" -ne 0 ]]; then
  echo "paired sampling failed" >&2
  exit 1
fi

"$PYTHON" - "$ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for report_path in sorted((root / "sample").glob("*/seed*/context/gate_report.json")):
    key = report_path.parts[-4]
    seed = int(report_path.parts[-3].removeprefix("seed"))
    context = json.loads(report_path.read_text())
    legacy = json.loads((report_path.parents[1] / "legacy" / "gate_report.json").read_text())
    if context["initial"] != legacy["initial"]:
        raise RuntimeError(f"paired initial states disagree: {key}/seed{seed}")
    real = context["real"]
    row = {
        "dataset": key,
        "seed": seed,
        "steps": context["steps"],
        "selected_scale": context["context_scale"],
        "transition_step": context["trace"][-1]["role_phase_transition_step"],
        "initial": context["initial"],
        "legacy": legacy["final"],
        "context": context["final"],
        "real": real,
    }
    for mode in ("legacy", "context"):
        row[f"{mode}_errors"] = {
            metric: abs(float(row[mode][metric]) - float(real[metric]))
            / max(abs(float(real[metric])), 1.0)
            for metric in ("triangles", "clustering", "lcc", "sum_degree_sq")
        }
    rows.append(row)
(root / "summary.json").write_text(
    json.dumps({"format": "dihug-rooted-context-end-to-end-v1", "rows": rows}, indent=2, sort_keys=True)
)
print("dataset\tseed\tscale\tlegacy_tri\tcontext_tri\treal_tri\tlegacy_clust\tcontext_clust\treal_clust")
for row in rows:
    print(
        f"{row['dataset']}\t{row['seed']}\t{row['selected_scale']:.2f}\t"
        f"{row['legacy']['triangles']:.0f}\t{row['context']['triangles']:.0f}\t{row['real']['triangles']:.0f}\t"
        f"{row['legacy']['clustering']:.6f}\t{row['context']['clustering']:.6f}\t{row['real']['clustering']:.6f}"
    )
PY

echo "[DONE] $ROOT/summary.json" | tee -a "$LOG_ROOT/driver.log"

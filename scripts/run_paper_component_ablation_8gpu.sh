#!/usr/bin/env bash
# Frozen-protocol component ablation: paired Binomial initial graphs.
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-3}"
STEPS="${STEPS:-10000}"
OUT_ROOT="${OUT_ROOT:-results/paper_component_ablation}"
LOG_ROOT="${LOG_ROOT:-logs/paper_component_ablation}"
read -r -a GPUS <<< "$GPU_LIST"
[[ "${#GPUS[@]}" -gt 0 ]] || { echo "GPU_LIST is empty" >&2; exit 2; }
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

data_path() {
  case "$1" in
    imdb) echo "data/IMDB_movie_subgraphs/subgraph_000" ;;
    pubmed) echo "data/PubMed_baseline_subgraphs/subgraph_000" ;;
    *) return 2 ;;
  esac
}

context_path() {
  case "$1" in
    imdb) echo "results/rooted_context_freeze/imdb/rooted_context_residual.pt" ;;
    pubmed) echo "results/rooted_context_freeze/pubmed/rooted_context_residual.pt" ;;
    *) return 2 ;;
  esac
}

run_mode() {
  local gpu="$1" dataset="$2" seed="$3" mode="$4"
  local data context output log steps="$STEPS"
  data="$(data_path "$dataset")"
  context="$(context_path "$dataset")"
  output="$OUT_ROOT/$dataset/seed${seed}/$mode"
  log="$LOG_ROOT/${dataset}_s${seed}_${mode}.log"
  mkdir -p "$output"
  local extra=(--context-scale 0 --phase-policy auto --selection-mode model)
  case "$mode" in
    prior)
      steps=0
      ;;
    random)
      extra=(--context-scale 0 --phase-policy auto --selection-mode random
        --closure-candidate-fraction 0 --targeted-role-candidate-fraction 0)
      ;;
    relocation)
      extra=(--context-scale 0 --phase-policy relocation_only --selection-mode model
        --closure-candidate-fraction 0 --targeted-role-candidate-fraction 0.5)
      ;;
    switch)
      extra=(--context-scale 0 --phase-policy switch_only --selection-mode model
        --role-weight 0 --closure-candidate-fraction 1
        --targeted-role-candidate-fraction 0)
      ;;
    full)
      extra=(--context-scale 0 --phase-policy auto --selection-mode model)
      ;;
    context)
      extra=(--context-scale -1 --phase-policy auto --selection-mode model)
      ;;
    *) echo "unknown component mode: $mode" >&2; return 2 ;;
  esac
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_role_guided_replacement_gate.py \
    --dataset-dir "$data" --context-checkpoint "$context" \
    --output-dir "$output" --seed "$seed" --device cuda \
    --sampling-backend cpu --family-count-mode binomial \
    --steps "$steps" --candidates 16 --metrics-every "$(( STEPS / 10 ))" \
    --model-time-horizon-steps "$(( STEPS * 2 ))" \
    --role-weight 0.05 --gain-scale 10 --schedule-power 1 --schedule-floor 1 \
    --role-error-per-node-threshold 5.5 --closure-candidate-fraction 1 \
    --auto-role-balance 0.5 --targeted-role-candidate-fraction 0.5 \
    --context-refresh-steps 20 "${extra[@]}" > "$log" 2>&1
}

run_worker() {
  local gpu="$1" dataset="$2" seed="$3"
  for mode in prior random relocation switch full context; do
    echo "[ABLATION-START] gpu=$gpu dataset=$dataset seed=$seed mode=$mode"
    run_mode "$gpu" "$dataset" "$seed" "$mode"
  done
  echo "[ABLATION-DONE] gpu=$gpu dataset=$dataset seed=$seed"
}

pids=()
slot=0
for dataset in imdb pubmed; do
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    gpu="${GPUS[$((slot % ${#GPUS[@]}))]}"
    run_worker "$gpu" "$dataset" "$seed" & pids+=("$!")
    slot=$((slot + 1))
    sleep 3
  done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "component ablation failed" >&2; exit 1; }

python - "$OUT_ROOT" "$(git rev-parse HEAD)" <<'PY'
import csv, json, pathlib, statistics, sys

root = pathlib.Path(sys.argv[1])
rows = []
for report_path in sorted(root.glob("*/seed*/*/gate_report.json")):
    report = json.loads(report_path.read_text())
    dataset = report_path.parts[-4]
    seed = int(report_path.parts[-3].removeprefix("seed"))
    mode = report_path.parts[-2]
    row = {
        "dataset": dataset, "seed": seed, "mode": mode,
        "steps": int(report["steps"]),
        "selection_mode": report["selection_mode"],
        "phase_policy": report["phase_policy"],
        "initial_graph_source": report["initial_graph_source"],
    }
    for prefix in ("initial", "final", "real"):
        for metric in (
            "edges", "lcc", "triangles", "clustering", "transitivity",
            "assortativity", "sum_degree_sq", "max_degree",
        ):
            row[f"{prefix}_{metric}"] = report[prefix][metric]
    row.update({f"actions_{k}": v for k, v in report["action_counts"].items()})
    rows.append(row)

for dataset in {row["dataset"] for row in rows}:
    for seed in {row["seed"] for row in rows if row["dataset"] == dataset}:
        group = [row for row in rows if row["dataset"] == dataset and row["seed"] == seed]
        initials = {json.dumps({k: v for k, v in row.items() if k.startswith("initial_")}, sort_keys=True) for row in group}
        if len(initials) != 1:
            raise RuntimeError(f"paired initial mismatch: {dataset}/seed{seed}")

with (root / "component_runs.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader(); writer.writerows(rows)

summary = []
for dataset in sorted({row["dataset"] for row in rows}):
    for mode in ("prior", "random", "relocation", "switch", "full", "context"):
        group = [row for row in rows if row["dataset"] == dataset and row["mode"] == mode]
        item = {"dataset": dataset, "mode": mode, "n": len(group)}
        for metric in (
            "edges", "lcc", "triangles", "clustering", "transitivity",
            "assortativity", "sum_degree_sq", "max_degree",
        ):
            values = [float(row[f"final_{metric}"]) for row in group]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = statistics.pstdev(values)
        summary.append(item)
with (root / "component_ablation.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]), delimiter="\t")
    writer.writeheader(); writer.writerows(summary)
(root / "component_ablation.json").write_text(json.dumps({
    "format": "dihug-frozen-component-ablation-v1",
    "commit": sys.argv[2], "family_count_mode": "binomial",
    "paired_initials_verified": True, "rows": summary,
}, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[COMPONENT-ABLATION] complete: $OUT_ROOT/component_ablation.tsv"

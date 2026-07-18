#!/usr/bin/env bash
# Reproducible backend benchmark for the frozen role-guided sampler.
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

GPU_LIST="${GPU_LIST:-3 4 5}"
REPETITIONS="${REPETITIONS:-3}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-500}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
OUT_ROOT="${OUT_ROOT:-results/paper_efficiency}"
LOG_ROOT="${LOG_ROOT:-logs/paper_efficiency}"
PYTHON="${PYTHON:-python}"
read -r -a GPUS <<< "$GPU_LIST"
[[ "${#GPUS[@]}" -ge 3 ]] || { echo "GPU_LIST needs three GPUs" >&2; exit 2; }
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

dataset_dir() {
  case "$1" in
    imdb) echo "${IMDB_DATA:-data/IMDB_movie_subgraphs/subgraph_000}" ;;
    pubmed) echo "${PUBMED_DATA:-data/PubMed_baseline_subgraphs/subgraph_000}" ;;
    dblp) echo "${DBLP_DATA:-data/DBLP_four_area_subgraphs/subgraph_000}" ;;
    *) return 2 ;;
  esac
}

checkpoint_args() {
  case "$1" in
    imdb)
      echo "--context-checkpoint ${IMDB_CONTEXT:-results/rooted_context_freeze/imdb/rooted_context_residual.pt} --context-scale 0"
      ;;
    pubmed)
      echo "--context-checkpoint ${PUBMED_CONTEXT:-results/rooted_context_freeze/pubmed/rooted_context_residual.pt} --context-scale -1"
      ;;
    dblp)
      echo "--model-checkpoint ${DBLP_MODEL:-results/hgen_official_fullgraph_pilot_0716/dblp/seed0/model/mixed_edit_diffusion.pt}"
      ;;
    *) return 2 ;;
  esac
}

run_one() {
  local gpu="$1" dataset="$2" backend="$3" repetition="$4" steps="$5"
  local output="$OUT_ROOT/$dataset/${backend}_r${repetition}"
  local log="$LOG_ROOT/${dataset}_${backend}_r${repetition}.log"
  local rss="$output/max_rss_kb.txt"
  local data
  data="$(dataset_dir "$dataset")"
  read -r -a checkpoint <<< "$(checkpoint_args "$dataset")"
  local backend_args=()
  case "$backend" in
    cpu_eager)
      backend_args=(--device cpu --sampling-backend cpu)
      ;;
    gpu_resident_eager)
      backend_args=(--device cuda --sampling-backend gpu --gpu-proposal-pool-size 0 --gpu-disable-triton)
      ;;
    gpu_streamed_eager)
      backend_args=(--device cuda --sampling-backend gpu --gpu-proposal-pool-size 4096 --gpu-proposal-chunk-size 1024 --gpu-disable-triton)
      ;;
    gpu_streamed_triton)
      backend_args=(--device cuda --sampling-backend gpu --gpu-proposal-pool-size 4096 --gpu-proposal-chunk-size 1024)
      ;;
    *) echo "unknown backend: $backend" >&2; return 2 ;;
  esac
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu" /usr/bin/time -f '%M' -o "$rss" \
    "$PYTHON" -u scripts/run_role_guided_replacement_gate.py \
    --dataset-dir "$data" "${checkpoint[@]}" --output-dir "$output" \
    --seed "$((1000 + repetition))" --family-count-mode binomial \
    --steps "$steps" --candidates 16 --metrics-every "$steps" \
    --model-time-horizon-steps "$((steps * 2))" \
    --role-weight 0.05 --gain-scale 10 --schedule-power 1 --schedule-floor 1 \
    --role-error-per-node-threshold 5.5 --closure-candidate-fraction 1.0 \
    --auto-role-balance 0.5 --targeted-role-candidate-fraction 0.5 \
    --context-refresh-steps 20 "${backend_args[@]}" > "$log" 2>&1
}

run_dataset() {
  local gpu="$1" dataset="$2"
  local backends=(cpu_eager gpu_resident_eager gpu_streamed_eager gpu_streamed_triton)
  for backend in "${backends[@]}"; do
    run_one "$gpu" "$dataset" "$backend" 0 "$WARMUP_STEPS"
    rm -rf "$OUT_ROOT/$dataset/${backend}_r0"
    for repetition in $(seq 1 "$REPETITIONS"); do
      run_one "$gpu" "$dataset" "$backend" "$repetition" "$BENCHMARK_STEPS"
    done
  done
}

pids=()
for item in "0 imdb" "1 pubmed" "2 dblp"; do
  read -r slot dataset <<< "$item"
  run_dataset "${GPUS[$slot]}" "$dataset" & pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "efficiency benchmark failed" >&2; exit 1; }

"$PYTHON" - "$OUT_ROOT" "$REPETITIONS" "$(git rev-parse HEAD)" <<'PY'
import csv
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
repetitions = int(sys.argv[2])
commit = sys.argv[3]
rows = []
for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    for report_path in sorted(dataset_dir.glob("*_r*/gate_report.json")):
        report = json.loads(report_path.read_text())
        name = report_path.parent.name
        backend, repetition_text = name.rsplit("_r", 1)
        initial_edges = int(report["initial"]["edges"])
        final_edges = int(report["final"]["edges"])
        if initial_edges != final_edges:
            raise RuntimeError(f"family-count invariant failed: {report_path}")
        rss_path = report_path.parent / "max_rss_kb.txt"
        rows.append({
            "commit": commit,
            "dataset": dataset_dir.name,
            "backend": backend,
            "repetition": int(repetition_text),
            "steps": int(report["steps"]),
            "sampling_seconds": float(report["sampling_seconds"]),
            "steps_per_second": float(report["sampling_steps_per_second"]),
            "peak_vram_allocated_mb": float(report.get("gpu_peak_memory_allocated_bytes", 0)) / 2**20,
            "peak_vram_reserved_mb": float(report.get("gpu_peak_memory_reserved_bytes", 0)) / 2**20,
            "max_rss_mb": float(rss_path.read_text().strip()) / 1024.0,
            "initial_edges": initial_edges,
            "final_edges": final_edges,
            "family_count_invariant": 1,
            "gpu_use_triton": int(bool(report.get("gpu_use_triton", False))),
            "gpu_proposal_pool_size": int(report.get("gpu_proposal_pool_size", 0)),
        })

with (root / "efficiency_runs.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader(); writer.writerows(rows)

summary = []
for dataset in sorted({row["dataset"] for row in rows}):
    for backend in sorted({row["backend"] for row in rows}):
        group = [row for row in rows if row["dataset"] == dataset and row["backend"] == backend]
        if len(group) != repetitions:
            continue
        summary.append({
            "dataset": dataset,
            "backend": backend,
            "n": len(group),
            "sampling_seconds_mean": statistics.fmean(row["sampling_seconds"] for row in group),
            "sampling_seconds_std": statistics.pstdev(row["sampling_seconds"] for row in group),
            "steps_per_second_mean": statistics.fmean(row["steps_per_second"] for row in group),
            "peak_vram_reserved_mb_max": max(row["peak_vram_reserved_mb"] for row in group),
            "max_rss_mb_max": max(row["max_rss_mb"] for row in group),
            "family_count_invariant": 1,
        })
with (root / "efficiency.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]), delimiter="\t")
    writer.writeheader(); writer.writerows(summary)
(root / "efficiency.json").write_text(json.dumps({
    "format": "dihug-paper-efficiency-v1", "commit": commit,
    "repetitions": repetitions, "rows": summary,
}, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[PAPER-EFFICIENCY] complete: $OUT_ROOT/efficiency.tsv"

#!/usr/bin/env bash
# Matched local HGEN 40% split for IMDB and DBLP. No test-metric tuning.
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

GPU_LIST="${GPU_LIST:-6 7}"
OUT_ROOT="${OUT_ROOT:-results/hgen40_matched_imdb_dblp}"
LOG_ROOT="${LOG_ROOT:-logs/hgen40_matched_imdb_dblp}"
SPLIT_SEED="${SPLIT_SEED:-481516234}"
TEST_SHARE="${TEST_SHARE:-0.4}"
SAMPLE_STEPS="${SAMPLE_STEPS:-320}"
SAMPLE_CANDIDATES="${SAMPLE_CANDIDATES:-16}"
HGEN_EPOCHS="${HGEN_EPOCHS:-5}"
HGEN_ROOT="${HGEN_ROOT:-/data2/lyh/gnn_project/HGEN}"
HGEN_PYTHON="${HGEN_PYTHON:-/data2/lyh/miniconda3/envs/hgen_env/bin/python}"
PYTHON="${PYTHON:-python}"
read -r -a GPUS <<< "$GPU_LIST"
[[ "${#GPUS[@]}" -ge 2 ]] || { echo "GPU_LIST needs two GPUs" >&2; exit 2; }
[[ -x "$HGEN_PYTHON" ]] || { echo "missing HGEN Python: $HGEN_PYTHON" >&2; exit 2; }
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

dataset_dir() {
  case "$1" in
    imdb) echo "${IMDB_DATA:-data/IMDB_movie_subgraphs/subgraph_000}" ;;
    dblp) echo "${DBLP_DATA:-data/DBLP_four_area_subgraphs/subgraph_000}" ;;
    *) return 2 ;;
  esac
}

hgen_name() {
  case "$1" in imdb) echo IMDB_movie ;; dblp) echo DBLP_four_area ;; *) return 2 ;; esac
}

prior_path() {
  case "$1" in
    imdb) echo "${IMDB_PRIOR:-results/imdb_exchangeable_compact_s0_3/prior/exchangeable_role_prior.json}" ;;
    dblp) echo "${DBLP_PRIOR:-results/hgen_official_fullgraph_pilot_0716/dblp/shared_prior/exchangeable_role_prior.json}" ;;
    *) return 2 ;;
  esac
}

checkpoint_path() {
  case "$1" in
    imdb) echo "${IMDB_MODEL:-results/imdb_exchangeable_compact_s0_3/model/mixed_edit_diffusion.pt}" ;;
    dblp) echo "${DBLP_MODEL:-results/hgen_official_fullgraph_pilot_0716/dblp/seed0/model/mixed_edit_diffusion.pt}" ;;
    *) return 2 ;;
  esac
}

run_dataset() {
  local gpu="$1" dataset="$2"
  local data prior checkpoint hgen_dataset base log_dir
  data="$(dataset_dir "$dataset")"
  prior="$(prior_path "$dataset")"
  checkpoint="$(checkpoint_path "$dataset")"
  hgen_dataset="$(hgen_name "$dataset")"
  base="$OUT_ROOT/$dataset"
  log_dir="$LOG_ROOT/$dataset"
  mkdir -p "$base" "$log_dir"
  for required in "$data" "$prior" "$checkpoint" "$HGEN_ROOT/hgen.py"; do
    [[ -e "$required" ]] || { echo "missing required path: $required" >&2; return 2; }
  done

  "$PYTHON" scripts/build_hgen_edge_split_target.py \
    --dataset-dir "$data" --generate-split --output-dir "$base/target" \
    --test-share "$TEST_SHARE" --split-seed "$SPLIT_SEED" \
    > "$log_dir/build_target.log" 2>&1
  local edge_count
  edge_count="$($PYTHON -c "import json; print(json.load(open('$base/target/split_report.json'))['generation_budget']['num_edges'])")"

  for seed in {0..7}; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_exchangeable_role_prior_gate.py \
      --mode sample --output-dir "$base/dihug_seed${seed}" \
      --prior "$prior" --checkpoint "$checkpoint" --device cuda --seed "$seed" \
      --edge-view-count "$edge_count" --edge-view-source hgen_split \
      --sample-steps "$SAMPLE_STEPS" --sample-candidates "$SAMPLE_CANDIDATES" \
      --sample-actions switch --motif-refinement-stage --metrics-every "$SAMPLE_STEPS" \
      > "$log_dir/dihug_seed${seed}.log" 2>&1
  done

  (
    cd "$HGEN_ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" "$HGEN_PYTHON" -u hgen.py \
      --dataset "$hgen_dataset" --n-epochs "$HGEN_EPOCHS"
  ) > "$log_dir/hgen.log" 2>&1
  "$PYTHON" scripts/parse_hgen_baseline_log.py \
    --log "$log_dir/hgen.log" --dataset "$dataset" \
    --output "$base/hgen_baseline.json"

  local generated=()
  for seed in {0..7}; do generated+=("$base/dihug_seed${seed}/generated_graph.json"); done
  "$HGEN_PYTHON" scripts/evaluate_hgen_edge_split_metrics.py \
    --target "$base/target/target_graph.json" --generated "${generated[@]}" \
    --output "$base/dihug_metrics.json" > "$log_dir/dihug_metrics.log" 2>&1

  "$PYTHON" - "$base" "$dataset" "$SPLIT_SEED" "$TEST_SHARE" <<'PY'
import json
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
dataset = sys.argv[2]
split_seed = int(sys.argv[3])
test_share = float(sys.argv[4])
split = json.loads((base / "target/split_report.json").read_text())
hgen = json.loads((base / "hgen_baseline.json").read_text())
dihug = json.loads((base / "dihug_metrics.json").read_text())
target = split["target_metrics"]

def relative_close(left, right, tolerance=1.0e-5):
    return abs(float(left) - float(right)) <= tolerance * max(abs(float(left)), 1.0)

checks = {
    "triangles": relative_close(target["triangles"], hgen["real"]["triangles"]),
    "clustering": relative_close(target["clustering"], hgen["real"]["clustering"]),
    "lcc": relative_close(target["lcc"], hgen["real"]["lcc"]),
}
payload = {
    "format": "dihug-hgen40-matched-local-v1",
    "dataset": dataset,
    "split_seed": split_seed,
    "test_share": test_share,
    "target_fingerprint": target,
    "hgen_real_fingerprint": hgen["real"],
    "fingerprint_checks": checks,
    "directly_matched": all(checks.values()),
    "dihug": dihug,
    "hgen": hgen,
    "hgen_motif": {"available_in_local_repository": False},
    "paper_comparison_rule": (
        "direct-local-split" if all(checks.values()) else "separate-local-protocol"
    ),
}
(base / "matched_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

run_dataset "${GPUS[0]}" imdb & imdb_pid=$!
sleep 15
run_dataset "${GPUS[1]}" dblp & dblp_pid=$!
failed=0
wait "$imdb_pid" || failed=1
wait "$dblp_pid" || failed=1
[[ "$failed" -eq 0 ]] || { echo "HGEN40 matched evaluation failed" >&2; exit 1; }

"$PYTHON" - "$OUT_ROOT" "$(git rev-parse HEAD)" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
payload = {
    "format": "dihug-hgen40-imdb-dblp-bundle-v1",
    "commit": sys.argv[2],
    "rows": [
        json.loads((root / dataset / "matched_summary.json").read_text())
        for dataset in ("imdb", "dblp")
    ],
}
(root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "[HGEN40-MATCHED] complete: $OUT_ROOT/summary.json"

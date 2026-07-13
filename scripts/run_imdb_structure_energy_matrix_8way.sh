#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-results/imdb_structure_energy_matrix_seed${SEED}}"
mkdir -p "${OUT_ROOT}"

run_gate() {
  local name="$1"
  local max_features="$2"
  local learning_rate="$3"
  local moment_l2="$4"
  local curveball="$5"
  local out_dir="${OUT_ROOT}/${name}"
  mkdir -p "${out_dir}"
  python scripts/run_structure_energy_gates.py \
    --output-dir "${out_dir}" \
    --seed "${SEED}" \
    --null-chains 4 \
    --null-steps 30000 \
    --null-burn-in 10000 \
    --energy-chains 2 \
    --energy-steps 500000 \
    --energy-burn-in 200000 \
    --thin 5000 \
    --max-features "${max_features}" \
    --nce-l2 0.1 \
    --nce-max-weight 8.0 \
    --moment-rounds 40 \
    --moment-steps-per-round 4000 \
    --moment-samples-per-round 4 \
    --moment-learning-rate "${learning_rate}" \
    --moment-l2 "${moment_l2}" \
    --moment-tolerance 0.10 \
    --structure-tolerance 0.10 \
    --rhat-threshold 1.10 \
    --min-ess-per-chain 20 \
    --curveball-probability "${curveball}" \
    > "${out_dir}/run.log" 2>&1
}

declare -a pids=()
run_gate f24_lr025_l201 24 0.25 0.01 0.0 & pids+=("$!")
run_gate f32_lr025_l201 32 0.25 0.01 0.0 & pids+=("$!")
run_gate f48_lr025_l201 48 0.25 0.01 0.0 & pids+=("$!")
run_gate f64_lr025_l201 64 0.25 0.01 0.0 & pids+=("$!")
run_gate f32_lr050_l201 32 0.50 0.01 0.0 & pids+=("$!")
run_gate f32_lr0125_l201 32 0.125 0.01 0.0 & pids+=("$!")
run_gate f32_lr025_l205 32 0.25 0.05 0.0 & pids+=("$!")
run_gate f32_lr025_curve50 32 0.25 0.01 0.5 & pids+=("$!")

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done

python - "${OUT_ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for report_path in sorted(root.glob("*/gate_report.json")):
    report = json.loads(report_path.read_text())
    gate3 = report["gate3"]
    rows.append(
        {
            "name": report_path.parent.name,
            "gate1": report["gate1"]["passed"],
            "gate2": report["gate2"]["passed"],
            "gate3": gate3["passed"],
            "triangles": gate3["summary"]["triangles"]["mean"],
            "clustering": gate3["summary"]["clustering"]["mean"],
            "tri_error": gate3["triangle_relative_error"],
            "cluster_error": gate3["clustering_relative_error"],
            "tri_rhat": gate3["summary"]["triangles"]["r_hat"],
            "tri_ess": gate3["summary"]["triangles"]["ess_min"],
            "edge_jaccard": gate3["pairwise_edge_jaccard_mean"],
        }
    )
(root / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
for row in rows:
    print(
        f"{row['name']:<24} tri={row['triangles']:8.2f} "
        f"C={row['clustering']:.4f} err=({row['tri_error']:.3f},"
        f"{row['cluster_error']:.3f}) Rhat={row['tri_rhat']} "
        f"gate3={int(row['gate3'])}"
    )
PY

if (( failures > 0 )); then
  echo "${failures} structure-energy jobs failed" >&2
  exit 1
fi

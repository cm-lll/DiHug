#!/usr/bin/env bash
# Role-conditioned context with explicit terminal/noop supervision.
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

OUT_ROOT="${OUT_ROOT:-results/role_stop_context_gate_0720}"
LOG_ROOT="${LOG_ROOT:-logs/role_stop_context_gate_0720}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

OLD_ROOT="${OLD_ROOT:-/data2/lyh/gnn_project/DiHug}"
JOBS=(
  "imdb_role|data/IMDB_movie_subgraphs/subgraph_000|${OLD_ROOT}/results/imdb_exchangeable_compact_s0_3/model/mixed_edit_diffusion.pt|role"
  "pubmed_role|data/PubMed_baseline_subgraphs/subgraph_000|${OLD_ROOT}/results/pubmed_role_corruption_gate_0716/f100_all/model/mixed_edit_diffusion.pt|role"
  "dblp_role|data/DBLP_four_area_subgraphs/subgraph_000|${OLD_ROOT}/results/hgen_official_fullgraph_pilot_0716/dblp/seed0/model/mixed_edit_diffusion.pt|role"
  "syn200_role|${OLD_ROOT}/data/HGEN_syn_200_subgraphs/subgraph_000|${OLD_ROOT}/results/hgen_official_fullgraph_pilot_0716/syn200/seed0/model/mixed_edit_diffusion.pt|role"
  "acm_role|data/ACM_subgraphs/subgraph_000|results/acm_rs_context_paired_gate_0720/legacy_model/mixed_edit_diffusion.pt|role"
  "imdb_no_role|data/IMDB_movie_subgraphs/subgraph_000|${OLD_ROOT}/results/imdb_exchangeable_compact_s0_3/model/mixed_edit_diffusion.pt|no_role"
  "pubmed_no_role|data/PubMed_baseline_subgraphs/subgraph_000|${OLD_ROOT}/results/pubmed_role_corruption_gate_0716/f100_all/model/mixed_edit_diffusion.pt|no_role"
  "syn200_no_role|${OLD_ROOT}/data/HGEN_syn_200_subgraphs/subgraph_000|${OLD_ROOT}/results/hgen_official_fullgraph_pilot_0716/syn200/seed0/model/mixed_edit_diffusion.pt|no_role"
)

pids=()
for gpu in "${!JOBS[@]}"; do
  IFS='|' read -r name dataset checkpoint variant <<< "${JOBS[$gpu]}"
  [[ -d "${dataset}" ]] || { echo "missing dataset: ${dataset}" >&2; exit 2; }
  [[ -f "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 2; }
  role_args=()
  [[ "${variant}" == "role" ]] && role_args+=(--use-role-condition)
  echo "[ROLE-STOP-LAUNCH] gpu=${gpu} job=${name} variant=${variant}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u scripts/run_rooted_context_residual_gate.py \
    --dataset-dir "${dataset}" --legacy-checkpoint "${checkpoint}" \
    --output-dir "${OUT_ROOT}/${name}" --device cuda --seed 0 \
    --trajectories 4 --examples-per-trajectory 12 \
    --switches-per-example 128 --candidates 16 --closure-fraction 1.0 \
    --terminal-noop-examples-per-trajectory 2 --stop-loss-weight 0.25 \
    --epochs 15 --batch-size 2 --hidden-dim 64 --heads 4 --layers 2 \
    "${role_args[@]}" > "${LOG_ROOT}/${name}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
(( status == 0 )) || {
  echo "at least one role-stop context gate failed" >&2
  exit 1
}

python - "${OUT_ROOT}" "${JOBS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for spec in sys.argv[2:]:
    name, dataset, checkpoint, variant = spec.split("|")
    report = json.loads((root / name / "gate_report.json").read_text())
    rows.append(
        {
            "job": name,
            "dataset": dataset,
            "variant": variant,
            "applicable": report.get("applicable", True),
            "relative_utility_gain": report.get("relative_utility_gain", 0.0),
            "residual_scale": report.get("residual_scale", 0.0),
            "continue_fraction": report.get("validation_continue_fraction", 0.0),
            "base_stop_accuracy": report.get("base_stop_accuracy", 0.0),
            "context_stop_accuracy": report.get("context_stop_accuracy", 0.0),
            "base_terminal_noop_rate": report.get("base_terminal_noop_rate", 0.0),
            "context_terminal_noop_rate": report.get("context_terminal_noop_rate", 0.0),
            "non_regression": report.get("non_regression", 0.0),
        }
    )
summary = {"format": "dihug-role-stop-context-gate-v1", "rows": rows}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2), flush=True)
PY

echo "[ROLE-STOP-DONE] ${OUT_ROOT}/summary.json"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TAG="${TAG:-edge_energy_edit_gate_0719}"
OUT_ROOT="${OUT_ROOT:-results/$TAG}"
LOG_ROOT="${LOG_ROOT:-logs/$TAG}"
SEED="${SEED:-71}"
GPU_LIST=(${GPU_LIST:-0 1 2 3})
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/driver.log"

if [[ ${#GPU_LIST[@]} -lt 4 ]]; then
  echo "GPU_LIST must contain four GPU ids" >&2
  exit 2
fi

launch() {
  local gpu="$1" name="$2" data="$3" legacy="$4" alignment="$5"
  shift 5
  for path in "$legacy" "$alignment"; do
    [[ -f "$path" ]] || { echo "missing artifact: $path" >&2; exit 2; }
  done
  echo "[LAUNCH] dataset=$name gpu=$gpu" | tee -a "$LOG_ROOT/driver.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_edge_energy_edit_gate.py \
    --dataset-dir "$data" \
    --legacy-checkpoint "$legacy" \
    --alignment-checkpoint "$alignment" \
    --output-dir "$OUT_ROOT/$name" \
    --seed "$SEED" --device cuda \
    --candidates 16 --train-candidate-sets 4 \
    --validation-candidate-sets 8 --epochs 100 "$@" \
    > "$LOG_ROOT/$name.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
}

PIDS=()
NAMES=()
OLD_ROOT="${OLD_ROOT:-/data2/lyh/gnn_project/DiHug}"
launch "${GPU_LIST[0]}" imdb data/IMDB_movie_subgraphs/subgraph_000 \
  "$OLD_ROOT/results/imdb_exchangeable_compact_s0_3/model/mixed_edit_diffusion.pt" \
  results/latent_alignment_gate_0719_v2/imdb/latent_alignment.pt \
  --trajectories 6
launch "${GPU_LIST[1]}" pubmed data/PubMed_baseline_subgraphs/subgraph_000 \
  "$OLD_ROOT/results/pubmed_role_corruption_gate_0716/f100_all/model/mixed_edit_diffusion.pt" \
  results/latent_alignment_gate_0719_v2/pubmed/latent_alignment.pt \
  --trajectories 6
launch "${GPU_LIST[2]}" dblp data/DBLP_four_area_subgraphs/subgraph_000 \
  "$OLD_ROOT/results/hgen_official_fullgraph_pilot_0716/dblp/seed0/model/mixed_edit_diffusion.pt" \
  results/latent_alignment_gate_0719_v2/dblp/latent_alignment.pt \
  --trajectories 4
launch "${GPU_LIST[3]}" syn200 \
  "${SYN200_DATA:-$OLD_ROOT/data/HGEN_syn_200_subgraphs/subgraph_000}" \
  "$OLD_ROOT/results/hgen_official_adaptive_0716/syn200/seed0/model/mixed_edit_diffusion.pt" \
  results/latent_alignment_gate_0719_v2/syn200/latent_alignment.pt \
  --trajectories 8

failed=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "[DONE] dataset=${NAMES[$index]}" | tee -a "$LOG_ROOT/driver.log"
  else
    echo "[FAILED] dataset=${NAMES[$index]}" | tee -a "$LOG_ROOT/driver.log"
    failed=1
  fi
done

OUT_ROOT="$OUT_ROOT" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
rows = []
for name in ("imdb", "pubmed", "dblp", "syn200"):
    path = root / name / "edge_energy_report.json"
    if path.exists():
        report = json.loads(path.read_text())
        rows.append(
            {
                "dataset": name,
                "passed": report["passed"],
                "weights": report["action_weights"],
                **report["overall"],
            }
        )
    else:
        rows.append({"dataset": name, "passed": False, "status": "missing"})
real = [row for row in rows if row["dataset"] in {"imdb", "pubmed", "dblp"}]
real_improved = sum(row.get("expected_utility_gain", 0.0) > 0.01 for row in real)
no_material_regression = all(
    row.get("selected_utility_gain", -1.0) >= -0.05 for row in rows
)
summary = {
    "format": "dihug-edge-energy-edit-gate-matrix-v1",
    "rows": rows,
    "real_datasets_improved": real_improved,
    "no_material_regression": no_material_regression,
    "advance_to_sampling": real_improved >= 2 and no_material_regression,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print("[EDGE-ENERGY-SUMMARY] " + json.dumps(summary, sort_keys=True))
PY

exit "$failed"

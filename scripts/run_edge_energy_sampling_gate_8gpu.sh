#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TAG="${TAG:-edge_energy_sampling_gate_0719}"
OUT_ROOT="${OUT_ROOT:-results/$TAG}"
LOG_ROOT="${LOG_ROOT:-logs/$TAG}"
EDGE_ROOT="${EDGE_ROOT:-results/edge_energy_edit_gate_0719_v2}"
OLD_ROOT="${OLD_ROOT:-/data2/lyh/gnn_project/DiHug}"
STEPS="${STEPS:-2000}"
SEED="${SEED:-0}"
GPU_LIST=(${GPU_LIST:-0 1 2 3 4 5 6 7})
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/driver.log"

if [[ ${#GPU_LIST[@]} -lt 8 ]]; then
  echo "GPU_LIST must contain eight GPU ids" >&2
  exit 2
fi

common=(
  --seed "$SEED" --device cuda --sampling-backend cpu
  --steps "$STEPS" --candidates 16 --metrics-every "$((STEPS / 10))"
  --model-time-horizon-steps "$((STEPS * 2))"
  --family-count-mode binomial
  --role-weight 0.05 --gain-scale 10 --schedule-power 1 --schedule-floor 1
  --role-error-per-node-threshold 5.5
  --closure-candidate-fraction 1.0
  --auto-role-balance 0.5
  --targeted-role-candidate-fraction 0.5
  --edge-energy-refresh-steps 20
)

launch_pair() {
  local base_gpu="$1" fused_gpu="$2" name="$3" data="$4" legacy="$5" alignment="$6"
  local fusion="$EDGE_ROOT/$name/edge_energy_fusion.pt"
  for path in "$legacy" "$alignment" "$fusion"; do
    [[ -f "$path" ]] || { echo "missing artifact: $path" >&2; exit 2; }
  done
  mkdir -p "$OUT_ROOT/$name/base" "$OUT_ROOT/$name/fused"
  echo "[PAIR] dataset=$name base_gpu=$base_gpu fused_gpu=$fused_gpu" \
    | tee -a "$LOG_ROOT/driver.log"
  CUDA_VISIBLE_DEVICES="$base_gpu" python -u scripts/run_role_guided_replacement_gate.py \
    --dataset-dir "$data" --model-checkpoint "$legacy" \
    --output-dir "$OUT_ROOT/$name/base" "${common[@]}" \
    > "$LOG_ROOT/${name}_base.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("${name}_base")
  CUDA_VISIBLE_DEVICES="$fused_gpu" python -u scripts/run_role_guided_replacement_gate.py \
    --dataset-dir "$data" --model-checkpoint "$legacy" \
    --edge-energy-checkpoint "$fusion" \
    --edge-energy-alignment-checkpoint "$alignment" \
    --output-dir "$OUT_ROOT/$name/fused" "${common[@]}" \
    > "$LOG_ROOT/${name}_fused.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("${name}_fused")
}

PIDS=()
NAMES=()
launch_pair "${GPU_LIST[0]}" "${GPU_LIST[1]}" imdb \
  data/IMDB_movie_subgraphs/subgraph_000 \
  "$OLD_ROOT/results/imdb_exchangeable_compact_s0_3/model/mixed_edit_diffusion.pt" \
  results/latent_alignment_gate_0719_v2/imdb/latent_alignment.pt
launch_pair "${GPU_LIST[2]}" "${GPU_LIST[3]}" pubmed \
  data/PubMed_baseline_subgraphs/subgraph_000 \
  "$OLD_ROOT/results/pubmed_role_corruption_gate_0716/f100_all/model/mixed_edit_diffusion.pt" \
  results/latent_alignment_gate_0719_v2/pubmed/latent_alignment.pt
launch_pair "${GPU_LIST[4]}" "${GPU_LIST[5]}" dblp \
  data/DBLP_four_area_subgraphs/subgraph_000 \
  "$OLD_ROOT/results/hgen_official_fullgraph_pilot_0716/dblp/seed0/model/mixed_edit_diffusion.pt" \
  results/latent_alignment_gate_0719_v2/dblp/latent_alignment.pt
launch_pair "${GPU_LIST[6]}" "${GPU_LIST[7]}" syn200 \
  "${SYN200_DATA:-$OLD_ROOT/data/HGEN_syn_200_subgraphs/subgraph_000}" \
  "$OLD_ROOT/results/hgen_official_adaptive_0716/syn200/seed0/model/mixed_edit_diffusion.pt" \
  results/latent_alignment_gate_0719_v2/syn200/latent_alignment.pt

failed=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "[DONE] ${NAMES[$index]}" | tee -a "$LOG_ROOT/driver.log"
  else
    echo "[FAILED] ${NAMES[$index]}" | tee -a "$LOG_ROOT/driver.log"
    failed=1
  fi
done

OUT_ROOT="$OUT_ROOT" STEPS="$STEPS" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
rows = []
for name in ("imdb", "pubmed", "dblp", "syn200"):
    base_path = root / name / "base" / "gate_report.json"
    fused_path = root / name / "fused" / "gate_report.json"
    if not base_path.exists() or not fused_path.exists():
        rows.append(
            {
                "dataset": name,
                "status": "unavailable",
                "reason": "paired sampling report missing; inspect dataset logs",
            }
        )
        continue
    base = json.loads(base_path.read_text())
    fused = json.loads(fused_path.read_text())
    row = {
        "dataset": name,
        "initial": base["initial"],
        "real": base["real"],
        "base": base["final"],
        "fused": fused["final"],
        "base_actions": base["action_counts"],
        "fused_actions": fused["action_counts"],
        "base_transition": base["trace"][-1]["role_phase_transition_step"],
        "fused_transition": fused["trace"][-1]["role_phase_transition_step"],
    }
    rows.append(row)
summary = {
    "format": "dihug-edge-energy-short-sampling-pair-v1",
    "steps": int(os.environ["STEPS"]),
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
for row in rows:
    print("[EDGE-ENERGY-SAMPLING] " + json.dumps(row, sort_keys=True))
PY

exit "$failed"

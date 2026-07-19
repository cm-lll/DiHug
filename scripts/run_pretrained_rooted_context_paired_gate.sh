#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_DIR="${DATASET_DIR:-data/IMDB_movie_subgraphs/subgraph_000}"
LEGACY_CHECKPOINT="${LEGACY_CHECKPOINT:-/data2/lyh/gnn_project/DiHug/results/imdb_exchangeable_compact_s0_3/model/mixed_edit_diffusion.pt}"
ALIGNMENT_CHECKPOINT="${ALIGNMENT_CHECKPOINT:-results/latent_alignment_gate_0719_v2/imdb/latent_alignment.pt}"
OUT_ROOT="${OUT_ROOT:-results/pretrained_rooted_context_gate_0719/imdb}"
LOG_ROOT="${LOG_ROOT:-logs/pretrained_rooted_context_gate_0719/imdb}"
GPU_LIST=(${GPU_LIST:-0 1 2})
SEED="${SEED:-71}"
TRAJECTORIES="${TRAJECTORIES:-4}"
EXAMPLES="${EXAMPLES:-12}"
EPOCHS="${EPOCHS:-10}"
CANDIDATES="${CANDIDATES:-16}"

if [[ ${#GPU_LIST[@]} -lt 3 ]]; then
  echo "GPU_LIST must contain at least three GPU ids" >&2
  exit 2
fi
for path in "$LEGACY_CHECKPOINT" "$ALIGNMENT_CHECKPOINT"; do
  [[ -f "$path" ]] || { echo "missing artifact: $path" >&2; exit 2; }
done
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

common=(
  --dataset-dir "$DATASET_DIR"
  --legacy-checkpoint "$LEGACY_CHECKPOINT"
  --device cuda
  --seed "$SEED"
  --trajectories "$TRAJECTORIES"
  --examples-per-trajectory "$EXAMPLES"
  --switches-per-example 64
  --candidates "$CANDIDATES"
  --epochs "$EPOCHS"
  --batch-size 2
)

CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" python -u scripts/run_rooted_context_residual_gate.py \
  "${common[@]}" \
  --output-dir "$OUT_ROOT/a_legacy_random" \
  > "$LOG_ROOT/a_legacy_random.log" 2>&1 &
pid_a=$!

CUDA_VISIBLE_DEVICES="${GPU_LIST[1]}" python -u scripts/run_rooted_context_residual_gate.py \
  "${common[@]}" \
  --alignment-checkpoint "$ALIGNMENT_CHECKPOINT" \
  --freeze-context-encoder \
  --output-dir "$OUT_ROOT/b_pretrained_legacy" \
  > "$LOG_ROOT/b_pretrained_legacy.log" 2>&1 &
pid_b=$!

CUDA_VISIBLE_DEVICES="${GPU_LIST[2]}" python -u scripts/run_rooted_context_residual_gate.py \
  "${common[@]}" \
  --alignment-checkpoint "$ALIGNMENT_CHECKPOINT" \
  --freeze-context-encoder \
  --state-protocol closed_form_marginal \
  --output-dir "$OUT_ROOT/c_pretrained_closed_form" \
  > "$LOG_ROOT/c_pretrained_closed_form.log" 2>&1 &
pid_c=$!

status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
wait "$pid_c" || status=1
if [[ $status -ne 0 ]]; then
  echo "at least one paired context gate failed; inspect $LOG_ROOT" >&2
  exit "$status"
fi

OUT_ROOT="$OUT_ROOT" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
rows = []
for name in ("a_legacy_random", "b_pretrained_legacy", "c_pretrained_closed_form"):
    report = json.loads((root / name / "gate_report.json").read_text())
    rows.append(
        {
            "configuration": name,
            "base_selected_utility": report["base_selected_utility"],
            "context_selected_utility": report["context_selected_utility"],
            "utility_gain": report["utility_gain"],
            "residual_scale": report["residual_scale"],
            "state_protocol": report["state_protocol"],
            "context_encoder_frozen": report["context_encoder_frozen"],
        }
    )
summary = {"format": "dihug-pretrained-rooted-context-paired-gate-v1", "rows": rows}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
for row in rows:
    print("[PRETRAINED-CONTEXT-PAIR] " + json.dumps(row, sort_keys=True))
PY

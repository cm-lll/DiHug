#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT_ROOT="${OUT_ROOT:-results/imdb_exchangeable_role_switch_8seed}"
LOG_ROOT="${LOG_ROOT:-logs/imdb_exchangeable_role_switch_8seed}"
SEED_OFFSET="${SEED_OFFSET:-0}"
STEPS="${STEPS:-20000}"
CANDIDATES="${CANDIDATES:-64}"
STARTUP_STAGGER_SECONDS="${STARTUP_STAGGER_SECONDS:-5}"
CHECKPOINT="${CHECKPOINT:-results/imdb_mixed_edit_gates/model/bdrs_s0_c16/mixed_edit_diffusion.pt}"
PRIOR="$OUT_ROOT/exchangeable_role_prior.json"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

if [[ ! -f "$PRIOR" ]]; then
  python scripts/run_exchangeable_role_prior_gate.py \
    --mode fit-gate \
    --dataset-dir data/IMDB_movie_subgraphs/subgraph_000 \
    --output-dir "$OUT_ROOT/prior_gate" \
    --components 64 --role-model categorical --prior-samples 16 --seed 0 \
    > "$LOG_ROOT/prior_gate.log" 2>&1
  cp "$OUT_ROOT/prior_gate/exchangeable_role_prior.json" "$PRIOR"
fi

pids=()
for gpu in $(seq 0 7); do
  seed=$((SEED_OFFSET + gpu))
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_exchangeable_role_prior_gate.py \
    --mode sample --output-dir "$OUT_ROOT/seed_${seed}" \
    --prior "$PRIOR" --checkpoint "$CHECKPOINT" \
    --device cuda --seed "$seed" --sample-steps "$STEPS" \
    --sample-candidates "$CANDIDATES" --sample-actions switch \
    --allow-kernel-override \
    --metrics-every 1000 > "$LOG_ROOT/seed_${seed}.log" 2>&1 &
  pids+=("$!")
  sleep "$STARTUP_STAGGER_SECONDS"
done
for pid in "${pids[@]}"; do wait "$pid"; done

python - "$OUT_ROOT" "$SEED_OFFSET" > "$OUT_ROOT/summary.json" <<'PY'
import json, pathlib, sys
root, offset = pathlib.Path(sys.argv[1]), int(sys.argv[2])
rows = []
for seed in range(offset, offset + 8):
    report = json.loads((root / f"seed_{seed}" / "sampling_report.json").read_text())
    rows.append({"seed": seed, **report["final"]})
keys = tuple(key for key in rows[0] if key != "seed")
summary = {
    "rows": rows,
    "mean": {key: sum(row[key] for row in rows) / len(rows) for key in keys},
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
echo "[ROLE-SWITCH-8GPU] complete: $OUT_ROOT/summary.json"

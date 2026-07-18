#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/lyh/gnn_project/DiHug}"
CKPT="${CKPT:-$ROOT/results/imdb_relocation_gate_s0/relocation_diffusion.pt}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/imdb_relocation_temperature_gate_8gpu}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imdb_relocation_temperature_gate_8gpu}"
STEPS="${STEPS:-10000}"

source /data2/lyh/miniconda3/etc/profile.d/conda.sh
conda activate sparse_block
cd "$ROOT"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

names=(temp0 temp001 temp003 temp010 temp025 temp050 temp100 temp200)
temperatures=(0 0.01 0.03 0.10 0.25 0.50 1.00 2.00)
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_relocation_diffusion.py \
    --output-dir "$OUT_ROOT/${names[$gpu]}" \
    --seed 0 --device cuda \
    --test-only "$CKPT" \
    --sample-steps "$STEPS" \
    --sample-candidates 64 \
    --sample-temperature "${temperatures[$gpu]}" \
    --metrics-every 1000 \
    > "$LOG_ROOT/${names[$gpu]}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if (( status != 0 )); then exit "$status"; fi

python - "$OUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
targets = {
    "triangles": 655.0,
    "clustering": 0.4689355333166192,
    "sum_degree_sq": 442168.0,
    "max_degree": 497.0,
    "lcc": 1653.0,
}
rows = []
for path in root.glob("*/sampling_report.json"):
    payload = json.loads(path.read_text(encoding="utf-8"))
    final = payload["final_metrics"]
    errors = {
        key: abs(float(final[key]) - value) / max(abs(value), 1.0e-12)
        for key, value in targets.items()
    }
    rows.append(
        {
            "name": path.parent.name,
            "temperature": payload["sampling_config"]["temperature"],
            "final": final,
            "errors": errors,
            "max_error": max(errors.values()),
        }
    )
rows.sort(key=lambda row: (row["max_error"], row["temperature"]))
(root / "temperature_gate_summary.json").write_text(
    json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
)
print("name\ttemp\ttri\tC\tsumdeg2\tmaxdeg\tLCC\tmax_err")
for row in rows:
    final = row["final"]
    print(
        f'{row["name"]}\t{row["temperature"]:.2f}\t'
        f'{final["triangles"]:.0f}\t{final["clustering"]:.4f}\t'
        f'{final["sum_degree_sq"]:.0f}\t{final["max_degree"]:.0f}\t'
        f'{final["lcc"]:.0f}\t{row["max_error"]:.4f}'
    )
PY

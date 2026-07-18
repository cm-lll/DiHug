#!/usr/bin/env bash
# PubMed HGEN 40% comparison with target-fingerprint protection.
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

GPU="${GPU:-0}"
OUT_ROOT="${OUT_ROOT:-results/hgen40_matched_pubmed}"
LOG_ROOT="${LOG_ROOT:-logs/hgen40_matched_pubmed}"
HGEN_ROOT="${HGEN_ROOT:-/data2/lyh/gnn_project/HGEN}"
HGEN_PYTHON="${HGEN_PYTHON:-/data2/lyh/miniconda3/envs/hgen_env/bin/python}"
DATA="${PUBMED_DATA:-data/PubMed_baseline_subgraphs/subgraph_000}"
# These are the artifacts used by the nested validation protocol.  Do not
# substitute similarly named role-corruption checkpoints: their prior produces
# a different initial-state distribution and invalidates the selected C64/320
# contract.
PRIOR="${PUBMED_PRIOR:-results/pubmed_exchangeable_compact_switch_gate_0716/prior_c512/exchangeable_role_prior.json}"
MODEL="${PUBMED_MODEL:-results/pubmed_exchangeable_compact_switch_gate_0716/model/mixed_edit_diffusion.pt}"
EXPECTED_PRIOR_SHA256="${EXPECTED_PRIOR_SHA256:-dd22f752210c00ad4960e05369ca2b387848a90874d4ac1c05eb04d23f5bc319}"
EXPECTED_MODEL_SHA256="${EXPECTED_MODEL_SHA256:-727aa1c0e25d1276ba9c8a7bb1a135f40f728788c0585448e24de79405d23330}"
SPLIT_SEED="${SPLIT_SEED:-481516234}"
SAMPLE_STEPS="${SAMPLE_STEPS:-320}"
SAMPLE_CANDIDATES="${SAMPLE_CANDIDATES:-64}"
HGEN_EPOCHS="${HGEN_EPOCHS:-5}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
for required in "$DATA" "$PRIOR" "$MODEL" "$HGEN_ROOT/hgen.py" "$HGEN_PYTHON"; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 2; }
done

actual_prior_sha256="$(sha256sum "$PRIOR" | awk '{print $1}')"
actual_model_sha256="$(sha256sum "$MODEL" | awk '{print $1}')"
[[ "$actual_prior_sha256" == "$EXPECTED_PRIOR_SHA256" ]] || {
  echo "PubMed prior SHA256 mismatch: $actual_prior_sha256" >&2
  echo "expected nested-selected artifact: $EXPECTED_PRIOR_SHA256" >&2
  exit 3
}
[[ "$actual_model_sha256" == "$EXPECTED_MODEL_SHA256" ]] || {
  echo "PubMed model SHA256 mismatch: $actual_model_sha256" >&2
  echo "expected nested-selected artifact: $EXPECTED_MODEL_SHA256" >&2
  exit 3
}

python scripts/build_hgen_edge_split_target.py --dataset-dir "$DATA" \
  --generate-split --output-dir "$OUT_ROOT/target" --test-share 0.4 \
  --split-seed "$SPLIT_SEED" > "$LOG_ROOT/build_target.log" 2>&1
edge_count="$(python -c "import json; print(json.load(open('$OUT_ROOT/target/split_report.json'))['generation_budget']['num_edges'])")"

for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/run_exchangeable_role_prior_gate.py \
    --mode sample --output-dir "$OUT_ROOT/dihug_seed${seed}" \
    --prior "$PRIOR" --checkpoint "$MODEL" --device cuda --seed "$seed" \
    --edge-view-count "$edge_count" --edge-view-source hgen_split \
    --sample-steps "$SAMPLE_STEPS" --sample-candidates "$SAMPLE_CANDIDATES" \
    --sample-actions relocation_switch --motif-refinement-stage \
    --metrics-every "$SAMPLE_STEPS" > "$LOG_ROOT/dihug_seed${seed}.log" 2>&1
done

(
  cd "$HGEN_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" "$HGEN_PYTHON" -u hgen.py \
    --dataset PubMed --n-epochs "$HGEN_EPOCHS"
) > "$LOG_ROOT/hgen.log" 2>&1
python scripts/parse_hgen_baseline_log.py --log "$LOG_ROOT/hgen.log" \
  --dataset pubmed --output "$OUT_ROOT/hgen_baseline.json"

generated=()
for seed in {0..7}; do generated+=("$OUT_ROOT/dihug_seed${seed}/generated_graph.json"); done
"$HGEN_PYTHON" scripts/evaluate_hgen_edge_split_metrics.py \
  --target "$OUT_ROOT/target/target_graph.json" --generated "${generated[@]}" \
  --output "$OUT_ROOT/dihug_metrics.json" > "$LOG_ROOT/dihug_metrics.log" 2>&1

python - "$OUT_ROOT" "$SPLIT_SEED" "$PRIOR" "$actual_prior_sha256" \
  "$MODEL" "$actual_model_sha256" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
split = json.loads((root / "target/split_report.json").read_text())
hgen = json.loads((root / "hgen_baseline.json").read_text())
dihug = json.loads((root / "dihug_metrics.json").read_text())
target = split["target_metrics"]
def close(a, b, tol=1e-5): return abs(float(a)-float(b)) <= tol*max(abs(float(a)),1.0)
checks = {key: close(target[key], hgen["real"][key]) for key in ("triangles", "clustering", "lcc")}
payload = {
    "format": "dihug-hgen40-matched-pubmed-v1",
    "dataset": "pubmed", "split_seed": int(sys.argv[2]),
    "target_fingerprint": target, "hgen_real_fingerprint": hgen["real"],
    "fingerprint_checks": checks, "directly_matched": all(checks.values()),
    "paper_comparison_rule": "direct-local-split" if all(checks.values()) else "separate-local-protocol",
    "selection_contract": "validation-selected relocation_switch C64 320 steps",
    "artifact_contract": {
        "prior": str(pathlib.Path(sys.argv[3])),
        "prior_sha256": sys.argv[4],
        "model": str(pathlib.Path(sys.argv[5])),
        "model_sha256": sys.argv[6],
    },
    "dihug": dihug, "hgen": hgen,
    "hgen_motif": {"available_in_local_repository": False},
}
(root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "[HGEN40-PUBMED] complete: $OUT_ROOT/summary.json"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-results/role_prior_strength_ablation_0716}"
LOG_ROOT="${LOG_ROOT:-logs/role_prior_strength_ablation_0716}"
IMDB_DATA="${IMDB_DATA:-data/IMDB_movie_subgraphs/subgraph_000}"
PUBMED_DATA="${PUBMED_DATA:-data/PubMed_baseline_subgraphs/subgraph_000}"
IMDB_CKPT="${IMDB_CKPT:-results/imdb_exchangeable_compact_s0_3/model/mixed_edit_diffusion.pt}"
PUBMED_CKPT="${PUBMED_CKPT:-results/pubmed_exchangeable_compact_switch_gate_0716/model/mixed_edit_diffusion.pt}"
MIXING_STEPS_PER_EDGE="${MIXING_STEPS_PER_EDGE:-20}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
for required in "$IMDB_DATA" "$PUBMED_DATA" "$IMDB_CKPT" "$PUBMED_CKPT"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
done

run_strength() {
  local gpu="$1" dataset_name="$2" dataset="$3" components="$4"
  local rho="$5" steps="$6" checkpoint="$7"
  local rho_tag="${rho//./p}"
  local name="${dataset_name}_rho${rho_tag}"
  local out="$OUT_ROOT/strength/$name"
  local log="$LOG_ROOT/$name.log"
  mkdir -p "$out"

  echo "[START] gpu=$gpu dataset=$dataset_name rho=$rho steps=$steps" | tee -a "$LOG_ROOT/launcher.log"
  "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
    --mode fit-gate --dataset-dir "$dataset" --output-dir "$out/prior" \
    --components "$components" --role-model categorical \
    --role-component-sampling systematic --role-correlation-strength "$rho" \
    --randomization-steps-per-edge "$MIXING_STEPS_PER_EDGE" \
    --prior-samples 2 --seed 0 \
    > "$log" 2>&1

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
    --mode sample --output-dir "$out/sample" \
    --prior "$out/prior/exchangeable_role_prior.json" \
    --checkpoint "$checkpoint" --device cuda --seed 0 \
    --sample-steps "$steps" --sample-candidates 64 \
    --sample-actions relocation_switch --motif-refinement-stage \
    --metrics-every "$((steps / 10))" \
    >> "$log" 2>&1
  echo "[DONE] gpu=$gpu dataset=$dataset_name rho=$rho" | tee -a "$LOG_ROOT/launcher.log"
}

# Paired one-seed screen. Every setting uses the same learned denoiser, sampling
# seed, candidate budget, and Havel-Hakimi mixing budget; only rho changes.
pids=()
run_strength 0 pubmed "$PUBMED_DATA" 512 1.00 5000 "$PUBMED_CKPT" & pids+=("$!")
sleep 10
run_strength 1 pubmed "$PUBMED_DATA" 512 0.75 5000 "$PUBMED_CKPT" & pids+=("$!")
sleep 10
run_strength 2 pubmed "$PUBMED_DATA" 512 0.50 5000 "$PUBMED_CKPT" & pids+=("$!")
sleep 10
run_strength 3 pubmed "$PUBMED_DATA" 512 0.25 5000 "$PUBMED_CKPT" & pids+=("$!")
sleep 10
run_strength 4 pubmed "$PUBMED_DATA" 512 0.00 5000 "$PUBMED_CKPT" & pids+=("$!")
sleep 10
run_strength 5 imdb "$IMDB_DATA" 64 1.00 20000 "$IMDB_CKPT" & pids+=("$!")
sleep 10
run_strength 6 imdb "$IMDB_DATA" 64 0.50 20000 "$IMDB_CKPT" & pids+=("$!")
sleep 10
run_strength 7 imdb "$IMDB_DATA" 64 0.00 20000 "$IMDB_CKPT" & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[[ "$failed" -eq 0 ]] || { echo "role-strength screen failed" >&2; exit 1; }

# Construction-bias control: rho remains one while only the number of uniform
# legal 2-switches after Havel-Hakimi changes. This stage never uses a GPU or a
# denoiser and therefore isolates the initializer itself.
pids=()
for dataset_name in imdb pubmed; do
  if [[ "$dataset_name" == "imdb" ]]; then
    dataset="$IMDB_DATA"; components=64
  else
    dataset="$PUBMED_DATA"; components=512
  fi
  for mixing in 2 10 20 50; do
    out="$OUT_ROOT/mixing/${dataset_name}_${mixing}E"
    log="$LOG_ROOT/mixing_${dataset_name}_${mixing}E.log"
    mkdir -p "$out"
    (
      "$PYTHON" scripts/run_exchangeable_role_prior_gate.py \
        --mode fit-gate --dataset-dir "$dataset" --output-dir "$out" \
        --components "$components" --role-model categorical \
        --role-component-sampling systematic --role-correlation-strength 1.0 \
        --randomization-steps-per-edge "$mixing" --prior-samples 8 --seed 100 \
        > "$log" 2>&1
    ) &
    pids+=("$!")
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PYTHON" - "$OUT_ROOT" "$IMDB_DATA" "$PUBMED_DATA" <<'PY'
import json
import pathlib
import sys

from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics

root = pathlib.Path(sys.argv[1])
datasets = {
    "imdb": pathlib.Path(sys.argv[2]),
    "pubmed": pathlib.Path(sys.argv[3]),
}
real = {name: graph_metrics(load_typed_graph(path, active_only=True)) for name, path in datasets.items()}

strength_rows = []
for prior_report_path in sorted((root / "strength").glob("*/prior/prior_gate_report.json")):
    run_dir = prior_report_path.parents[1]
    sample_path = run_dir / "sample" / "sampling_report.json"
    prior_report = json.loads(prior_report_path.read_text())
    sample = json.loads(sample_path.read_text())
    dataset_name = run_dir.name.split("_rho", 1)[0]
    prior_payload = json.loads((run_dir / "prior" / "exchangeable_role_prior.json").read_text())
    target = real[dataset_name]
    initial = sample["initial"]
    final = sample["final"]
    strength_rows.append({
        "dataset": dataset_name,
        "rho": prior_payload["role_correlation_strength"],
        "randomization_steps_per_edge": prior_payload["randomization_steps_per_edge"],
        "initial": initial,
        "final": final,
        "triangle_gap_recovery": (final["triangles"] - initial["triangles"]) / max(target["triangles"] - initial["triangles"], 1.0),
        "clustering_gap_recovery": (final["clustering"] - initial["clustering"]) / max(target["clustering"] - initial["clustering"], 1.0e-12),
        "triangle_relative_error": abs(final["triangles"] - target["triangles"]) / max(target["triangles"], 1.0),
        "clustering_relative_error": abs(final["clustering"] - target["clustering"]) / max(target["clustering"], 1.0e-12),
        "prior_total_degree_tv": prior_report["role_prior"]["total_degree_tv_mean"],
        "prior_family_degree_tv": prior_report["role_prior"]["family_degree_tv_mean"],
        "prior_relation_degree_correlation_error": prior_report["role_prior"]["relation_degree_correlation_error_mean"],
    })

mixing_rows = []
for report_path in sorted((root / "mixing").glob("*/prior_gate_report.json")):
    run_dir = report_path.parent
    dataset_name, mixing_tag = run_dir.name.rsplit("_", 1)
    report = json.loads(report_path.read_text())
    mixing_rows.append({
        "dataset": dataset_name,
        "randomization_steps_per_edge": float(mixing_tag[:-1]),
        "triangles_mean": report["role_prior"]["metrics_mean"]["triangles"],
        "clustering_mean": report["role_prior"]["metrics_mean"]["clustering"],
        "transitivity_mean": report["role_prior"]["metrics_mean"]["transitivity"],
        "total_degree_tv": report["role_prior"]["total_degree_tv_mean"],
        "family_degree_tv": report["role_prior"]["family_degree_tv_mean"],
        "relation_degree_correlation_error": report["role_prior"]["relation_degree_correlation_error_mean"],
    })

summary = {
    "format": "dihug-role-prior-strength-ablation-v1",
    "inference_uses_clean_graph": False,
    "real_metrics_used_for_posthoc_evaluation_only": True,
    "real": real,
    "strength_rows": sorted(strength_rows, key=lambda row: (row["dataset"], -row["rho"])),
    "mixing_rows": sorted(mixing_rows, key=lambda row: (row["dataset"], row["randomization_steps_per_edge"])),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[ROLE-PRIOR-STRENGTH] complete: $OUT_ROOT/summary.json"

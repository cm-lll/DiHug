#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source /data2/lyh/miniconda3/etc/profile.d/conda.sh
conda activate sparse_block

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# ── shared paths ──
IMDB_STRUCTURE="results/imdb_standard_switch_diffusion_gpu7/structure_checkpoint.json"
IMDB_MODEL="results/imdb_standard_switch_diffusion_gpu7/guided_train/switch_diffusion.pt"
IMDB_DATASET="data/IMDB_movie_subgraphs/subgraph_000"
STAGE_A_DIR="${STAGE_A_DIR:-results/imdb_step_ablation_macro_v2_8gpu}"
STAGE_B_DIR="${STAGE_B_DIR:-results/imdb_10graph_macro_v2}"
STAGE_C_DIR="${STAGE_C_DIR:-results/pubmed_standard_diffusion_macro_v2}"
LOG_DIR="${LOG_DIR:-logs/imdb_pubmed_3stage_macro_v2}"
mkdir -p "$STAGE_A_DIR" "$STAGE_B_DIR" "$STAGE_C_DIR" "$LOG_DIR"

# ═══════════════════════════════════════════════════════
# Stage A: diffusion-step ablation (8 GPUs, 8 configs)
# ═══════════════════════════════════════════════════════
run_stage_a() {
  echo "[STAGE-A] beginning diffusion-step ablation on 8 GPUs"
  local configs=(
    "10 2000"
    "20 1000"
    "50 400"
    "100 200"
    "200 100"
    "500 40"
    "1000 20"
    "2000 10"
  )

  for gpu in 0 1 2 3 4 5 6 7; do
    local T_val K_val
    read -r T_val K_val <<< "${configs[$gpu]}"
    local name="ablation_T${T_val}_K${K_val}"
    echo "[STAGE-A] GPU=$gpu $name"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_switch_diffusion.py \
      --dataset-dir "$IMDB_DATASET" \
      --structure-checkpoint "$IMDB_STRUCTURE" \
      --output-dir "$STAGE_A_DIR/$name" \
      --seed 0 --device cuda \
      --test-only "$IMDB_MODEL" \
      --diffusion-steps "$T_val" \
      --switches-per-diffusion-step "$K_val" \
      --randomization-steps 20000 \
      --sample-candidates 64 \
      --sample-temperature 1.0 \
      --sampling-mode model \
      --switch-kernel tensor \
      --metrics-every 2000 \
      > "$LOG_DIR/stage_a_${name}.log" 2>&1 &
  done
  wait

  # Select best (T,K) by max(tri_err, clust_err)
  python3 - "$STAGE_A_DIR" <<'PY'
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1])
best_score = float("inf")
best_name = None
best_T = best_K = None
for report_path in sorted(root.glob("ablation_T*/sampling_report.json")):
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    name = report_path.parent.name
    tri = float(payload["final_metrics"]["triangles"])
    clust = float(payload["final_metrics"]["clustering"])
    tri_err = abs(tri - 655.0) / 655.0
    clust_err = abs(clust - 0.4689355333) / 0.4689355333
    score = max(tri_err, clust_err)
    T_val = name.split("_T")[1].split("_")[0]
    K_val = name.split("_K")[1]
    config = payload["sampling_config"]
    expected = {
        "diffusion_steps": int(T_val),
        "switches_per_diffusion_step": int(K_val),
        "requested_steps": 20_000,
        "candidates_per_step": 64,
    }
    actual = {key: int(config[key]) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"macro diffusion contract failed for {name}: "
            f"expected={expected} actual={actual}"
        )
    print(f"{name}: T={T_val} K={K_val} tri={tri:.0f} clust={clust:.4f} tri_err={tri_err:.4f} clust_err={clust_err:.4f} score={score:.4f}")
    is_better = score < best_score - 1.0e-12
    is_canonical_tie = (
        abs(score - best_score) <= 1.0e-12
        and (
            best_T is None
            or abs(math.log2(int(T_val) / 100.0))
            < abs(math.log2(int(best_T) / 100.0))
        )
    )
    if is_better or is_canonical_tie:
        best_score = score
        best_name = name
        best_T = T_val
        best_K = K_val
with open(pathlib.Path(root) / "best_config.txt", "w") as fh:
    fh.write(f"T={best_T}\nK={best_K}\nname={best_name}\nscore={best_score}\n")
print(f"\nBEST: T={best_T} K={best_K} score={best_score:.4f}")
PY
  echo "[STAGE-A] done. best config written to $STAGE_A_DIR/best_config.txt"
}


# ═══════════════════════════════════════════════════════
# Stage B: 10-graph official results
# ═══════════════════════════════════════════════════════
run_stage_b() {
  local best_T best_K
  best_T=$(grep "^T=" "$STAGE_A_DIR/best_config.txt" | cut -d= -f2)
  best_K=$(grep "^K=" "$STAGE_A_DIR/best_config.txt" | cut -d= -f2)
  echo "[STAGE-B] using best (T=$best_T, K=$best_K) from Stage A"

  # First round: seeds 0-7 on GPU 0-7
  echo "[STAGE-B] round 1: seeds 0-7"
  for seed in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$seed" python scripts/run_switch_diffusion.py \
      --dataset-dir "$IMDB_DATASET" \
      --structure-checkpoint "$IMDB_STRUCTURE" \
      --output-dir "$STAGE_B_DIR/seed${seed}" \
      --seed "$seed" --device cuda \
      --test-only "$IMDB_MODEL" \
      --diffusion-steps "$best_T" \
      --switches-per-diffusion-step "$best_K" \
      --randomization-steps 20000 \
      --sample-candidates 64 \
      --sample-temperature 1.0 \
      --sampling-mode model \
      --switch-kernel tensor \
      --metrics-every 2000 \
      > "$LOG_DIR/stage_b_seed${seed}.log" 2>&1 &
  done
  wait

  # Second round: seeds 8-9 on GPU 0-1
  echo "[STAGE-B] round 2: seeds 8-9"
  for seed in 8 9; do
    local gpu=$((seed - 8))
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_switch_diffusion.py \
      --dataset-dir "$IMDB_DATASET" \
      --structure-checkpoint "$IMDB_STRUCTURE" \
      --output-dir "$STAGE_B_DIR/seed${seed}" \
      --seed "$seed" --device cuda \
      --test-only "$IMDB_MODEL" \
      --diffusion-steps "$best_T" \
      --switches-per-diffusion-step "$best_K" \
      --randomization-steps 20000 \
      --sample-candidates 64 \
      --sample-temperature 1.0 \
      --sampling-mode model \
      --switch-kernel tensor \
      --metrics-every 2000 \
      > "$LOG_DIR/stage_b_seed${seed}.log" 2>&1 &
  done
  wait

  # Aggregate 10-seed summary
  python3 - "$STAGE_B_DIR" "$STAGE_A_DIR/best_config.txt" "$IMDB_DATASET" <<'PY'
import json, pathlib, sys, statistics, math
from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics

root = pathlib.Path(sys.argv[1])
real = graph_metrics(load_typed_graph(sys.argv[3], active_only=True))
best = dict(
    line.strip().split("=", 1)
    for line in pathlib.Path(sys.argv[2]).read_text().splitlines()
    if "=" in line
)
expected_config = {
    "diffusion_steps": int(best["T"]),
    "switches_per_diffusion_step": int(best["K"]),
    "requested_steps": 20_000,
    "candidates_per_step": 64,
}
triangles = []; clustering = []; transitivity = []; assort = []; lcc = []
for report_path in sorted(root.glob("seed*/sampling_report.json")):
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    config = payload["sampling_config"]
    actual_config = {key: int(config[key]) for key in expected_config}
    if actual_config != expected_config:
        raise RuntimeError(
            f"Stage B diffusion contract failed: {report_path}: "
            f"expected={expected_config} actual={actual_config}"
        )
    contract = payload["inference_contract"]
    if contract.get("clean_graphlet_target") or not contract.get("learned_graphlet_condition"):
        raise RuntimeError(f"Stage B inference contract failed: {report_path}: {contract}")
    m = payload["final_metrics"]
    triangles.append(m["triangles"])
    clustering.append(m["clustering"])
    transitivity.append(m["transitivity"])
    assort.append(m["assortativity"])
    lcc.append(m["lcc"])
def stats(vals, target):
    vals = [float(v) for v in vals]
    return {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals)>1 else 0.0,
            "min": min(vals), "max": max(vals),
            "rel_err": abs(statistics.mean(vals)-target)/max(abs(target), 1.0e-12)}
report = {
    "triangles": stats(triangles, real["triangles"]),
    "clustering": stats(clustering, real["clustering"]),
    "transitivity": stats(transitivity, real["transitivity"]),
    "assortativity": stats(assort, real["assortativity"]),
    "lcc": stats(lcc, real["lcc"]),
    "n": len(triangles)
}
with open(root / "aggregate_10graph.json", "w") as fh:
    json.dump(report, fh, indent=2)
for k, v in report.items():
    if isinstance(v, dict):
        print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}  err={v['rel_err']:.4%}  [{v['min']:.1f}, {v['max']:.1f}]")
PY
  echo "[STAGE-B] 10-graph results aggregated to $STAGE_B_DIR/aggregate_10graph.json"
}


# ═══════════════════════════════════════════════════════
# Stage C: PubMed standard diffusion
# ═══════════════════════════════════════════════════════
run_stage_c() {
  local PUBMED_DATASET="data/PubMed_baseline_subgraphs/subgraph_000"
  local PUBMED_CKPT="${PUBMED_STRUCTURE_CKPT:-results/pubmed_switch_structure_seed0/structure_checkpoint.json}"
  local PUBMED_NPZ="$STAGE_C_DIR/data_build/switch_dataset.npz"
  local T_val=100 K_val=200

  if [[ ! -s "$PUBMED_CKPT" ]]; then
    echo "[STAGE-C] building canonical PubMed structure checkpoint"
    mkdir -p "$(dirname "$PUBMED_CKPT")"
    python scripts/build_switch_structure_checkpoint.py \
      --dataset-dir "$PUBMED_DATASET" \
      --output "$PUBMED_CKPT" \
      --seed 0 --null-chains 4 --null-steps 5000 --null-thin 500 \
      --max-features 64 \
      --feature-prefix triangle/ \
      --feature-prefix root_triangle/ \
      --feature-prefix root_triangle_total/ \
      --reserve-prefix triangle/=-1 \
      --reserve-prefix root_triangle_total/=-1 \
      > "$LOG_DIR/stage_c_structure.log" 2>&1
  fi

  # Build canonical switch dataset (single-process)
  if [[ ! -s "$PUBMED_NPZ" ]]; then
    echo "[STAGE-C] building canonical switch dataset (single-process)"
    mkdir -p "$STAGE_C_DIR/data_build"
    python scripts/run_switch_diffusion.py \
      --dataset-dir "$PUBMED_DATASET" \
      --structure-checkpoint "$PUBMED_CKPT" \
      --output-dir "$STAGE_C_DIR/data_build" \
      --seed 0 --device cpu \
      --conditioning-mode state_only \
      --diffusion-steps "$T_val" \
      --switches-per-diffusion-step "$K_val" \
      --trajectories 4 --record-every 50 \
      --train-candidates 32 --build-dataset-only \
      > "$LOG_DIR/stage_c_data.log" 2>&1
  fi

  # Train state_only + utility_kl + learned_condition
  echo "[STAGE-C] training PubMed standard diffusion"
  CUDA_VISIBLE_DEVICES=0 python scripts/run_switch_diffusion.py \
    --dataset-dir "$PUBMED_DATASET" \
    --structure-checkpoint "$PUBMED_CKPT" \
    --dataset-cache "$PUBMED_NPZ" \
    --output-dir "$STAGE_C_DIR/train" \
    --seed 0 --device cuda \
    --conditioning-mode state_only --training-objective utility_kl \
    --utility-temperature 1 --learned-condition --learned-condition-ridge 1e-6 \
    --diffusion-steps "$T_val" \
    --switches-per-diffusion-step "$K_val" \
    --epochs 1 --batch-size 64 \
    --hidden-dim 256 --layers 3 --dropout 0 \
    --train-only \
    > "$LOG_DIR/stage_c_train.log" 2>&1

  # Generate 10 graphs with T=100, K=200
  echo "[STAGE-C] generating 10 PubMed graphs (T=$T_val, K=$K_val)"

  for seed in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$seed" python scripts/run_switch_diffusion.py \
      --structure-checkpoint "$PUBMED_CKPT" \
      --output-dir "$STAGE_C_DIR/seed${seed}" \
      --seed "$seed" --device cuda \
      --test-only "$STAGE_C_DIR/train/switch_diffusion.pt" \
      --diffusion-steps "$T_val" \
      --switches-per-diffusion-step "$K_val" \
      --randomization-steps 20000 \
      --sample-candidates 64 \
      --sample-temperature 0 \
      --sampling-mode model \
      --switch-kernel tensor \
      --learned-condition-noop \
      --learned-condition-noop-patience 1000 \
      --metrics-every 2000 \
      > "$LOG_DIR/stage_c_seed${seed}.log" 2>&1 &
  done
  wait

  for seed in 8 9; do
    local gpu=$((seed - 8))
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_switch_diffusion.py \
      --structure-checkpoint "$PUBMED_CKPT" \
      --output-dir "$STAGE_C_DIR/seed${seed}" \
      --seed "$seed" --device cuda \
      --test-only "$STAGE_C_DIR/train/switch_diffusion.pt" \
      --diffusion-steps "$T_val" \
      --switches-per-diffusion-step "$K_val" \
      --randomization-steps 20000 \
      --sample-candidates 64 \
      --sample-temperature 0 \
      --sampling-mode model \
      --switch-kernel tensor \
      --learned-condition-noop \
      --learned-condition-noop-patience 1000 \
      --metrics-every 2000 \
      > "$LOG_DIR/stage_c_seed${seed}.log" 2>&1 &
  done
  wait

  # Aggregate PubMed 10-seed
  python3 - "$STAGE_C_DIR" "$PUBMED_DATASET" <<'PY'
import json, pathlib, sys, statistics
from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics

root = pathlib.Path(sys.argv[1])
real = graph_metrics(load_typed_graph(sys.argv[2], active_only=True))
triangles = []; clustering = []; transitivity = []; assortativity = []; lcc = []
for report_path in sorted(root.glob("seed*/sampling_report.json")):
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    config = payload["sampling_config"]
    expected = {
        "diffusion_steps": 100,
        "switches_per_diffusion_step": 200,
        "requested_steps": 20_000,
        "candidates_per_step": 64,
    }
    actual = {key: int(config[key]) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"PubMed macro diffusion contract failed: {report_path}: "
            f"expected={expected} actual={actual}"
        )
    if float(config["temperature"]) != 0.0:
        raise RuntimeError(
            f"PubMed sampling must use deterministic learned utility: {report_path}: "
            f"temperature={config['temperature']}"
        )
    if not bool(config["learned_condition_noop"]):
        raise RuntimeError(
            f"PubMed sampling must reject non-positive learned utility: {report_path}"
        )
    contract = payload["inference_contract"]
    if contract.get("clean_graphlet_target") or not contract.get("learned_graphlet_condition"):
        raise RuntimeError(
            f"PubMed inference contract failed: {report_path}: {contract}"
        )
    m = payload["final_metrics"]
    triangles.append(m["triangles"])
    clustering.append(m["clustering"])
    transitivity.append(m["transitivity"])
    assortativity.append(m["assortativity"])
    lcc.append(m["lcc"])
def stats(vals, target):
    vals = [float(v) for v in vals]
    return {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals)>1 else 0.0,
            "min": min(vals), "max": max(vals),
            "rel_err": abs(statistics.mean(vals)-target)/max(abs(target), 1.0e-12)}
report = {
    "triangles": stats(triangles, real["triangles"]),
    "clustering": stats(clustering, real["clustering"]),
    "transitivity": stats(transitivity, real["transitivity"]),
    "assortativity": stats(assortativity, real["assortativity"]),
    "lcc": stats(lcc, real["lcc"]),
    "n": len(triangles)
}
with open(root / "aggregate_10graph.json", "w") as fh:
    json.dump(report, fh, indent=2)
for k, v in report.items():
    if isinstance(v, dict):
        print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}  err={v['rel_err']:.4%}")
PY
  echo "[STAGE-C] PubMed 10-graph done: $STAGE_C_DIR/aggregate_10graph.json"
}


# ── main ──
echo "[3STAGE] starting at $(date)"
run_stage_a
run_stage_b
run_stage_c
echo "[3STAGE] all stages completed at $(date)"

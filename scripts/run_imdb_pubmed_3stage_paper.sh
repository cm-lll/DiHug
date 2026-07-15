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
STAGE_A_DIR="results/imdb_step_ablation_8gpu"
STAGE_B_DIR="results/imdb_10graph_official"
STAGE_C_DIR="results/pubmed_standard_diffusion"
LOG_DIR="logs/imdb_pubmed_3stage"
mkdir -p "$STAGE_A_DIR" "$STAGE_B_DIR" "$STAGE_C_DIR" "$LOG_DIR"

# ═══════════════════════════════════════════════════════
# Stage A: diffusion-step ablation (8 GPUs, 8 configs)
# ═══════════════════════════════════════════════════════
run_stage_a() {
  echo "[STAGE-A] beginning diffusion-step ablation on 8 GPUs"
  declare -A CONFIGS
  CONFIGS=(
    [0]="T=0,K=20000"
    [1]="T=100,K=20000"
    [2]="T=50,K=400"
    [3]="T=100,K=200"
    [4]="T=200,K=100"
    [5]="T=500,K=40"
    [6]="T=1000,K=20"
    [7]="T=2000,K=10"
  )

  for gpu in 0 1 2 3 4 5 6 7; do
    local spec="${CONFIGS[$gpu]}"
    local T_val="${spec#T=}"; T_val="${T_val%,*}"
    local K_val="${spec##*,K=}"
    local name="ablation_T${T_val}_K${K_val}"
    echo "[STAGE-A] GPU=$gpu $name"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_switch_diffusion.py \
      --dataset-dir "$IMDB_DATASET" \
      --structure-checkpoint "$IMDB_STRUCTURE" \
      --output-dir "$STAGE_A_DIR/$name" \
      --seed 0 --device cuda \
      --test-only "$IMDB_MODEL" \
      --randomization-steps 20000 \
      --sample-steps "$T_val" \
      --sample-candidates "$K_val" \
      --sample-temperature 1.0 \
      --sampling-mode model \
      --switch-kernel tensor \
      --metrics-every 2000 \
      > "$LOG_DIR/stage_a_${name}.log" 2>&1 &
  done
  wait

  # Select best (T,K) by max(tri_err, clust_err)
  python3 - "$STAGE_A_DIR" /data2/lyh/gnn_project/DiHug/imdb_step_ablation_8gpu <<'PY'
import json, pathlib, sys
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
    print(f"{name}: T={T_val} K={K_val} tri={tri:.0f} clust={clust:.4f} tri_err={tri_err:.4f} clust_err={clust_err:.4f} score={score:.4f}")
    if score < best_score:
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
      --randomization-steps 20000 \
      --sample-steps "$best_T" \
      --sample-candidates "$best_K" \
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
      --randomization-steps 20000 \
      --sample-steps "$best_T" \
      --sample-candidates "$best_K" \
      --sample-temperature 1.0 \
      --sampling-mode model \
      --switch-kernel tensor \
      --metrics-every 2000 \
      > "$LOG_DIR/stage_b_seed${seed}.log" 2>&1 &
  done
  wait

  # Aggregate 10-seed summary
  python3 - "$STAGE_B_DIR" <<'PY'
import json, pathlib, sys, statistics, math
root = pathlib.Path(sys.argv[1])
triangles = []; clustering = []; transitivity = []; assort = []; lcc = []
for report_path in sorted(root.glob("seed*/sampling_report.json")):
    m = json.loads(report_path.read_text(encoding="utf-8"))["final_metrics"]
    triangles.append(m["triangles"])
    clustering.append(m["clustering"])
    transitivity.append(m["transitivity"])
    assort.append(m["assortativity"])
    lcc.append(m["lcc"])
def stats(vals, target):
    vals = [float(v) for v in vals]
    return {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals)>1 else 0.0,
            "min": min(vals), "max": max(vals),
            "rel_err": abs(statistics.mean(vals)-target)/target}
report = {
    "triangles": stats(triangles, 655.0),
    "clustering": stats(clustering, 0.4689355333),
    "transitivity": stats(transitivity, 0.008432985415082602),
    "assortativity": stats(assort, -0.16497348176967325),
    "lcc": stats(lcc, 1653.0),
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
  local PUBMED_CKPT="$STAGE_C_DIR/structure_checkpoint.json"
  local PUBMED_NPZ="/tmp/pubmed_switch_dataset.npz"

  # Build canonical dataset once
  echo "[STAGE-C] building canonical PubMed structure checkpoint"
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

  # Build canonical switch dataset (single-process)
  if [[ ! -f "$PUBMED_NPZ" ]]; then
    echo "[STAGE-C] building canonical switch dataset (single-process)"
    python scripts/run_switch_diffusion.py \
      --dataset-dir "$PUBMED_DATASET" \
      --structure-checkpoint "$PUBMED_CKPT" \
      --output-dir "$STAGE_C_DIR/data_build" \
      --seed 0 --device cpu \
      --trajectories 8 --forward-steps 2000 --record-every 10 \
      --train-candidates 32 --epochs 1 --train-only \
      > "$LOG_DIR/stage_c_data.log" 2>&1
    # Copy npz to /tmp cache
    cp "$STAGE_C_DIR/data_build/switch_dataset.npz" "$PUBMED_NPZ" 2>/dev/null || true
  fi

  # Train state_only + utility_kl + learned_condition
  echo "[STAGE-C] training PubMed standard diffusion"
  CUDA_VISIBLE_DEVICES=0 python scripts/run_switch_diffusion.py \
    --dataset-dir "$PUBMED_DATASET" \
    --structure-checkpoint "$PUBMED_CKPT" \
    --output-dir "$STAGE_C_DIR/train" \
    --seed 0 --device cuda \
    --trajectories 8 --forward-steps 2000 --record-every 10 \
    --train-candidates 32 --epochs 40 --batch-size 64 \
    --hidden-dim 256 --layers 3 --dropout 0.1 \
    > "$LOG_DIR/stage_c_train.log" 2>&1

  # Generate 10 graphs with T=100, K=200
  local T_val=100 K_val=200
  echo "[STAGE-C] generating 10 PubMed graphs (T=$T_val, K=$K_val)"

  for seed in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$seed" python scripts/run_switch_diffusion.py \
      --structure-checkpoint "$PUBMED_CKPT" \
      --output-dir "$STAGE_C_DIR/seed${seed}" \
      --seed "$seed" --device cuda \
      --test-only "$STAGE_C_DIR/train/switch_diffusion.pt" \
      --sample-steps "$T_val" \
      --sample-candidates "$K_val" \
      --sample-temperature 1.0 \
      --sampling-mode model \
      --switch-kernel tensor \
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
      --sample-steps "$T_val" \
      --sample-candidates "$K_val" \
      --sample-temperature 1.0 \
      --sampling-mode model \
      --switch-kernel tensor \
      --metrics-every 2000 \
      > "$LOG_DIR/stage_c_seed${seed}.log" 2>&1 &
  done
  wait

  # Aggregate PubMed 10-seed
  python3 - "$STAGE_C_DIR" 61137.0 0.335 <<'PY'
import json, pathlib, sys, statistics
root = pathlib.Path(sys.argv[1])
tri_target = float(sys.argv[2])
clust_target = float(sys.argv[3])
triangles = []; clustering = []; transitivity = []; lcc = []
for report_path in sorted(root.glob("seed*/sampling_report.json")):
    m = json.loads(report_path.read_text(encoding="utf-8"))["final_metrics"]
    triangles.append(m["triangles"])
    clustering.append(m["clustering"])
    transitivity.append(m["transitivity"])
    lcc.append(m["lcc"])
def stats(vals, target):
    vals = [float(v) for v in vals]
    return {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals)>1 else 0.0,
            "min": min(vals), "max": max(vals),
            "rel_err": abs(statistics.mean(vals)-target)/target}
report = {
    "triangles": stats(triangles, tri_target),
    "clustering": stats(clustering, clust_target),
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

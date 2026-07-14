#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET_DIR="${DATASET_DIR:-data/PubMed_baseline_subgraphs/subgraph_000}"
OUT_ROOT="${OUT_ROOT:-results/pubmed_switch_diffusion_8way}"
LOG_ROOT="${LOG_ROOT:-logs/pubmed_switch_diffusion_8way}"
STRUCTURE_CKPT="${STRUCTURE_CKPT:-$OUT_ROOT/structure_checkpoint.json}"
SEED_OFFSET="${SEED_OFFSET:-0}"
DEVICE="${DEVICE:-cpu}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

if [[ ! -f "$STRUCTURE_CKPT" ]]; then
  echo "[PUBMED-SWITCH] building anonymous triangle/root-triangle condition"
  python scripts/build_switch_structure_checkpoint.py \
    --dataset-dir "$DATASET_DIR" --output "$STRUCTURE_CKPT" \
    --seed 0 --null-chains 4 --null-steps 5000 --null-thin 500 \
    --max-features 64 \
    --feature-prefix triangle/ \
    --feature-prefix root_triangle/ \
    --feature-prefix root_triangle_total/ \
    --reserve-prefix triangle/=-1 \
    --reserve-prefix root_triangle_total/=-1 \
    >"$LOG_ROOT/structure_checkpoint.log" 2>&1
fi

train_job() {
  local worker="$1" seed="$2" name="train_s$2"
  CUDA_VISIBLE_DEVICES="$worker" python scripts/run_switch_diffusion.py \
    --dataset-dir "$DATASET_DIR" --structure-checkpoint "$STRUCTURE_CKPT" \
    --output-dir "$OUT_ROOT/$name" --seed "$seed" --device "$DEVICE" \
    --trajectories 4 --forward-steps 1000 --record-every 10 \
    --train-candidates 16 --epochs 40 --batch-size 32 \
    --hidden-dim 256 --layers 3 --dropout 0.1 --train-only \
    >"$LOG_ROOT/$name.log" 2>&1
}

sample_job() {
  local worker="$1" checkpoint="$2" name="$3" seed="$4"
  local mode="$5" steps="$6" candidates="$7"
  CUDA_VISIBLE_DEVICES="$worker" python scripts/run_switch_diffusion.py \
    --structure-checkpoint "$STRUCTURE_CKPT" --output-dir "$OUT_ROOT/$name" \
    --seed "$seed" --device "$DEVICE" --test-only "$checkpoint" \
    --sample-steps "$steps" --sample-candidates "$candidates" \
    --sample-temperature 0 --metrics-every 1000 --sampling-mode "$mode" \
    >"$LOG_ROOT/$name.log" 2>&1
}

echo "[PUBMED-SWITCH] stage A: eight matched reverse-kernel fits"
for worker in $(seq 0 7); do
  seed=$((SEED_OFFSET + worker))
  train_job "$worker" "$seed" &
done
wait

echo "[PUBMED-SWITCH] stage B: 2k-step causal gate"
for worker in $(seq 0 7); do
  seed=$((SEED_OFFSET + worker))
  sample_job "$worker" "$OUT_ROOT/train_s${seed}/switch_diffusion.pt" \
    "gate2k_s${seed}" "$seed" model 2000 16 &
done
wait

python - "$OUT_ROOT" <<'PY'
import json, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
ratios = []
for path in sorted(root.glob("gate2k_s*/sampling_report.json")):
    report = json.loads(path.read_text())
    initial = abs(report["initial_metrics"]["triangles"] - 61137.0)
    final = abs(report["final_metrics"]["triangles"] - 61137.0)
    ratios.append(final / max(initial, 1.0))
if len(ratios) != 8 or statistics.mean(ratios) >= 0.95:
    raise SystemExit(
        f"PubMed causal gate failed: runs={len(ratios)} mean_error_ratio={statistics.mean(ratios) if ratios else float('inf'):.4f}"
    )
print(f"[PUBMED-SWITCH] causal gate passed: mean_error_ratio={statistics.mean(ratios):.4f}")
PY

echo "[PUBMED-SWITCH] stage C: eight-seed 20k x 16 main evaluation"
for worker in $(seq 0 7); do
  seed=$((SEED_OFFSET + worker))
  sample_job "$worker" "$OUT_ROOT/train_s${seed}/switch_diffusion.pt" \
    "main_s${seed}" "$seed" model 20000 16 &
done
wait

echo "[PUBMED-SWITCH] stage D: paired null, upper bound, and compute ablations"
seed="$SEED_OFFSET"
checkpoint="$OUT_ROOT/train_s${seed}/switch_diffusion.pt"
sample_job 0 "$checkpoint" paired_random "$seed" random 20000 16 &
sample_job 1 "$checkpoint" paired_oracle "$seed" oracle 20000 16 &
sample_job 2 "$checkpoint" paired_c8 "$seed" model 20000 8 &
sample_job 3 "$checkpoint" paired_c32 "$seed" model 20000 32 &
sample_job 4 "$checkpoint" paired_10k "$seed" model 10000 16 &
wait

python scripts/summarize_switch_diffusion.py "$OUT_ROOT" \
  --real-triangles 61137 --real-clustering 0.335 \
  | tee "$LOG_ROOT/summary.tsv"
echo "[PUBMED-SWITCH] all jobs completed"

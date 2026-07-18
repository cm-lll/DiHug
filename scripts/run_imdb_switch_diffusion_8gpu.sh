#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STRUCTURE_CKPT="${STRUCTURE_CKPT:-results/imdb_structure_energy_gwesp_continuation_f48_v2_seed0/structure_energy_checkpoint.json}"
DATASET_DIR="${DATASET_DIR:-data/IMDB_movie_subgraphs/subgraph_000}"
OUT_ROOT="${OUT_ROOT:-results/imdb_switch_diffusion_8gpu}"
LOG_ROOT="${LOG_ROOT:-logs/imdb_switch_diffusion_8gpu}"
SEED_OFFSET="${SEED_OFFSET:-0}"
TRAJECTORIES="${TRAJECTORIES:-8}"
FORWARD_STEPS="${FORWARD_STEPS:-2000}"
RECORD_EVERY="${RECORD_EVERY:-10}"
EPOCHS="${EPOCHS:-40}"

if [[ ! -f "$STRUCTURE_CKPT" ]]; then
  echo "Missing structure checkpoint: $STRUCTURE_CKPT" >&2
  exit 1
fi
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

train_job() {
  local gpu="$1" seed="$2" name="train_s$2"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_switch_diffusion.py \
    --dataset-dir "$DATASET_DIR" \
    --structure-checkpoint "$STRUCTURE_CKPT" \
    --output-dir "$OUT_ROOT/$name" \
    --seed "$seed" --device cuda \
    --trajectories "$TRAJECTORIES" \
    --forward-steps "$FORWARD_STEPS" --record-every "$RECORD_EVERY" \
    --train-candidates 32 --epochs "$EPOCHS" --batch-size 64 \
    --hidden-dim 256 --layers 3 --dropout 0.1 --train-only \
    >"$LOG_ROOT/$name.log" 2>&1
}

sample_job() {
  local gpu="$1" checkpoint="$2" name="$3" seed="$4"
  local mode="$5" steps="$6" candidates="$7" temperature="$8"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_switch_diffusion.py \
    --structure-checkpoint "$STRUCTURE_CKPT" \
    --output-dir "$OUT_ROOT/$name" \
    --seed "$seed" --device cuda --test-only "$checkpoint" \
    --sample-steps "$steps" --sample-candidates "$candidates" \
    --sample-temperature "$temperature" --metrics-every 1000 \
    --sampling-mode "$mode" >"$LOG_ROOT/$name.log" 2>&1
}

echo "[SWITCH-DIFFUSION-8GPU] stage A: eight matched reverse-kernel fits"
for gpu in $(seq 0 7); do
  seed=$((SEED_OFFSET + gpu))
  train_job "$gpu" "$seed" &
done
wait

echo "[SWITCH-DIFFUSION-8GPU] stage B: eight-seed 20k x 64 main evaluation"
for gpu in $(seq 0 7); do
  seed=$((SEED_OFFSET + gpu))
  checkpoint="$OUT_ROOT/train_s${seed}/switch_diffusion.pt"
  sample_job "$gpu" "$checkpoint" "main_s${seed}" "$seed" model 20000 64 0 &
done
wait

echo "[SWITCH-DIFFUSION-8GPU] stage C: paired proposal and compute ablations"
seed="$SEED_OFFSET"
checkpoint="$OUT_ROOT/train_s${seed}/switch_diffusion.pt"
sample_job 0 "$checkpoint" "paired_random" "$seed" random 20000 64 0 &
sample_job 1 "$checkpoint" "paired_oracle" "$seed" oracle 20000 64 0 &
sample_job 2 "$checkpoint" "paired_c16" "$seed" model 20000 16 0 &
sample_job 3 "$checkpoint" "paired_c32" "$seed" model 20000 32 0 &
sample_job 4 "$checkpoint" "paired_10k_c64" "$seed" model 10000 64 0 &
sample_job 5 "$checkpoint" "paired_c128" "$seed" model 20000 128 0 &
sample_job 6 "$checkpoint" "paired_30k_c64" "$seed" model 30000 64 0 &
sample_job 7 "$checkpoint" "paired_temp01" "$seed" model 20000 64 0.1 &
wait

python scripts/summarize_switch_diffusion.py "$OUT_ROOT" \
  | tee "$LOG_ROOT/summary.tsv"
echo "[SWITCH-DIFFUSION-8GPU] all jobs completed"

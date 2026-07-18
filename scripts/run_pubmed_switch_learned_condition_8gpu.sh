#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/lyh/gnn_project/DiHug}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/pubmed_switch_learned_condition_8gpu}"
STRUCTURE_CHECKPOINT="${STRUCTURE_CHECKPOINT:-$ROOT/results/pubmed_switch_structure_seed0/structure_checkpoint.json}"
SEED_OFFSET="${SEED_OFFSET:-0}"

source /data2/lyh/miniconda3/etc/profile.d/conda.sh
conda activate sparse_block
cd "$ROOT"
mkdir -p "$OUT_ROOT"

run_one() {
  local gpu="$1"
  local seed="$2"
  local train_dir="$OUT_ROOT/learned_s${seed}_train"
  local sample_dir="$OUT_ROOT/learned_s${seed}_sample20k"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_switch_diffusion.py \
    --dataset-dir data/PubMed_baseline_subgraphs/subgraph_000 \
    --structure-checkpoint "$STRUCTURE_CHECKPOINT" \
    --output-dir "$train_dir" \
    --seed "$seed" --device cuda \
    --conditioning-mode state_only \
    --training-objective utility_kl --utility-temperature 1.0 \
    --learned-condition --learned-condition-ridge 1e-6 \
    --trajectories 4 --forward-steps 20000 --record-every 100 \
    --train-candidates 16 --epochs 1 --batch-size 64 \
    --hidden-dim 256 --layers 3 --dropout 0 \
    --train-only > "$OUT_ROOT/learned_s${seed}_train.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_switch_diffusion.py \
    --dataset-dir data/PubMed_baseline_subgraphs/subgraph_000 \
    --structure-checkpoint "$STRUCTURE_CHECKPOINT" \
    --output-dir "$sample_dir" \
    --seed "$seed" --device cuda \
    --test-only "$train_dir/switch_diffusion.pt" \
    --randomization-steps 20000 --sample-steps 20000 \
    --sample-candidates 64 --sample-temperature 0 \
    --sampling-mode model --metrics-every 2000 --switch-kernel tensor \
    --learned-condition-noop --learned-condition-noop-patience 1000 \
    > "$OUT_ROOT/learned_s${seed}_sample20k.log" 2>&1
}

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  run_one "$gpu" "$((SEED_OFFSET + gpu))" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"

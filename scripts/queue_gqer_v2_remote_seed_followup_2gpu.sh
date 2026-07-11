#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/jdt/gnn_project/DiHug}"
SOURCE_LOG_DIR="${SOURCE_LOG_DIR:-$ROOT/logs/gqer_v2_full100_seed2_0712}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/gqer_v2_seed_followup_0712}"
GPU_IMDB="${GPU_IMDB:-3}"
GPU_PUBMED="${GPU_PUBMED:-6}"

while pgrep -f '[r]un_gqer_v2_remote_2gpu_queue.sh' >/dev/null; do
  sleep 60
done

eval "$(conda shell.bash hook)"
conda activate dihug
cd "$ROOT"
mkdir -p "$LOG_DIR"

find_ckpt() {
  local name="$1" ckpt
  ckpt=$(find "$ROOT/output" -path "*${name}*" -name last.ckpt | sort | tail -1)
  [[ -n "$ckpt" ]] || { echo "Missing checkpoint for $name" >&2; exit 1; }
  printf '%s\n' "$ckpt"
}

run_sample() {
  local gpu="$1" experiment="$2" ckpt="$3" name="$4" seed="$5" projection="$6"
  shift 6
  echo "[$(date '+%F %T')] GPU $gpu starting $name" | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m dihug.main \
    "+experiment=$experiment" \
    general.wandb=disabled general.gpus=1 general.test_only="$ckpt" \
    general.run_test_after_train=false general.enable_test_sampling=true \
    general.enable_val_sampling=false general.test_variance=1 \
    general.final_model_samples_to_generate=1 general.test_sampling_full_steps=true \
    "general.test_sampling_seeds=[$seed]" \
    model.sampling_structure_swap_limiter="$projection" \
    general.name="$name" "$@" >"$LOG_DIR/$name.log" 2>&1
}

run_followup() {
  local gpu="$1" dataset="$2" experiment="$3" ckpt="$4"
  run_sample "$gpu" "$experiment" "$ckpt" "${dataset}_v2_s2_modelonly_seed3_full100" 3 false
  if [[ -f "$SOURCE_LOG_DIR/${dataset}_v2_s2_projected_strong_full100.log" ]]; then
    run_sample "$gpu" "$experiment" "$ckpt" "${dataset}_v2_s2_projected_strong_seed3_full100" 3 true \
      model.sampling_structure_swap_limiter_max_swaps=1024 \
      model.sampling_structure_swap_limiter_pair_top_adds=128 \
      model.sampling_structure_swap_limiter_pair_top_drops=128 \
      model.sampling_structure_swap_limiter_model_score_weight=0.0 \
      model.sampling_structure_swap_limiter_score_margin=10.0
  else
    run_sample "$gpu" "$experiment" "$ckpt" "${dataset}_v2_s2_projected_seed4_full100" 4 true
  fi
}

run_followup "$GPU_IMDB" imdb imdb_movie_hgt_l3h4_gqer_v2 "$(find_ckpt imdb_gqer_v2_s2)" &
PID_IMDB=$!
run_followup "$GPU_PUBMED" pubmed pubmed_hetero_xey_hgt_gqer_v2_pw20 "$(find_ckpt pubmed_gqer_v2_s2)" &
PID_PUBMED=$!

failed=0
wait "$PID_IMDB" || failed=1
wait "$PID_PUBMED" || failed=1
exit "$failed"

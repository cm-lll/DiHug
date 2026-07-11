#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/lyh/gnn_project/DiHug}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/gqer_v2_modelonly_followup_0712}"

while pgrep -f '[r]un_gqer_v2_full100_local_8gpu.sh' >/dev/null; do
  sleep 60
done

source /data2/lyh/miniconda3/etc/profile.d/conda.sh
conda activate sparse_block
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
mkdir -p "$LOG_DIR"

find_ckpt() {
  local name="$1" ckpt
  ckpt=$(find "$ROOT/output" -path "*${name}*" -name last.ckpt | sort | tail -1)
  [[ -n "$ckpt" ]] || { echo "Missing checkpoint for $name" >&2; exit 1; }
  printf '%s\n' "$ckpt"
}

launch() {
  local gpu="$1" experiment="$2" ckpt="$3" name="$4" seed="$5"
  echo "[$(date '+%F %T')] GPU $gpu starting $name" | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m dihug.main \
    "+experiment=$experiment" \
    general.wandb=disabled general.gpus=1 \
    general.test_only="$ckpt" \
    general.run_test_after_train=false \
    general.enable_test_sampling=true \
    general.enable_val_sampling=false \
    general.test_variance=1 \
    general.final_model_samples_to_generate=1 \
    general.test_sampling_full_steps=true \
    "general.test_sampling_seeds=[$seed]" \
    model.sampling_structure_swap_limiter=false \
    general.name="$name" >"$LOG_DIR/$name.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
}

PIDS=()
NAMES=()
launch 0 imdb_movie_hgt_l3h4_gqer_v2 "$(find_ckpt imdb_gqer_v2_w00_s1)" imdb_v2_control_s1_full100 1
launch 1 imdb_movie_hgt_l3h4_gqer_v2 "$(find_ckpt imdb_gqer_v2_s1)" imdb_v2_modelonly_s1_full100 1
launch 2 imdb_movie_hgt_l3h4_gqer "$(find_ckpt imdb_gqer_w01_s0)" imdb_v1_modelonly_s0_full100 0
launch 3 imdb_movie_hgt_l3h4_gqer "$(find_ckpt imdb_gqer_w01_s1)" imdb_v1_modelonly_s1_full100 1
launch 4 pubmed_hetero_xey_hgt_gqer_v2_pw20 "$(find_ckpt pubmed_gqer_v2_w00_s1)" pubmed_v2_control_s1_full100 1
launch 5 pubmed_hetero_xey_hgt_gqer_v2_pw20 "$(find_ckpt pubmed_gqer_v2_s1)" pubmed_v2_modelonly_s1_full100 1
launch 6 pubmed_hetero_xey_hgt_gqer_pw20 "$(find_ckpt pubmed_gqer_w01_s0)" pubmed_v1_modelonly_s0_full100 0
launch 7 pubmed_hetero_xey_hgt_gqer_pw20 "$(find_ckpt pubmed_gqer_w01_s1)" pubmed_v1_modelonly_s1_full100 1

failed=0
for index in "${!PIDS[@]}"; do
  wait "${PIDS[$index]}" || failed=1
  echo "[$(date '+%F %T')] finished ${NAMES[$index]}" | tee -a "$LOG_DIR/launcher.log"
done
exit "$failed"

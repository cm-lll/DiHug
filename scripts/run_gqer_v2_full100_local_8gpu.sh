#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/lyh/gnn_project/DiHug}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/gqer_v2_full100_3column_s0_0712}"
ENV_NAME="${ENV_NAME:-sparse_block}"

source /data2/lyh/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
mkdir -p "$LOG_DIR"

find_ckpt() {
  local name="$1"
  local ckpt
  ckpt=$(find "$ROOT/output" -path "*${name}*" -name last.ckpt | sort | tail -1)
  if [[ -z "$ckpt" ]]; then
    echo "Missing checkpoint for $name" >&2
    exit 1
  fi
  printf '%s\n' "$ckpt"
}

launch() {
  local gpu="$1" experiment="$2" ckpt="$3" name="$4" seed="$5" projection="$6"
  shift 6
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
    model.sampling_structure_swap_limiter="$projection" \
    general.name="$name" \
    "$@" >"$LOG_DIR/$name.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
}

IMDB_C0=$(find_ckpt imdb_gqer_v2_w00_s0)
IMDB_V0=$(find_ckpt imdb_gqer_v2_s0)
IMDB_V1=$(find_ckpt imdb_gqer_v2_s1)
PUBMED_C0=$(find_ckpt pubmed_gqer_v2_w00_s0)
PUBMED_V0=$(find_ckpt pubmed_gqer_v2_s0)
PUBMED_V1=$(find_ckpt pubmed_gqer_v2_s1)

PIDS=()
NAMES=()
launch 0 imdb_movie_hgt_l3h4_gqer_v2 "$IMDB_C0" imdb_v2_control_s0_full100 0 false
launch 1 imdb_movie_hgt_l3h4_gqer_v2 "$IMDB_V0" imdb_v2_modelonly_s0_full100 0 false
launch 2 imdb_movie_hgt_l3h4_gqer_v2 "$IMDB_V0" imdb_v2_projected_s0_full100 0 true
launch 3 imdb_movie_hgt_l3h4_gqer_v2 "$IMDB_V1" imdb_v2_projected_s1_full100 1 true
launch 4 pubmed_hetero_xey_hgt_gqer_v2_pw20 "$PUBMED_C0" pubmed_v2_control_s0_full100 0 false
launch 5 pubmed_hetero_xey_hgt_gqer_v2_pw20 "$PUBMED_V0" pubmed_v2_modelonly_s0_full100 0 false
launch 6 pubmed_hetero_xey_hgt_gqer_v2_pw20 "$PUBMED_V0" pubmed_v2_projected_s0_full100 0 true
launch 7 pubmed_hetero_xey_hgt_gqer_v2_pw20 "$PUBMED_V1" pubmed_v2_projected_s1_full100 1 true

failed=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "[$(date '+%F %T')] completed ${NAMES[$index]}" | tee -a "$LOG_DIR/launcher.log"
  else
    echo "[$(date '+%F %T')] FAILED ${NAMES[$index]}" | tee -a "$LOG_DIR/launcher.log"
    failed=1
  fi
done
exit "$failed"

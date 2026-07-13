#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/jdt/gnn_project/DiHug}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/imdb_absorbing_remote_0713}"
ENV_NAME="${ENV_NAME:-dihug}"
GPU_A="${GPU_A:-3}"
GPU_B="${GPU_B:-4}"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
cd "$ROOT"
mkdir -p "$LOG_DIR"

run_one() {
  local gpu="$1" seed="$2" name="$3"
  echo "[$(date '+%F %T')] GPU=$gpu $name seed=$seed uncapped" \
    | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m dihug.main \
    +experiment=imdb_movie_hgt_l3h4_family_set_absorbing \
    general.wandb=disabled general.gpus=1 \
    general.run_test_after_train=true \
    general.enable_test_sampling=true general.enable_val_sampling=false \
    general.test_variance=1 general.final_model_samples_to_generate=1 \
    general.test_sampling_full_steps=true general.test_sampling_metrics_every=0 \
    train.n_epochs=100 train.batch_size=1 train.save_model=true \
    train.seed="$seed" model.block_query_max_edges_per_family=0 \
    model.family_set_ranking_negatives_per_positive=16 \
    model.family_set_ranking_max_negatives=4096 \
    general.name="$name" >"$LOG_DIR/$name.log" 2>&1
}

run_one "$GPU_A" 2 imdb_absorb_uncapped_s2 &
run_one "$GPU_B" 3 imdb_absorb_uncapped_s3 &
wait

echo "[$(date '+%F %T')] ALL REMOTE ABSORBING RUNS DONE" \
  | tee -a "$LOG_DIR/launcher.log"

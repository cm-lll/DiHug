#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/lyh/gnn_project/DiHug}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/imdb_family_set_pilot_0713}"
ENV_NAME="${ENV_NAME:-sparse_block}"

source /data2/lyh/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
mkdir -p "$LOG_DIR"

COMMON=(
  +experiment=imdb_movie_hgt_l3h4_family_set
  general.wandb=disabled general.gpus=1
  general.run_test_after_train=false
  general.enable_test_sampling=false
  general.enable_val_sampling=false
  train.n_epochs=20 train.batch_size=1 train.save_model=true
)

run_one() {
  local gpu="$1" name="$2" seed="$3"
  shift 3
  echo "[$(date '+%F %T')] GPU=$gpu START $name seed=$seed $*" \
    | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m dihug.main \
    "${COMMON[@]}" "train.seed=$seed" "general.name=$name" "$@" \
    >"$LOG_DIR/$name.log" 2>&1
  echo "[$(date '+%F %T')] GPU=$gpu DONE $name" \
    | tee -a "$LOG_DIR/launcher.log"
}

run_one 0 imdb_fset_bce_control_s0 0 \
  model.family_set_ranking_loss_weight=0 model.edge_exist_weight=1.0 &
run_one 1 imdb_fset_main_s0 0 \
  model.family_set_ranking_loss_weight=1 model.edge_exist_weight=0.2 &
run_one 2 imdb_fset_pure_s0 0 \
  model.family_set_ranking_loss_weight=1 model.edge_exist_weight=0 &
run_one 3 imdb_fset_bce005_s0 0 \
  model.family_set_ranking_loss_weight=1 model.edge_exist_weight=0.05 &
run_one 4 imdb_fset_temp05_s0 0 \
  model.family_set_ranking_temperature=0.5 &
run_one 5 imdb_fset_temp20_s0 0 \
  model.family_set_ranking_temperature=2.0 &
run_one 6 imdb_fset_neg16_s0 0 \
  model.family_set_ranking_negatives_per_positive=16 \
  model.family_set_ranking_max_negatives=4096 &
run_one 7 imdb_fset_main_s1 1 \
  model.family_set_ranking_loss_weight=1 model.edge_exist_weight=0.2 &

wait
echo "[$(date '+%F %T')] ALL DONE" | tee -a "$LOG_DIR/launcher.log"


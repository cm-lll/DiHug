#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/lyh/gnn_project/DiHug}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/imdb_absorbing_pilot_0713}"
ENV_NAME="${ENV_NAME:-sparse_block}"
GPUS_CSV="${GPUS_CSV:-0,1,2,3,4,5,6,7}"
source /data2/lyh/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
mkdir -p "$LOG_DIR"

run_one() {
  local gpu="$1" name="$2" seed="$3"
  shift 3
  echo "[$(date '+%F %T')] GPU=$gpu $name seed=$seed $*" \
    | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m dihug.main \
    +experiment=imdb_movie_hgt_l3h4_family_set_absorbing \
    general.wandb=disabled general.gpus=1 \
    general.run_test_after_train=true \
    general.enable_test_sampling=true general.enable_val_sampling=false \
    general.test_variance=1 general.final_model_samples_to_generate=1 \
    general.test_sampling_full_steps=true general.test_sampling_metrics_every=0 \
    train.n_epochs=30 train.batch_size=1 train.save_model=true \
    train.seed="$seed" model.diffusion_steps=20 \
    model.family_set_ranking_negatives_per_positive=16 \
    model.family_set_ranking_max_negatives=4096 \
    general.name="$name" "$@" >"$LOG_DIR/$name.log" 2>&1
}

# Four seeds test the formal mechanism; the remaining jobs isolate the only
# training choices that materially affect the absorbing restoration ranking.
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
NAMES=(
  imdb_absorb_main_s0 imdb_absorb_main_s1 imdb_absorb_main_s2
  imdb_absorb_main_s3 imdb_absorb_purepl_s0 imdb_absorb_bce005_s0
  imdb_absorb_survival05_s0 imdb_absorb_survival20_s0
)
SEEDS=(0 1 2 3 0 0 0 0)
EXTRAS=("" "" "" "" "model.edge_exist_weight=0.0" \
  "model.edge_exist_weight=0.05" "model.absorbing_survival_power=0.5" \
  "model.absorbing_survival_power=2.0")
for idx in "${!GPUS[@]}"; do
  [[ "$idx" -lt "${#NAMES[@]}" ]] || break
  read -r -a extra_args <<< "${EXTRAS[$idx]}"
  run_one "${GPUS[$idx]}" "${NAMES[$idx]}" "${SEEDS[$idx]}" \
    "${extra_args[@]}" &
done
wait

echo "[$(date '+%F %T')] ALL ABSORBING PILOTS DONE" \
  | tee -a "$LOG_DIR/launcher.log"

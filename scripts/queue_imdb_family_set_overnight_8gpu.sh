#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/lyh/gnn_project/DiHug}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/imdb_family_set_overnight_0713}"
ENV_NAME="${ENV_NAME:-sparse_block}"
source /data2/lyh/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
mkdir -p "$LOG_DIR"

PILOT_NAMES=(
  imdb_fset_bce_control_s0 imdb_fset_main_s0 imdb_fset_pure_s0
  imdb_fset_bce005_s0 imdb_fset_temp05_s0 imdb_fset_temp20_s0
  imdb_fset_neg16_s0 imdb_fset_main_s1
)

sample_checkpoint() {
  local gpu="$1" source_name="$2" output_name="$3" seed="$4"
  local ckpt
  ckpt=$(find "$ROOT/output" -path "*${source_name}*" -name last.ckpt | sort | tail -1)
  [[ -n "$ckpt" ]] || { echo "missing checkpoint: $source_name"; return 1; }
  echo "[$(date '+%F %T')] GPU=$gpu SAMPLE $source_name -> $output_name" \
    | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m dihug.main \
    +experiment=imdb_movie_hgt_l3h4_family_set \
    general.wandb=disabled general.gpus=1 \
    general.test_only="$ckpt" \
    general.run_test_after_train=false \
    general.enable_test_sampling=true general.enable_val_sampling=false \
    general.test_variance=1 general.final_model_samples_to_generate=1 \
    general.test_sampling_full_steps=true general.test_sampling_metrics_every=0 \
    general.test_sampling_seeds="[$seed]" \
    model.sampling_structure_swap_limiter=false \
    model.sampling_exact_k_final_only=true \
    general.name="$output_name" \
    >"$LOG_DIR/$output_name.log" 2>&1
}

echo "[$(date '+%F %T')] STAGE-A full100 for all pilot checkpoints" \
  | tee -a "$LOG_DIR/launcher.log"
pids=()
for gpu in $(seq 0 7); do
  seed=0
  [[ "${PILOT_NAMES[$gpu]}" == "imdb_fset_main_s1" ]] && seed=1
  sample_checkpoint "$gpu" "${PILOT_NAMES[$gpu]}" \
    "${PILOT_NAMES[$gpu]}_ep20_full100" "$seed" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

train_and_sample() {
  local gpu="$1" name="$2" seed="$3"
  shift 3
  echo "[$(date '+%F %T')] GPU=$gpu TRAIN100 $name seed=$seed $*" \
    | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m dihug.main \
    +experiment=imdb_movie_hgt_l3h4_family_set \
    general.wandb=disabled general.gpus=1 \
    general.run_test_after_train=false general.enable_test_sampling=false \
    general.enable_val_sampling=false train.n_epochs=100 train.batch_size=1 \
    train.save_model=true train.seed="$seed" general.name="${name}_train100" \
    "$@" >"$LOG_DIR/${name}_train100.log" 2>&1
  sample_checkpoint "$gpu" "${name}_train100" "${name}_full100" "$seed"
}

echo "[$(date '+%F %T')] STAGE-B train100 robustness matrix" \
  | tee -a "$LOG_DIR/launcher.log"
train_and_sample 0 imdb_fset100_control_s0 0 \
  model.family_set_ranking_loss_weight=0 model.edge_exist_weight=1.0 &
train_and_sample 1 imdb_fset100_control_s1 1 \
  model.family_set_ranking_loss_weight=0 model.edge_exist_weight=1.0 &
train_and_sample 2 imdb_fset100_neg16_s0 0 \
  model.family_set_ranking_negatives_per_positive=16 \
  model.family_set_ranking_max_negatives=4096 &
train_and_sample 3 imdb_fset100_neg16_s1 1 \
  model.family_set_ranking_negatives_per_positive=16 \
  model.family_set_ranking_max_negatives=4096 &
train_and_sample 4 imdb_fset100_neg16_s2 2 \
  model.family_set_ranking_negatives_per_positive=16 \
  model.family_set_ranking_max_negatives=4096 &
train_and_sample 5 imdb_fset100_main_s1 1 &
train_and_sample 6 imdb_fset100_main_s2 2 &
train_and_sample 7 imdb_fset100_temp05_s0 0 \
  model.family_set_ranking_temperature=0.5 &
wait
echo "[$(date '+%F %T')] ALL STAGES DONE" | tee -a "$LOG_DIR/launcher.log"


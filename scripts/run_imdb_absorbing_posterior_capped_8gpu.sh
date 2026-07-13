#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/lyh/gnn_project/DiHug}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/imdb_absorbing_posterior_capped_0713}"
ENV_NAME="${ENV_NAME:-sparse_block}"
source /data2/lyh/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
mkdir -p "$LOG_DIR"

COMMON_TRAIN=(
  +experiment=imdb_movie_hgt_l3h4_absorbing_posterior
  general.wandb=disabled general.gpus=1
  general.run_test_after_train=false
  general.enable_test_sampling=false general.enable_val_sampling=false
  train.n_epochs=200 train.batch_size=1 train.save_model=true
  model.absorbing_probability_require_full_candidates=false
)

COMMON_SAMPLE=(
  +experiment=imdb_movie_hgt_l3h4_absorbing_posterior
  general.wandb=disabled general.gpus=1
  general.run_test_after_train=false
  general.enable_test_sampling=true general.enable_val_sampling=false
  general.test_variance=1 general.final_model_samples_to_generate=1
  general.test_sampling_full_steps=true
  model.absorbing_probability_match_checkpoint_family_count=true
  model.absorbing_probability_require_full_candidates=false
)

sample_from_ckpt() {
  local gpu="$1" ckpt="$2" name="$3"
  echo "[$(date '+%F %T')] SAMPLE GPU=$gpu $name" | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m dihug.main \
    "${COMMON_SAMPLE[@]}" \
    general.test_only="$ckpt" \
    general.name="$name" \
    >"$LOG_DIR/${name}.log" 2>&1
}

train_and_sample() {
  local gpu="$1" seed="$2" name="$3" cap="$4"
  shift 4
  echo "[$(date '+%F %T')] TRAIN GPU=$gpu seed=$seed $name cap=$cap $*" \
    | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m dihug.main \
    "${COMMON_TRAIN[@]}" \
    train.seed="$seed" \
    model.block_query_max_edges_per_family="$cap" \
    general.name="${name}" \
    "$@" \
    >"$LOG_DIR/${name}.log" 2>&1

  local ckpt
  ckpt=$(find "$ROOT/output" -path "*${name}*" -name last.ckpt | sort | tail -1)
  if [[ -z "$ckpt" ]]; then
    echo "[$(date '+%F %T')] MISSING checkpoint for $name" \
      | tee -a "$LOG_DIR/launcher.log"
    return 1
  fi
  sample_from_ckpt "$gpu" "$ckpt" "${name}_full100"
}

# ── GPU 0-2: posterior main, cap=5000, seeds 0/1/2 ──
train_and_sample 0 0 imdb_abs_post_s0_cap5k 5000 &
train_and_sample 1 1 imdb_abs_post_s1_cap5k 5000 &
train_and_sample 2 2 imdb_abs_post_s2_cap5k 5000 &

# ── GPU 3: pure calibrated BCE (PL=0) ──
train_and_sample 3 0 imdb_abs_post_pure_bce_s0 5000 \
  model.family_set_ranking_loss_weight=0 &

# ── GPU 4: family-count matching disabled ──
train_and_sample 4 0 imdb_abs_post_nomatch_s0 5000 \
  model.absorbing_probability_match_checkpoint_family_count=false &

# ── GPU 5: probability calibration disabled ──
train_and_sample 5 0 imdb_abs_post_nocal_s0 5000 \
  model.absorbing_probability_calibration=false &

# ── GPU 6: reduced cap 2500 ──
train_and_sample 6 0 imdb_abs_post_s0_cap25k 2500 &

# ── GPU 7: adaptive high-cap (10000 → 7500 → 5000) ──
gpu=7 seed=0 name=imdb_abs_post_s0_cap10k cap=10000
echo "[$(date '+%F %T')] TRAIN GPU=$gpu seed=$seed $name cap=$cap (adaptive)" \
  | tee -a "$LOG_DIR/launcher.log"

train_ok=0
for attempt_cap in 10000 7500 5000; do
  CUDA_VISIBLE_DEVICES="$gpu" python -m dihug.main \
    "${COMMON_TRAIN[@]}" \
    train.seed="$seed" \
    model.block_query_max_edges_per_family="$attempt_cap" \
    general.name="${name}" \
    >"$LOG_DIR/${name}.log" 2>&1 && train_ok=1 && break

  if grep -q "out of memory\|OOM\|CUDA error" "$LOG_DIR/${name}.log" 2>/dev/null; then
    echo "[$(date '+%F %T')] OOM at cap=$attempt_cap, retrying cap=$(
      [[ $attempt_cap -eq 10000 ]] && echo 7500 || echo 5000
    )" | tee -a "$LOG_DIR/launcher.log"
    # Clean up partial output before next attempt
    mv "$LOG_DIR/${name}.log" "$LOG_DIR/${name}_cap${attempt_cap}_oom.log"
    # Remove partial checkpoint dir
    rm -rf $(find "$ROOT/output" -path "*${name}*" -maxdepth 3 -type d 2>/dev/null | head -1) 2>/dev/null || true
  else
    # Non-OOM failure, don't retry
    echo "[$(date '+%F %T')] FAILED $name (non-OOM)" | tee -a "$LOG_DIR/launcher.log"
    break
  fi
done

if [[ "$train_ok" -eq 1 ]]; then
  ckpt=$(find "$ROOT/output" -path "*${name}*" -name last.ckpt | sort | tail -1)
  if [[ -n "$ckpt" ]]; then
    sample_from_ckpt "$gpu" "$ckpt" "${name}_full100"
  fi
fi

wait
echo "[$(date '+%F %T')] ALL DONE" | tee -a "$LOG_DIR/launcher.log"

#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/lyh/gnn_project/DiHug}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/imdb_absorbing_v2_mechanisms_0713}"
ENV_NAME="${ENV_NAME:-sparse_block}"
source /data2/lyh/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
mkdir -p "$LOG_DIR"

train_one() {
  local gpu="$1" seed="$2" name="$3"
  echo "[$(date '+%F %T')] TRAIN gpu=$gpu seed=$seed name=$name" \
    | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m dihug.main \
    +experiment=imdb_movie_hgt_l3h4_family_set_absorbing_v2 \
    general.wandb=disabled general.gpus=1 \
    general.run_test_after_train=false general.enable_test_sampling=false \
    general.enable_val_sampling=false train.n_epochs=30 train.batch_size=1 \
    train.save_model=true train.seed="$seed" model.diffusion_steps=20 \
    model.family_set_ranking_negatives_per_positive=16 \
    model.family_set_ranking_max_negatives=4096 general.name="$name" \
    >"$LOG_DIR/${name}_train.log" 2>&1
}

sample_one() {
  local gpu="$1" ckpt="$2" name="$3"
  shift 3
  echo "[$(date '+%F %T')] SAMPLE gpu=$gpu name=$name $*" \
    | tee -a "$LOG_DIR/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m dihug.main \
    +experiment=imdb_movie_hgt_l3h4_family_set_absorbing_v2 \
    general.wandb=disabled general.gpus=1 general.test_only="$ckpt" \
    general.run_test_after_train=false general.enable_test_sampling=true \
    general.enable_val_sampling=false general.test_variance=1 \
    general.final_model_samples_to_generate=1 \
    general.test_sampling_full_steps=true general.test_sampling_metrics_every=0 \
    model.diffusion_steps=20 general.name="$name" "$@" \
    >"$LOG_DIR/${name}.log" 2>&1
}

# Train only two checkpoints. Every sampling variant below shares the same s0
# checkpoint so differences are caused by reverse mechanics, not optimization.
train_one 0 0 imdb_absorb_v2_s0 &
train_one 1 1 imdb_absorb_v2_s1 &
wait

CKPT0=$(find output -path '*imdb_absorb_v2_s0*' -name last.ckpt | sort | tail -1)
CKPT1=$(find output -path '*imdb_absorb_v2_s1*' -name last.ckpt | sort | tail -1)
[[ -n "$CKPT0" && -n "$CKPT1" ]] || {
  echo "missing v2 checkpoints" | tee -a "$LOG_DIR/launcher.log"
  exit 1
}

sample_one 0 "$CKPT0" v2_s0_monotone \
  model.sampling_absorbing_refinement_mode=monotone \
  model.sampling_absorbing_protect_existing=true &
sample_one 1 "$CKPT0" v2_s0_global_refine \
  model.sampling_absorbing_refinement_mode=monotone \
  model.sampling_absorbing_protect_existing=false &
sample_one 2 "$CKPT0" v2_s0_edgepp_a010 \
  model.sampling_absorbing_active_node_fraction=0.10 &
sample_one 3 "$CKPT0" v2_s0_edgepp_a025 \
  model.sampling_absorbing_active_node_fraction=0.25 &
sample_one 4 "$CKPT0" v2_s0_edgepp_a050 \
  model.sampling_absorbing_active_node_fraction=0.50 &
sample_one 5 "$CKPT0" v2_s0_edgepp_nobudget \
  model.sampling_absorbing_degree_budget_strict=false &
sample_one 6 "$CKPT0" v2_s0_edgepp_deterministic \
  model.sampling_edge_selection=deterministic_exact_k &
sample_one 7 "$CKPT1" v2_s1_edgepp_a025 \
  model.sampling_absorbing_active_node_fraction=0.25 &
wait

echo "[$(date '+%F %T')] ALL V2 MECHANISM RUNS DONE" \
  | tee -a "$LOG_DIR/launcher.log"


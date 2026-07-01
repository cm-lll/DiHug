#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_pubmed_edge_struct_residual_8gpu}"
mkdir -p "$LOG_DIR"

N_EPOCHS="${N_EPOCHS:-100}"
SAMPLING_SKIP="${SAMPLING_SKIP:-10}"
TRAIN_SEED="${TRAIN_SEED:-0}"
TEST_VARIANCE="${TEST_VARIANCE:-3}"
SAMPLES_TO_GENERATE="${SAMPLES_TO_GENERATE:-3}"

COMMON_OVERRIDES=(
  general.wandb=disabled
  general.gpus=1
  train.n_epochs="${N_EPOCHS}"
  train.batch_size=1
  train.save_model=true
  train.seed="${TRAIN_SEED}"
  general.run_test_after_train=true
  general.enable_test_sampling=true
  general.enable_val_sampling=false
  general.enable_val_pred_metrics=false
  general.test_variance="${TEST_VARIANCE}"
  general.final_model_samples_to_generate="${SAMPLES_TO_GENERATE}"
  general.final_model_chains_to_save=0
  general.test_sampling_full_steps=false
  general.sampling_skip="${SAMPLING_SKIP}"
  general.test_sampling_metrics_every=1
  general.verbose_sampling=true
  general.log_sampling_triangle_trace=true
  model.use_sparse_hetero_y=true
  model.use_edge_struct_features=true
  model.edge_struct_feature_dim=8
  model.edge_struct_hidden_dim=64
  model.edge_struct_residual_scale=1.0
  model.edge_struct_use_family_y=false
  model.use_sparse_family_y=false
  model.use_family_y_film=false
  model.use_family_y_in_attention=false
  model.use_family_y_in_edge_film=false
  model.sparse_family_y_degree_bins=5
  model.sampling_edge_selection=gumbel_exact_k
  model.sampling_use_reverse_posterior=true
  model.sampling_reverse_posterior_mix_mode=alpha_bar_s
  model.sampling_reverse_posterior_mix_scale=0.2
  model.sampling_gumbel_temperature=0.01
  model.degree_pair_dist_loss_weight=0.0
  model.closure_pos_loss_weight=0.0
)

launch() {
  local gpu="$1"
  local name="$2"
  shift 2
  local log_file="$LOG_DIR/${name}.out"
  echo "[launch] gpu=${gpu} name=${name}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export CONDA_ENV="${CONDA_ENV:-sparse_block}"
    EXPERIMENT=pubmed_hetero_xey_family_pw20 bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON_OVERRIDES[@]}" \
      "general.name=${name}" \
      "$@"
  ) >"$log_file" 2>&1 &
  echo "$!" >"$LOG_DIR/${name}.pid"
}

# 1) Candidate-edge structural residual only: common-neighbor closure,
# endpoint degree roles, current-edge state, and same-component connectivity.
launch 0 pubmed_edge_struct_only_seed${TRAIN_SEED}

# 2) Same residual with stronger scale. This tests whether the zero-init
# residual learns but is too weak at scale 1.0.
launch 1 pubmed_edge_struct_scale2_seed${TRAIN_SEED} \
  model.edge_struct_residual_scale=2.0

# 3) Family-y only on y->E FiLM, plus family-conditioned structural residual.
# This is the closest version to "global_y + family_y guides each edge update".
launch 2 pubmed_edge_struct_famy_edgefilm_seed${TRAIN_SEED} \
  model.use_sparse_family_y=true \
  model.use_family_y_film=true \
  model.use_family_y_in_edge_film=true \
  model.edge_struct_use_family_y=true

# 4) Same as 3, stronger structural residual.
launch 3 pubmed_edge_struct_famy_edgefilm_scale2_seed${TRAIN_SEED} \
  model.use_sparse_family_y=true \
  model.use_family_y_film=true \
  model.use_family_y_in_edge_film=true \
  model.edge_struct_use_family_y=true \
  model.edge_struct_residual_scale=2.0

# 5) Family-y affects both E->attention FiLM and y->E FiLM. This checks whether
# direct message-passing conditioning helps or duplicates the E-update signal.
launch 4 pubmed_edge_struct_famy_both_seed${TRAIN_SEED} \
  model.use_sparse_family_y=true \
  model.use_family_y_film=true \
  model.use_family_y_in_attention=true \
  model.use_family_y_in_edge_film=true \
  model.edge_struct_use_family_y=true

# 6) Add a light learned degree-bin pair distribution objective.
launch 5 pubmed_edge_struct_famy_dpjs002_seed${TRAIN_SEED} \
  model.use_sparse_family_y=true \
  model.use_family_y_film=true \
  model.use_family_y_in_edge_film=true \
  model.edge_struct_use_family_y=true \
  model.degree_pair_dist_loss_weight=0.02 \
  model.degree_pair_dist_loss_type=js

# 7) Stronger degree-bin pair objective.
launch 6 pubmed_edge_struct_famy_dpjs005_seed${TRAIN_SEED} \
  model.use_sparse_family_y=true \
  model.use_family_y_film=true \
  model.use_family_y_in_edge_film=true \
  model.edge_struct_use_family_y=true \
  model.degree_pair_dist_loss_weight=0.05 \
  model.degree_pair_dist_loss_type=js

# 8) Same training target as 7, but test with degree-pair exact-K ranking.
# This separates "learned role distribution" from the sampling-side role prior.
launch 7 pubmed_edge_struct_famy_dpjs005_degpair_sample_seed${TRAIN_SEED} \
  model.use_sparse_family_y=true \
  model.use_family_y_film=true \
  model.use_family_y_in_edge_film=true \
  model.edge_struct_use_family_y=true \
  model.degree_pair_dist_loss_weight=0.05 \
  model.degree_pair_dist_loss_type=js \
  model.sampling_edge_selection=gumbel_exact_k_degree_pair \
  model.sampling_degree_pair_strength=0.5

wait
echo "[done] logs: ${LOG_DIR}"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_hetero_xey_family_y_ablation_8gpu}"
mkdir -p "$LOG_DIR"

COMMON_OVERRIDES=(
  general.wandb=disabled
  general.gpus=1
  train.n_epochs="${N_EPOCHS:-100}"
  train.batch_size=1
  train.save_model=true
  general.run_test_after_train=true
  general.enable_test_sampling=true
  general.enable_val_sampling=false
  general.enable_val_pred_metrics=false
  general.test_variance=1
  general.final_model_samples_to_generate=1
  general.final_model_chains_to_save=0
  general.test_sampling_full_steps=false
  general.sampling_skip="${SAMPLING_SKIP:-10}"
  general.test_sampling_metrics_every=1
  model.use_sparse_hetero_y=true
  model.use_sparse_family_y=true
  model.use_family_y_film=true
  model.use_family_y_in_attention=false
  model.use_family_y_in_edge_film=false
  model.sparse_family_y_degree_bins="${FAMILY_Y_BINS:-5}"
  model.sampling_edge_selection=gumbel_exact_k
  model.sampling_use_reverse_posterior=true
  model.sampling_reverse_posterior_mix_mode=alpha_bar_s
  model.sampling_reverse_posterior_mix_scale=0.2
  model.sampling_gumbel_temperature=0.01
)

launch() {
  local gpu="$1"
  local name="$2"
  local seed="$3"
  shift 3
  local log_file="$LOG_DIR/${name}.out"
  echo "[launch] gpu=${gpu} name=${name} seed=${seed}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export CONDA_ENV="${CONDA_ENV:-sparse_block}"
    EXPERIMENT=pubmed_hetero_xey_family_pw20 bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON_OVERRIDES[@]}" \
      "general.name=${name}" \
      "train.seed=${seed}" \
      "$@"
  ) >"$log_file" 2>&1 &
  echo "$!" >"$LOG_DIR/${name}.pid"
}

# family_y only enters y->E FiLM. This is closest to DiGress/SD's y-conditioned
# edge update path and is the recommended first comparison.
launch 0 hetero_xey_family_y_edgefilm_seed0 0 \
  model.use_family_y_in_edge_film=true
launch 1 hetero_xey_family_y_edgefilm_seed1 1 \
  model.use_family_y_in_edge_film=true

# family_y only enters the E->attention FiLM input. This tests whether family
# state should affect node message aggregation directly.
launch 2 hetero_xey_family_y_attention_seed0 0 \
  model.use_family_y_in_attention=true
launch 3 hetero_xey_family_y_attention_seed1 1 \
  model.use_family_y_in_attention=true

# family_y enters both paths. This tests whether the extra signal is helpful or
# duplicated/too strong.
launch 4 hetero_xey_family_y_both_seed0 0 \
  model.use_family_y_in_attention=true \
  model.use_family_y_in_edge_film=true
launch 5 hetero_xey_family_y_both_seed1 1 \
  model.use_family_y_in_attention=true \
  model.use_family_y_in_edge_film=true

# Recommended path plus the current soft degree-pair distribution auxiliary.
launch 6 hetero_xey_family_y_edgefilm_dpjs005_seed0 0 \
  model.use_family_y_in_edge_film=true \
  model.degree_pair_dist_loss_type=js \
  model.degree_pair_dist_loss_weight=0.05 \
  model.degree_pair_dist_loss_warmup_epochs=10
launch 7 hetero_xey_family_y_edgefilm_dpjs005_seed1 1 \
  model.use_family_y_in_edge_film=true \
  model.degree_pair_dist_loss_type=js \
  model.degree_pair_dist_loss_weight=0.05 \
  model.degree_pair_dist_loss_warmup_epochs=10

echo "[launch] all jobs submitted. Logs: $LOG_DIR"
wait
echo "[done] family-y sweep finished. Logs: $LOG_DIR"

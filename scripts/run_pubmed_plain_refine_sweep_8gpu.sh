#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_plain_refine_sweep_100ep}"
EPOCHS="${EPOCHS:-100}"
EDGE_FRACTION="${EDGE_FRACTION:-0.1}"
TEST_VARIANCE="${TEST_VARIANCE:-1}"
SAMPLE_SEEDS="${SAMPLE_SEEDS:-[0]}"
RUN_TEST_AFTER_TRAIN="${RUN_TEST_AFTER_TRAIN:-true}"
ENABLE_TEST_SAMPLING="${ENABLE_TEST_SAMPLING:-true}"
CHECK_VAL_EVERY="${CHECK_VAL_EVERY:-50}"

mkdir -p "$LOG_DIR"

launch() {
  local gpu="$1"
  local refine_tag="$2"
  local refine_balance="$3"
  local refine_iter="$4"
  local seed="$5"
  shift 5
  local name="plain_refine${refine_tag}_seed${seed}_ep${EPOCHS}"

  echo "[launch] gpu=${gpu} refine=${refine_tag} balance=${refine_balance} iter=${refine_iter} seed=${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  CONDA_ENV="${CONDA_ENV:-sparse_block}" \
  EXPERIMENT=pubmed_query_gate_pw20 \
    bash scripts/run_pubmed_baseline_single.sh \
      general.gpus=1 \
      general.name="$name" \
      general.resume=null \
      general.resume_full=false \
      general.run_test_after_train="$RUN_TEST_AFTER_TRAIN" \
      general.enable_test_sampling="$ENABLE_TEST_SAMPLING" \
      general.test_variance="$TEST_VARIANCE" \
      "general.test_sampling_seeds=${SAMPLE_SEEDS}" \
      general.test_sampling_full_steps=false \
      general.sampling_skip=25 \
      general.check_val_every_n_epochs="$CHECK_VAL_EVERY" \
      train.n_epochs="$EPOCHS" \
      train.seed="$seed" \
      model.edge_fraction="$EDGE_FRACTION" \
      model.hetero_metis_relation_balance_power=0.5 \
      model.hetero_metis_refine_degree_balance="$refine_balance" \
      model.hetero_metis_refine_max_iter="$refine_iter" \
      model.use_query_context_gate=false \
      model.use_two_hop_structure=false \
      model.use_typed_two_hop_structure=false \
      model.use_endpoint_role_residual=false \
      model.family_role_loss_weight=0 \
      model.sampling_ranking_intervention_diag=false \
      "$@" \
      >"$LOG_DIR/${name}.out" 2>&1 &
}

# Plain partition/refinement check:
# keep the training/sampling path simple and only sweep the degree-refinement
# strength. Later, once a partition setting wins, test two-hop/posterior on it.
launch 0 no_refine false 0 0
launch 1 no_refine false 0 1

launch 2 20 true 20 0
launch 3 20 true 20 1

launch 4 50 true 50 0
launch 5 50 true 50 1

launch 6 200 true 200 0
launch 7 200 true 200 1

wait
echo "[done] plain refine sweep finished; logs: $LOG_DIR"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_twohop_partition_preserve_100ep}"
EPOCHS="${EPOCHS:-100}"
EDGE_FRACTION="${EDGE_FRACTION:-0.1}"
POSTERIOR_SCALE="${POSTERIOR_SCALE:-0.5}"
TEST_VARIANCE="${TEST_VARIANCE:-3}"
SAMPLE_SEEDS="${SAMPLE_SEEDS:-[0,1,2]}"
CHECK_VAL_EVERY="${CHECK_VAL_EVERY:-50}"

mkdir -p "$LOG_DIR"

# Prepare PubMed once, then activate once in the launcher. Launching each child
# through run_pubmed_baseline_single.sh can silently stop after the sourced env
# helper when run under nohup/background shells on some machines.
RAW_ADJ="$ROOT/data/raw/PubMed_baseline/adj_matrix.p"
RAW_NODE="$ROOT/data/raw/PubMed_baseline/new_node_dict.p"
OUT_DIR="$ROOT/data/PubMed_baseline_subgraphs"
META="$OUT_DIR/subgraph_000/meta.json"
NODES="$OUT_DIR/subgraph_000/nodes.pt"
EDGES="$OUT_DIR/subgraph_000/edges.pt"

if [[ "${FORCE_CONVERT:-0}" == "1" || ! -f "$META" || ! -f "$NODES" || ! -f "$EDGES" || "$RAW_ADJ" -nt "$META" || "$RAW_NODE" -nt "$META" ]]; then
  echo "[convert] PubMed_baseline -> PubMed_baseline_subgraphs (full graph)"
  python dataprocess/scripts/convert_hgen_pubmed_baseline.py \
    --input-dir "$ROOT/data/raw/PubMed_baseline" \
    --out-dir "$OUT_DIR"
else
  echo "[convert] skip: PubMed_baseline_subgraphs already prepared (use FORCE_CONVERT=1 to refresh)"
fi

# shellcheck source=scripts/activate_env.sh
source "$ROOT/scripts/activate_env.sh"

launch() {
  local gpu="$1"
  local tag="$2"
  local refine_balance="$3"
  local refine_iter="$4"
  local preserve="$5"
  local penalty="$6"
  local seed="$7"
  shift 7

  local name="twohop_part_${tag}_seed${seed}_ep${EPOCHS}"
  echo "[launch] gpu=${gpu} tag=${tag} refine=${refine_balance}/${refine_iter} preserve=${preserve} penalty=${penalty} seed=${seed}"

  CUDA_VISIBLE_DEVICES="$gpu" \
    python -m dihug.main \
      +experiment=pubmed_query_gate_twohop_pw20 \
      general.wandb=disabled \
      general.gpus=1 \
      general.name="$name" \
      general.resume=null \
      general.resume_full=false \
      general.run_test_after_train=true \
      general.enable_test_sampling=true \
      general.test_variance="$TEST_VARIANCE" \
      "general.test_sampling_seeds=${SAMPLE_SEEDS}" \
      general.test_sampling_full_steps=false \
      'general.sampling_time_schedule=[100,86,71,57,43,29,14,0]' \
      general.check_val_every_n_epochs="$CHECK_VAL_EVERY" \
      general.profile_training_efficiency=false \
      train.n_epochs="$EPOCHS" \
      train.seed="$seed" \
      model.edge_fraction="$EDGE_FRACTION" \
      model.use_edge_state_update=false \
      model.hetero_metis_relation_balance_power=0.5 \
      model.hetero_metis_refine_degree_balance="$refine_balance" \
      model.hetero_metis_refine_max_iter="$refine_iter" \
      model.hetero_metis_refine_preserve_high_high="$preserve" \
      model.hetero_metis_refine_preserve_high_quantile=0.8 \
      model.hetero_metis_refine_preserve_penalty_weight="$penalty" \
      model.use_query_context_gate=false \
      model.use_two_hop_structure=true \
      model.use_typed_two_hop_structure=false \
      model.two_hop_structure_hidden_dim=64 \
      model.two_hop_structure_scale=0.5 \
      model.two_hop_structure_schedule=fixed \
      model.use_endpoint_role_residual=false \
      model.family_role_loss_weight=0 \
      model.sampling_edge_selection=gumbel_exact_k \
      model.sampling_exact_k_connectivity_repair=true \
      model.sampling_use_reverse_posterior=true \
      model.sampling_reverse_posterior_mix_weights=null \
      model.sampling_reverse_posterior_mix_mode=alpha_bar_s \
      "model.sampling_reverse_posterior_mix_scale=${POSTERIOR_SCALE}" \
      model.sampling_ranking_intervention_diag=true \
      "$@" \
      >"$LOG_DIR/${name}.out" 2>&1 &
}

# Same two-hop training mode as the previous best run, only the train-time
# hetero-METIS refinement policy changes.
launch 0 r200_base true 200 false 0.0 0
launch 1 r200_base true 200 false 0.0 1

launch 2 r20_mild true 20 false 0.0 0
launch 3 r20_mild true 20 false 0.0 1

launch 4 r50_preserve_w2 true 50 true 2.0 0
launch 5 r50_preserve_w2 true 50 true 2.0 1

launch 6 r200_preserve_w2 true 200 true 2.0 0
launch 7 r200_preserve_w2 true 200 true 2.0 1

wait
echo "[done] two-hop partition-preserving sweep finished; logs: $LOG_DIR"

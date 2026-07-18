#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="logs/logs_equivariance_layers12"
mkdir -p "$LOG_DIR"

CKPT="output/2026-06-20/22-43-34-pubmed_ef010_noedge_posterior_pw20_ep300/output/sparse_diffusion/checkpoints/pubmed_ef010_noedge_posterior_pw20_ep300/last.ckpt"
if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
fi

COMMON=(
  general.gpus=1
  "general.test_only=$CKPT"
  general.test_sampling_full_steps=false
  "general.sampling_time_schedule=[100,75,50,25,0]"
  general.enable_test_sampling=true
  general.test_variance=1
  general.verbose_sampling=false
  general.forward_equivariance_diag=true
  model.edge_fraction=0.1
  model.use_edge_state_update=false
  model.exist_pos_weight=20
  model.sampling_calibrate_exist_pos_weight=true
  model.sampling_block_marginal_init=true
  model.sampling_block_template_init=false
  model.sampling_use_reverse_posterior=true
  model.sampling_block_family_budget_projection=false
  model.sampling_edge_selection=deterministic_exact_k
  model.sampling_exact_k_connectivity_repair=false
)

launch() {
  local gpu="$1"
  local permutation_seed="$2"
  local exactk_diag="$3"
  local name="$4"

  CUDA_VISIBLE_DEVICES="$gpu" CONDA_ENV=sparse_block \
    nohup bash scripts/run_pubmed_baseline_single.sh \
      "${COMMON[@]}" \
      train.seed=0 \
      "general.forward_equivariance_permutation_seed=$permutation_seed" \
      "general.exactk_equivariance_diag=$exactk_diag" \
      "general.name=$name" \
      > "$LOG_DIR/${name}.out" 2>&1 &
  echo "GPU${gpu}: ${name}, PID=$!"
}

# Layer 1: paired pure forward on the same computation graph/query/time.
launch 0 1000 false layer1_forward_perm1000
launch 1 1001 false layer1_forward_perm1001
launch 2 1002 false layer1_forward_perm1002
launch 3 1003 false layer1_forward_perm1003

# Layer 2: same paired forward plus fixed-candidate deterministic exact-K.
launch 4 2000 true layer2_exactk_perm2000
launch 5 2001 true layer2_exactk_perm2001
launch 6 2002 true layer2_exactk_perm2002
launch 7 2003 true layer2_exactk_perm2003

echo "Logs: $ROOT/$LOG_DIR"

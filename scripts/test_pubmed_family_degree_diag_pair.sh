#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_family_degree_diag_pair}"
RUN_NAME="${RUN_NAME:-twohop_fixed_a0p5_seed1_ep100}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
mkdir -p "$LOG_DIR"

CKPT="$(find output/2026-06-23 \
  -path "*/output/sparse_diffusion/checkpoints/${RUN_NAME}/last.ckpt" \
  -type f -print -quit)"
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "Missing checkpoint for ${RUN_NAME}" >&2
  exit 1
fi

run_case() {
  local gpu="$1"
  local scale="$2"
  local tag="${scale//./p}"
  local name="familydiag_c${tag}_seed${SAMPLE_SEED}"

  CUDA_VISIBLE_DEVICES="$gpu" \
  CONDA_ENV="${CONDA_ENV:-sparse_block}" \
  EXPERIMENT=pubmed_query_gate_twohop_pw20 \
    bash scripts/run_pubmed_baseline_single.sh \
      general.gpus=1 \
      "general.test_only=$CKPT" \
      "general.name=$name" \
      general.run_test_after_train=false \
      general.enable_test_sampling=true \
      general.test_variance=1 \
      "general.test_sampling_seeds=[${SAMPLE_SEED}]" \
      general.test_sampling_full_steps=false \
      'general.sampling_time_schedule=[100,86,71,57,43,29,14,0]' \
      model.edge_fraction=0.1 \
      model.use_query_context_gate=false \
      model.use_two_hop_structure=true \
      model.two_hop_structure_hidden_dim=64 \
      model.two_hop_structure_scale=0.5 \
      model.two_hop_structure_schedule=fixed \
      model.sampling_use_reverse_posterior=true \
      model.sampling_reverse_posterior_mix_weights=null \
      model.sampling_reverse_posterior_mix_mode=alpha_bar_s \
      "model.sampling_reverse_posterior_mix_scale=$scale" \
      model.sampling_ranking_intervention_diag=true \
      >"$LOG_DIR/${name}.out" 2>&1
}

run_case 0 0.5 &
run_case 1 0.65 &
wait

run_05="$(find output -maxdepth 2 -type d \
  -name "*-familydiag_c0p5_seed${SAMPLE_SEED}" | sort | tail -n 1)"
run_065="$(find output -maxdepth 2 -type d \
  -name "*-familydiag_c0p65_seed${SAMPLE_SEED}" | sort | tail -n 1)"
if [[ -z "$run_05" || -z "$run_065" ]]; then
  echo "Could not locate paired Hydra output directories" >&2
  exit 1
fi

DIAG_DIR="${DIAG_DIR:-output/family_degree_diag_seed${SAMPLE_SEED}}"
ENV_NAME="${CONDA_ENV:-sparse_block}"
ENV_PREFIX="${CONDA_PREFIX:-/data2/lyh/miniconda3/envs/$ENV_NAME}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" scripts/analyze_family_degree_roles.py \
    --generated "c0p5=$run_05/generated_samples.pkl" \
    --generated "c0p65=$run_065/generated_samples.pkl" \
    --intervention "c0p5=$run_05/posterior_intervention.jsonl" \
    --intervention "c0p65=$run_065/posterior_intervention.jsonl" \
    --output-dir "$DIAG_DIR"

echo "[done] paired family-degree diagnostics written to $DIAG_DIR"

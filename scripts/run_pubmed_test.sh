#!/usr/bin/env bash
# PubMed 单图：从 checkpoint 仅做测试采样（不训练）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/activate_env.sh
source "$ROOT/scripts/activate_env.sh"

CKPT="${1:-}"
if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  echo "Usage: $0 /path/to/last.ckpt [hydra overrides...]" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m dihug.main +experiment=pubmed_single_train \
  "general.test_only=$CKPT" \
  general.enable_test_sampling=true \
  general.test_variance=3 \
  general.sampling_skip=5 \
  general.gpus=1 \
  general.run_test_after_train=false \
  general.wandb=disabled \
  "${@:2}"

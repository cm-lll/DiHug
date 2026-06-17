#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/activate_env.sh
source "$ROOT/scripts/activate_env.sh"
python -m dihug.main +experiment=imdb_single_train general.wandb=disabled "$@"

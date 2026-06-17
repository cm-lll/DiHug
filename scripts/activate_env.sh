#!/usr/bin/env bash
# Activate the Python environment used by DiHuG launcher scripts.
#
# Override when needed:
#   CONDA_ENV=my_env bash scripts/run_dblp_single.sh
#   CONDA_BASE=/path/to/miniconda3 bash scripts/run_pubmed_single.sh
#   DIHUG_SKIP_CONDA=1 bash scripts/run_pubmed_single.sh

if [[ "${DIHUG_SKIP_CONDA:-0}" == "1" ]]; then
  return 0 2>/dev/null || exit 0
fi

CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-dihug}}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -n "${CONDA_BASE:-}" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  :
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  CONDA_BASE="$HOME/miniconda3"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  CONDA_BASE="$HOME/anaconda3"
else
  echo "Could not find conda. Set CONDA_BASE or activate the environment manually." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" == "$CONDA_ENV" ]]; then
  :
elif ! conda activate "$CONDA_ENV"; then
  echo "Could not activate conda environment '$CONDA_ENV'." >&2
  echo "Create one first, pass CONDA_ENV=<name>, or set DIHUG_SKIP_CONDA=1 after activating manually." >&2
  exit 1
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

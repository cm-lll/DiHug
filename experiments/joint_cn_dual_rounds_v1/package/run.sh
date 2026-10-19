#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON="${PYTHON:-/usr/local/bin/python3.11}"
exec "$PYTHON" -u -m dual.cli "$@"

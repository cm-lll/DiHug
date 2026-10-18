#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
test -s SHA256SUMS || { echo 'Missing SHA256SUMS; refusing incomplete package' >&2; exit 2; }
sha256sum --quiet -c SHA256SUMS
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${PYTHON_BIN:-/usr/local/bin/python3.11}" -u run.py "$@"

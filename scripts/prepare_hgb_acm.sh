#!/usr/bin/env bash
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

RAW_DIR="${RAW_DIR:-data/ACM_subgraphs/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-data/ACM_subgraphs/subgraph_000}"
FILE_ID="${FILE_ID:-1NVT_IHhPDS8dwMmsrnTRHj90F7OZu0WY}"
EXPECTED_SHA256="${EXPECTED_SHA256:-0ccd838e545e8f16e3dc84356da2f51dfd2290c32e37a784e374dd510c76578d}"
mkdir -p "$RAW_DIR"

if [[ ! -f "$RAW_DIR/ACM.mat" ]]; then
  python -m gdown "$FILE_ID" -O "$RAW_DIR/ACM.mat"
fi
echo "$EXPECTED_SHA256  $RAW_DIR/ACM.mat" | sha256sum -c -
python scripts/prepare_hgb_acm.py \
  --mat "$RAW_DIR/ACM.mat" \
  --output "$OUTPUT_DIR" \
  --expected-sha256 "$EXPECTED_SHA256"

python - "$OUTPUT_DIR" <<'PY'
import json, sys
from pathlib import Path
from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics

path = Path(sys.argv[1])
graph = load_typed_graph(path, active_only=True)
graph.validate_disjoint_family_domains()
print(json.dumps({
    "types": graph.type_names,
    "nodes": graph.num_nodes,
    "families": {name: len(edges) for name, edges in graph.family_edges.items()},
    "metrics": graph_metrics(graph),
}, indent=2, sort_keys=True))
PY

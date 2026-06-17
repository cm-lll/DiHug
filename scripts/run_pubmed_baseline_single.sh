#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 1) 转换 HGEN baseline -> DiHuG 全图 subgraph 格式。
# 默认只在缺失或 raw 数据更新时转换；converter 会清空 processed 缓存，
# 所以不要在每次训练前无条件运行。需要强制刷新时：
#   FORCE_CONVERT=1 bash scripts/run_pubmed_baseline_single.sh ...
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
python -m dihug.main +experiment=pubmed_baseline_single_train general.wandb=disabled "$@"

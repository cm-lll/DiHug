#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
SOURCE_URL="${HGEN_DATA_URL:-https://www.dropbox.com/sh/lmryy7r4la3owgj/AAB2eKhCp3UUEP5Nb8foJxUla?dl=1}"
EXPECTED_SHA256="${HGEN_DATA_SHA256:-f0b1ee615e7bcdfc667ec52aa11a3dae7644fef19c2f903dac6a79015f01e2ab}"
CACHE_ROOT="${HGEN_CACHE_ROOT:-data/external/hgen_official}"
OUTPUT_ROOT="${HGEN_OUTPUT_ROOT:-data}"
ARCHIVE="$CACHE_ROOT/hgen_official_data.zip"
EXTRACTED="$CACHE_ROOT/extracted"
mkdir -p "$CACHE_ROOT"

if [[ ! -f "$ARCHIVE" ]]; then
  curl -L --fail --retry 3 --retry-delay 2 "$SOURCE_URL" -o "$ARCHIVE"
fi

actual_sha256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "HGEN archive checksum mismatch: $actual_sha256" >&2
  exit 1
fi

rm -rf "$EXTRACTED"
mkdir -p "$EXTRACTED"
"$PYTHON" - "$ARCHIVE" "$EXTRACTED" <<'PY'
import pathlib
import shutil
import sys
import zipfile

archive = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2]).resolve()
with zipfile.ZipFile(archive) as source:
    for member in source.infolist():
        relative = pathlib.PurePosixPath(member.filename.lstrip("/"))
        if not relative.parts:
            continue
        target = destination.joinpath(*relative.parts).resolve()
        if destination not in target.parents:
            raise RuntimeError(f"unsafe ZIP member: {member.filename}")
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open(member) as reader, target.open("wb") as writer:
            shutil.copyfileobj(reader, writer)
PY

convert() {
  local source="$1" output="$2" name="$3"
  shift 3
  rm -rf "$output"
  "$PYTHON" dataprocess/scripts/convert_hgen_graph_to_subgraphs.py \
    --input-dir "$EXTRACTED/$source" --out-dir "$output" \
    --dataset-name "$name" "$@"
}

convert DBLP_four_area "$OUTPUT_ROOT/DBLP_four_area_subgraphs" DBLP_four_area \
  --preset dblp_four_area
convert syn_100 "$OUTPUT_ROOT/HGEN_syn_100_subgraphs" HGEN_syn_100 \
  --node-types A,B,C
convert syn_200 "$OUTPUT_ROOT/HGEN_syn_200_subgraphs" HGEN_syn_200 \
  --node-types A,B,C
convert syn_500 "$OUTPUT_ROOT/HGEN_syn_500_subgraphs" HGEN_syn_500 \
  --node-types A,B,C

HGEN_OUTPUT_ROOT="$OUTPUT_ROOT" "$PYTHON" - <<'PY'
import os
from pathlib import Path

from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics

root = Path(os.environ["HGEN_OUTPUT_ROOT"])
datasets = {
    "DBLP_four_area": root / "DBLP_four_area_subgraphs/subgraph_000",
    "syn_100": root / "HGEN_syn_100_subgraphs/subgraph_000",
    "syn_200": root / "HGEN_syn_200_subgraphs/subgraph_000",
    "syn_500": root / "HGEN_syn_500_subgraphs/subgraph_000",
}
for name, path in datasets.items():
    graph = load_typed_graph(path, active_only=True)
    graph.validate()
    metrics = graph_metrics(graph)
    print(
        f"[HGEN-OFFICIAL] {name}: nodes={graph.num_nodes} "
        f"edges={graph.num_edges} triangles={metrics['triangles']:.0f} "
        f"clustering={metrics['clustering']:.6f}"
    )
PY

echo "[HGEN-OFFICIAL] prepared and verified (sha256=$actual_sha256)"

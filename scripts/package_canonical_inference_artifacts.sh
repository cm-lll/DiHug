#!/usr/bin/env bash
# Package the two self-contained paper checkpoints for manual cross-machine transfer.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT="${OUTPUT:-paper_tables/canonical_inference_artifacts_189fb51.tar.gz}"
IMDB_CONTEXT="${IMDB_CONTEXT:-results/rooted_context_freeze/imdb/rooted_context_residual.pt}"
PUBMED_CONTEXT="${PUBMED_CONTEXT:-results/rooted_context_freeze/pubmed/rooted_context_residual.pt}"
EXPECTED_IMDB_SHA256="1bcbc727106d2caed86a69a81296669ed63ead1dcc3cbcd527f9cfe0288fed52"
EXPECTED_PUBMED_SHA256="2d1e4f009902494fae86a3c6bcb8466f142f8598e3ba7d1a629d30eecc534249"

for required in "$IMDB_CONTEXT" "$PUBMED_CONTEXT"; do
  [[ -f "$required" ]] || { echo "missing canonical artifact: $required" >&2; exit 2; }
done

imdb_sha="$(sha256sum "$IMDB_CONTEXT" | awk '{print $1}')"
pubmed_sha="$(sha256sum "$PUBMED_CONTEXT" | awk '{print $1}')"
[[ "$imdb_sha" == "$EXPECTED_IMDB_SHA256" ]] || { echo "IMDB artifact hash mismatch" >&2; exit 3; }
[[ "$pubmed_sha" == "$EXPECTED_PUBMED_SHA256" ]] || { echo "PubMed artifact hash mismatch" >&2; exit 3; }

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/canonical_inference_artifacts" "$(dirname "$OUTPUT")"
cp "$IMDB_CONTEXT" "$stage/canonical_inference_artifacts/imdb_context.pt"
cp "$PUBMED_CONTEXT" "$stage/canonical_inference_artifacts/pubmed_context.pt"

python - "$stage/canonical_inference_artifacts/manifest.json" \
  "$imdb_sha" "$pubmed_sha" "$(git rev-parse HEAD)" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "format": "dihug-canonical-inference-artifacts-v1",
    "source_commit": sys.argv[4],
    "artifacts": {
        "imdb_context.pt": {"sha256": sys.argv[2]},
        "pubmed_context.pt": {"sha256": sys.argv[3]},
    },
    "data_files": {
        "data/IMDB_movie_subgraphs/subgraph_000/edges.pt": "7d372b11d6bea2ba6f1d68b2d6ab936e8ebb8fe9ded57c8ea70f9314e1bfd2ec",
        "data/IMDB_movie_subgraphs/subgraph_000/nodes.pt": "c5204a8aab7255e6d5b6cfa9dc7096dae4773645c562f8cdf45c9afe6a66b03e",
        "data/IMDB_movie_subgraphs/processed/vocab.json": "845584bbcafedf5d9ffcd9fe968ea0ac584857eba07bd3b6053bde57f4ad80c9",
        "data/PubMed_baseline_subgraphs/subgraph_000/edges.pt": "0255bfc987601975f2b52f7cd7808bea16bee0db90c20dde595447e5f7201b27",
        "data/PubMed_baseline_subgraphs/subgraph_000/nodes.pt": "9d39e366e88d8d1b05ec5f2c5a8e0486be4e2a4866f9b8ff9012b15ecf2c0f30",
        "data/PubMed_baseline_subgraphs/processed/vocab.json": "537500aa0ee1024bdfcfe04de69335a6a9c74f04bb4afb18ed55f00bae717783",
    },
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

tar -C "$stage" -czf "$OUTPUT" canonical_inference_artifacts
sha256sum "$OUTPUT" | tee "${OUTPUT}.sha256"
echo "[CANONICAL-PACKAGE] wrote $OUTPUT"

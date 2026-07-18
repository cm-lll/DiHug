#!/usr/bin/env bash
# Verify a manually transferred canonical bundle, then run inference only.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BUNDLE="${BUNDLE:-paper_tables/canonical_inference_artifacts_189fb51.tar.gz}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-results/canonical_inference_artifacts}"
MACHINE_TAG="${MACHINE_TAG:-remote_canonical_audit}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-3}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
OUT_ROOT="${OUT_ROOT:-results/canonical_inference_audit_${MACHINE_TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/canonical_inference_audit_${MACHINE_TAG}}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"

[[ -f "$BUNDLE" ]] || { echo "missing transferred bundle: $BUNDLE" >&2; exit 2; }
rm -rf "$ARTIFACT_ROOT"
mkdir -p "$(dirname "$ARTIFACT_ROOT")" "$LOG_ROOT"
tar -xzf "$BUNDLE" -C "$(dirname "$ARTIFACT_ROOT")"
extracted_root="$(dirname "$ARTIFACT_ROOT")/canonical_inference_artifacts"
if [[ "$extracted_root" != "$ARTIFACT_ROOT" ]]; then
  mv "$extracted_root" "$ARTIFACT_ROOT"
fi

IMDB_CONTEXT="$ARTIFACT_ROOT/imdb_context.pt"
PUBMED_CONTEXT="$ARTIFACT_ROOT/pubmed_context.pt"
MANIFEST="$ARTIFACT_ROOT/manifest.json"
for required in "$IMDB_CONTEXT" "$PUBMED_CONTEXT" "$MANIFEST"; do
  [[ -f "$required" ]] || { echo "bundle is missing: $required" >&2; exit 2; }
done

python - "$MANIFEST" "$IMDB_CONTEXT" "$PUBMED_CONTEXT" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
if manifest.get("format") != "dihug-canonical-inference-artifacts-v1":
    raise RuntimeError("unexpected canonical artifact manifest format")
for name, path_text in zip(("imdb_context.pt", "pubmed_context.pt"), sys.argv[2:]):
    actual = hashlib.sha256(pathlib.Path(path_text).read_bytes()).hexdigest()
    expected = manifest["artifacts"][name]["sha256"]
    if actual != expected:
        raise RuntimeError(f"artifact SHA256 mismatch for {name}: {actual} != {expected}")
for path_text, expected in manifest["data_files"].items():
    path = pathlib.Path(path_text)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"data SHA256 mismatch for {path}: {actual} != {expected}")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

if [[ "$VERIFY_ONLY" == "1" ]]; then
  echo "[CANONICAL-AUDIT] artifact verification passed"
  exit 0
fi

env IMDB_CONTEXT="$IMDB_CONTEXT" PUBMED_CONTEXT="$PUBMED_CONTEXT" \
  MACHINE_TAG="$MACHINE_TAG" SEED_START="$SEED_START" SEED_END="$SEED_END" \
  GPU_LIST="$GPU_LIST" FAMILY_COUNT_MODE=binomial \
  OUT_ROOT="$OUT_ROOT" LOG_ROOT="$LOG_ROOT" \
  bash scripts/run_paper_final_context_main_8gpu.sh

cp "$MANIFEST" "$OUT_ROOT/artifact_manifest.json"
echo "[CANONICAL-AUDIT] complete: $OUT_ROOT/aggregate.json"

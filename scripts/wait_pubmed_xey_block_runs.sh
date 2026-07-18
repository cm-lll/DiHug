#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-logs/logs_pubmed_xey_structural_block_vs_accumulate_8gpu}"
POLL_SECONDS="${POLL_SECONDS:-60}"
BLOCK_NAMES="${BLOCK_NAMES:-xey_struct_block_bce xey_struct_block_dpjs xey_struct_block_closure_retry xey_struct_block_combo_mild_retry}"
RUN_AFTER="${RUN_AFTER:-}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

is_running() {
  local name="$1"
  ps -u "${USER}" -o args= | grep -F "python -m dihug.main" | grep -F "general.name=${name}" >/dev/null 2>&1
}

last_epoch() {
  local log_file="$1"
  if [[ -f "$log_file" ]]; then
    grep -E "Epoch [0-9]+ finished" "$log_file" | tail -n 1 || true
  fi
}

has_failure() {
  local log_file="$1"
  [[ -f "$log_file" ]] && grep -E "Training crashed|OutOfMemory|Error executing job|Traceback" "$log_file" >/dev/null 2>&1
}

echo "[wait-block] $(date '+%F %T') log_dir=${LOG_DIR}"
echo "[wait-block] names=${BLOCK_NAMES}"

while true; do
  running=0
  for name in ${BLOCK_NAMES}; do
    if is_running "$name"; then
      running=$((running + 1))
    fi
  done

  echo "[wait-block] $(date '+%F %T') running=${running}"
  for name in ${BLOCK_NAMES}; do
    log_file="${LOG_DIR}/${name}.out"
    status="stopped"
    if is_running "$name"; then
      status="running"
    fi
    latest="$(last_epoch "$log_file")"
    echo "  - ${name}: ${status}${latest:+ | ${latest}}"
  done

  if [[ "$running" -eq 0 ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

failed=0
for name in ${BLOCK_NAMES}; do
  log_file="${LOG_DIR}/${name}.out"
  if has_failure "$log_file"; then
    echo "[wait-block][fail] ${name}: failure pattern found in ${log_file}"
    failed=1
  else
    echo "[wait-block][ok] ${name}: no failure pattern found"
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "[wait-block] block runs finished with failures; not running RUN_AFTER"
  exit "$failed"
fi

echo "[wait-block] all block runs finished cleanly"
if [[ -n "$RUN_AFTER" ]]; then
  echo "[wait-block] running RUN_AFTER: ${RUN_AFTER}"
  bash -lc "$RUN_AFTER"
fi

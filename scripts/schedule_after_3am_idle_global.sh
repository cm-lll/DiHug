#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-logs/logs_pubmed_after_3am_idle_global_scheduler}"
mkdir -p "$LOG_DIR"

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
START_AFTER="${START_AFTER:-03:00}"
POLL_SECONDS="${POLL_SECONDS:-600}"
IDLE_MEM_MB="${IDLE_MEM_MB:-700}"
IDLE_UTIL_PCT="${IDLE_UTIL_PCT:-5}"
START_CMD="${START_CMD:-bash scripts/run_pubmed_xey_global_structural_8gpu.sh}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/scheduler.log"
}

seconds_now() {
  date '+%H:%M:%S' | awk -F: '{print $1 * 3600 + $2 * 60 + $3}'
}

seconds_for_hm() {
  local hm="$1"
  awk -F: '{print $1 * 3600 + $2 * 60}' <<<"$hm"
}

sleep_until_start_after() {
  local now target wait_s
  now="$(seconds_now)"
  target="$(seconds_for_hm "$START_AFTER")"
  if (( now < target )); then
    wait_s=$((target - now))
    log "waiting ${wait_s}s until ${START_AFTER}"
    sleep "$wait_s"
  else
    log "current time already past ${START_AFTER}"
  fi
}

gpu_idle() {
  local gpu="$1"
  local line util mem
  line="$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
    | awk -F, -v g="$gpu" '$1 + 0 == g {gsub(/ /, ""); print $2" "$3}')"
  [[ -n "$line" ]] || return 1
  util="$(awk '{print $1}' <<<"$line")"
  mem="$(awk '{print $2}' <<<"$line")"
  [[ "$util" -le "$IDLE_UTIL_PCT" && "$mem" -le "$IDLE_MEM_MB" ]]
}

all_gpus_idle() {
  local gpu
  for gpu in "${GPUS[@]}"; do
    gpu_idle "$gpu" || return 1
  done
  return 0
}

gpu_snapshot() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits \
    | awk '{print "[gpu] "$0}' | tee -a "$LOG_DIR/scheduler.log" >/dev/null
}

log "scheduler started; start_after=${START_AFTER}, poll=${POLL_SECONDS}s, idle_mem<=${IDLE_MEM_MB}MB, idle_util<=${IDLE_UTIL_PCT}%, gpus=${GPUS[*]}"
log "start_cmd=${START_CMD}"

sleep_until_start_after

while true; do
  if all_gpus_idle; then
    log "all target GPUs idle; launching global command"
    gpu_snapshot
    (
      set +e
      eval "$START_CMD"
      rc=$?
      echo "GLOBAL_TASK_EXIT=$rc"
      exit "$rc"
    ) >"$LOG_DIR/global_launcher.out" 2>&1 &
    echo "$!" >"$LOG_DIR/global_launcher.pid"
    log "global launcher pid=$(cat "$LOG_DIR/global_launcher.pid"), log=$LOG_DIR/global_launcher.out"
    exit 0
  fi
  log "GPUs not idle yet; next check in ${POLL_SECONDS}s"
  gpu_snapshot
  sleep "$POLL_SECONDS"
done

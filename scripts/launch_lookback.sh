#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
mkdir -p logs
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="logs/lookback_${stamp}.log"
runner="logs/lookback_${stamp}_runner.sh"
cp scripts/run_lookback_queue.sh "$runner"
export LOOKBACK_STOP_FILE="logs/lookback_${stamp}.exit" LOOKBACK_SYNC_LOG="logs/hf_sync_lookback_${stamp}.log"
nohup setsid bash "$runner" > "$log" 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/lookback.pid
printf '%s\n' "$log" > logs/lookback_latest_log.txt
printf 'PID %s; log %s; exit %s\n' "$!" "$log" "$LOOKBACK_STOP_FILE"

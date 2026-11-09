#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
mkdir -p logs
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="logs/final_reference_${stamp}.log"
runner="logs/final_reference_${stamp}_runner.sh"
cp scripts/run_final_reference_queue.sh "$runner"
export REFERENCE_STOP_FILE="logs/final_reference_${stamp}.exit" REFERENCE_SYNC_LOG="logs/hf_sync_final_reference_${stamp}.log"
nohup setsid bash "$runner" > "$log" 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/final_reference.pid
printf '%s\n' "$log" > logs/final_reference_latest_log.txt
printf 'PID %s; log %s; exit %s\n' "$!" "$log" "$REFERENCE_STOP_FILE"

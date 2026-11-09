#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
mkdir -p logs
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="logs/model_updates_${stamp}.log"
runner="logs/model_updates_${stamp}_runner.sh"
cp scripts/run_model_updates_queue.sh "$runner"
export UPDATE_STOP_FILE="logs/model_updates_${stamp}.exit" UPDATE_SYNC_LOG="logs/hf_sync_model_updates_${stamp}.log"
nohup bash "$runner" > "$log" 2>&1 < /dev/null &
echo "$!" > logs/model_updates_queue.pid
printf '%s\n' "$log" > logs/model_updates_latest_log.txt
printf 'PID: %s\nLog: %s/%s\nExit: %s/%s\n' "$(cat logs/model_updates_queue.pid)" "$PWD" "$log" "$PWD" "$UPDATE_STOP_FILE"

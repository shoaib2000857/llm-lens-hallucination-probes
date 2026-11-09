#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
mkdir -p logs
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="logs/recovery_${stamp}.log"
runner="logs/recovery_${stamp}_runner.sh"
cp scripts/run_recovery_queue.sh "$runner"
export RECOVERY_STOP_FILE="logs/recovery_${stamp}.exit" RECOVERY_SYNC_LOG="logs/hf_sync_recovery_${stamp}.log"
nohup bash "$runner" > "$log" 2>&1 < /dev/null &
echo "$!" > logs/recovery_queue.pid
printf '%s\n' "$log" > logs/recovery_latest_log.txt
printf 'PID: %s\nLog: %s/%s\nExit: %s/%s\n' "$(cat logs/recovery_queue.pid)" "$PWD" "$log" "$PWD" "$RECOVERY_STOP_FILE"

#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
mkdir -p logs
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="logs/replication_${stamp}.log"
runner="logs/replication_${stamp}_runner.sh"
cp scripts/run_trajectory_replication_queue.sh "$runner"
export REPLICATION_STOP_FILE="logs/replication_${stamp}.exit" REPLICATION_SYNC_LOG="logs/hf_sync_replication_${stamp}.log"
nohup setsid bash "$runner" > "$log" 2>&1 < /dev/null &
echo "$!" > logs/replication_queue.pid
printf '%s\n' "$log" > logs/replication_latest_log.txt
printf 'PID: %s\nLog: %s/%s\nExit: %s/%s\n' "$(cat logs/replication_queue.pid)" "$PWD" "$log" "$PWD" "$REPLICATION_STOP_FILE"

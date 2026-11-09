#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
mkdir -p logs
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="logs/distribution_${stamp}.log"
runner="logs/distribution_${stamp}_runner.sh"
cp scripts/run_distribution_queue.sh "$runner"
export ROUND_STOP_FILE="logs/distribution_${stamp}.exit" ROUND_SYNC_LOG="logs/hf_sync_distribution_${stamp}.log"
nohup bash "$runner" > "$log" 2>&1 < /dev/null &
echo "$!" > logs/distribution_queue.pid
printf '%s\n' "$log" > logs/distribution_latest_log.txt
printf 'PID: %s\nLog: %s/%s\nExit status: %s/%s\n' "$(cat logs/distribution_queue.pid)" "$PWD" "$log" "$PWD" "$ROUND_STOP_FILE"

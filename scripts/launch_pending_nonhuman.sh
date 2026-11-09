#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
mkdir -p logs
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="logs/pending_nonhuman_${stamp}.log"
runner="logs/pending_nonhuman_${stamp}_runner.sh"
cp scripts/run_pending_nonhuman_queue.sh "$runner"
export PENDING_STOP_FILE="logs/pending_nonhuman_${stamp}.exit" PENDING_SYNC_LOG="logs/hf_sync_pending_nonhuman_${stamp}.log"
nohup setsid bash "$runner" > "$log" 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/pending_nonhuman.pid
printf '%s\n' "$log" > logs/pending_nonhuman_latest_log.txt
printf 'PID %s; log %s\n' "$!" "$log"

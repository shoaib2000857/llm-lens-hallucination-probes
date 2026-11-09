#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
mkdir -p logs
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="logs/reviewer_closure_${stamp}.log"
runner="logs/reviewer_closure_${stamp}_runner.sh"
cp scripts/run_reviewer_closure_queue.sh "$runner"
export CLOSURE_STOP_FILE="logs/reviewer_closure_${stamp}.exit" CLOSURE_SYNC_LOG="logs/hf_sync_reviewer_closure_${stamp}.log"
nohup setsid bash "$runner" > "$log" 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/reviewer_closure.pid
printf '%s\n' "$log" > logs/reviewer_closure_latest_log.txt
printf 'PID %s; log %s; exit %s\n' "$!" "$log" "$CLOSURE_STOP_FILE"

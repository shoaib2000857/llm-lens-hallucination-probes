#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
mkdir -p logs
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="logs/final_nonhuman_${stamp}.log"
cp scripts/run_final_nonhuman_queue.sh "logs/final_nonhuman_${stamp}_runner.sh"
export FINAL_NONHUMAN_STOP="logs/final_nonhuman_${stamp}.exit" FINAL_NONHUMAN_SYNC_LOG="logs/hf_sync_final_nonhuman_${stamp}.log"
nohup setsid bash "logs/final_nonhuman_${stamp}_runner.sh" > "$log" 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/final_nonhuman.pid
printf '%s\n' "$log" > logs/final_nonhuman_latest_log.txt
printf 'PID%s;log%s\n' "$!" "$log"

#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
mkdir -p logs
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="logs/remaining_baselines_${stamp}.log"
cp scripts/run_remaining_baselines_queue.sh "logs/remaining_baselines_${stamp}_runner.sh"
export BASELINES_STOP_FILE="logs/remaining_baselines_${stamp}.exit" BASELINES_SYNC_LOG="logs/hf_sync_remaining_baselines_${stamp}.log"
nohup setsid bash "logs/remaining_baselines_${stamp}_runner.sh" > "$log" 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/remaining_baselines.pid
printf '%s\n' "$log" > logs/remaining_baselines_latest_log.txt
printf 'PID %s; log %s\n' "$!" "$log"

#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
mkdir -p logs
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="logs/resilient_judging_${stamp}.log"
cp scripts/run_resilient_judging_queue.sh "logs/resilient_judging_${stamp}_runner.sh"
export RESILIENT_STOP_FILE="logs/resilient_judging_${stamp}.exit" RESILIENT_SYNC_LOG="logs/hf_sync_resilient_judging_${stamp}.log"
nohup setsid bash "logs/resilient_judging_${stamp}_runner.sh" > "$log" 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/resilient_judging.pid
printf '%s\n' "$log" > logs/resilient_judging_latest_log.txt
printf 'PID%s; log%s\n' "$!" "$log"

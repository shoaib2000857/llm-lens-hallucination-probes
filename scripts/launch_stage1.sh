#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="logs/stage1_${stamp}.log"
nohup bash scripts/run_stage1_queue.sh > "$log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > logs/stage1_queue.pid
echo "$log" > logs/stage1_latest_log.txt
echo "PID: $pid"
echo "Log: $PWD/$log"
echo "Watch: tail -f '$PWD/$log'"
echo 'Resume after interruption: bash scripts/launch_stage1.sh'

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
log="logs/stage2_$(date -u +%Y%m%dT%H%M%SZ).log"
runner="${log%.log}_runner.sh"
cp scripts/run_stage2_queue.sh "$runner"
nohup bash "$runner" > "$log" 2>&1 < /dev/null &
echo "$!" > logs/stage2_queue.pid
printf '%s\n' "$log" > logs/stage2_latest_log.txt
printf 'PID: %s\nLog: %s/%s\n' "$(cat logs/stage2_queue.pid)" "$PWD" "$log"

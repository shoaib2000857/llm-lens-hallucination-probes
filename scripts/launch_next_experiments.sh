#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
stamp=$(date -u +%Y%m%dT%H%M%SZ)
runner="logs/next_${stamp}_runner.sh"
# Snapshot shell stages so later edits do not corrupt a running shell's read offset.
cp scripts/run_stage3_queue.sh "logs/stage3_${stamp}_runner.sh"
cp scripts/run_stage4_queue.sh "logs/stage4_${stamp}_runner.sh"
printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\nbash %q\nif [ "${FULL_ARCHITECTURES:-1}" = 1 ]; then bash %q; fi\n' "$PWD" "logs/stage3_${stamp}_runner.sh" "logs/stage4_${stamp}_runner.sh" > "$runner"
log="logs/next_${stamp}.log"
nohup bash "$runner" > "$log" 2>&1 < /dev/null &
echo "$!" > logs/next_experiments.pid
printf '%s\n' "$log" > logs/next_experiments_latest_log.txt
printf 'Queued PID: %s\nLog: %s/%s\n' "$(cat logs/next_experiments.pid)" "$PWD" "$log"

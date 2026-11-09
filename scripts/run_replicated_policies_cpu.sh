#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
echo 'RESOURCE PLAN: CPU-only simulations on existing weights/features, four CPU threads,zero workers,RAM target<8GiB,reserve1GiB prediction caches; no generator inference or new training. Replay kernels/checksums fixed. Two replicated generator trajectories,three predeclared drift thresholds,90 simulations/configuration; cache reused across thresholds. Thresholds are not selected using test labels.'
failures=0
for seed in 43 44;do
 for threshold in 0.025 0.05 0.10;do
  if ! .venv/bin/python scripts/simulate_probe_updates.py --trajectory-prefix "updates_v1r$seed" --replay "artifacts/model_updates_v1/replay_s$seed" --output "results/replicated_policy_v1/s$seed/drift_$threshold" --cache-root "results/replicated_policy_v1/s$seed/cache" --drift-threshold "$threshold" --device cpu --threads 4;then failures=$((failures+1));echo "Policy seed$seed threshold$threshold failed; independent settings continue.";fi
 done
done
printf '%s\n' "$failures" > logs/replicated_policies_failures.txt
echo "Replicated CPU policies processed;failures:$failures"
if [ "$failures" -gt 0 ];then exit 2;fi

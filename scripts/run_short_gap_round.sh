#!/usr/bin/env bash
set -uo pipefail
cd /home/jovyan/hallucination-portability
source ~/.bashrc >/dev/null 2>&1 || true
export HF_HOME="$PWD/artifacts/hf_cache" TMPDIR="$PWD/artifacts/tmp" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
mkdir -p logs
exec 9>artifacts/gpu_stage.lock
flock 9
status=0
.venv/bin/python scripts/run_sampling_disagreement.py --samples 2 --draws 2 --pilot || status=$?
if [ "$status" -eq 0 ];then
 .venv/bin/python scripts/run_sampling_disagreement.py --samples 200 --draws 5 || status=$?
fi
printf '%s\n' "$status" > logs/short_gap_round.exit
.venv/bin/python scripts/hf_sync.py --help >/dev/null 2>&1 || true
exit "$status"

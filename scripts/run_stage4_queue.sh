#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
set +u
source "$HOME/.bashrc"
set -u
export HF_HOME="$PWD/artifacts/hf_cache" TMPDIR="$PWD/artifacts/tmp"
export OMP_NUM_THREADS=${CPU_THREADS:-8} MKL_NUM_THREADS=${CPU_THREADS:-8} OPENBLAS_NUM_THREADS=${CPU_THREADS:-8} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec 9>artifacts/gpu_stage.lock
flock 9
trap 'code=$?; echo "$code" > logs/stage4_queue.exit' EXIT
rm -f logs/stage4_queue.exit
echo 'RESOURCE PLAN: full-token M2/M3/M4; one GPU process, batch2, estimated <30 GiB VRAM and <16 GiB host working memory; original ~359 GiB corpus streamed with bounded cache. Repeated disk reads can dominate. Allow potentially >12 hours; inspect pilot timing. No LLM weights loaded.'
.venv/bin/python scripts/hf_sync.py --watch --interval 120 --stop-file logs/stage4_queue.exit > logs/hf_sync_stage4.log 2>&1 9>&- &
.venv/bin/python scripts/probe_gpu_preflight.py --raw --batch "${RAW_BATCH_SIZE:-2}"
# Full-label matched scratch/warm and frozen/depth-mapped transfer first.
# Dense small-label curves run in stage3 on the cheaper probes.
.venv/bin/python scripts/run_probe_suite.py --probes M2 M3 M4 --budgets full --threads "${CPU_THREADS:-8}" --raw-batch "${RAW_BATCH_SIZE:-2}"
.venv/bin/python scripts/summarize_probe_results.py

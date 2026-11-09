#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
set +u
source "$HOME/.bashrc"
set -u
export HF_HOME="$PWD/artifacts/hf_cache" TMPDIR="$PWD/artifacts/tmp"
export OMP_NUM_THREADS=${CPU_THREADS:-8} MKL_NUM_THREADS=${CPU_THREADS:-8} OPENBLAS_NUM_THREADS=${CPU_THREADS:-8} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p logs "$TMPDIR"
# Wait for the labeling stage if it holds the GPU lock. Never start a competing GPU job.
exec 9>artifacts/gpu_stage.lock
flock 9
trap 'code=$?; echo "$code" > logs/stage3_queue.exit' EXIT
rm -f logs/stage3_queue.exit
.venv/bin/python scripts/analyze_judges.py
for corpus in ministral3_8b qwen3_8b; do
 .venv/bin/python scripts/prepare_probe_features.py --corpus "$corpus"
done
echo 'RESOURCE PLAN: one GPU training process; pooled probes typically <8 GiB VRAM, bounded <16 GiB host working memory; ~8 GiB pooled cache; <=10 GiB probe checkpoints/results; 8 CPU threads default; no DataLoader subprocesses. Full duration measured in logs, not a fixed 12-hour promise.'
.venv/bin/python scripts/hf_sync.py --watch --interval 120 --stop-file logs/stage3_queue.exit > logs/hf_sync_stage3.log 2>&1 9>&- &
.venv/bin/python scripts/probe_gpu_preflight.py --batch 8
.venv/bin/python scripts/run_probe_suite.py --threads "${CPU_THREADS:-8}"
.venv/bin/python scripts/run_single_layer_baseline.py
.venv/bin/python scripts/run_alignment.py
.venv/bin/python scripts/summarize_probe_results.py

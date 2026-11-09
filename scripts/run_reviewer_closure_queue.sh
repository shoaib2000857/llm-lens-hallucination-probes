#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
set +u
source "$HOME/.bashrc"
set -u
export HF_HOME="$PWD/artifacts/hf_cache" TMPDIR="$PWD/artifacts/tmp" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p logs "$TMPDIR"
exec 9>artifacts/gpu_stage.lock
flock -n 9 || { echo 'Another project GPU stage holds lock'; exit 1; }
stop="${CLOSURE_STOP_FILE:-logs/reviewer_closure.exit}"
trap 'code=$?; echo "$code" > "$stop"' EXIT
rm -f "$stop"
echo 'RESOURCE PLAN: existing representations only; zero LLM generation or loading. 36 judge-specific pooled fits plus 180 raw context-grouped target-reference fits (4 corpora x3 probes x3 seeds x5 folds), then paired-reference and source-dev threshold reports. One GPU process; raw batch8 previously measured peak reserved2.19 GiB; RAM target below16 GiB, 8 CPU threads, zero workers; reserve30 GiB output. Raw reads may total multiple TiB. Runtime could exceed12 hours depending early stopping and shared I/O; resume checkpoints and completed fits reused. No raw hidden-state uploads.'
.venv/bin/python -c "import shutil; assert shutil.disk_usage('.').free>30*2**30"
.venv/bin/python scripts/hf_sync.py --watch --interval 120 --stop-file "$stop" > "${CLOSURE_SYNC_LOG:-logs/hf_sync_reviewer_closure.log}" 2>&1 9>&- &
.venv/bin/python scripts/run_label_source_robustness.py
.venv/bin/python scripts/portability_reference_report.py --pooled-only
.venv/bin/python scripts/run_distribution_controls.py --corpora distribution_v1_hotpotqa_ministral3_8b distribution_v1_hotpotqa_qwen3_8b distribution_v1_pubmedqa_ministral3_8b distribution_v1_pubmedqa_qwen3_8b --probes M2 M3 M4 --adaptations scratch --raw-batch 8
.venv/bin/python scripts/portability_reference_report.py
printf 'Reviewer closure computations complete; human validation, faithful published baselines, final analysis and manuscript remain pending.\n'

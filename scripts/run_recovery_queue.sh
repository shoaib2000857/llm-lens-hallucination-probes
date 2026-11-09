#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
set +u
source "$HOME/.bashrc"
set -u
export HF_HOME="$PWD/artifacts/hf_cache" TMPDIR="$PWD/artifacts/tmp"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p logs "$TMPDIR"
exec 9>artifacts/gpu_stage.lock
flock -n 9 || { echo 'Another project GPU stage holds the lock'; exit 1; }
stop="${RECOVERY_STOP_FILE:-logs/recovery_queue.exit}"
trap 'code=$?; echo "$code" > "$stop"' EXIT
rm -f "$stop"
echo 'RESOURCE PLAN: reuse all existing representations; no LLM loading, generation, judging or new hidden-state dumps. 162 maximum raw M2/M3/M4 fits = 3 updates x3 architectures x3 seeds x3 label budgets x2 initialization methods; single-class budgets skip. BF16 probe compute, batch8, one GPU process, eight CPU threads, no DataLoader workers. Real batch8 pilot peak reserved2.19 GiB and allocated1.76 GiB; full-fit optimizer state and allocator variation may increase this. Host process RAM target below16 GiB, checkpoints/results reserve20 GiB. Repeated sequential raw-shard reads may process multiple TiB over the queue. Prior full source fits took37–48 minutes at their recorded settings; total runtime may exceed24 hours and remains dataset/early-stopping dependent. Atomic per-epoch resume and completed-fit skipping. Lightweight schedules run first for quick outputs.'
.venv/bin/python -c "import shutil; print('Free GiB:',shutil.disk_usage('.').free/2**30); assert shutil.disk_usage('.').free>20*2**30"
.venv/bin/python scripts/hf_sync.py --watch --interval 120 --stop-file "$stop" > "${RECOVERY_SYNC_LOG:-logs/hf_sync_recovery.log}" 2>&1 9>&- &
.venv/bin/python scripts/audit_model_updates.py
.venv/bin/python scripts/simulate_probe_updates.py
.venv/bin/python scripts/run_raw_update_recovery.py --batch "${RAW_BATCH:-8}" --epochs "${RECOVERY_EPOCHS:-25}"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compare_update_recovery.py
.venv/bin/python scripts/summarize_model_updates.py
.venv/bin/python scripts/audit_model_updates.py
echo 'Recovery/schedule queue complete. Replicated generator trajectories, human annotation, remaining dataset coverage, architectural ablations, literature and manuscript work remain pending.'

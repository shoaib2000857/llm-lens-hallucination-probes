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
stop="${PENDING_STOP_FILE:-logs/pending_nonhuman.exit}"
trap 'code=$?; echo "$code" > "$stop"' EXIT
rm -f "$stop"
echo 'RESOURCE PLAN: sequential Lookback replay on six already-generated seed43/44 checkpoints (42000 responses), then108 maximum full-token probe fits:6 checkpoints x3 architectures x3 probe seeds x2 initializations, target-label budget500. Reuse existing tensors/labels; no new generator training or judging. One GPU process,8 CPU threads,zero DataLoader workers. Measured Lookback peak<21GiB; measured raw-probe batch8 pilot2.19GiB reserved. Host RAM target<32GiB; reserve20GiB output. Replay likely30-60min; probe recovery provisional12-36h, dependent on early stopping/shared disk throughput. Full training budget controls already exist; do not repeat them. Atomic shard/epoch resume and completed-fit skipping. No human-label dependency.'
.venv/bin/python -c "import shutil; assert shutil.disk_usage('.').free>20*2**30"
corpora=(updates_v1r43_qwen3_8b_step000050 updates_v1r43_qwen3_8b_step000150 updates_v1r43_qwen3_8b_step000300 updates_v1r44_qwen3_8b_step000050 updates_v1r44_qwen3_8b_step000150 updates_v1r44_qwen3_8b_step000300)
.venv/bin/python scripts/hf_sync.py --watch --interval 120 --stop-file "$stop" > "${PENDING_SYNC_LOG:-logs/hf_sync_pending_nonhuman.log}" 2>&1 9>&- &
.venv/bin/python scripts/extract_lookback_features.py --corpora "${corpora[@]}" --batch 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/evaluate_lookback_baseline.py --targets "${corpora[@]}"
.venv/bin/python scripts/run_raw_update_recovery.py --corpora "${corpora[@]}" --budgets 500 --adaptations scratch warm --probes M2 M3 M4 --seeds 42 43 44 --batch 8 --threads 8 --epochs 25
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/build_saved_result_publication.py
.venv/bin/python scripts/summarize_trajectory_replication.py
echo 'Nonhuman replication queue complete. Unimplemented published baselines, missing domain coverage and paper work remain explicit in TASKS.md.'

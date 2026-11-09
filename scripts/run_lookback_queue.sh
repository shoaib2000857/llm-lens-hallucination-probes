#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
set +u
source "$HOME/.bashrc"
set -u
export HF_HOME="$PWD/artifacts/hf_cache" TMPDIR="$PWD/artifacts/tmp" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p logs "$TMPDIR"
exec 9>artifacts/gpu_stage.lock
flock -n 9 || { echo 'Another project GPU stage holds lock'; exit 1; }
stop="${LOOKBACK_STOP_FILE:-logs/lookback.exit}"
trap 'code=$?; echo "$code" > "$stop"' EXIT
rm -f "$stop"
echo 'RESOURCE PLAN: replay39000 existing responses:14000 source +4000 distribution +21000 first-trajectory checkpoint cases. No regeneration or new judge calls. Saved prompt/response token IDs unchanged. Capture only96 selected attention rows per layer and immediately reduce to per-head mean features; retain no full attention maps or model gradients. Batch4, one GPU process,8 CPU threads,zero DataLoader workers. Real source/domain pilots passed; peak reserved15.56 GiB Qwen and17.64 GiB Ministral. Host RAM target<32 GiB; reserve3 GiB compact features/results. Typical source batch4 took0.16s; longer Hotpot batches0.64-0.91s. Provisional total45-120min plus classifier fitting, variable context lengths/shared I/O. Atomic128-case feature shards, checksum resume. Cases lacking response marker in original truncated inputs are explicitly unavailable; all probe comparisons use identical supported target IDs. This is the published attention-feature/default linear method adapted to response-level QA labels, not the original span benchmark.'
.venv/bin/python -c "import shutil; assert shutil.disk_usage('.').free>3*2**30"
.venv/bin/python scripts/hf_sync.py --watch --interval 120 --stop-file "$stop" > "${LOOKBACK_SYNC_LOG:-logs/hf_sync_lookback.log}" 2>&1 9>&- &
.venv/bin/python scripts/extract_lookback_features.py --corpora ministral3_8b distribution_v1_hotpotqa_ministral3_8b distribution_v1_pubmedqa_ministral3_8b qwen3_8b distribution_v1_hotpotqa_qwen3_8b distribution_v1_pubmedqa_qwen3_8b updates_v1_qwen3_8b_step000050 updates_v1_qwen3_8b_step000150 updates_v1_qwen3_8b_step000300 --batch 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/evaluate_lookback_baseline.py --include-updates
printf 'Lookback feature-method comparison complete. Human validation and original span-benchmark reproduction remain separate.\n'

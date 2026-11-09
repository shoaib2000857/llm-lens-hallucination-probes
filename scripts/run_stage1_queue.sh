#!/usr/bin/env bash
set -euo pipefail
set +u
source "$HOME/.bashrc"
set -u
cd "$(dirname "$0")/.."
mkdir -p logs artifacts/tmp
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME="$PWD/artifacts/hf_cache" TMPDIR="$PWD/artifacts/tmp"
exec 9>artifacts/gpu_stage.lock
flock -n 9 || { echo 'Another project GPU stage is running; retry after its smoke test exits.'; exit 1; }
echo 'RESOURCE PLAN: one GPU process; tune batches 1/2/4/8/16/32; <=32 GiB measured tuning limit; 4 CPU threads; 0 workers; ~359 GiB FP16 states for both models, plus ~30 GiB model cache; runtime printed from measurements. Full run requested by terminal operator.'
status_file=logs/stage1_queue.exit
sync_pid=''
finish() {
  code=$?
  echo "$code" > "$status_file"
  if [[ -n "$sync_pid" ]]; then
    echo "GPU queue exited with code $code. HF sync is draining pending uploads (PID $sync_pid)."
  fi
  exit "$code"
}
trap finish EXIT
rm -f "$status_file"
if [[ "${HF_SYNC:-1}" == 1 ]]; then
  .venv/bin/python -c 'from huggingface_hub import HfApi,get_token; assert get_token(), "Authenticate first: export HF_HOME=$PWD/artifacts/hf_cache; .venv/bin/hf auth login"; HfApi().whoami(); print("HF authentication verified")'
  .venv/bin/python -u scripts/hf_sync.py --watch --stop-file "$status_file" \
    --repo "${HF_SYNC_REPO:-ShoaibSSM/hallucination-portability-revision}" \
    > logs/hf_sync.log 2>&1 &
  sync_pid=$!
  echo "$sync_pid" > logs/hf_sync.pid
fi
for model in ministral3_8b qwen3_8b; do
  echo "[$(date -u +%FT%TZ)] Starting 100-row pilot: $model"
  .venv/bin/python -u scripts/extract_representations.py \
    --config "configs/stage1_${model}.json" --output "artifacts/stage1_${model}" \
    --limit 100 --batch-size 2 --autotune
  PYTHONPATH=src .venv/bin/python scripts/validate_extraction.py "artifacts/stage1_${model}" --minimum 100
  echo "[$(date -u +%FT%TZ)] Pilot passed; continuing full corpus: $model"
  .venv/bin/python -u scripts/extract_representations.py \
    --config "configs/stage1_${model}.json" --output "artifacts/stage1_${model}" \
    --limit 0 --batch-size 2 --autotune
  PYTHONPATH=src .venv/bin/python scripts/validate_extraction.py "artifacts/stage1_${model}" --minimum 7000
done
echo 'Stage 1 complete: two new-model representation corpora and confidence features. Next: judge labels and probe training/adaptation.'

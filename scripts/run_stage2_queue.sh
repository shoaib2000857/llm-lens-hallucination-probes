#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
set +u
source "$HOME/.bashrc"
set -u
export HF_HOME="$PWD/artifacts/hf_cache" TMPDIR="$PWD/artifacts/tmp"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p logs "$TMPDIR"
exec 9>artifacts/gpu_stage.lock
flock -n 9 || { echo 'Another project GPU stage holds the lock'; exit 1; }
trap 'code=$?; echo "$code" > logs/stage2_queue.exit' EXIT
rm -f logs/stage2_queue.exit
limit=${LIMIT:-0}
batch=${BATCH_SIZE:-32}
echo "RESOURCE PLAN: one judge GPU process; estimated 18-36 GiB VRAM (automatic batch reduction on OOM) at batch $batch; <32 GiB RAM; <=25 GB new model downloads; <2 GB labels/logs; 4 CPU threads, 0 loader workers. Three judges x two saved corpora. Runtime to estimate from pilot."
.venv/bin/python scripts/hf_sync.py --watch --interval 120 --stop-file logs/stage2_queue.exit > logs/hf_sync_stage2.log 2>&1 9>&- &
for judge in mistral qwen phi; do
 case "$judge" in
 mistral) model=mistralai/Mistral-7B-Instruct-v0.2; revision=63a8b081895390a26e140280378bc85ec8bce07a;;
 qwen) model=Qwen/Qwen3-8B; revision=b968826d9c46dd6066d109eabc6255188de91218;;
 phi) model=microsoft/Phi-4-mini-instruct; revision=cfbefacb99257ffa30c83adab238a50856ac3083;;
 esac
 for corpus in ministral3_8b qwen3_8b; do
  if [ "$limit" -eq 0 ]; then
    .venv/bin/python scripts/judge_representations.py --representations "artifacts/stage1_${corpus}/hidden_states" --output "artifacts/stage1_${corpus}/judge_${judge}_v3" --model "$model" --revision "$revision" --limit 100 --batch-size "$batch"
    .venv/bin/python scripts/validate_judge_pilot.py "artifacts/stage1_${corpus}/judge_${judge}_v3"
  fi
  .venv/bin/python scripts/judge_representations.py --representations "artifacts/stage1_${corpus}/hidden_states" --output "artifacts/stage1_${corpus}/judge_${judge}_v3" --model "$model" --revision "$revision" --limit "$limit" --batch-size "$batch"
 done
done
.venv/bin/python scripts/analyze_judges.py

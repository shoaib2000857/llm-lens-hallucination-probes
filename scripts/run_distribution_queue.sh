#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
set +u
source "$HOME/.bashrc"
set -u
export HF_HOME="$PWD/artifacts/hf_cache" TMPDIR="$PWD/artifacts/tmp"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 EXTRACTION_THREADS=8 POOL_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p logs "$TMPDIR"
exec 9>artifacts/gpu_stage.lock
flock -n 9 || { echo 'Another project GPU stage is running'; exit 1; }
stop="${ROUND_STOP_FILE:-logs/distribution_queue.exit}"
trap 'code=$?; echo "$code" > "$stop"' EXIT
rm -f "$stop"
limit=${LIMIT:-0}
echo 'RESOURCE PLAN: 4000 generations + 12000 judgments; one GPU model at a time; extraction batch16 with OOM backoff, roughly 18–34 GiB VRAM; <32 GiB RAM; approximately 103 GiB hidden tensors plus <5 GiB pooled/results; eight CPU threads, zero loader workers. Runtime must be estimated from measured progress; no 12-hour guarantee. Existing cached checkpoints reused. Large tensors excluded from routine HF backup.'
.venv/bin/python scripts/prepare_distribution_round.py
.venv/bin/python scripts/distribution_input_preflight.py
.venv/bin/python scripts/hf_sync.py --watch --interval 120 --stop-file "$stop" > "${ROUND_SYNC_LOG:-logs/hf_sync_distribution.log}" 2>&1 9>&- &
corpora=()
for dataset in hotpotqa pubmedqa; do
 for generator in ministral3_8b qwen3_8b; do
  corpus="distribution_v1_${dataset}_${generator}"
  corpora+=("$corpus")
  .venv/bin/python scripts/extract_representations.py --config "configs/$corpus.json" --output "artifacts/stage1_$corpus" --limit "$limit" --batch-size "${EXTRACTION_BATCH:-16}"
 done
done
for judge in mistral qwen phi; do
 case "$judge" in
 mistral) model=mistralai/Mistral-7B-Instruct-v0.2; revision=63a8b081895390a26e140280378bc85ec8bce07a;;
 qwen) model=Qwen/Qwen3-8B; revision=b968826d9c46dd6066d109eabc6255188de91218;;
 phi) model=microsoft/Phi-4-mini-instruct; revision=cfbefacb99257ffa30c83adab238a50856ac3083;;
 esac
 for corpus in "${corpora[@]}"; do
  .venv/bin/python scripts/judge_representations.py --representations "artifacts/stage1_$corpus/hidden_states" --output "artifacts/stage1_$corpus/judge_${judge}_v3" --model "$model" --revision "$revision" --limit "$limit" --batch-size "${JUDGE_BATCH:-8}" --max-input 8192
 done
done
.venv/bin/python scripts/analyze_judges.py --corpora "${corpora[@]}"
for corpus in "${corpora[@]}"; do
 .venv/bin/python scripts/prepare_probe_features.py --corpus "$corpus"
done
.venv/bin/python scripts/evaluate_distribution_round.py --corpora "${corpora[@]}" --raw-batch "${RAW_BATCH:-8}"
if [ "$limit" -eq 0 ] && [ "${TARGET_CONTROLS:-1}" -eq 1 ]; then
 .venv/bin/python scripts/run_distribution_controls.py --corpora "${corpora[@]}"
fi
.venv/bin/python scripts/distribution_tables.py
echo 'Distribution round complete. Controlled model updates, human labels and raw-token target controls remain pending.'

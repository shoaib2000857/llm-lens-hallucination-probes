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
flock -n 9 || { echo 'Another project GPU stage holds the lock'; exit 1; }
stop="${UPDATE_STOP_FILE:-logs/model_updates_queue.exit}"
trap 'code=$?; echo "$code" > "$stop"' EXIT
rm -f "$stop"
echo 'RESOURCE PLAN: one BF16 Qwen3-8B parent + rank8 query/value LoRA, 300 optimizer updates at microbatch1/accumulation4/sequence768, checkpoints50/150/300. Training estimated24–34 GiB VRAM with allocation guard; extraction batch16 with OOM backoff. Eight CPU threads, zero loader workers, <32 GiB host process RAM. Approximately554 GiB new hidden tensors for 3x7000 generations, ~12 GiB pooled caches, <1 GiB adapters/optimizer and several GiB probe checkpoints. 21,000 generations and63,000 judgments. Runtime must be measured: plan for an overnight queue, potentially over12 hours; restartable at every stage. One GPU model at a time, parent cache reused, no duplicated base checkpoints or routine hidden-tensor HF uploads.'
.venv/bin/python -c "import shutil; free=shutil.disk_usage('.').free; print('Free disk GiB:',free/2**30); assert free>650*2**30, 'Need at least 650 GiB free for this round and reserve'"
.venv/bin/python scripts/hf_sync.py --watch --interval 120 --stop-file "$stop" > "${UPDATE_SYNC_LOG:-logs/hf_sync_model_updates.log}" 2>&1 9>&- &
.venv/bin/python scripts/prepare_model_updates.py
.venv/bin/python scripts/train_model_updates.py --output artifacts/model_updates_v1/qwen_hotpot
.venv/bin/python scripts/build_update_configs.py
.venv/bin/python scripts/measure_update_drift.py
corpora=()
for step in 000050 000150 000300; do
 corpus="updates_v1_qwen3_8b_step${step}";corpora+=("$corpus")
 .venv/bin/python scripts/extract_representations.py --config "configs/$corpus.json" --output "artifacts/stage1_$corpus" --limit 0 --batch-size "${EXTRACTION_BATCH:-16}"
done
for judge in mistral qwen phi; do
 case "$judge" in
 mistral) model=mistralai/Mistral-7B-Instruct-v0.2; revision=63a8b081895390a26e140280378bc85ec8bce07a;;
 qwen) model=Qwen/Qwen3-8B; revision=b968826d9c46dd6066d109eabc6255188de91218;;
 phi) model=microsoft/Phi-4-mini-instruct; revision=cfbefacb99257ffa30c83adab238a50856ac3083;;
 esac
 for corpus in "${corpora[@]}"; do
  .venv/bin/python scripts/judge_representations.py --representations "artifacts/stage1_$corpus/hidden_states" --output "artifacts/stage1_$corpus/judge_${judge}_v3" --model "$model" --revision "$revision" --limit 0 --batch-size "${JUDGE_BATCH:-32}"
 done
done
.venv/bin/python scripts/analyze_judges.py --corpora "${corpora[@]}"
for corpus in "${corpora[@]}"; do
 .venv/bin/python scripts/prepare_probe_features.py --corpus "$corpus"
done
.venv/bin/python scripts/run_update_probes.py --corpora "${corpora[@]}"
.venv/bin/python scripts/update_confound_baselines.py --corpora "${corpora[@]}"
.venv/bin/python scripts/summarize_model_updates.py
echo 'Controlled update queue complete. One trajectory; replicated trajectories, raw-token recovery, update schedules and human labels remain pending.'

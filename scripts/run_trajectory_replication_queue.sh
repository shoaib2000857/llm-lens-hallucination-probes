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
stop="${REPLICATION_STOP_FILE:-logs/replication_queue.exit}"
trap 'code=$?; echo "$code" > "$stop"' EXIT
rm -f "$stop"
echo 'RESOURCE PLAN: two independent Qwen3-8B rank8 LoRA seeds43/44 on the identical pinned adaptation manifest, 300 optimizer updates each and checkpoints50/150/300. 42,000 new generations and126,000 judge outcomes; one GPU model at a time. BF16 parent, training microbatch1/accumulation4/sequence768; measured prior training below19 GiB reserved,34 GiB allocation guard. Extraction batch16 with OOM backoff, budget up to34 GiB VRAM; eight CPU threads, no loader processes, RAM below32 GiB per heavyweight process. Approximately1,108 GiB new FP16 hidden tensors plus24 GiB pooled caches and checkpoint/result reserve. Requires1,200 GiB free. Cached parent/judges reused; no duplicate base checkpoints, no routine tensor HF uploads. Runtime may exceed24 hours; extraction, judging and fits resume. 144 new frozen evaluations and up to540 pooled scratch/warm fits; source trajectory reused rather than regenerated.'
.venv/bin/python -c "import shutil; print('Free disk GiB:',shutil.disk_usage('.').free/2**30); assert shutil.disk_usage('.').free>1200*2**30,'Need 1200 GiB free for replication and reserve'"
.venv/bin/python scripts/hf_sync.py --watch --interval 120 --stop-file "$stop" > "${REPLICATION_SYNC_LOG:-logs/hf_sync_replication.log}" 2>&1 9>&- &
corpora=()
for seed in 43 44; do
 adapter="artifacts/model_updates_v1/qwen_hotpot_s${seed}"
 .venv/bin/python scripts/train_model_updates.py --config "configs/model_updates_replication_s${seed}.json" --output "$adapter"
 .venv/bin/python scripts/build_update_configs.py --adapters "$adapter" --prefix "updates_v1r${seed}"
 .venv/bin/python scripts/measure_update_drift.py --adapters "$adapter" --output "artifacts/model_updates_v1/replay_s${seed}" --result-dir "results/trajectory_replication_v1/s${seed}"
 for step in 000050 000150 000300; do
  corpus="updates_v1r${seed}_qwen3_8b_step${step}";corpora+=("$corpus")
  .venv/bin/python scripts/extract_representations.py --config "configs/$corpus.json" --output "artifacts/stage1_$corpus" --limit 0 --batch-size "${EXTRACTION_BATCH:-16}"
 done
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
.venv/bin/python scripts/run_update_probes.py --corpora "${corpora[@]}" --budgets 100 500 full
.venv/bin/python scripts/update_confound_baselines.py --corpora "${corpora[@]}"
.venv/bin/python scripts/summarize_trajectory_replication.py
echo 'Trajectory replication complete. Remaining: domain replication, more checkpoints, additional benchmarks/ablations, human validation and manuscript/literature/figures.'

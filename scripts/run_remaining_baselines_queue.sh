#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
set +u
source "$HOME/.bashrc"
set -u
export HF_HOME="$PWD/artifacts/hf_cache" TMPDIR="$PWD/artifacts/tmp"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 EXTRACTION_THREADS=8 POOL_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p logs "$TMPDIR"
stop="${BASELINES_STOP_FILE:-logs/remaining_baselines.exit}"
trap 'code=$?; echo "$code" > "$stop"' EXIT
rm -f "$stop"
echo 'RESOURCE PLAN: SpikeScore PPL600 saved-response cases x20 continuation turns, then4000 new context-supported NQ/Trivia responses,12000 judge calls, frozen probe evaluation and matched target-trained grouped OOF controls. One GPU process,8 CPU threads,zero DataLoader workers. Spike tiny pilot peak19.70GiB; corrected raw-NLL3-turn256-token pilot gates execution; full run uses20 turns. Extraction starts batch8 with existing OOM backoff; prior measured batches supported. RAM target<32GiB, reserve160GiB storage (roughly103GiB new FP16 states plus checkpoints/results); existing model caches reused. Provisional queue1-3days, depending on measured decoding and early stopping, no completion-time guarantee. Large tensors stay local; metadata, scores, best checkpoints and logs sync to HF. No human-label dependency. NQ is10-document retrieval-QA, not context-free NQ-Open; Trivia first1000 unique validation cases with authentic Wikipedia contexts. Original prompt truncation protocol preserved and recorded.'
.venv/bin/python -c "import shutil; assert shutil.disk_usage('.').free>160*2**30"
# Medium pilot is a separate short process; never overlap it with the main GPU job.
while [ ! -f results/spikescore_medium_v2/complete.json ]; do
 pid=$(cat logs/spikescore_medium.pid)
 if ! kill -0 "$pid" 2>/dev/null; then echo 'Medium SpikeScore pilot failed; inspect logs/spikescore_medium_v2.log'; exit 1; fi
 sleep 5
done
exec 9>artifacts/gpu_stage.lock
flock -n 9 || { echo 'Another project GPU stage holds the lock'; exit 1; }
.venv/bin/python scripts/hf_sync.py --watch --interval 120 --stop-file "$stop" > "${BASELINES_SYNC_LOG:-logs/hf_sync_remaining_baselines.log}" 2>&1 9>&- &
if [ ! -f results/spikescore_ppl_v1/complete.json ]; then
 .venv/bin/python scripts/run_spikescore_ppl.py --samples 200 --turns 20 --batch 4 --max-new 256
else
 echo 'Reuse completed600-case SpikeScore generation; evaluation checks exact saved subset IDs.'
fi
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/evaluate_spikescore_ppl.py
.venv/bin/python scripts/prepare_distribution_v2.py
corpora=()
for dataset in nq_retrieval triviaqa; do
 for generator in ministral3_8b qwen3_8b; do
  corpus="distribution_v2_${dataset}_${generator}"
  corpora+=("$corpus")
  # Eight-case smoke checks before scaling each context distribution/model.
  .venv/bin/python scripts/extract_representations.py --config "configs/$corpus.json" --output "artifacts/stage1_$corpus" --limit 8 --batch-size 4
  .venv/bin/python scripts/validate_extraction.py "artifacts/stage1_$corpus" --minimum 8
  .venv/bin/python scripts/extract_representations.py --config "configs/$corpus.json" --output "artifacts/stage1_$corpus" --limit 0 --batch-size 8
  .venv/bin/python scripts/validate_extraction.py "artifacts/stage1_$corpus" --minimum 1000
 done
done
for judge in mistral qwen phi; do
 case "$judge" in
 mistral) model=mistralai/Mistral-7B-Instruct-v0.2; revision=63a8b081895390a26e140280378bc85ec8bce07a;;
 qwen) model=Qwen/Qwen3-8B; revision=b968826d9c46dd6066d109eabc6255188de91218;;
 phi) model=microsoft/Phi-4-mini-instruct; revision=cfbefacb99257ffa30c83adab238a50856ac3083;;
 esac
 for corpus in "${corpora[@]}"; do
  .venv/bin/python scripts/judge_representations.py --representations "artifacts/stage1_$corpus/hidden_states" --output "artifacts/stage1_$corpus/judge_${judge}_v3" --model "$model" --revision "$revision" --limit 0 --batch-size 8 --max-input 8192
 done
done
.venv/bin/python scripts/analyze_judges.py --corpora "${corpora[@]}"
for corpus in "${corpora[@]}"; do
 .venv/bin/python scripts/prepare_probe_features.py --corpus "$corpus"
done
.venv/bin/python scripts/evaluate_distribution_round.py --corpora "${corpora[@]}" --raw-batch 8
.venv/bin/python scripts/run_distribution_controls.py --corpora "${corpora[@]}" --probes mean_linear last_linear last_mlp M0 M1 M2 M3 M4 --raw-batch 8
echo 'SpikeScore adaptation and new context-supported distribution queue complete; fixed-text alignment/ablations/manuscript and human analyses remain explicit.'

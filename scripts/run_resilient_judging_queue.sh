#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
set +u
source "$HOME/.bashrc"
set -u
export HF_HOME="$PWD/artifacts/hf_cache" TMPDIR="$PWD/artifacts/tmp" PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 POOL_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p logs "$TMPDIR"
exec 9>artifacts/gpu_stage.lock
flock -n 9 || { echo 'Another project GPU stage holds lock'; exit 1; }
stop="${RESILIENT_STOP_FILE:-logs/resilient_judging.exit}"
trap 'code=$?; echo "$code" > "$stop"' EXIT
rm -f "$stop"
echo 'RESOURCE PLAN: resume remaining12000 judge records, existing4000 generations reused. Explicit full-context fallback30000 tokens; single GPU,8 CPU threads,zero loader workers; batch starts4 with OOM backoff to1. Long full-context inference may exceed30GiB but bounded memory failures become missing labels, not fabricated verdicts. Host RAM target<32GiB; reserve30GiB checkpoints/results. Unsupported lengths recorded without truncation. Estimates depend on long-context throughput; potentially1-3days including controls. Coverage guard remains70percent; failed corpora do not stop independent corpora/probes.'
.venv/bin/python scripts/hf_sync.py --watch --interval 120 --stop-file "$stop" > "${RESILIENT_SYNC_LOG:-logs/hf_sync_resilient_judging.log}" 2>&1 9>&- &
failures=0
run_stage() {
 local stage="$1";shift
 echo "START $stage"
 if "$@"; then
  LAST_STAGE_OK=1;printf '%s,%s,complete\n' "$(date -u +%FT%TZ)" "$stage" >> logs/resilient_stage_status.csv
 else
  local code=$?;LAST_STAGE_OK=0;failures=$((failures+1));printf '%s,%s,failed_exit_%s\n' "$(date -u +%FT%TZ)" "$stage" "$code" >> logs/resilient_stage_status.csv
  echo "Stage $stage failed with exit$code; preserving outputs and continuing independent stages."
 fi
}
corpora=(distribution_v2_nq_retrieval_ministral3_8b distribution_v2_nq_retrieval_qwen3_8b distribution_v2_triviaqa_ministral3_8b distribution_v2_triviaqa_qwen3_8b)
for judge in mistral qwen phi; do
 case "$judge" in
 mistral) model=mistralai/Mistral-7B-Instruct-v0.2;revision=63a8b081895390a26e140280378bc85ec8bce07a;;
 qwen) model=Qwen/Qwen3-8B;revision=b968826d9c46dd6066d109eabc6255188de91218;;
 phi) model=microsoft/Phi-4-mini-instruct;revision=cfbefacb99257ffa30c83adab238a50856ac3083;;
 esac
 for corpus in "${corpora[@]}";do
  run_stage "judge_${judge}_${corpus}" .venv/bin/python scripts/judge_representations.py --representations "artifacts/stage1_$corpus/hidden_states" --output "artifacts/stage1_$corpus/judge_${judge}_v3" --model "$model" --revision "$revision" --limit 0 --batch-size 4 --max-input 8192 --fallback-input 30000 --skip-unsupported
 done
done
for corpus in "${corpora[@]}";do
 run_stage "consensus_$corpus" .venv/bin/python scripts/analyze_judges.py --corpora "$corpus"
 if [ "$LAST_STAGE_OK" -ne 1 ];then continue;fi
 run_stage "features_$corpus" .venv/bin/python scripts/prepare_probe_features.py --corpus "$corpus"
 if [ "$LAST_STAGE_OK" -ne 1 ];then echo "Coverage/readiness guard blocks training for$corpus; next corpus continues.";continue;fi
 for probe in mean_linear last_linear last_mlp M0 M1 M2 M3 M4;do
  run_stage "transfer_${corpus}_$probe" .venv/bin/python scripts/evaluate_distribution_round.py --corpora "$corpus" --probes "$probe" --raw-batch 8
  run_stage "controls_${corpus}_$probe" .venv/bin/python scripts/run_distribution_controls.py --corpora "$corpus" --probes "$probe" --raw-batch 8
 done
done
echo "Queue processed; failed stages:$failures. Inspect logs/resilient_stage_status.csv; missing labels remain explicit."
if [ "$failures" -gt 0 ];then exit 2;fi

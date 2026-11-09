#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
set +u
source "$HOME/.bashrc"
set -u
export HF_HOME="$PWD/artifacts/hf_cache" TMPDIR="$PWD/artifacts/tmp" PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 EXTRACTION_THREADS=8 POOL_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p logs "$TMPDIR"
exec 9>artifacts/gpu_stage.lock
flock -n 9 || { echo 'Another project GPU stage holds lock'; exit 1; }
stop="${FINAL_NONHUMAN_STOP:-logs/final_nonhuman.exit}"
trap 'code=$?; echo "$code" > "$stop"' EXIT
rm -f "$stop"
echo 'RESOURCE PLAN: same-text alignment500 anchors/direction,30 controlled source ablation fits,2 new modern same-family8B checkpoints,22000 generated responses (14000 SQuAD train/dev/test plus8000 other-domain cases),66000 judge records,240 frozen transfer evaluations,48 full-setting SQuAD reference fits and1920 maximum other-domain grouped control fits. Single heavyweight GPU job,8 CPU threads,zero DataLoader workers. Frozen BF16 generation estimated18-34GiB, extraction batch4 smoke then8 with OOM backoff. Raw probe pilot passed. RAM target<32GiB; reserve750GiB disk, roughly580GiB FP16 new states plus36GiB bounded two-worker model downloads and checkpoint/results. Repeated raw training reads multipleTiB sequentially; no full dataset RAM load. Provisional3-6days including early stopping, long-context judges and shared I/O. Resume/checksum checks, per-stage error isolation and continuous essential HF backup. Original NQ is retrievalQA, fixed plain prompts retained; no human labels used.'
.venv/bin/python -c "import shutil; assert shutil.disk_usage('.').free>750*2**30"
test -f results/fixed_text_alignment_pilot/complete.json
test -f results/architecture_ablation_pilot/summary.json
.venv/bin/python scripts/hf_sync.py --watch --interval 120 --stop-file "$stop" > "${FINAL_NONHUMAN_SYNC_LOG:-logs/hf_sync_final_nonhuman.log}" 2>&1 9>&- &
failures=0
run_stage(){
 local name="$1";shift
 echo "START $name"
 if "$@";then LAST_STAGE_OK=1;printf '%s,%s,complete\n' "$(date -u +%FT%TZ)" "$name" >> logs/final_nonhuman_stage_status.csv
 else local code=$?;LAST_STAGE_OK=0;failures=$((failures+1));printf '%s,%s,failed_exit_%s\n' "$(date -u +%FT%TZ)" "$name" "$code" >> logs/final_nonhuman_stage_status.csv;echo "Stage$name failed; independent stages continue.";fi
}
run_stage saved_evidence env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/analyze_remaining_saved_evidence.py
run_stage matched_text_alignment .venv/bin/python scripts/run_fixed_text_alignment.py --anchors 500 --batch 4
for probe in M1_uniform capacity_global M2_no_positions M3_no_layer_positions M4_no_token_positions;do
 run_stage "ablation_$probe" .venv/bin/python scripts/run_architecture_ablations.py --probes "$probe"
done
# Downloads run with two workers in a separate CPU-only process while early GPU
# stages execute; duplicate consolidated weights are excluded.
if [ ! -f artifacts/samefamily_v1/downloads_complete.json ];then
 run_stage checkpoint_downloads .venv/bin/python scripts/download_samefamily_models.py
fi
run_stage samefamily_selection .venv/bin/python scripts/prepare_samefamily_round.py
mapfile -t corpora < <(.venv/bin/python -c "import json;print('\n'.join(json.load(open('artifacts/samefamily_v2/plan.json'))['corpora']))")
ready=()
for corpus in "${corpora[@]}";do
 run_stage "extraction_smoke_$corpus" .venv/bin/python scripts/extract_representations.py --config "configs/$corpus.json" --output "artifacts/stage1_$corpus" --limit 8 --batch-size 4
 if [ "$LAST_STAGE_OK" -ne 1 ];then continue;fi
 run_stage "validate_smoke_$corpus" .venv/bin/python scripts/validate_extraction.py "artifacts/stage1_$corpus" --minimum 8
 if [ "$LAST_STAGE_OK" -ne 1 ];then continue;fi
 run_stage "extract_full_$corpus" .venv/bin/python scripts/extract_representations.py --config "configs/$corpus.json" --output "artifacts/stage1_$corpus" --limit 0 --batch-size 8
 if [ "$LAST_STAGE_OK" -ne 1 ];then continue;fi
 count=1000;if [[ "$corpus" == *'_squad_'* ]];then count=7000;fi
 run_stage "validate_full_$corpus" .venv/bin/python scripts/validate_extraction.py "artifacts/stage1_$corpus" --minimum "$count"
 if [ "$LAST_STAGE_OK" -eq 1 ];then ready+=("$corpus");fi
done
for judge in mistral qwen phi;do
 case "$judge" in
 mistral) model=mistralai/Mistral-7B-Instruct-v0.2;revision=63a8b081895390a26e140280378bc85ec8bce07a;;
 qwen) model=Qwen/Qwen3-8B;revision=b968826d9c46dd6066d109eabc6255188de91218;;
 phi) model=microsoft/Phi-4-mini-instruct;revision=cfbefacb99257ffa30c83adab238a50856ac3083;;
 esac
 for corpus in "${ready[@]}";do
  run_stage "judge_${judge}_$corpus" .venv/bin/python scripts/judge_representations.py --representations "artifacts/stage1_$corpus/hidden_states" --output "artifacts/stage1_$corpus/judge_${judge}_v3" --model "$model" --revision "$revision" --limit 0 --batch-size 4 --max-input 8192 --fallback-input 30000 --skip-unsupported
 done
done
for corpus in "${ready[@]}";do
 run_stage "consensus_$corpus" .venv/bin/python scripts/analyze_judges.py --corpora "$corpus"
 if [ "$LAST_STAGE_OK" -ne 1 ];then continue;fi
 run_stage "features_$corpus" .venv/bin/python scripts/prepare_probe_features.py --corpus "$corpus"
 if [ "$LAST_STAGE_OK" -ne 1 ];then continue;fi
 run_stage "transfer_$corpus" .venv/bin/python scripts/evaluate_samefamily_round.py --corpus "$corpus"
 if [[ "$corpus" == *'_squad_'* ]];then
  run_stage "full_reference_$corpus" .venv/bin/python scripts/train_samefamily_references.py --corpus "$corpus"
 else
  for probe in mean_linear last_linear last_mlp M0 M1 M2 M3 M4;do
   run_stage "controls_${corpus}_$probe" .venv/bin/python scripts/run_distribution_controls.py --corpora "$corpus" --probes "$probe" --raw-batch 8
  done
 fi
done
echo "Final nonhuman queue processed;failed stages:$failures. Paper/interpretation and genuine human validation remain separate."
if [ "$failures" -gt 0 ];then exit 2;fi

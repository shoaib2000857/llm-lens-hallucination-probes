#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/hallucination-portability
set +u
source "$HOME/.bashrc"
set -u
export HF_HOME="$PWD/artifacts/hf_cache" TMPDIR="$PWD/artifacts/tmp" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p logs "$TMPDIR"
exec 9>artifacts/gpu_stage.lock
flock -n 9 || { echo 'Another project GPU stage holds lock'; exit 1; }
stop="${REFERENCE_STOP_FILE:-logs/final_reference.exit}"
trap 'code=$?; echo "$code" > "$stop"' EXIT
rm -f "$stop"
echo 'RESOURCE PLAN: 54 full-budget raw scratch fits only: two replicated trajectories x3 snapshots x3 architectures x3 probe seeds. Existing tensors and labels; no LLM loading, generation or judging. Previously measured raw batch8 reserved2.19 GiB, RAM target<16 GiB; 8 CPU threads, zero workers, one GPU process. Reserve30 GiB output. Comparable completed fits took20.7/21.7/30.2 minutes: roughly18-28 hours total at similar throughput; early stopping/shared I/O may extend this. Resume per epoch; completed fits reused. Essential HF sync, no tensor uploads. Cross-label/model/domain evaluation first, then fits and final checks.'
.venv/bin/python -c "import shutil; assert shutil.disk_usage('.').free>30*2**30"
.venv/bin/python scripts/hf_sync.py --watch --interval 120 --stop-file "$stop" > "${REFERENCE_SYNC_LOG:-logs/hf_sync_final_reference.log}" 2>&1 9>&- &
.venv/bin/python scripts/evaluate_label_source_transfer.py
.venv/bin/python scripts/run_raw_update_recovery.py --corpora updates_v1r43_qwen3_8b_step000050 updates_v1r43_qwen3_8b_step000150 updates_v1r43_qwen3_8b_step000300 updates_v1r44_qwen3_8b_step000050 updates_v1r44_qwen3_8b_step000150 updates_v1r44_qwen3_8b_step000300 --budgets full --adaptations scratch --batch 8
.venv/bin/python scripts/portability_reference_report.py
.venv/bin/python scripts/summarize_trajectory_replication.py
.venv/bin/python - <<'PY'
import json
from pathlib import Path
missing=json.loads(Path('results/portability_reference_v1/missing_references.json').read_text())
assert not missing,missing
print('All saved transfer experiments have target-trained references.')
PY
printf 'Reference closure complete. Human annotations, verified published comparisons, final scientific analysis and manuscript remain pending.\n'

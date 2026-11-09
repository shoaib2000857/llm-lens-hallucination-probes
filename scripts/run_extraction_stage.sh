#!/usr/bin/env bash
set -euo pipefail
set +u
source "$HOME/.bashrc"
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME="$PWD/artifacts/hf_cache"
export TMPDIR="$PWD/artifacts/tmp"
mkdir -p "$TMPDIR" logs
config="${1:?usage: run_extraction_stage.sh CONFIG OUTPUT [LIMIT=100] [BATCH=2]}"
output="${2:?output directory required}"
limit="${3:-100}"
batch="${4:-2}"
# Serialize GPU jobs launched through this wrapper; no unrelated process is killed.
exec 9>artifacts/gpu_stage.lock
flock -n 9 || { echo 'Another project GPU stage holds the lock'; exit 1; }
exec .venv/bin/python -u scripts/extract_representations.py --config "$config" --output "$output" --limit "$limit" --batch-size "$batch" --autotune

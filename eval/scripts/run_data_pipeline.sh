#!/usr/bin/env bash
# Build the SFT dataset: LLM cleaning -> answer sidecar -> SFT/eval splits ->
# validation -> public benchmarks.
#
# Usage: bash scripts/run_data_pipeline.sh [extra args for extract_answers.py]
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/5 LLM cleaning (dedicated cleaner model) =="
if [ -n "${DEEPSEEK_CLEAN_API_KEY:-}" ] || [ -f ../api_clean.txt ]; then
  python data/clean_with_llm.py || \
    echo "[warn] LLM cleaning failed; build_dataset.py will fall back to raw CoTs"
else
  echo "[warn] no cleaning key found (api_clean.txt or DEEPSEEK_CLEAN_API_KEY); skip LLM cleaning"
fi

echo "== 2/5 extract final answers from teacher answers =="
python data/extract_answers.py "$@"

echo "== 3/5 build SFT + eval splits =="
python data/build_dataset.py

echo "== 4/5 validate dataset =="
python data/validate_dataset.py

echo "== 5/5 (optional) download public benchmarks =="
python data/download_benchmarks.py --bench gsm8k math500 amc23 || \
  echo "[warn] benchmark download failed (no network / dataset id changed); provide your own JSONL instead"

echo "done. next: bash scripts/run_training.sh"

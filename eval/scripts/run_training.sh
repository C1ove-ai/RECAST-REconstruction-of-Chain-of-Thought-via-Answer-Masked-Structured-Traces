#!/usr/bin/env bash
# Train the student model (QLoRA by default) and optionally merge the adapter.
# Env overrides: MODEL=... OUT=... METHOD=... EXTRA="..."
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
OUT="${OUT:-runs/sft_qwen1.5b}"
METHOD="${METHOD:-qlora}"
EXTRA="${EXTRA:-}"

echo "== training $MODEL ($METHOD) -> $OUT =="
python train/train_sft.py \
  --model "$MODEL" \
  --method "$METHOD" \
  --data-dir data/processed \
  --output-dir "$OUT" \
  --merge \
  $EXTRA

echo "done. eval with:"
echo "  python harness/run_eval.py --model $MODEL --adapter $OUT/final_adapter --data data/benchmarks/gsm8k.jsonl --name sft_gsm8k"

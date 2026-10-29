#!/usr/bin/env bash
# Evaluate base student / SFT student (and optionally the teacher) on the
# in-domain held-out split and public benchmarks, then build the comparison table.
#
# Env overrides:
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct
#   ADAPTER=runs/sft_qwen1.5b/final_adapter
#   TEACHER_MODEL=deepseek-chat TEACHER_LIMIT=200   (set TEACHER_MODEL="" to skip)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ADAPTER="${ADAPTER:-runs/sft_qwen1.5b/final_adapter}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-chat}"
TEACHER_LIMIT="${TEACHER_LIMIT:-200}"

DATASETS=()
[ -f data/processed/eval_test.jsonl ] && DATASETS+=("data/processed/eval_test.jsonl")
[ -f data/benchmarks/gsm8k.jsonl ] && DATASETS+=("data/benchmarks/gsm8k.jsonl")
[ -f data/benchmarks/math500.jsonl ] && DATASETS+=("data/benchmarks/math500.jsonl")
[ ${#DATASETS[@]} -eq 0 ] && { echo "no eval datasets found"; exit 1; }

for DATA in "${DATASETS[@]}"; do
  NAME=$(basename "$DATA" .jsonl)
  echo "== $NAME =="
  python harness/run_eval.py --model "$MODEL" --data "$DATA" \
      --name "base_${NAME}" --k 1
  if [ -d "$ADAPTER" ]; then
    python harness/run_eval.py --model "$MODEL" --adapter "$ADAPTER" --data "$DATA" \
        --name "sft_${NAME}" --k 1
  else
    echo "[warn] adapter not found: $ADAPTER (skip sft)"
  fi
  if [ -n "$TEACHER_MODEL" ] && [ "$TEACHER_MODEL" != "none" ]; then
    python harness/run_eval.py --provider teacher --model "$TEACHER_MODEL" --data "$DATA" \
        --name "teacher_${NAME}" --k 1 --limit "$TEACHER_LIMIT" || \
      echo "[warn] teacher eval failed (check API key / base-url)"
  fi
done

python harness/compare_results.py results/*.json --baseline base --out results/summary.md
echo "done. see results/summary.md"

# faithfulness probe (optional; needs recorded generations for the SFT model)
# python harness/faithfulness.py --model "$MODEL" --adapter "$ADAPTER" \
#   --data data/processed/eval_test.jsonl \
#   --generations results/sft_eval_test.generations.jsonl

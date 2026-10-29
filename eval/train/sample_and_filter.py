#!/usr/bin/env python3
"""Stage-2 rejection-sampling fine-tuning (RFT / STaR-style self-training).

Procedure:
  1. Run the SFT student on the training questions.
  2. Keep only sampled solutions whose final answer is correct (verifiable for
     math, no reward model / preference labels needed).
  3. Write the accepted samples as a new SFT JSONL and train again with
     train/train_sft.py.

This is usually a better stage-2 move than DPO here: DPO needs preference
pairs, while math gives us free correctness labels.

Example:
    python train/sample_and_filter.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter ../runs/sft_qwen1.5b/final_adapter \
        --data data/processed/sft_train.jsonl \
        --out data/processed/sft_rft.jsonl --num-samples 4 --temperature 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
EVAL_ROOT = TRAIN_DIR.parent
sys.path.insert(0, str(EVAL_ROOT / "harness"))

from math_verify import extract_final_answer, is_correct  # noqa: E402
from run_eval import generate_hf, load_hf_model  # noqa: E402


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Rejection-sampling SFT data generation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default="", help="SFT adapter (optional)")
    parser.add_argument("--data", default=str(EVAL_ROOT / "data" / "processed" / "sft_train.jsonl"))
    parser.add_argument("--out", default=str(EVAL_ROOT / "data" / "processed" / "sft_rft.jsonl"))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stats", default=str(EVAL_ROOT / "data" / "processed" / "rft_stats.json"))
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)

    rows = load_jsonl(args.data)
    if args.limit > 0:
        rows = rows[: args.limit]

    model, tokenizer = load_hf_model(args.model, args.adapter, args.load_4bit, args.device)

    accepted: list[dict] = []
    stats = {"questions": 0, "questions_with_answer": 0, "samples": 0,
             "accepted": 0, "skipped_no_answer": 0, "accepted_questions": 0}

    prompts: list[list[dict]] = []
    question_rows: list[dict] = []
    for row in rows:
        messages = row.get("messages", [])
        ref = ((row.get("meta") or {}).get("answer") or "").strip()
        if not messages or not ref:
            stats["skipped_no_answer"] += 1
            continue
        prompts.append(messages[:-1])  # system + user
        question_rows.append(row)

    stats["questions"] = len(question_rows)
    stats["questions_with_answer"] = len(question_rows)

    # generate in chunks so a long run prints progress and saves partial work
    chunk = max(1, args.batch_size * 4)
    for start in range(0, len(prompts), chunk):
        batch_prompts = prompts[start:start + chunk]
        gens, _ = generate_hf(
            model, tokenizer, batch_prompts,
            k=args.num_samples, temperature=args.temperature, top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=3072, batch_size=args.batch_size,
        )
        for row, samples in zip(question_rows[start:start + chunk], gens):
            ref = (row.get("meta") or {}).get("answer", "")
            kept_for_question = 0
            for index, generation in enumerate(samples):
                stats["samples"] += 1
                pred = extract_final_answer(generation)
                if not is_correct(pred, ref):
                    continue
                stats["accepted"] += 1
                kept_for_question += 1
                accepted.append({
                    "id": f"{row.get('id')}#rft{index}",
                    "source": f"rft:{row.get('source', '')}",
                    "messages": list(row["messages"][:-1]) + [
                        {"role": "assistant", "content": generation.strip()}
                    ],
                    "meta": {
                        "record_key": row.get("id"),
                        "answer": ref,
                        "answer_type": (row.get("meta") or {}).get("answer_type", "unknown"),
                        "answer_method": "rft-self-generated",
                        "cot_tokens_est": None,
                        "source": "rft",
                    },
                })
            if kept_for_question:
                stats["accepted_questions"] += 1
        print(f"[rft] processed {min(start + chunk, len(prompts))}/{len(prompts)} questions, "
              f"accepted {stats['accepted']} samples", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in accepted:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    accept_rate = stats["accepted"] / stats["samples"] if stats["samples"] else 0.0
    stats.update({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_model": args.model,
        "adapter": args.adapter,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "accept_rate": round(accept_rate, 4),
        "output": str(out_path),
    })
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] {stats['accepted']}/{stats['samples']} samples accepted "
          f"(rate {accept_rate:.1%}) -> {out_path}")
    print(f"[out]  {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

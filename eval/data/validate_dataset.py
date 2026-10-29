#!/usr/bin/env python3
"""Validate a processed SFT dataset before uploading it to the training server.

Checks: JSONL integrity, message roles, non-empty assistant turns, dialogue /
meta leakage in the assistant target, duplicate ids/questions, split
consistency, occasional missing final answers, and (optionally) token length
against a tokenizer's chat template.

Example:
    python data/validate_dataset.py
    python data/validate_dataset.py --data-dir data/processed --tokenizer Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
EVAL_ROOT = DATA_DIR.parent
sys.path.insert(0, str(DATA_DIR))

from text_clean import find_all_leaks, question_key  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"[error] {path.name}:{lineno}: invalid JSON: {exc}")
    return rows


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Validate processed SFT/eval JSONL")
    parser.add_argument("--data-dir", default=str(DATA_DIR / "processed"))
    parser.add_argument("--max-seq-len", type=int, default=0,
                        help="warn when a sample is longer than this after chat templating")
    parser.add_argument("--tokenizer", default="",
                        help="HF tokenizer name/path for the chat-template length check")
    parser.add_argument("--allow-leak", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    errors: list[str] = []
    warnings: list[str] = []

    expected = ["sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl"]
    present = [name for name in expected if (data_dir / name).exists()]
    if not present:
        sys.exit(f"[fatal] no SFT splits found in {data_dir}")
    missing_splits = [name for name in expected if name not in present]
    if missing_splits:
        warnings.append(f"missing splits (ok for tiny datasets): {missing_splits}")

    all_ids: Counter = Counter()
    all_questions: Counter = Counter()
    totals: dict[str, int] = {}
    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"could not load tokenizer {args.tokenizer}: {exc}")

    for name in present:
        rows = load_jsonl(data_dir / name)
        totals[name] = len(rows)
        if len(rows) == 0:
            warnings.append(f"{name}: empty split")
        max_tokens = 0
        for idx, row in enumerate(rows):
            loc = f"{name}:{idx}"
            if "id" not in row:
                errors.append(f"{loc}: missing id")
            all_ids[str(row.get("id"))] += 1
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                errors.append(f"{loc}: bad messages")
                continue
            roles = [m.get("role") for m in messages]
            if roles[-1] != "assistant":
                errors.append(f"{loc}: last role must be assistant, got {roles[-1]}")
            if "user" not in roles:
                errors.append(f"{loc}: no user message")
            assistant = messages[-1].get("content", "")
            if not assistant.strip():
                errors.append(f"{loc}: empty assistant content")
            leaks = find_all_leaks(assistant)
            leak_terms = leaks["dialogue"] + leaks["meta"]
            if leak_terms and not args.allow_leak:
                warnings.append(f"{loc}: leak terms {leak_terms}")
            if "meta" in row and row["meta"].get("question"):
                all_questions[question_key(row["meta"]["question"])] += 1
            if tokenizer is not None:
                try:
                    text = tokenizer.apply_chat_template(messages, tokenize=False)
                    n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                    max_tokens = max(max_tokens, n_tokens)
                    if args.max_seq_len and n_tokens > args.max_seq_len:
                        warnings.append(f"{loc}: {n_tokens} tokens > max_seq_len {args.max_seq_len}")
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"{loc}: chat template failed: {exc}")
                    tokenizer = None  # avoid spamming
        if tokenizer is not None:
            print(f"[{name}] {len(rows)} samples | max tokens={max_tokens}")
        else:
            print(f"[{name}] {len(rows)} samples")

    dup_ids = [k for k, v in all_ids.items() if v > 1]
    if dup_ids:
        warnings.append(f"duplicate ids: {dup_ids[:5]} (total {len(dup_ids)})")
    dup_questions = [k for k, v in all_questions.items() if v > 1]
    if dup_questions:
        warnings.append(f"duplicate questions across splits: {len(dup_questions)}")

    eval_test = data_dir / "eval_test.jsonl"
    if eval_test.exists():
        rows = load_jsonl(eval_test)
        missing = sum(1 for r in rows if not r.get("answer"))
        if missing:
            warnings.append(f"eval_test: {missing}/{len(rows)} rows without reference answer")
        print(f"[eval_test] {len(rows)} rows (missing answers: {missing})")

    print("\n== summary ==")
    print(f"files: {', '.join(f'{k}={v}' for k, v in totals.items())}")
    for warning in warnings:
        print(f"[warn] {warning}")
    for error in errors:
        print(f"[error] {error}")
    if errors:
        return 1
    print("[ok] dataset passed validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

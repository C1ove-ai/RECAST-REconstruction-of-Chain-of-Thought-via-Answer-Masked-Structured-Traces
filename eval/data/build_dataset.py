#!/usr/bin/env python3
"""Build the student SFT dataset from decoded RECAST runs.

Input : output/decoded_*.json  (question + teacher answer + decoded CoT)
        data/answers.json             (final answers extracted from teacher answers)
Output: data/processed/sft_{train,val,test}.jsonl   chat-format SFT samples
        data/processed/eval_{val,test}.jsonl        question+reference rows for harness/run_eval.py
        data/processed/dataset_stats.json

Target variants (--target):
  cot_answer       (default) assistant = cleaned reconstructed CoT + "最终答案：<answer>"
  cot_only                   assistant = cleaned reconstructed CoT only
  source_only                assistant = the teacher's raw answer (ablation baseline)
  cot_plus_source            assistant = CoT + teacher's full answer (the "everything" variant;
                             documented as NOT recommended: duplicates info and teaches the
                             teacher's dialogue presentation style)

Example:
    python data/build_dataset.py --decoded "../output/decoded_*.json"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
EVAL_ROOT = DATA_DIR.parent
RECAST_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(DATA_DIR))

from rec_io import load_answers, load_cleaned, load_decoded_records  # noqa: E402
from text_clean import (  # noqa: E402
    clean_cot,
    clean_source_answer,
    estimate_tokens,
    extract_answer_regex,
    find_all_leaks,
    guess_answer_type,
    question_key,
)

STUDENT_SYSTEM = "你是一个严谨的数学推理助手。解答时先写出清晰的推理步骤，最后单独给出最终答案。"
DEFAULT_STUDENT_SYSTEM_FILE = EVAL_ROOT / "system_prompt_student.txt"
TARGET_CHOICES = ("cot_answer", "cot_only", "source_only", "cot_plus_source")


def build_assistant(target: str, cot: str, answer: str | None, source_answer: str) -> str:
    if target == "cot_only":
        return cot
    if target == "cot_answer":
        return f"{cot}\n\n最终答案：{answer}" if answer else cot
    if target == "source_only":
        return clean_source_answer(source_answer)
    if target == "cot_plus_source":
        return f"{cot}\n\n{clean_source_answer(source_answer)}"
    raise ValueError(f"unknown target: {target}")


def resolve_answer(record: dict, answers: dict[str, dict]) -> tuple[str | None, str]:
    entry = answers.get(record["record_key"])
    if entry and entry.get("answer"):
        return str(entry["answer"]), entry.get("method", "sidecar")
    parsed = extract_answer_regex(clean_source_answer(record.get("source_answer", "")))
    if parsed["answer"]:
        return str(parsed["answer"]), f"regex:{parsed['method']}"
    return None, "none"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return float(ordered[idx])


def token_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0}
    return {
        "min": min(values),
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "max": max(values),
        "mean": round(sum(values) / len(values), 1),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_sizes(n: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    if n >= 10 and (val_ratio > 0 or test_ratio > 0):
        n_val = max(1, round(n * val_ratio)) if val_ratio > 0 else 0
        n_test = max(1, round(n * test_ratio)) if test_ratio > 0 else 0
    elif n >= 5:
        n_val = 1 if val_ratio > 0 else 0
        n_test = 1 if test_ratio > 0 else 0
    else:
        n_val = n_test = 0
    # keep a reasonable training set
    while n - n_val - n_test < max(1, n // 2) and (n_val > 0 or n_test > 0):
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        else:
            n_test -= 1
    return n_val, n_test


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Build SFT dataset from decoded RECAST runs")
    parser.add_argument("--decoded", nargs="+", default=[str(RECAST_ROOT / "output" / "decoded_*.json")])
    parser.add_argument("--answers", default=str(DATA_DIR / "answers.json"))
    parser.add_argument("--cleaned", default=str(DATA_DIR / "cleaned_cots.json"),
                        help="LLM-cleaned CoT overlay from data/clean_with_llm.py")
    parser.add_argument("--no-cleaned", action="store_true", help="ignore the LLM-cleaned overlay")
    parser.add_argument("--out-dir", default=str(DATA_DIR / "processed"))
    parser.add_argument("--target", choices=TARGET_CHOICES, default="cot_answer")
    parser.add_argument("--system-file", default="", help="override the student system prompt with a text file")
    parser.add_argument("--no-system", action="store_true", help="omit the system message")
    parser.add_argument("--min-cot-chars", type=int, default=200)
    parser.add_argument("--max-cot-chars", type=int, default=40000)
    parser.add_argument("--keep-leak", action="store_true",
                        help="keep samples whose CoT contains dialogue/meta leakage")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-eval-files", action="store_true")
    args = parser.parse_args()

    if args.system_file:
        system_prompt = Path(args.system_file).read_text(encoding="utf-8-sig").strip()
    elif args.no_system:
        system_prompt = ""
    elif DEFAULT_STUDENT_SYSTEM_FILE.exists():
        system_prompt = DEFAULT_STUDENT_SYSTEM_FILE.read_text(encoding="utf-8-sig").strip()
    else:
        system_prompt = STUDENT_SYSTEM

    records = load_decoded_records(args.decoded)
    if args.limit > 0:
        records = records[: args.limit]
    # newest runs first: when the same question appears in several runs, the
    # dedupe keeps the newest (usually best) decoding.
    records.sort(key=lambda rec: rec.get("_source_stem", ""), reverse=True)
    answers = load_answers(args.answers)
    cleaned = {} if args.no_cleaned else load_cleaned(args.cleaned)
    print(f"[info] {len(records)} decoded record(s), {len(answers)} extracted answer(s), "
          f"{len(cleaned)} LLM-cleaned CoT(s), target={args.target}")

    dropped: dict[str, int] = {}
    missing_answer = 0
    used_cleaned = 0
    cot_sources: dict[str, int] = {"llm_clean": 0, "raw": 0}
    samples: list[dict] = []
    seen_questions: dict[str, str] = {}

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    for record in records:
        key = record["record_key"]
        question = (record.get("question") or "").strip()
        cleaned_entry = cleaned.get(key, {})
        if cleaned_entry.get("status") == "ok" and cleaned_entry.get("cleaned_cot"):
            cot = clean_cot(cleaned_entry["cleaned_cot"])
            cot_source = "llm_clean"
        else:
            cot = clean_cot(record.get("decoded_cot") or "")
            cot_source = "raw"
        source_answer = record.get("source_answer") or ""

        if not question:
            drop("missing_question")
            continue
        if not cot:
            drop("missing_cot")
            continue
        if len(cot) < args.min_cot_chars:
            drop("short_cot")
            continue
        if len(cot) > args.max_cot_chars:
            drop("too_long_cot")
            continue

        leaks = find_all_leaks(cot)
        leak_terms = leaks["dialogue"] + leaks["meta"]
        if leak_terms and not args.keep_leak:
            drop("leak")
            continue

        answer, answer_method = resolve_answer(record, answers)
        if answer is None:
            missing_answer += 1

        assistant = build_assistant(args.target, cot, answer, source_answer)
        if not assistant.strip():
            drop("empty_assistant")
            continue

        qkey = question_key(question)
        if qkey in seen_questions:
            drop("duplicate_question")
            continue
        seen_questions[qkey] = key

        message_system = [{"role": "system", "content": system_prompt}] if system_prompt else []
        cot_sources[cot_source] += 1
        if cot_source == "llm_clean":
            used_cleaned += 1
        samples.append({
            "id": key,
            "source": record.get("_source_stem", ""),
            "messages": message_system + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant},
            ],
            "meta": {
                "record_key": key,
                "question": question,
                "answer": answer,
                "answer_type": guess_answer_type(answer),
                "answer_method": answer_method,
                "cot_source": cot_source,
                "cot_chars": len(cot),
                "cot_tokens_est": estimate_tokens(cot),
                "assistant_chars": len(assistant),
                "assistant_tokens_est": estimate_tokens(assistant),
                "leak_terms": leak_terms,
            },
        })

    if not samples:
        sys.exit(f"[fatal] no usable samples (dropped: {dropped})")

    rng = random.Random(args.seed)
    order = list(range(len(samples)))
    rng.shuffle(order)
    n_val, n_test = split_sizes(len(samples), args.val_ratio, args.test_ratio)
    test_idx = order[:n_test]
    val_idx = order[n_test:n_test + n_val]
    train_idx = order[n_test + n_val:]

    splits = {
        "train": [samples[i] for i in train_idx],
        "val": [samples[i] for i in val_idx],
        "test": [samples[i] for i in test_idx],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        if not rows:
            continue
        write_jsonl(out_dir / f"sft_{name}.jsonl", rows)

    if not args.no_eval_files:
        for name in ("val", "test"):
            rows = [
                {
                    "id": s["id"],
                    "source": s["source"],
                    "question": s["meta"]["question"],
                    "answer": s["meta"]["answer"],
                }
                for s in splits[name]
            ]
            if rows:
                write_jsonl(out_dir / f"eval_{name}.jsonl", rows)

    all_assistant = [s["meta"]["assistant_tokens_est"] for s in samples]
    all_cot = [s["meta"]["cot_tokens_est"] for s in samples]
    stats = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target": args.target,
        "system_prompt": system_prompt,
        "decoded_sources": [str(p) for p in args.decoded],
        "input_records": len(records),
        "cleaned_available": len(cleaned),
        "used_cleaned": used_cleaned,
        "cot_sources": cot_sources,
        "kept_samples": len(samples),
        "dropped": dropped,
        "missing_answer": missing_answer,
        "splits": {name: len(rows) for name, rows in splits.items()},
        "assistant_tokens_est": token_stats(all_assistant),
        "cot_tokens_est": token_stats(all_cot),
        "seed": args.seed,
    }
    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[done] kept={len(samples)} train/val/test="
          f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} "
          f"dropped={dropped} missing_answer={missing_answer}")
    print(f"[out]  {out_dir}")
    if not splits["test"]:
        print("[warn] test split is empty (too few samples); eval_test.jsonl was not written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

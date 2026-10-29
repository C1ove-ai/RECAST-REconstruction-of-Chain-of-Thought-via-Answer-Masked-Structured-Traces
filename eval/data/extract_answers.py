#!/usr/bin/env python3
"""Extract the final answer from each teacher (closed-model) answer.

The SFT target needs an explicit final answer, but the dataset only stores the
closed model's full answer (a teacher-student dialogue in the current data).
This script produces `data/answers.json`, a sidecar consumed by
`build_dataset.py`.

Modes:
  regex   - only local heuristics (fast, no cost, may miss/garble proofs)
  api     - only the LLM extractor (accurate, one cheap API call per record)
  hybrid  - regex when confident, LLM otherwise (default)

Example:
    python data/extract_answers.py --decoded "../output/decoded_*.json"

The API key is read from api.txt (or DEEPSEEK_API_KEY).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
EVAL_ROOT = DATA_DIR.parent
RECAST_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(DATA_DIR))
sys.path.insert(0, str(EVAL_ROOT / "harness"))

from rec_io import load_decoded_records  # noqa: E402
from text_clean import (  # noqa: E402
    clean_source_answer,
    extract_answer_regex,
    guess_answer_type,
)
from api_client import ChatClient, mask_key  # noqa: E402

EXTRACT_PROMPT = """你是严格的答案抽取器。下面是一道题和某个模型对它的完整解答。
请只抽取该解答给出的最终答案，不要解释、不要重新推理、不要加任何前缀、引号或格式说明。
- 数值/表达式题：输出最简形式的数值或表达式；
- 证明题：输出最终要证明的结论（一句话，尽量短）；
- 如果解答中确实没有给出明确答案，输出：无

【问题】
{question}

【解答】
{answer}
"""


def regex_extract(record: dict) -> dict:
    parsed = extract_answer_regex(clean_source_answer(record.get("source_answer", "")))
    answer = parsed["answer"]
    return {
        "answer": answer,
        "answer_type": guess_answer_type(answer),
        "method": f"regex:{parsed['method']}",
        "confidence": parsed["confidence"],
    }


def load_reference_answers(path: str | Path | None) -> dict[str, str]:
    """Map benchmark row id -> ground-truth answer (e.g. from collected test-set rows).

    Used when the teacher responses were collected on a benchmark that ships
    reference answers: prefer those over post-hoc extraction.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        sys.exit(f"[fatal] reference jsonl not found: {p}")
    mapping: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        row_id = item.get("id")
        answer = item.get("answer") or item.get("reference_answer")
        if row_id is not None and answer:
            mapping[str(row_id)] = str(answer).strip()
    return mapping


def api_extract(client: ChatClient, record: dict, max_tokens: int) -> dict:
    prompt = EXTRACT_PROMPT.format(
        question=record.get("question", ""),
        answer=clean_source_answer(record.get("source_answer", "")),
    )
    resp = client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    text = resp["content"].strip().strip('"').strip()
    answer = None if text in {"", "无", "None", "none", "N/A"} else text
    return {
        "answer": answer,
        "answer_type": guess_answer_type(answer),
        "method": "api",
        "confidence": 0.9 if answer else 0.0,
        "usage": _usage_to_dict(resp.get("usage")),
    }


def _usage_to_dict(usage) -> dict | None:
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Extract final answers from teacher answers")
    parser.add_argument("--decoded", nargs="+", default=[str(RECAST_ROOT / "output" / "decoded_*.json")],
                        help="decoded run JSON(s) or glob(s)")
    parser.add_argument("--out", default=str(DATA_DIR / "answers.json"))
    parser.add_argument("--mode", choices=["regex", "api", "hybrid"], default="hybrid")
    parser.add_argument("--api", default=str(RECAST_ROOT / "api.txt"), help="api key file")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="hybrid mode: use regex when confidence >= this")
    parser.add_argument("--reference-jsonl", default="",
                        help="benchmark rows with ground-truth answers; matched by decoded record id")
    parser.add_argument("--sleep", type=float, default=0.5, help="seconds between API calls")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="re-extract existing answers")
    args = parser.parse_args()

    records = load_decoded_records(args.decoded)
    if args.limit > 0:
        records = records[: args.limit]
    reference_answers = load_reference_answers(args.reference_jsonl)
    if reference_answers:
        print(f"[info] {len(reference_answers)} ground-truth answers loaded from {args.reference_jsonl}")

    out_path = Path(args.out)
    answers: dict[str, dict] = {}
    if out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8-sig"))
        answers = payload.get("answers", payload) if isinstance(payload, dict) else {}

    client: ChatClient | None = None
    if args.mode in {"api", "hybrid"}:
        client = ChatClient(model=args.model, base_url=args.base_url, api_file=args.api)
        print(f"[info] extractor model={args.model} base_url={args.base_url} "
              f"key={mask_key(client.key)} mode={args.mode}")

    stats = {"total": 0, "regex": 0, "api": 0, "reference": 0, "skipped": 0, "missing": 0}
    for idx, record in enumerate(records, 1):
        key = record["record_key"]
        stats["total"] += 1
        if key in answers and not args.force:
            stats["skipped"] += 1
            continue

        reference = reference_answers.get(str(record.get("id")))
        result: dict | None = None
        if reference:
            result = {
                "answer": reference,
                "answer_type": guess_answer_type(reference),
                "method": "testset_reference",
                "confidence": 1.0,
            }
            stats["reference"] += 1
        elif args.mode in {"regex", "hybrid"}:
            result = regex_extract(record)
            if args.mode == "hybrid" and result["confidence"] < args.confidence_threshold:
                result = None
            if result is not None:
                stats["regex"] += 1

        if result is None:
            assert client is not None
            try:
                result = api_extract(client, record, args.max_tokens)
            except Exception as exc:  # noqa: BLE001 - keep going, record the failure
                print(f"[warn] record {key}: API extraction failed: {exc}", file=sys.stderr)
                answers[key] = {
                    "answer": None,
                    "answer_type": "unknown",
                    "method": "failed",
                    "confidence": 0.0,
                    "error": str(exc),
                }
                continue
            stats["api"] += 1
            time.sleep(args.sleep)

        answers[key] = result
        if result.get("answer") is None:
            stats["missing"] += 1
        print(f"[{idx}/{len(records)}] {key} -> {result.get('answer')!r} "
              f"({result.get('method')}, {result.get('answer_type')})")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "model": args.model,
        "base_url": args.base_url,
        "source": [str(p) for p in args.decoded],
        "stats": stats,
        "answers": answers,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {len(answers)} answers -> {out_path} | stats={stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

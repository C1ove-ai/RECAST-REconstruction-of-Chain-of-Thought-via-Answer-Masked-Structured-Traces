#!/usr/bin/env python3
"""Download/format public math benchmarks into the harness JSONL schema:

    {"id": "...", "source": "gsm8k", "question": "...", "answer": "..."}

Presets use Hugging Face dataset ids that are reasonably stable, but dataset
ids/fields do change -- every preset can be overridden with:
    --hf <dataset> --config <config> --split <split> \
    --question-field <field> --answer-field <field> --name <name>

Examples:
    python data/download_benchmarks.py --bench gsm8k math500
    python data/download_benchmarks.py --bench aime24 --hf Maxwell-Jia/AIME_2024
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
BENCH_DIR = DATA_DIR / "benchmarks"

BENCHES: dict[str, dict] = {
    "gsm8k": {
        "hf": "openai/gsm8k", "config": "main", "split": "test",
        "question": "question", "answer": "answer",
    },
    "math500": {
        "hf": "HuggingFaceH4/MATH-500", "config": None, "split": "test",
        "question": "problem", "answer": "answer",
    },
    "amc23": {
        "hf": "math-ai/amc23", "config": None, "split": "test",
        "question": "question", "answer": "answer",
    },
    "aime24": {
        "hf": "Maxwell-Jia/AIME_2024", "config": None, "split": "train",
        "question": "Problem", "answer": "Answer",
    },
}


def clean_answer(value, source: str) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    text = str(value).strip()
    if source == "gsm8k" and "####" in text:
        text = text.split("####")[-1].strip()
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    return text


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Format public math benchmarks as JSONL")
    parser.add_argument("--bench", nargs="+", choices=sorted(BENCHES), default=["gsm8k", "math500"])
    parser.add_argument("--out-dir", default=str(BENCH_DIR))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--hf", default="", help="override HF dataset id for a single custom benchmark")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--name", default="custom")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[fatal] datasets package missing: pip install datasets")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.hf:
        specs = [{
            "name": args.name, "hf": args.hf, "config": args.config, "split": args.split,
            "question": args.question_field, "answer": args.answer_field,
        }]
    else:
        specs = [{"name": name, **BENCHES[name]} for name in args.bench]

    for spec in specs:
        name = spec["name"]
        print(f"[{name}] loading {spec['hf']} (config={spec['config']}, split={spec['split']}) ...")
        try:
            ds = load_dataset(spec["hf"], spec["config"], split=spec["split"])
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {name}: {exc}\n"
                  f"        dataset id/fields may have changed; rerun with "
                  f"--hf <dataset> --split <split> --question-field ... --answer-field ...",
                  file=sys.stderr)
            continue

        rows = []
        for idx, item in enumerate(ds):
            if args.limit and idx >= args.limit:
                break
            question = str(item.get(spec["question"], "")).strip()
            answer = clean_answer(item.get(spec["answer"], ""), name)
            if not question:
                continue
            rows.append({
                "id": f"{name}-{idx}",
                "source": name,
                "question": question,
                "answer": answer,
            })

        out_path = out_dir / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[ok] {name}: {len(rows)} rows -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

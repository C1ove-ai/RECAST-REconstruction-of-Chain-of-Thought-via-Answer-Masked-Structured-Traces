#!/usr/bin/env python3
"""Download and normalize standard math test sets into testset/.

Output schema (one JSON object per line):
    {"id", "source", "question", "answer", "solution"?, "subject"?, "level"?}

Sources (public research datasets, follow their upstream licenses):
  - GSM8K test.jsonl : openai/grade-school-math (MIT)
  - MATH-500 test.jsonl : HuggingFaceH4/MATH-500 (MATH subset, MIT)

Usage:
    python testset/download_testset.py                 # both
    python testset/download_testset.py --bench math500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

TESTDIR = Path(__file__).resolve().parent

SOURCES = {
    "gsm8k": {
        "urls": [
            "https://raw.githubusercontent.com/openai/grade-school-math/master/"
            "grade_school_math/data/test.jsonl",
        ],
        "out": "gsm8k_test.jsonl",
    },
    "math500": {
        "urls": [
            "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl",
            "https://hf-mirror.com/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl",
        ],
        "out": "math500_test.jsonl",
    },
}


def fetch(url: str, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (recast-testset)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_jsonl(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.decode("utf-8", "replace").splitlines() if line.strip()]


def normalize_gsm8k(items: list[dict]) -> list[dict]:
    rows = []
    for idx, item in enumerate(items, 1):
        full_answer = item.get("answer", "")
        final = full_answer.split("####")[-1].strip() if "####" in full_answer else full_answer.strip()
        rows.append({
            "id": f"gsm8k-{idx:04d}",
            "source": "gsm8k",
            "question": item.get("question", "").strip(),
            "answer": final,
            "solution": full_answer.strip(),
        })
    return rows


def normalize_math500(items: list[dict]) -> list[dict]:
    rows = []
    for idx, item in enumerate(items, 1):
        rows.append({
            "id": f"math500-{idx:04d}",
            "source": "math500",
            "question": item.get("problem", "").strip(),
            "answer": str(item.get("answer", "")).strip(),
            "solution": str(item.get("solution", "")).strip(),
            "subject": item.get("subject"),
            "level": item.get("level"),
            "unique_id": item.get("unique_id"),
        })
    return rows


NORMALIZERS = {"gsm8k": normalize_gsm8k, "math500": normalize_math500}


def download_bench(bench: str, out_dir: Path = TESTDIR) -> Path:
    spec = SOURCES[bench]
    out_path = out_dir / spec["out"]
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[{bench}] cached: {out_path}")
        return out_path

    last_error: Exception | None = None
    for url in spec["urls"]:
        try:
            print(f"[{bench}] downloading {url} ...")
            data = fetch(url)
            items = parse_jsonl(data)
            rows = NORMALIZERS[bench](items)
            out_dir.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
            print(f"[{bench}] ok: {len(rows)} rows -> {out_path} (sha256 {digest[:16]}...)")
            return out_path
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_error = exc
            print(f"[{bench}] failed {url}: {type(exc).__name__}: {str(exc)[:200]}")
    raise RuntimeError(f"[{bench}] all mirrors failed: {last_error}")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Download standard math test sets")
    parser.add_argument("--bench", nargs="+", choices=["gsm8k", "math500"], default=["gsm8k", "math500"])
    parser.add_argument("--out-dir", default=str(TESTDIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest = {
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "sources": {name: SOURCES[name] for name in args.bench},
        "files": {},
    }
    for bench in args.bench:
        path = download_bench(bench, out_dir)
        manifest["files"][bench] = {
            "path": path.name,
            "rows": sum(1 for _ in path.open(encoding="utf-8")),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] manifest -> {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

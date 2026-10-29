#!/usr/bin/env python3
"""Shared loading utilities for decoded RECAST outputs.

A decoded run is the JSON written by `recast_decode.py`, i.e.
`{"records": [{"id": 1, "question": ..., "source_answer": ..., "decoded_cot": ...}]}`.

Each record gets a stable `record_key` = "<run-stem>:<id>" so that answer
sidecars, SFT samples and eval rows can be joined across scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

EVAL_ROOT = Path(__file__).resolve().parent.parent
RECAST_ROOT = EVAL_ROOT.parent


def expand_paths(patterns: Iterable[str | Path]) -> list[Path]:
    """Expand shell globs (and plain paths) into existing files, sorted."""
    paths: list[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        if any(ch in str(pattern) for ch in "*?["):
            paths.extend(sorted(p.parent.glob(p.name)))
        elif p.exists():
            paths.append(p)
    # de-dupe, keep order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def load_decoded_records(patterns: Iterable[str | Path]) -> list[dict]:
    records: list[dict] = []
    for path in expand_paths(patterns):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        recs = payload.get("records", []) if isinstance(payload, dict) else payload
        for rec in recs:
            item = dict(rec)
            item["_source_file"] = path.name
            item["_source_stem"] = path.stem
            item["record_key"] = f"{path.stem}:{rec.get('id')}"
            records.append(item)
    # stable unique keys even if the same run was passed twice
    seen: dict[str, int] = {}
    for item in records:
        key = item["record_key"]
        if key in seen:
            seen[key] += 1
            item["record_key"] = f"{key}#dup{seen[key]}"
        else:
            seen[key] = 0
    return records


def load_answers(path: str | Path) -> dict[str, dict]:
    """Load the answer sidecar produced by data/extract_answers.py."""
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8-sig"))
    answers = payload.get("answers", payload) if isinstance(payload, dict) else {}
    if not isinstance(answers, dict):
        return {}
    out: dict[str, dict] = {}
    for key, value in answers.items():
        if isinstance(value, str):
            out[str(key)] = {"answer": value, "answer_type": "unknown", "method": "sidecar"}
        elif isinstance(value, dict):
            out[str(key)] = value
    return out


def load_cleaned(path: str | Path) -> dict[str, dict]:
    """Load the LLM-cleaned CoT overlay produced by data/clean_with_llm.py."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    cleaned = payload.get("cleaned", payload) if isinstance(payload, dict) else {}
    return cleaned if isinstance(cleaned, dict) else {}

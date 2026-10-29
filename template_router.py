#!/usr/bin/env python3
"""Choose an elicitation pattern (prompt template) for a dataset.

The pattern decides *how the target model externalises its reasoning*:
  math        -> math_solver   (step-by-step derivation + verification)
  code        -> pair_programming (driver-navigator + tests)
  science     -> expert_panel  (multi-perspective decomposition)
  logic       -> socratic      (clarify goals/constraints/assumptions)
  planning    -> socratic      (clarify then plan)
  open_ended  -> retrospective (solve, then review/expand requirements)
  general     -> socratic

Selection order:
  1. dataset name -> registry["datasets"]
  2. keyword scoring over the first rows (math / code / science / logic / ...)
  3. fallback profile "general"
An explicit --profile overrides 1-3; an explicit --pattern overrides the result.

Examples:
    python template_router.py --data testset/math500_test.jsonl
    python template_router.py --source humaneval --question "Write a function ..."
    python template_router.py --profile code --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
PATTERN_DIR = BASE / "templates" / "patterns"
REGISTRY_PATH = PATTERN_DIR / "registry.json"

_NORMALIZE_SUFFIXES = ("_test", "-test", "_valid", "-valid", "_validation", "_dev", "_sample", "_subset")


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    if not path.exists():
        sys.exit(f"[fatal] pattern registry not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_source(source: str | Path | None) -> str:
    if not source:
        return ""
    name = Path(str(source)).name
    name = re.sub(r"\.(jsonl|json|txt)$", "", name, flags=re.I)
    name = name.lower().strip()
    for suffix in _NORMALIZE_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    name = re.sub(r"[^a-z0-9_]+", "_", name).strip("_")
    return name


def _dataset_profile(source_key: str, registry: dict) -> str | None:
    datasets = {normalize_source(k): v for k, v in registry.get("datasets", {}).items()}
    if source_key in datasets:
        return datasets[source_key]
    # allow prefix matches such as "math500_v2"
    for name, profile in datasets.items():
        if name and source_key.startswith(name + "_"):
            return profile
    return None


def _score_keywords(rows: list[dict], registry: dict, limit: int = 20) -> dict[str, int]:
    keywords = registry.get("keywords", {})
    scores = {profile: 0 for profile in registry.get("profiles", {})}
    for row in rows[:limit]:
        text = f"{row.get('question', '')}\n{row.get('prompt', '')}".lower()
        if not text.strip():
            continue
        for profile, words in keywords.items():
            scores.setdefault(profile, 0)
            for word in words:
                if word and word.lower() in text:
                    scores[profile] += 1
    return scores


def infer_profile(rows: list[dict], source_hint: str | None = None,
                  registry: dict | None = None) -> dict:
    registry = registry or load_registry()
    source_key = normalize_source(source_hint)
    mapped = _dataset_profile(source_key, registry)
    if mapped:
        return {"profile": mapped, "reason": "dataset_map", "source_key": source_key,
                "scores": {}, "pattern": None}

    scores = _score_keywords(rows, registry)
    # deterministic tie-break follows the preference order below
    order = ["math", "code", "science", "logic", "planning", "open_ended"]
    best_profile, best_score = "general", 0
    for profile in order:
        score = scores.get(profile, 0)
        if score > best_score:
            best_profile, best_score = profile, score
    if best_score <= 0:
        return {"profile": "general", "reason": "fallback", "source_key": source_key,
                "scores": scores, "pattern": None}
    return {"profile": best_profile, "reason": "keywords", "source_key": source_key,
            "scores": scores, "pattern": None}


def select_patterns(profile: str, registry: dict | None = None) -> dict:
    registry = registry or load_registry()
    spec = registry.get("profiles", {}).get(profile)
    if not spec:
        spec = registry["profiles"]["general"]
        profile = "general"
    return {
        "profile": profile,
        "primary": spec["primary"],
        "secondary": spec.get("secondary", ""),
        "description": spec.get("description", ""),
    }


def pattern_path(pattern: str, registry: dict | None = None) -> Path:
    registry = registry or load_registry()
    spec = registry.get("patterns", {}).get(pattern)
    if not spec:
        raise KeyError(f"unknown pattern: {pattern}")
    return PATTERN_DIR / spec["file"]


def load_pattern_text(pattern: str, registry: dict | None = None) -> str:
    path = pattern_path(pattern, registry)
    if not path.exists():
        raise FileNotFoundError(f"pattern prompt not found: {path}")
    text = path.read_text(encoding="utf-8-sig").strip()
    if "{user_query}" not in text:
        raise ValueError(f"{path} has no {{user_query}} placeholder")
    return text


def route(rows: list[dict], source_hint: str | None = None,
          profile_override: str = "", pattern_override: str = "auto",
          registry: dict | None = None) -> dict:
    registry = registry or load_registry()
    profile_info = infer_profile(rows, source_hint, registry)
    profile = profile_override or profile_info["profile"]
    if profile not in registry.get("profiles", {}):
        sys.exit(f"[fatal] unknown profile: {profile}")
    selected = select_patterns(profile, registry)
    pattern = selected["primary"] if pattern_override == "auto" else pattern_override
    if pattern not in registry.get("patterns", {}):
        sys.exit(f"[fatal] unknown pattern: {pattern}")
    return {
        "source": source_hint or "",
        "source_key": profile_info["source_key"],
        "profile": profile,
        "profile_reason": profile_info["reason"],
        "profile_scores": profile_info["scores"],
        "pattern": pattern,
        "secondary": selected["secondary"],
        "pattern_file": str(pattern_path(pattern, registry)),
        "pattern_description": registry["patterns"][pattern].get("description", ""),
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Choose an elicitation pattern for a dataset")
    parser.add_argument("--data", default="", help="dataset JSONL (question/answer rows)")
    parser.add_argument("--source", default="", help="dataset/source name (e.g. gsm8k, humaneval)")
    parser.add_argument("--question", default="", help="a single question for ad-hoc routing")
    parser.add_argument("--profile", default="", help="override profile (math/code/science/logic/planning/open_ended)")
    parser.add_argument("--pattern", default="auto", help="override pattern id (or 'auto')")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    rows: list[dict] = []
    source_hint = args.source
    if args.question:
        rows = [{"question": args.question}]
    elif args.data:
        path = Path(args.data)
        if not path.exists():
            sys.exit(f"[fatal] data not found: {path}")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        source_hint = source_hint or path.stem
    else:
        sys.exit("[fatal] provide --data or --question")

    result = route(rows, source_hint, args.profile, args.pattern)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"source        : {result['source']}")
    print(f"profile       : {result['profile']} ({result['profile_reason']})")
    print(f"primary       : {result['pattern']}  [{result['pattern_file']}]")
    print(f"secondary     : {result['secondary'] or '-'}")
    print(f"description   : {result['pattern_description']}")
    if result.get("profile_scores"):
        print(f"keyword scores: {result['profile_scores']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

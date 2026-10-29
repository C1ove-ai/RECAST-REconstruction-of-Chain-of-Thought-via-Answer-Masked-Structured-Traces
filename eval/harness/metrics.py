#!/usr/bin/env python3
"""Metrics for math-reasoning evaluation.

Records are per-question dicts produced by harness/run_eval.py:
    {"id": ..., "preds": ["..."], "correct": [true, false, ...],
     "tokens": [123, ...], "ref": "..."}

Metric definitions:
  pass@1 : accuracy of the first sample (greedy decode when temperature=0)
  avg@k  : mean accuracy over all k samples (average accuracy / avg@k)
  maj@k  : majority-vote accuracy over the k sampled answers
  invalid: fraction of samples from which no final answer could be extracted

Also provides bootstrap confidence intervals; use them on AIME-sized sets
(30 questions) or any small benchmark.
"""

from __future__ import annotations

import random
from collections import Counter

from math_verify import normalize_answer


def pass_at_1(records: list[dict]) -> float:
    values = [1.0 if rec["correct"][0] else 0.0 for rec in records if rec.get("correct")]
    return sum(values) / len(values) if values else 0.0


def avg_at_k(records: list[dict]) -> float:
    values = [1.0 if c else 0.0 for rec in records for c in rec.get("correct", [])]
    return sum(values) / len(values) if values else 0.0


def maj_at_k(records: list[dict]) -> float:
    hits = 0
    total = 0
    for rec in records:
        preds = [normalize_answer(p) for p in rec.get("preds", []) if p]
        if not preds or not rec.get("ref"):
            continue
        total += 1
        winner = Counter(preds).most_common(1)[0][0]
        if winner == normalize_answer(rec["ref"]):
            hits += 1
    return hits / total if total else 0.0


def invalid_rate(records: list[dict]) -> float:
    total = 0
    empty = 0
    for rec in records:
        for pred in rec.get("preds", []):
            total += 1
            if not pred:
                empty += 1
    return empty / total if total else 0.0


def token_stats(records: list[dict]) -> dict[str, float]:
    values = [t for rec in records for t in rec.get("tokens", []) if t is not None]
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))],
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 1),
    }


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return (round(lo, 4), round(hi, 4))


def summarize(records: list[dict]) -> dict:
    scored = [rec for rec in records if rec.get("scored")]
    values = [1.0 if rec["correct"][0] else 0.0 for rec in scored if rec.get("correct")]
    return {
        "num_questions": len(records),
        "num_scored": len(scored),
        "k": max((len(rec.get("preds", [])) for rec in scored), default=0),
        "pass@1": round(pass_at_1(scored), 4),
        "avg@k": round(avg_at_k(scored), 4),
        "maj@k": round(maj_at_k(scored), 4),
        "pass@1_ci95": list(bootstrap_ci(values)),
        "invalid_rate": round(invalid_rate(scored), 4),
        "tokens": token_stats(scored),
    }

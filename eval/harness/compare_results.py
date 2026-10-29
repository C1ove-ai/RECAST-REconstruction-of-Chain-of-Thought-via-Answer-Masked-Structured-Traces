#!/usr/bin/env python3
"""Merge `harness/run_eval.py` summaries into one markdown comparison table.

Example:
    python harness/compare_results.py results/*.json --out results/summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_summary(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def fmt(value, ndigits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def data_stem(path: str) -> str:
    return Path(path).stem if path else "-"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Compare run_eval summaries")
    parser.add_argument("results", nargs="+")
    parser.add_argument("--baseline", default="", help="name substring of the baseline run")
    parser.add_argument("--out", default="", help="write markdown table here")
    args = parser.parse_args()

    rows = [load_summary(path) for path in args.results]
    baseline = next((r for r in rows if args.baseline and args.baseline in r["name"]), None)

    header = ("| run | data | k | pass@1 | avg@k | maj@k | invalid | tokens(mean) "
              "| pass@1 Δ |")
    lines = [header, "|" + "---|" * 10]
    for row in sorted(rows, key=lambda r: (data_stem(r["data"]), -r.get("metrics", {}).get("pass@1", 0))):
        metrics = row.get("metrics", {})
        delta = ""
        if baseline is not None and row is not baseline and data_stem(row["data"]) == data_stem(baseline["data"]):
            diff = metrics.get("pass@1", 0) - baseline.get("metrics", {}).get("pass@1", 0)
            delta = f"{diff:+.3f}"
        lines.append(
            f"| {row.get('name')} | {data_stem(row.get('data', ''))} | {row.get('k')} "
            f"| {fmt(metrics.get('pass@1'))} | {fmt(metrics.get('avg@k'))} "
            f"| {fmt(metrics.get('maj@k'))} | {fmt(metrics.get('invalid_rate'))} "
            f"| {fmt(metrics.get('tokens', {}).get('mean'), 1)} | {delta} |"
        )

    table = "\n".join(lines)
    print(table)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table + "\n", encoding="utf-8")
        print(f"[out] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

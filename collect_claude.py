#!/usr/bin/env python3
"""Batch-collect Claude responses on a local test set.

Fills the {user_query} placeholder of a teacher-student prompt with each
question, calls the relay once per question, and saves one JSONL record per
response (resumable, concurrent, error-tolerant).

Prompt selection:
  --pattern auto (default) routes the dataset to a reasoning pattern via
  template_router.py: math -> math_solver, code -> pair_programming,
  science -> expert_panel, logic/planning -> socratic, open-ended ->
  retrospective. The chain is: pattern -> secondary pattern -> medium -> short.
  --pattern none restores the old long -> medium -> short chain.
  --pattern <id> forces one pattern; --profile <name> forces a dataset profile.

For each prompt the original question is tried first, then a LaTeX-light
variant. The first success is recorded with `prompt_variant`/`question_variant`;
failed attempts are kept in `attempts`.

Usage:
    python collect_claude.py                          # MATH-500, all 500
    python collect_claude.py --limit 5                # smoke test
    python collect_claude.py --bench gsm8k --limit 100
    python collect_claude.py --data testset/math500_test.jsonl --workers 4
    python collect_claude.py --bench math500 --pattern none      # old chain
    python collect_claude.py --bench math500 --pattern math_solver
    python collect_claude.py --bench math500 --profile code      # manual profile
    python collect_claude.py --prompt-file my.txt                # single custom prompt

The test set is downloaded automatically on first use (see testset/).
Config defaults to config.relay.json in the repository root,
or pass --config / --url / --model / --api-key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "testset"))

from ask_claude import PROMPT  # noqa: E402
from download_testset import download_bench  # noqa: E402
from template_router import (  # noqa: E402
    load_pattern_text,
    load_registry,
    pattern_path,
    route,
)

DEFAULT_BENCH = "math500"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.7

# fallback chain: long (ask_claude.PROMPT) -> medium -> short
MEDIUM_PROMPT_FILE = BASE / "templates" / "ask_claude_medium.txt"
SHORT_PROMPT_FILE = BASE / "templates" / "ask_claude_short.txt"


def mask_key(key: str) -> str:
    return key[:7] + "..." + key[-4:] if len(key) > 12 else "***"


def _read_template(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8-sig").strip()
    return text or None


def _generic_chain(args: argparse.Namespace) -> list[dict]:
    """long -> medium -> short prompt variants used to dodge relay-side filters."""
    chain: list[dict] = [{"name": "long", "path": "ask_claude.PROMPT", "text": PROMPT}]
    for name, path in (("medium", MEDIUM_PROMPT_FILE), ("short", SHORT_PROMPT_FILE)):
        text = _read_template(path)
        if text and "{user_query}" in text:
            chain.append({"name": name, "path": str(path), "text": text})
        elif text:
            print(f"[warn] {path} has no {{user_query}} placeholder; skipped", file=sys.stderr)
        else:
            print(f"[warn] prompt fallback missing: {path}", file=sys.stderr)
    if args.no_fallback:
        chain = chain[:1]
    return chain


def build_prompt_chain(args: argparse.Namespace, rows: list[dict] | None = None,
                       source_hint: str = "") -> tuple[list[dict], dict]:
    """Build the prompt chain and return (chain, routing info).

    Default: route the dataset to a reasoning pattern (math_solver / pair_programming /
    expert_panel / socratic / retrospective), then fall back to the generic
    medium/short prompts if the relay filters the pattern prompt.
    `--pattern none` restores the old long -> medium -> short chain.
    """
    if args.prompt_file:
        path = Path(args.prompt_file)
        if not path.exists():
            sys.exit(f"[fatal] prompt file not found: {path}")
        text = _read_template(path)
        if not text or "{user_query}" not in text:
            sys.exit(f"[fatal] {path} has no {{user_query}} placeholder")
        return [{"name": "custom", "path": str(path), "text": text}], {
            "profile": None, "pattern": None, "secondary": None, "reason": "custom_prompt"}

    registry = load_registry()

    if args.pattern == "none":
        return _generic_chain(args), {
            "profile": None, "pattern": None, "secondary": None, "reason": "disabled"}

    if args.pattern != "auto" and args.pattern not in registry.get("patterns", {}):
        sys.exit(f"[fatal] unknown pattern: {args.pattern}; options: auto, none, "
                 f"{', '.join(sorted(registry.get('patterns', {})))}")
    if args.profile and args.profile not in registry.get("profiles", {}):
        sys.exit(f"[fatal] unknown profile: {args.profile}; options: "
                 f"{', '.join(sorted(registry.get('profiles', {})))}")

    info = route(rows or [], source_hint, args.profile, args.pattern, registry)
    pattern_names: list[str] = [info["pattern"]]
    if info.get("secondary") and info["secondary"] != info["pattern"]:
        pattern_names.append(info["secondary"])

    chain: list[dict] = []
    for name in pattern_names:
        try:
            chain.append({
                "name": f"pattern:{name}",
                "path": str(pattern_path(name, registry)),
                "text": load_pattern_text(name, registry),
            })
        except Exception as exc:  # noqa: BLE001 - registry/file drift should not kill the run
            print(f"[warn] pattern {name} unavailable: {exc}", file=sys.stderr)
    if not chain:
        print("[warn] no pattern prompt available; falling back to generic chain", file=sys.stderr)
        info = dict(info)
        info["pattern"] = None
        return _generic_chain(args), info
    # generic prompts remain the final filter fallback
    for name, path in (("medium", MEDIUM_PROMPT_FILE), ("short", SHORT_PROMPT_FILE)):
        text = _read_template(path)
        if text and "{user_query}" in text:
            chain.append({"name": name, "path": str(path), "text": text})
    if args.no_fallback:
        chain = chain[:1]
    return chain, info


def load_config(args: argparse.Namespace) -> dict:
    candidates = [Path(args.config)] if args.config else [BASE / "config.relay.json"]
    cfg: dict = {}
    cfg_path: Path | None = None
    for path in candidates:
        if path and path.exists():
            cfg = json.loads(path.read_text(encoding="utf-8-sig"))
            cfg_path = path
            break
    base_url = (args.url or cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
    model = args.model or cfg.get("model") or "claude-sonnet-5-5"
    api_key = args.api_key or cfg.get("api_key")
    if not api_key:
        sys.exit("[fatal] no API key: pass --api-key, or put api_key in config.relay.json")
    return {
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "config_path": str(cfg_path) if cfg_path else "",
    }


def fill_prompt(question: str, template: str = PROMPT) -> str:
    if "{user_query}" not in template:
        sys.exit("[fatal] prompt template has no {user_query} placeholder")
    return template.replace("{user_query}", question)


def plain_math(text: str) -> str:
    """LaTeX-light version of a question, used only when the relay filters the
    original formatting (some questions trigger a spurious content_filter)."""
    q = text.replace("$", "")
    q = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", q)
    q = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", q)
    for pattern, replacement in (
        (r"\\theta", "theta"), (r"\\pi", "pi"), (r"\\alpha", "alpha"),
        (r"\\beta", "beta"), (r"\\gamma", "gamma"), (r"\\phi", "phi"),
        (r"\\le(?:q)?", "<="), (r"\\ge(?:q)?", ">="), (r"\\ne(?:q)?", "!="),
        (r"\\times|\\cdot", "*"), (r"\\left|\\right", ""),
        (r"\\,|\\;|\\!|\\ ", " "), (r"\\[a-zA-Z]+", ""),
    ):
        q = re.sub(pattern, replacement, q)
    q = q.replace("{", "").replace("}", "")
    q = re.sub(r"\s+", " ", q).strip()
    return q


def call_claude(prompt: str, cfg: dict, args: argparse.Namespace) -> dict:
    body = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last_error = ""
    for attempt in range(1, args.retries + 1):
        started = time.perf_counter()
        req = urllib.request.Request(
            f"{cfg['base_url']}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {cfg['api_key']}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
            elapsed = time.perf_counter() - started
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            if isinstance(content, list):  # some relays return content blocks
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            if choice.get("finish_reason") == "content_filter":
                # retrying the identical prompt does not help; process_row falls
                # back to a LaTeX-light question variant instead
                return {"error": "content_filter"}
            if not content.strip():
                last_error = f"empty response (finish_reason={choice.get('finish_reason')})"
                if attempt < args.retries:
                    time.sleep(min(2 ** attempt, 20))
                    continue
                return {"error": last_error}
            return {
                "response": content,
                "usage": data.get("usage") or {},
                "finish_reason": choice.get("finish_reason"),
                "elapsed_s": round(elapsed, 2),
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:1000]
            last_error = f"HTTP {exc.code}: {detail}"
            if exc.code in (400, 401, 403, 404):
                break
            time.sleep(min(2 ** attempt, 30))
        except Exception as exc:  # noqa: BLE001 - retry transient network errors
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt, 30))
    return {"error": last_error or "unknown error"}


def _attempts(question: str) -> list[tuple[str, str]]:
    variants = [("original", question)]
    plain = plain_math(question)
    if plain and plain != question:
        variants.append(("plain_math", plain))
    return variants


def process_row(row: dict, cfg: dict, args: argparse.Namespace) -> dict:
    question = str(row.get("question") or "").strip()
    chain = getattr(args, "prompt_chain", [{"name": "long", "text": PROMPT}])
    prompt_variant = None
    question_variant = None
    errors: list[str] = []
    attempts_log: list[dict] = []
    result: dict = {"error": "empty question"}

    if question:
        for template in chain:
            for qvariant, qtext in _attempts(question):
                result = call_claude(fill_prompt(qtext, template["text"]), cfg, args)
                if not result.get("error"):
                    prompt_variant = template["name"]
                    question_variant = qvariant
                    errors = []
                    break
                error_text = str(result.get("error"))
                errors.append(f"{template['name']}/{qvariant}: {error_text}")
                attempts_log.append({
                    "prompt_variant": template["name"],
                    "question_variant": qvariant,
                    "error": error_text,
                })
            if prompt_variant is not None:
                break
        if prompt_variant is None and errors:
            result = {"error": "; ".join(errors)}

    record = {
        "id": row.get("id"),
        "source": row.get("source", args.bench),
        "question": question,
        "dataset_profile": (getattr(args, "pattern_info", {}) or {}).get("profile"),
        "pattern": (getattr(args, "pattern_info", {}) or {}).get("pattern"),
        "prompt_variant": prompt_variant,
        "question_variant": question_variant,
        "attempts": attempts_log,
        "answer": row.get("answer"),
        "reference_answer": row.get("answer"),
        "subject": row.get("subject"),
        "level": row.get("level"),
        "model": cfg["model"],
        "response": result.get("response"),
        "usage": result.get("usage"),
        "finish_reason": result.get("finish_reason"),
        "elapsed_s": result.get("elapsed_s"),
        "error": result.get("error"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    return record


def load_rows(path: Path, args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.start:
        rows = rows[args.start:]
    if args.limit > 0:
        rows = rows[: args.limit]
    return rows


def existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(str(json.loads(line).get("id")))
            except json.JSONDecodeError:
                continue
    return ids


def read_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Batch-collect Claude responses on a test set")
    parser.add_argument("--bench", choices=["math500", "gsm8k"], default=DEFAULT_BENCH)
    parser.add_argument("--data", default="", help="custom test-set JSONL (overrides --bench)")
    parser.add_argument("--out", default="", help="output JSONL (default: runs/claude_<bench>_<stamp>.jsonl)")
    parser.add_argument("--config", default="", help="relay config JSON")
    parser.add_argument("--url", default="", help="base URL override")
    parser.add_argument("--model", default="", help="model id override")
    parser.add_argument("--api-key", default="", help="API key override (avoid using on shared shells)")
    parser.add_argument("--prompt-file", default="",
                        help="custom prompt template containing {user_query} (disables the fallback chain)")
    parser.add_argument("--pattern", default="auto",
                        help="auto|none|math_solver|socratic|expert_panel|pair_programming|red_blue|retrospective")
    parser.add_argument("--profile", default="",
                        help="override dataset profile: math|code|science|logic|planning|open_ended|general")
    parser.add_argument("--no-fallback", action="store_true",
                        help="only use the first template (pattern or long); no medium/short fallback")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite the output file")
    parser.add_argument("--retry-errors", action="store_true",
                        help="drop errored records from an existing output file and retry them")
    parser.add_argument("--dry-run", action="store_true", help="print the filled prompt and exit")
    args = parser.parse_args()

    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            sys.exit(f"[fatal] data not found: {data_path}")
        bench_label = data_path.stem
    else:
        bench_label = args.bench
        data_path = BASE / "testset" / f"{args.bench}_test.jsonl"
        if not data_path.exists():
            print(f"[info] {data_path} missing, downloading ...")
            download_bench(args.bench, BASE / "testset")

    rows = load_rows(data_path, args)
    if not rows:
        sys.exit(f"[fatal] no rows in {data_path}")

    prompt_chain, pattern_info = build_prompt_chain(
        args, rows, (rows[0].get("source") if rows and rows[0].get("source") else bench_label)
    )
    args.prompt_chain = prompt_chain
    args.pattern_info = pattern_info
    prompt_chain_label = ", ".join(item["name"] for item in prompt_chain)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out) if args.out else BASE / "runs" / f"claude_{bench_label}_{stamp}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.force and out_path.exists():
        out_path.unlink()
    if args.retry_errors and out_path.exists() and not args.force:
        kept = [rec for rec in read_records(out_path) if not rec.get("error")]
        with out_path.open("w", encoding="utf-8") as fh:
            for rec in kept:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[retry-errors] kept {len(kept)} successful record(s)")
    done_ids = set() if args.force else existing_ids(out_path)
    pending = [row for row in rows if str(row.get("id")) not in done_ids]
    if not pending:
        print(f"[done] all {len(rows)} rows already collected -> {out_path}")
        return 0

    cfg = load_config(args)
    print(f"[info] data={data_path} rows={len(rows)} pending={len(pending)} "
          f"model={cfg['model']} base_url={cfg['base_url']} key={mask_key(cfg['api_key'])} "
          f"temp={args.temperature} max_tokens={args.max_tokens} workers={args.workers} "
          f"profile={pattern_info.get('profile')} pattern={pattern_info.get('pattern')} "
          f"prompt_chain={prompt_chain_label}")
    print(f"[out]  {out_path}")
    if args.dry_run:
        question = str(pending[0].get("question", ""))
        for item in prompt_chain:
            print(f"===== prompt variant: {item['name']} ({item.get('path', '')}) =====")
            print(fill_prompt(question, item["text"]))
            print()
        return 0
    if len(pending) >= 20:
        print(f"[warn] this run will make {len(pending)} API calls; use --limit for a smoke test")

    started = time.perf_counter()
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    variant_counts: dict[str, int] = {}
    errors = 0
    finished = 0

    mode = "w" if (args.force or not out_path.exists()) else "a"
    with out_path.open(mode, encoding="utf-8") as fh:
        if args.workers <= 1:
            for row in pending:
                record = process_row(row, cfg, args)
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                finished += 1
                if record.get("error"):
                    errors += 1
                    print(f"[{finished}/{len(pending)}] {record['id']}: ERROR {record['error'][:160]}", flush=True)
                else:
                    usage = record.get("usage") or {}
                    for key in total_usage:
                        total_usage[key] += usage.get(key) or 0
                    variant_key = f"{record.get('prompt_variant')}/{record.get('question_variant')}"
                    variant_counts[variant_key] = variant_counts.get(variant_key, 0) + 1
                    print(f"[{finished}/{len(pending)}] {record['id']}: ok "
                          f"[{variant_key}] finish={record['finish_reason']} "
                          f"{record['elapsed_s']}s tokens={usage.get('completion_tokens')}",
                          flush=True)
                if args.sleep:
                    time.sleep(args.sleep)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_row, row, cfg, args): row for row in pending}
                for future in as_completed(futures):
                    try:
                        record = future.result()
                    except Exception as exc:  # noqa: BLE001
                        row = futures[future]
                        record = {"id": row.get("id"), "question": row.get("question"),
                                  "error": f"{type(exc).__name__}: {exc}", "response": None}
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                    finished += 1
                    if record.get("error"):
                        errors += 1
                        print(f"[{finished}/{len(pending)}] {record['id']}: ERROR {str(record['error'])[:160]}", flush=True)
                    else:
                        usage = record.get("usage") or {}
                        for key in total_usage:
                            total_usage[key] += usage.get(key) or 0
                        variant_key = f"{record.get('prompt_variant')}/{record.get('question_variant')}"
                        variant_counts[variant_key] = variant_counts.get(variant_key, 0) + 1
                        print(f"[{finished}/{len(pending)}] {record['id']}: ok "
                              f"[{variant_key}] finish={record['finish_reason']} "
                              f"{record['elapsed_s']}s tokens={usage.get('completion_tokens')}",
                              flush=True)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data": str(data_path),
        "out": str(out_path),
        "bench": bench_label,
        "model": cfg["model"],
        "base_url": cfg["base_url"],
        "config_path": cfg["config_path"],
        "dataset_profile": pattern_info.get("profile"),
        "pattern": pattern_info.get("pattern"),
        "pattern_secondary": pattern_info.get("secondary"),
        "routing_reason": pattern_info.get("reason"),
        "profile_scores": pattern_info.get("scores") or {},
        "num_rows": len(rows),
        "num_pending_at_start": len(pending),
        "num_finished": finished,
        "num_errors": errors,
        "usage_totals": total_usage,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "workers": args.workers,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "prompt_chain": [
            {
                "name": item["name"],
                "path": item.get("path", ""),
                "chars": len(item["text"]),
                "sha256": hashlib.sha256(item["text"].encode("utf-8")).hexdigest(),
                "text": item["text"],
            }
            for item in prompt_chain
        ],
        "prompt_variant_counts": variant_counts,
        "prompt_template_sha256": hashlib.sha256(prompt_chain[0]["text"].encode("utf-8")).hexdigest(),
        "prompt_template_chars": len(prompt_chain[0]["text"]),
        "prompt_template": prompt_chain[0]["text"],
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {finished} collected, {errors} errors, "
          f"usage={total_usage}, elapsed={summary['elapsed_s']}s")
    print(f"[out] {out_path}")
    print(f"[out] {summary_path}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

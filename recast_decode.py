#!/usr/bin/env python3
"""RECAST decoder runner.

Reads api.txt / decoder_template.txt / decode_dataset.txt from the RECAST
folder, fills the template placeholders {问题} and {完整回答} with each
(question, closed-source full answer) pair, calls the DeepSeek official API,
and stores the reconstructed rationales (CoT) next to the run.

Usage:
    python recast_decode.py                 # run all records
    python recast_decode.py --dry-run       # only print prompts, no API call
    python recast_decode.py --limit 1       # first N records only
    python recast_decode.py --model deepseek-reasoner
    python recast_decode.py --min-ratio 1.5 # decoded CoT >= 1.5x source-answer tokens
    python recast_decode.py --min-tokens 1500 --no-min  # absolute floor / disable

A system prompt (system_prompt.txt, if present) is prepended to force the
decoder into the closed model's first-person persona. Use --no-system to
disable it, or --system <path> to point at another file.

Length floor: the reconstructed rationale must be at least as long as the
visible closed-model answer, which is the observable proxy for its reasoning
trace. The floor is max(--min-tokens, ceil(--min-ratio * answer_tokens)).
Token counts of the decoded output come from the API usage; answer tokens are
estimated with TOKENS_PER_CHAR, calibrated on DeepSeek usage for this
Chinese-heavy text. Short outputs are re-asked with an expansion instruction
(--expand-retries).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
API_FILE = ROOT / "api.txt"
TEMPLATE_FILE = ROOT / "decoder_template.txt"
SYSTEM_FILE = ROOT / "system_prompt.txt"
DATASET_FILE = ROOT / "decode_dataset.txt"
OUT_DIR = ROOT / "output"

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
MAX_OUTPUT_TOKENS = 8192  # deepseek-chat hard cap

# Chinese-heavy text on DeepSeek: ~0.635 tokens per character. Calibrated from
# the API `usage` counts of this pipeline itself (0.63-0.64 across runs).
TOKENS_PER_CHAR = 0.635

KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]+")
Q_HEAD = re.compile(r"^\s*(\d+)\s*q\s*[.．、]?\s*问题\s*[:：]")
A_HEAD = re.compile(r"^\s*(\d+)\s*a\s*[.．、]?\s*闭源模型的完整回答\s*[:：]")

# The dialogue in the source answer is only a presentation shell; these terms
# must not leak into the reconstructed rationale.
DIALOGUE_LEAK_TERMS = ("老师", "学生", "同学", "课堂", "师生", "对话")


def find_dialogue_leaks(cot: str) -> list[str]:
    return [term for term in DIALOGUE_LEAK_TERMS if term in cot]


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) * TOKENS_PER_CHAR))


def length_instruction(required: int) -> str:
    chars = round(required / TOKENS_PER_CHAR)
    return (
        f"\n\n长度要求：这份思考过程不得少于约 {required} tokens（约 {chars} 汉字）。"
        f"请把每一步的判断依据、考虑过又放弃的思路、验证过程写充分；"
        f"但不能重复、注水，也不能为凑长度堆砌无信息内容；"
        f"更不要写任何关于这份回答是怎么呈现的、角色怎么安排的内容。"
    )


def load_api_key(path: Path = API_FILE) -> str:
    """Accept both `sk-...` alone and a labelled line like `cot-decoder：sk-...`."""
    env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return env_key
    if not path.exists():
        sys.exit(f"[fatal] api file not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    match = KEY_RE.search(text)
    if not match:
        sys.exit(f"[fatal] no `sk-...` key found in {path}")
    return match.group(0)


def mask_key(key: str) -> str:
    return key[:7] + "..." + key[-4:] if len(key) > 12 else "***"


def _strip_outer_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text.strip()


def parse_dataset(text: str) -> list[dict[str, str]]:
    """Parse repeated blocks:
        1q.问题:"..."
        1a.闭源模型的完整回答:"..."
    The answer body may span multiple lines; a new question/answer head starts a new block.
    """
    lines = text.splitlines()
    blocks: dict[str, dict[str, str]] = {}
    current: tuple[int, str] | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal current, buf
        if current is None:
            return
        rid, kind = current
        body = _strip_outer_quotes("\n".join(buf))
        blocks.setdefault(rid, {})[kind] = body
        current, buf = None, []

    for line in lines:
        mq = Q_HEAD.match(line)
        ma = A_HEAD.match(line)
        if mq:
            flush()
            current = (int(mq.group(1)), "question")
            buf = [line[mq.end():]]
        elif ma:
            flush()
            current = (int(ma.group(1)), "answer")
            buf = [line[ma.end():]]
        elif current is not None:
            buf.append(line)
    flush()

    records: list[dict[str, str]] = []
    for rid in sorted(blocks):
        item = blocks[rid]
        missing = {"question", "answer"} - item.keys()
        if missing:
            sys.exit(f"[fatal] record {rid} is missing {sorted(missing)}")
        records.append({"id": rid, "question": item["question"], "answer": item["answer"]})
    if not records:
        sys.exit(f"[fatal] no Q/A records parsed from {DATASET_FILE}")
    return records


def parse_jsonl_dataset(path: Path) -> list[dict[str, str]]:
    """Read a JSONL dataset with {question, answer|response|source_answer} rows.

    Used to feed collected teacher answers (e.g. Claude responses) directly into
    the decoder without round-tripping through decode_dataset.txt. `response` is
    preferred over `answer` because batch-collected files keep the reference
    answer under `answer` and the teacher's full answer under `response`.
    """
    records: list[dict[str, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.exit(f"[fatal] {path}:{lineno}: invalid JSON: {exc}")
        question = str(item.get("question") or item.get("problem") or "").strip()
        answer = str(item.get("source_answer") or item.get("response")
                     or item.get("answer") or "").strip()
        if not question or not answer:
            sys.exit(f"[fatal] {path}:{lineno}: row needs question and answer/response")
        records.append({
            "id": item.get("id", lineno),
            "question": question,
            "answer": answer,
        })
    if not records:
        sys.exit(f"[fatal] no Q/A records parsed from {path}")
    return records


def build_prompt(template: str, question: str, answer: str) -> str:
    if "{问题}" not in template and "{完整回答}" not in template:
        sys.exit("[fatal] template has neither {问题} nor {完整回答} placeholder")
    return template.replace("{问题}", question).replace("{完整回答}", answer)


def _http_status(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    return status


def call_decoder(client: Any, model: str, messages: list[dict[str, str]], temperature: float,
                 max_tokens: int, retries: int) -> tuple[str, Any, float]:
    """One completion with retries. Fails fast on auth / balance errors."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            latency = time.perf_counter() - started
            cot = (resp.choices[0].message.content or "").strip()
            return cot, resp, latency
        except Exception as exc:  # noqa: BLE001 - surface the API error verbatim
            last_exc = exc
            status = _http_status(exc)
            if status in (401, 403, 402):
                raise
            if attempt == retries:
                break
            wait = min(2 ** attempt, 20)
            print(f"  [retry {attempt}/{retries - 1}] {type(exc).__name__}: {exc} "
                  f"-> sleeping {wait}s", flush=True)
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def _entry_score(entry: dict[str, Any]) -> tuple[int, int, int, int]:
    """Prefer: clean+long enough > long enough > clean > longest."""
    clean = not entry["leaks"]
    floor_ok = entry["floor_ok"]
    return (1 if (clean and floor_ok) else 0,
            1 if floor_ok else 0,
            1 if clean else 0,
            entry["tokens"] or 0)


def decode_with_constraints(client: Any, model: str, base_messages: list[dict[str, str]],
                            temperature: float, max_tokens: int, retries: int,
                            required: int, expand_retries: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Decode and enforce two constraints: output tokens >= `required`, and no
    dialogue/presentation leakage. Re-asks with targeted feedback, keeping the
    best attempt."""
    attempts: list[dict[str, Any]] = []
    history = list(base_messages)
    best: dict[str, Any] | None = None
    max_calls = 1 + max(0, expand_retries)

    for call in range(1, max_calls + 1):
        cot, resp, latency = call_decoder(client, model, history, temperature, max_tokens, retries)
        usage = resp.usage
        got = getattr(usage, "completion_tokens", None) if usage else None
        leaks = find_dialogue_leaks(cot)
        floor_ok = required <= 0 or (got is not None and got >= required)
        attempts.append({
            "call": call,
            "completion_tokens": got,
            "chars": len(cot),
            "dialogue_leak_terms": leaks,
            "floor_ok": floor_ok,
            "latency_s": round(latency, 3),
        })
        entry = {
            "cot": cot,
            "resp": resp,
            "latency": latency,
            "tokens": got,
            "leaks": leaks,
            "floor_ok": floor_ok,
        }
        if best is None or _entry_score(entry) > _entry_score(best):
            best = entry

        if (floor_ok and not leaks) or call == max_calls:
            break

        reasons: list[str] = []
        if not floor_ok:
            got_s = got if got is not None else "?"
            reasons.append(
                f"上一版输出偏短（约 {got_s} tokens），未达到 {required} tokens 的长度要求，"
                f"请把被省略的中间判断（备选思路、排除理由、验证细节）写足"
            )
        if leaks:
            reasons.append(
                "上一版出现了关于呈现形式的表述（" + "、".join(leaks) + "），"
                "请删除这类内容：不要描述回答的形式、角色安排或自己如何组织文字，"
                "只保留第一人称的推理链本身"
            )
        feedback = "；".join(reasons) + "。重新输出一个完整版本；不要保留任何关于长度、上一版输出或修改说明的元话语。"
        history = list(base_messages) + [
            {"role": "assistant", "content": best["cot"]},
            {"role": "user", "content": feedback},
        ]
        tag: list[str] = []
        if not floor_ok:
            tag.append(f"长度 {got} < {required}")
        if leaks:
            tag.append(f"对话残留 {','.join(leaks)}")
        print(f"  [retry] 第 {call} 次输出" + "，".join(tag) + "，重新生成…", flush=True)

    assert best is not None
    return best, attempts


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:  # Windows consoles default to GBK and choke on Chinese content
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="RECAST CoT decoder runner")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek model name")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--limit", type=int, default=0, help="decode only first N records")
    parser.add_argument("--retries", type=int, default=5, help="network retries per API call")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--dry-run", action="store_true", help="print prompts, skip API")
    parser.add_argument("--out", default="", help="explicit output directory")
    parser.add_argument("--dataset", default="", help="override decode_dataset.txt path")
    parser.add_argument("--template", default="", help="override decoder_template.txt path")
    parser.add_argument("--api", default="", help="override api.txt path")
    parser.add_argument("--system", default="", help="override system_prompt.txt path")
    parser.add_argument("--no-system", action="store_true", help="disable system prompt")
    parser.add_argument("--min-tokens", type=int, default=0,
                        help="absolute minimum completion tokens per record")
    parser.add_argument("--min-ratio", type=float, default=1.0,
                        help="minimum completion tokens as a multiple of the source answer's "
                             "estimated tokens (default 1.0; 0 disables the ratio floor)")
    parser.add_argument("--expand-retries", type=int, default=2,
                        help="extra expansion attempts when output is below the floor")
    parser.add_argument("--no-min", action="store_true", help="disable length enforcement")
    args = parser.parse_args()

    template_path = Path(args.template) if args.template else TEMPLATE_FILE
    dataset_path = Path(args.dataset) if args.dataset else DATASET_FILE
    api_path = Path(args.api) if args.api else API_FILE
    system_path = Path(args.system) if args.system else SYSTEM_FILE

    system_prompt: str | None = None
    if not args.no_system:
        if args.system and not system_path.exists():
            sys.exit(f"[fatal] system prompt not found: {system_path}")
        if system_path.exists():
            system_prompt = system_path.read_text(encoding="utf-8-sig").strip() or None

    template = template_path.read_text(encoding="utf-8-sig")
    if dataset_path.suffix.lower() == ".jsonl":
        records = parse_jsonl_dataset(dataset_path)
    else:
        records = parse_dataset(dataset_path.read_text(encoding="utf-8-sig"))
    if args.limit > 0:
        records = records[: args.limit]

    prompts = [build_prompt(template, r["question"], r["answer"]) for r in records]

    items: list[dict[str, Any]] = []
    for rec, prompt in zip(records, prompts):
        est = estimate_tokens(rec["answer"])
        required = 0
        if not args.no_min:
            required = max(args.min_tokens, math.ceil(args.min_ratio * est))
        system_effective = system_prompt
        if required > 0:
            note = length_instruction(required)
            system_effective = (system_effective + note) if system_effective else note.strip()
        messages: list[dict[str, str]] = []
        if system_effective:
            messages.append({"role": "system", "content": system_effective})
        messages.append({"role": "user", "content": prompt})
        effective_max = args.max_tokens
        if required > 0:
            effective_max = min(MAX_OUTPUT_TOKENS, max(args.max_tokens, required + 512))
        items.append({
            "id": rec["id"],
            "question": rec["question"],
            "answer": rec["answer"],
            "prompt": prompt,
            "messages": messages,
            "system_effective": system_effective,
            "estimated_source_tokens": est,
            "required_tokens": required,
            "effective_max_tokens": effective_max,
        })

    if args.dry_run:
        if system_prompt:
            print(f"===== base system prompt ({len(system_prompt)} chars) =====")
            print(system_prompt)
            print()
        for item in items:
            print(f"===== record {item['id']} prompt ({len(item['prompt'])} chars) "
                  f"| source≈{item['estimated_source_tokens']} tok "
                  f"| min {item['required_tokens']} tok "
                  f"| max_tokens {item['effective_max_tokens']} =====")
            print(item["prompt"])
            print()
        print(f"[dry-run] {len(items)} prompt(s), no API call made")
        return 0

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("[fatal] openai package is missing: pip install openai")

    key = load_api_key(api_path)
    system_note = f"on ({len(system_prompt)} chars)" if system_prompt else "off"
    if args.no_min:
        min_note = "off"
    else:
        min_note = f"max({args.min_tokens}, {args.min_ratio}x answer tokens)"
    print(f"[info] key {mask_key(key)} | base_url {args.base_url} | model {args.model} "
          f"| temperature {args.temperature} | max_tokens {args.max_tokens} "
          f"| system {system_note} | min {min_note}")

    client = OpenAI(api_key=key, base_url=args.base_url, timeout=args.timeout)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    failures = 0
    below_floor = 0
    for idx, item in enumerate(items, 1):
        rec = item
        print(f"\n[decode {idx}/{len(items)}] record {item['id']} "
              f"({len(item['prompt'])} prompt chars) | "
              f"source≈{item['estimated_source_tokens']} tok | "
              f"min {item['required_tokens']} tok | max_tokens {item['effective_max_tokens']}",
              flush=True)
        try:
            best, attempts = decode_with_constraints(
                client, args.model, item["messages"], args.temperature,
                item["effective_max_tokens"], args.retries,
                item["required_tokens"], args.expand_retries,
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            status = _http_status(exc)
            print(f"  [fail] HTTP {status}: {exc}", flush=True)
            if status in (401, 402):
                print("  [hint] 401 = key invalid; 402 = account balance exhausted.",
                      file=sys.stderr)
            results.append({
                "id": item["id"],
                "question": item["question"],
                "source_answer": item["answer"],
                "system_prompt": item["system_effective"],
                "prompt": item["prompt"],
                "decoded_cot": None,
                "estimated_source_tokens": item["estimated_source_tokens"],
                "min_tokens_required": item["required_tokens"],
                "length_ok": False,
                "attempt_token_counts": [],
                "error": f"HTTP {status}: {exc}" if status else f"{type(exc).__name__}: {exc}",
                "created_at": datetime.now().isoformat(timespec="seconds"),
            })
            continue

        cot = best["cot"]
        resp = best["resp"]
        usage = resp.usage
        tokens = getattr(usage, "completion_tokens", None) if usage else None
        required = item["required_tokens"]
        length_ok = required <= 0 or (tokens is not None and tokens >= required)
        if not length_ok:
            below_floor += 1
        finish = resp.choices[0].finish_reason
        leaks = find_dialogue_leaks(cot)

        results.append({
            "id": item["id"],
            "question": item["question"],
            "source_answer": item["answer"],
            "system_prompt": item["system_effective"],
            "prompt": item["prompt"],
            "decoded_cot": cot,
            "dialogue_leak_terms": leaks,
            "estimated_source_tokens": item["estimated_source_tokens"],
            "min_tokens_required": required,
            "length_ok": length_ok,
            "attempt_token_counts": [a["completion_tokens"] for a in attempts],
            "finish_reason": finish,
            "latency_s": round(best["latency"], 3),
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": tokens,
                "total_tokens": getattr(usage, "total_tokens", None),
            } if usage else None,
            "response_id": getattr(resp, "id", None),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })

        floor_note = ""
        if required > 0:
            floor_note = f", {tokens if tokens is not None else '?'} tok " \
                         f"{'≥' if length_ok else '<'} {required} tok"
        print(f"-- decoded CoT ({len(cot)} chars, {best['latency']:.1f}s, "
              f"finish={finish}{floor_note}) --")
        if len(attempts) > 1:
            print("  [attempts] " + ", ".join(
                f"{a['completion_tokens']} tok" + ("/leak" if a["dialogue_leak_terms"] else "")
                for a in attempts))
        if not length_ok:
            print(f"  [warn] 长度未达标: {tokens} < {required} tokens "
                  f"(尝试 {len(attempts)} 次，取最长输出)")
        if finish == "length":
            print(f"  [warn] finish_reason=length: 输出被 max_tokens 截断")
        if leaks:
            print(f"  [warn] 对话体残留: {', '.join(leaks)}")
        print(cot)
        print("-" * 70)

    json_path = out_dir / f"decoded_{stamp}.json"
    md_path = out_dir / f"decoded_{stamp}.md"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "system_prompt": system_prompt,
        "template": template,
        "min_tokens": args.min_tokens,
        "min_ratio": args.min_ratio,
        "expand_retries": args.expand_retries,
        "tokens_per_char": TOKENS_PER_CHAR,
        "num_records": len(results),
        "num_failed": failures,
        "num_below_floor": below_floor,
        "records": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md: list[str] = [f"# RECAST decoded CoT ({stamp})", "",
                     f"- model: `{args.model}`",
                     f"- temperature: {args.temperature}",
                     f"- system prompt: {system_note}",
                     f"- min length: {min_note}",
                     f"- records: {len(results)} (failed: {failures}, below floor: {below_floor})", ""]
    for res in results:
        md += [f"## Record {res['id']}", "", "### 问题", "", res["question"], "",
               "### 解码出的思维链", "",
               res["decoded_cot"] or f"> **FAILED**: {res.get('error')}", ""]
        usage = res.get("usage") or {}
        tokens = usage.get("completion_tokens")
        if res.get("decoded_cot") and res.get("min_tokens_required", 0) > 0:
            ok = "✅" if res.get("length_ok") else "❌"
            md.append(f"> 输出 {tokens} tokens / 下限 {res['min_tokens_required']} tokens {ok}")
            md.append("")
        if res.get("dialogue_leak_terms"):
            md += [f"> ⚠ 对话体残留: {', '.join(res['dialogue_leak_terms'])}", ""]
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"\n[done] {len(results) - failures}/{len(results)} decoded, "
          f"{below_floor} below length floor")
    print(f"[out]  {json_path}")
    print(f"[out]  {md_path}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""LLM-based cleaning of reconstructed CoTs (dedicated DeepSeek cleaner).

Why: rule-based filtering can only *drop* a rationale that leaks dialogue /
meta narration. A dedicated cleaner can *repair* it (remove the classroom
shell, keep the reasoning) and normalize formatting, so fewer samples are lost.

Input : output/decoded_*.json
Output: data/cleaned_cots.json   (overlay keyed by record_key; raw data untouched)

The overlay is optional for build_dataset.py: it prefers an `ok` entry and
falls back to the raw decoded CoT otherwise. Rule-based leak/length checks
still run afterwards as a cheap safety net.

Key: api_clean.txt (or DEEPSEEK_CLEAN_API_KEY env var).
Example:
    python data/clean_with_llm.py --workers 4
    python data/clean_with_llm.py --only-leaky --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
EVAL_ROOT = DATA_DIR.parent
RECAST_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(DATA_DIR))
sys.path.insert(0, str(EVAL_ROOT / "harness"))

from rec_io import load_decoded_records  # noqa: E402
from text_clean import clean_cot, find_all_leaks, estimate_tokens  # noqa: E402
from api_client import ChatClient, mask_key  # noqa: E402

CLEAN_PROMPT = """你是训练数据清洗器。输入是一段由"解码模型"从某闭源模型的完整回答中还原出来的思维链（CoT），将作为小模型的监督训练目标。它可能混入三类噪声：
1) 对话/课堂痕迹：老师、学生、同学、课堂、对话体、谁在提问/谁在回答；
2) 元叙述：提到还原/解码/模板/闭源模型的完整回答/本次任务/长度要求，或描述自己如何组织回答、如何设计角色；
3) 把课堂形式换个说法保留下来：角色 A / 分角色叙述 / 呈现方式 / 模拟输入 / 回答的还原 等。

请把它清洗成"模型本人"的第一人称思维链：读起来应当像模型自己在解题，而不是在评论一份回答或一项任务。要求：
- 删除或改写所有老师/学生/课堂/对话痕迹，只保留"我"的思考；
- 凡是讨论这份回答如何呈现、如何被还原/解码/模拟、或涉及任务/要求/模板/输入的整句或整段，直接删除，不要换词保留；
- 不得出现"角色""分角色""呈现""模拟""还原""解码"等词；
- 也不要保留"第一人称""可审计""输出结构""处理阶段""重组"等关于任务或输出格式的描述，这类整句直接删除；
- 保留原有的阶段小标题（任务识别、用户需求分析、材料诊断、策略选择、关键验证、输出组织、自检与修正、一句话总结）及其顺序；某阶段删空时可保留标题或删掉该阶段，但不要用空话填充；
- 删除显式的"最终答案：…"或"答案是…"这类答案声明行（最终答案会在后续步骤统一追加）；
- 不修正数学错误，不补充新的数学内容；
- 删除元叙述后长度可以低于原文，但有效推理要尽量保留，不要为了凑长度保留或编造内容；
- 如果原文已经完全没有上述噪声，请直接原样返回（仅做必要的空白/空行归一），不要改写、压缩或重新组织；只有存在噪声时才做删除或改写；
- 只输出一个 JSON 对象，不要输出任何其他文字。

JSON 格式：
{"cleaned_cot": "清洗后的完整思维链，保留 markdown 小标题和换行", "removed": ["dialogue", "shell", "task_meta", "answer_line"], "notes": "一句话说明改了什么，可为空字符串"}

问题：
{question}

待清洗的思维链：
{cot}
"""


def build_prompt(question: str, cot: str) -> str:
    return CLEAN_PROMPT.replace("{question}", question or "").replace("{cot}", cot)


def parse_json_response(text: str) -> dict | None:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def normalize_removed(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def sentences_with_terms(text: str, terms: list[str], limit: int = 5) -> list[str]:
    parts = re.split(r"(?<=[。！？!?\n])", text or "")
    found: list[str] = []
    for part in parts:
        sentence = part.strip()
        if sentence and any(term in sentence for term in terms):
            found.append(sentence[:300])
            if len(found) >= limit:
                break
    return found


def usage_to_dict(usage) -> dict | None:
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def clean_one(client: ChatClient, record: dict, args: argparse.Namespace) -> dict:
    raw = clean_cot(record.get("decoded_cot") or "")
    question = (record.get("question") or "").strip()
    if not raw:
        return {"status": "empty_raw", "cleaned_cot": None, "removed": []}

    attempts: list[dict] = []
    feedback = ""
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    started = time.perf_counter()

    for attempt in range(1, args.attempts + 1):
        prompt = build_prompt(question, raw)
        if feedback:
            prompt += "\n\n补充要求：\n" + feedback
        try:
            resp = client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                json_mode=True,
            )
        except Exception as exc:  # noqa: BLE001 - record and let the caller decide
            return {
                "status": "api_error",
                "cleaned_cot": None,
                "removed": [],
                "error": f"{type(exc).__name__}: {exc}",
                "attempts": attempts,
                "latency_s": round(time.perf_counter() - started, 2),
            }

        usage = usage_to_dict(resp.get("usage"))
        if usage:
            for key in usage_total:
                usage_total[key] += usage.get(key) or 0

        payload = parse_json_response(resp.get("content", ""))
        if payload is None:
            attempts.append({"attempt": attempt, "status": "parse_error"})
            feedback = "上一版没有返回合法 JSON。请严格只输出一个 JSON 对象，包含 cleaned_cot / removed / notes 三个字段。"
            continue

        cleaned = clean_cot(str(payload.get("cleaned_cot", "")))
        attempts.append({
            "attempt": attempt,
            "status": "parsed",
            "cleaned_chars": len(cleaned),
            "latency_s": round(time.perf_counter() - started, 2),
        })
        if not cleaned:
            feedback = "cleaned_cot 为空。请输出清洗后的完整思维链，不要省略内容。"
            continue

        leaks = find_all_leaks(cleaned)
        leak_terms = leaks["dialogue"] + leaks["meta"]
        ratio = len(cleaned) / max(1, len(raw))

        if leak_terms:
            if attempt >= args.attempts:
                return {
                    "status": "still_leaky",
                    "cleaned_cot": None,
                    "removed": normalize_removed(payload.get("removed")),
                    "leak_terms": leak_terms,
                    "leak_sentences": sentences_with_terms(cleaned, leak_terms),
                    "attempts": attempts,
                    "usage": usage_total,
                    "latency_s": round(time.perf_counter() - started, 2),
                }
            bad_sentences = sentences_with_terms(cleaned, leak_terms)
            examples = "\n".join(f"- {sentence}" for sentence in bad_sentences)
            feedback = ("上一版仍包含禁用词：" + "、".join(leak_terms) + "。"
                        + (f"\n具体句子：\n{examples}\n" if examples else "")
                        + "请删除这些整句，或在不改变数学内容的前提下用第一人称重写，"
                          "确保最终文本不再出现这些词；不要解释清洗过程。")
            continue

        if ratio < args.min_length_ratio:
            if attempt >= args.attempts:
                return {
                    "status": "rejected_short",
                    "cleaned_cot": None,
                    "removed": normalize_removed(payload.get("removed")),
                    "length_ratio": round(ratio, 3),
                    "attempts": attempts,
                    "usage": usage_total,
                    "latency_s": round(time.perf_counter() - started, 2),
                }
            feedback = (f"上一版只有原文长度的 {ratio:.0%}，有效推理被删多了。"
                        f"请保留原有阶段内容，把长度恢复到原文 80% 以上。")
            continue

        if ratio > args.max_length_ratio:
            if attempt >= args.attempts:
                return {
                    "status": "rejected_long",
                    "cleaned_cot": None,
                    "removed": normalize_removed(payload.get("removed")),
                    "length_ratio": round(ratio, 3),
                    "attempts": attempts,
                    "usage": usage_total,
                    "latency_s": round(time.perf_counter() - started, 2),
                }
            feedback = (f"上一版长度是原文的 {ratio:.0%}，扩写过多。"
                        f"请只做删除/改写，不要新增推理内容，长度控制在原文附近。")
            continue

        return {
            "status": "ok",
            "cleaned_cot": cleaned,
            "removed": normalize_removed(payload.get("removed")),
            "notes": str(payload.get("notes", ""))[:300],
            "raw_chars": len(raw),
            "cleaned_chars": len(cleaned),
            "length_ratio": round(ratio, 3),
            "raw_tokens_est": estimate_tokens(raw),
            "cleaned_tokens_est": estimate_tokens(cleaned),
            "attempts": attempts,
            "usage": usage_total,
            "latency_s": round(time.perf_counter() - started, 2),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    return {
        "status": "failed",
        "cleaned_cot": None,
        "removed": [],
        "attempts": attempts,
        "usage": usage_total,
        "latency_s": round(time.perf_counter() - started, 2),
    }


def save_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def recompute_stats(cleaned: dict[str, dict]) -> dict:
    stats: dict[str, int] = {}
    tokens = 0
    for entry in cleaned.values():
        status = entry.get("status", "unknown")
        stats[status] = stats.get(status, 0) + 1
        usage = entry.get("usage") or {}
        tokens += usage.get("total_tokens") or 0
    return {"entries": len(cleaned), "status": stats, "total_tokens": tokens}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="LLM cleaning of reconstructed CoTs")
    parser.add_argument("--decoded", nargs="+",
                        default=[str(RECAST_ROOT / "output" / "decoded_*.json")])
    parser.add_argument("--out", default=str(DATA_DIR / "cleaned_cots.json"))
    parser.add_argument("--api", default=str(RECAST_ROOT / "api_clean.txt"),
                        help="key file for the cleaning model")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--min-length-ratio", type=float, default=0.4)
    parser.add_argument("--max-length-ratio", type=float, default=2.5)
    parser.add_argument("--workers", type=int, default=1, help="parallel API calls")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--record-key", nargs="*", default=[],
                        help="only clean these record_key(s), e.g. decoded_...:1")
    parser.add_argument("--force", action="store_true", help="re-clean existing entries")
    parser.add_argument("--only-leaky", action="store_true",
                        help="only clean records that the rule-based check flags")
    parser.add_argument("--dry-run", action="store_true", help="print one prompt and exit")
    args = parser.parse_args()

    records = load_decoded_records(args.decoded)
    if args.limit > 0:
        records = records[: args.limit]
    if args.record_key:
        wanted = set(args.record_key)
        records = [rec for rec in records if rec["record_key"] in wanted]
        if not records:
            sys.exit(f"[fatal] none of the requested record keys found: {args.record_key}")
    records.sort(key=lambda rec: rec.get("_source_stem", ""), reverse=True)

    out_path = Path(args.out)
    payload: dict = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                     "model": args.model,
                     "base_url": args.base_url,
                     "prompt_version": "clean_v1",
                     "source": [str(p) for p in args.decoded],
                     "stats": {},
                     "cleaned": {}}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8-sig"))
            payload["cleaned"] = existing.get("cleaned", {})
            payload["generated_at"] = existing.get("generated_at", payload["generated_at"])
        except json.JSONDecodeError:
            print(f"[warn] could not parse existing {out_path}, starting fresh", file=sys.stderr)

    cleaned: dict[str, dict] = payload["cleaned"]
    pending = [rec for rec in records
               if args.force or cleaned.get(rec["record_key"], {}).get("status") != "ok"]
    if args.only_leaky:
        pending = [rec for rec in pending
                   if any(find_all_leaks(clean_cot(rec.get("decoded_cot") or "")).values())]
    if not pending:
        print(f"[done] nothing to clean ({len(cleaned)} cached entries)")
        return 0

    if args.dry_run:
        rec = pending[0]
        print(f"===== prompt for {rec['record_key']} =====")
        print(build_prompt(rec.get("question", ""), clean_cot(rec.get("decoded_cot") or "")))
        return 0

    client = ChatClient(model=args.model, base_url=args.base_url,
                        api_file=args.api, key_env="DEEPSEEK_CLEAN_API_KEY")
    print(f"[info] cleaner model={args.model} base_url={args.base_url} "
          f"key={mask_key(client.key)} records={len(records)} pending={len(pending)} "
          f"workers={args.workers}")

    started = time.perf_counter()
    processed = 0

    def handle(record: dict, result: dict) -> None:
        nonlocal processed
        cleaned[record["record_key"]] = result
        processed += 1
        status = result.get("status")
        if status == "ok":
            print(f"[{processed}/{len(pending)}] {record['record_key']}: ok "
                  f"{result['raw_chars']}->{result['cleaned_chars']} chars "
                  f"({result['length_ratio']:.0%}, removed={result.get('removed')})", flush=True)
        else:
            print(f"[{processed}/{len(pending)}] {record['record_key']}: {status}", flush=True)
        if processed % 20 == 0:
            payload["stats"] = recompute_stats(cleaned)
            save_payload(out_path, payload)

    if args.workers <= 1:
        for record in pending:
            try:
                result = clean_one(client, record, args)
            except Exception as exc:  # noqa: BLE001
                result = {"status": "api_error", "error": f"{type(exc).__name__}: {exc}"}
            handle(record, result)
            if args.sleep:
                time.sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(clean_one, client, rec, args): rec for rec in pending}
            for future in as_completed(futures):
                record = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {"status": "api_error", "error": f"{type(exc).__name__}: {exc}"}
                handle(record, result)
                if args.sleep:
                    time.sleep(args.sleep)

    payload["stats"] = recompute_stats(cleaned)
    payload["elapsed_s"] = round(time.perf_counter() - started, 1)
    save_payload(out_path, payload)
    print(f"[done] {len(pending)} processed, stats={payload['stats']} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate a local HF model / LoRA adapter OR an OpenAI-compatible teacher API
on a math benchmark, and store generations + metrics.

Data schema (JSONL): {"id", "question", "answer", ...}
  - data/processed/eval_test.jsonl  (in-domain held-out, from build_dataset.py)
  - data/benchmarks/gsm8k.jsonl     (from download_benchmarks.py)

Examples:
    # local student checkpoint (LoRA adapter on top of the base model)
    python harness/run_eval.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter ../runs/sft_qwen1.5b/final_adapter \
        --data data/benchmarks/gsm8k.jsonl --name sft_qwen1.5b_gsm8k

    # base model, greedy
    python harness/run_eval.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --data data/benchmarks/math500.jsonl --name base_qwen1.5b_math500 --k 1

    # teacher via API (OpenAI-compatible; default DeepSeek)
    python harness/run_eval.py --provider teacher --model deepseek-chat \
        --data data/benchmarks/math500.jsonl --name teacher_math500 --k 1

    # majority vote over 8 samples
    python harness/run_eval.py --model ... --k 8 --temperature 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
EVAL_ROOT = HARNESS_DIR.parent
RECAST_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(HARNESS_DIR))

from math_verify import extract_final_answer, is_correct  # noqa: E402
from metrics import summarize  # noqa: E402

DEFAULT_SYSTEM = "你是一个严谨的数学推理助手。解答时先写出清晰的推理步骤，最后单独给出最终答案。"
DEFAULT_SYSTEM_FILE = EVAL_ROOT / "system_prompt_student.txt"


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def load_rows(path: str | Path, limit: int = 0) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit > 0 else rows


def build_messages(system: str, question: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    return messages


def resolve_system(args: argparse.Namespace) -> str:
    if args.system:
        return args.system
    if args.system_file:
        return Path(args.system_file).read_text(encoding="utf-8-sig").strip()
    if DEFAULT_SYSTEM_FILE.exists() and not args.no_system:
        return DEFAULT_SYSTEM_FILE.read_text(encoding="utf-8-sig").strip()
    return "" if args.no_system else DEFAULT_SYSTEM


def load_hf_model(model_path: str, adapter: str, load_4bit: bool, device: str,
                  merge_adapter: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    kwargs: dict = {"trust_remote_code": True}
    if load_4bit:
        from transformers import BitsAndBytesConfig

        if not torch.cuda.is_available():
            sys.exit("[fatal] --load-4bit requires CUDA")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = {"": 0}
    elif device == "cpu":
        kwargs["torch_dtype"] = torch.float32
        kwargs["device_map"] = "cpu"
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["torch_dtype"] = dtype
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if merge_adapter:
            if load_4bit:
                sys.exit("[fatal] cannot merge a 4-bit model; rerun without --load-4bit")
            model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def generate_hf(model, tokenizer, messages_list: list[list[dict]], *, k: int,
                temperature: float, top_p: float, max_new_tokens: int,
                max_prompt_tokens: int, batch_size: int) -> tuple[list[list[str]], list[list[int]]]:
    import torch

    if k > 1 and temperature <= 0:
        sys.exit("[fatal] k>1 requires --temperature > 0 (sampling)")
    device = next(model.parameters()).device
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in messages_list]
    generations: list[list[str]] = []
    token_counts: list[list[int]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_prompt_tokens)
        enc = {key: value.to(device) for key, value in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p,
                              num_return_sequences=k)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            output = model.generate(**enc, **gen_kwargs)
        for i in range(len(batch)):
            if temperature > 0:
                block = output[i * k:(i + 1) * k]
            else:
                block = output[i:i + 1]
            gens: list[str] = []
            counts: list[int] = []
            for seq in block:
                text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
                gens.append(text)
                counts.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
            generations.append(gens)
            token_counts.append(counts)
    return generations, token_counts


def generate_teacher(client, messages_list: list[list[dict]], *, k: int,
                     temperature: float, max_new_tokens: int) -> tuple[list[list[str]], list[list[int]], dict]:
    generations: list[list[str]] = []
    token_counts: list[list[int]] = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for messages in messages_list:
        gens: list[str] = []
        counts: list[int] = []
        for _ in range(k):
            resp = client.chat(messages, temperature=temperature if k > 1 or temperature > 0 else 0.0,
                               max_tokens=max_new_tokens)
            gens.append(resp["content"])
            usage = resp.get("usage")
            if usage is not None:
                totals["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                totals["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
                totals["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
                counts.append(getattr(usage, "completion_tokens", 0) or 0)
            else:
                counts.append(0)
        generations.append(gens)
        token_counts.append(counts)
    return generations, token_counts, totals


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Evaluate student/teacher on a math benchmark")
    parser.add_argument("--provider", choices=["hf", "teacher"], default="hf")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HF model path (hf) or API model name (teacher)")
    parser.add_argument("--adapter", default="", help="LoRA adapter path on top of --model")
    parser.add_argument("--data", required=True, help="benchmark/eval JSONL")
    parser.add_argument("--out-dir", default=str(EVAL_ROOT / "results"))
    parser.add_argument("--name", default="", help="result name (default: model+data stem)")
    parser.add_argument("--k", type=int, default=1, help="samples per question (maj@k needs k>1)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--merge-adapter", action="store_true",
                        help="merge the LoRA adapter into the base weights (fp16/bf16) before eval")
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    parser.add_argument("--system", default="", help="inline student system prompt")
    parser.add_argument("--system-file", default="", help="system prompt file")
    parser.add_argument("--no-system", action="store_true")
    parser.add_argument("--api", default=str(RECAST_ROOT / "api.txt"), help="teacher API key file")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import random

    import torch

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"[fatal] data not found: {data_path}")
    rows = load_rows(data_path, args.limit)
    if not rows:
        sys.exit("[fatal] empty dataset")
    system = resolve_system(args)
    messages_list = [build_messages(system, row["question"]) for row in rows]

    data_stem = data_path.stem
    name = args.name or f"{Path(args.model).name}_{data_stem}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] provider={args.provider} model={args.model} "
          f"data={data_path.name} rows={len(rows)} k={args.k} temp={args.temperature}")

    start = time.perf_counter()
    if args.provider == "hf":
        model, tokenizer = load_hf_model(args.model, args.adapter, args.load_4bit,
                                         args.device, args.merge_adapter)
        generations, token_counts = generate_hf(
            model, tokenizer, messages_list,
            k=args.k, temperature=args.temperature, top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens, batch_size=args.batch_size,
        )
        usage = {}
    else:
        from api_client import ChatClient, mask_key

        client = ChatClient(model=args.model, base_url=args.base_url, api_file=args.api)
        print(f"[info] teacher key={mask_key(client.key)}")
        generations, token_counts, usage = generate_teacher(
            client, messages_list, k=args.k, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    elapsed = time.perf_counter() - start

    records: list[dict] = []
    for row, gens, counts in zip(rows, generations, token_counts):
        ref = str(row.get("answer") or "").strip()
        preds = [extract_final_answer(gen) for gen in gens]
        scored = bool(ref)
        correct = [is_correct(pred, ref) for pred in preds] if scored else [False] * len(preds)
        records.append({
            "id": row.get("id", ""),
            "question": row.get("question", ""),
            "ref": ref,
            "scored": scored,
            "preds": preds,
            "correct": correct,
            "tokens": counts,
            "generations": gens,
        })

    summary = summarize(records)
    result = {
        "name": name,
        "provider": args.provider,
        "model": args.model,
        "adapter": args.adapter,
        "data": str(data_path),
        "num_rows": len(rows),
        "k": args.k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "system_prompt": system,
        "seed": args.seed,
        "elapsed_s": round(elapsed, 2),
        "usage": usage,
        "metrics": summary,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    summary_path = out_dir / f"{name}.json"
    gen_path = out_dir / f"{name}.generations.jsonl"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with gen_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"-- {name}: pass@1={summary['pass@1']:.3f} "
          f"avg@k={summary['avg@k']:.3f} maj@k={summary['maj@k']:.3f} "
          f"invalid={summary['invalid_rate']:.3f} "
          f"tokens(mean)={summary['tokens']['mean']} ({elapsed:.1f}s) --")
    print(f"[out] {summary_path}")
    print(f"[out] {gen_path}")
    return 0

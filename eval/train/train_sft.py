#!/usr/bin/env python3
"""SFT / LoRA / QLoRA training for a small reasoning student model.

Why not TRL: the training loop is plain transformers.Trainer + PEFT, which is
stable across TRL/transformers version churn and easy to audit.

Data format (from data/build_dataset.py): JSONL rows with
    {"messages": [{"role": "system", ...}, {"role": "user", ...},
                  {"role": "assistant", ...}], "meta": {...}}
Only the assistant tokens contribute to the loss.

Recommended (see docs/training_methods.md):
    python train/train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --data-dir data/processed --method qlora \
        --output-dir runs/sft_qwen1.5b

    # optional: merge the adapter into fp16/bf16 weights for fast eval
    python train/train_sft.py ... --merge
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
EVAL_ROOT = TRAIN_DIR.parent
sys.path.insert(0, str(EVAL_ROOT / "data"))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_sublist(haystack: list[int], needle: list[int]) -> int:
    if not needle:
        return -1
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return -1


def tokenize_example(tokenizer, messages: list[dict], max_len: int) -> dict | None:
    """Tokenize a full chat and mask everything except the assistant answer."""
    if not messages or messages[-1].get("role") != "assistant":
        return None
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:  # noqa: BLE001 - template unsupported
        return None

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        assistant_ids = tokenizer(messages[-1]["content"], add_special_tokens=False)["input_ids"]
        start = find_sublist(full_ids, assistant_ids)
        if start < 0:
            return None
        labels = ([-100] * start + assistant_ids
                  + [-100] * (len(full_ids) - start - len(assistant_ids)))
    else:
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    input_ids = full_ids[:max_len]
    labels = labels[:max_len]
    if not any(label != -100 for label in labels):
        return None
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def build_dataset(rows: list[dict], tokenizer, max_len: int, name: str) -> list[dict]:
    tokenized: list[dict] = []
    skipped = 0
    for row in rows:
        example = tokenize_example(tokenizer, row.get("messages", []), max_len)
        if example is None:
            skipped += 1
            continue
        tokenized.append(example)
    if not tokenized:
        sys.exit(f"[fatal] {name}: no tokenizable samples")
    lengths = sorted(len(ex["input_ids"]) for ex in tokenized)
    print(f"[data] {name}: {len(tokenized)} samples (skipped {skipped}), "
          f"tokens min/p50/max = {lengths[0]}/{lengths[len(lengths) // 2]}/{lengths[-1]}")
    return tokenized


class TokenizedDataset:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids, labels, attention = [], [], []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)
            attention.append(feature["attention_mask"] + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def detect_lora_targets(model) -> list[str]:
    suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = {name.split(".")[-1] for name, _ in model.named_modules() if name.split(".")[-1] in suffixes}
    if not found:
        sys.exit("[fatal] could not detect LoRA target modules; pass --lora-target-modules")
    return sorted(found)


def dtype_keyword() -> str:
    """transformers 5 renamed torch_dtype -> dtype."""
    import transformers

    major = int(transformers.__version__.split(".")[0])
    return "dtype" if major >= 5 else "torch_dtype"


def load_model_and_tokenizer(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs: dict = {"trust_remote_code": args.trust_remote_code}
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    kwargs[dtype_keyword()] = dtype

    if args.method == "qlora":
        from transformers import BitsAndBytesConfig

        if not torch.cuda.is_available():
            sys.exit("[fatal] QLoRA requires CUDA")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": 0}
    elif args.method == "full":
        kwargs["device_map"] = None
    else:  # lora on a full-precision base
        kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.config.use_cache = False

    if args.method in {"lora", "qlora"}:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if args.method == "qlora":
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args.gradient_checkpointing)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            if args.method == "lora":
                model.enable_input_require_grads()

        targets = args.lora_target_modules.split(",") if args.lora_target_modules else detect_lora_targets(model)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    return model, tokenizer


def merge_adapter(base_model: str, adapter_dir: Path, out_dir: Path, trust_remote_code: bool) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[merge] loading {base_model} in fp16/bf16 and merging {adapter_dir} ...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        **{dtype_keyword(): dtype},
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    merged_dir = out_dir / "merged"
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(merged_dir)
    print(f"[merge] saved merged model -> {merged_dir}")


def build_training_args(args, out_dir: Path, has_eval: bool):
    import torch
    from transformers import TrainingArguments

    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    fp16 = use_cuda and not bf16
    default_lr = {"qlora": 2e-4, "lora": 2e-4, "full": 1e-5}[args.method]
    fields = TrainingArguments.__dataclass_fields__
    eval_key = "eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"

    kwargs: dict = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr if args.lr > 0 else default_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim="adamw_torch",
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    if args.gradient_checkpointing:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if has_eval:
        kwargs[eval_key] = "steps"
        kwargs["eval_steps"] = args.eval_steps
    else:
        kwargs[eval_key] = "no"
    return TrainingArguments(**kwargs)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="SFT/LoRA/QLoRA for a reasoning student model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--method", choices=["qlora", "lora", "full"], default="qlora")
    parser.add_argument("--data-dir", default=str(EVAL_ROOT / "data" / "processed"))
    parser.add_argument("--train-file", default="", help="override sft_train.jsonl")
    parser.add_argument("--val-file", default="", help="override sft_val.jsonl")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=0.0, help="0 = method default")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="",
                        help="comma-separated; default = auto-detect")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--merge", action="store_true", help="also save a merged fp16/bf16 model")
    parser.add_argument("--dry-run-tokenize", action="store_true",
                        help="only tokenize a few samples and exit (debugging)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = Path(args.train_file) if args.train_file else data_dir / "sft_train.jsonl"
    val_path = Path(args.val_file) if args.val_file else data_dir / "sft_val.jsonl"
    if not train_path.exists():
        sys.exit(f"[fatal] train file not found: {train_path}")

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path) if val_path.exists() else []
    print(f"[data] train={len(train_rows)} val={len(val_rows)} from {data_dir}")

    if args.dry_run_tokenize:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        tokenized = build_dataset(train_rows[:5], tokenizer, args.max_seq_len, "dry-run")
        print(f"[dry-run] example tokenized keys: {list(tokenized[0])}")
        return 0

    import torch
    from transformers import Trainer

    model, tokenizer = load_model_and_tokenizer(args)

    train_dataset = TokenizedDataset(build_dataset(train_rows, tokenizer, args.max_seq_len, "train"))
    eval_dataset = None
    if val_rows:
        eval_dataset = TokenizedDataset(build_dataset(val_rows, tokenizer, args.max_seq_len, "val"))

    out_dir = Path(args.output_dir) if args.output_dir else EVAL_ROOT / "runs" / (
        f"sft_{args.method}_" + args.model.replace("/", "_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = build_training_args(args, out_dir, eval_dataset is not None)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    print(f"[train] method={args.method} model={args.model} out={out_dir}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    eval_metrics = trainer.evaluate() if eval_dataset is not None else {}

    if args.method in {"lora", "qlora"}:
        save_dir = out_dir / "final_adapter"
        trainer.model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    else:
        save_dir = out_dir / "final_model"
        trainer.save_model(str(save_dir))
        tokenizer.save_pretrained(save_dir)

    import peft
    import transformers

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": args.method,
        "base_model": args.model,
        "train_file": str(train_path),
        "val_file": str(val_path) if val_rows else "",
        "num_train": len(train_dataset),
        "num_val": len(eval_dataset) if eval_dataset is not None else 0,
        "hyperparameters": {
            "max_seq_len": args.max_seq_len,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "epochs": args.epochs,
            "lr": training_args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "lora_r": args.lora_r if args.method != "full" else None,
            "lora_alpha": args.lora_alpha if args.method != "full" else None,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
        "eval_metrics": eval_metrics,
        "save_dir": str(save_dir),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    (out_dir / "training_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] saved -> {save_dir}")

    if args.merge and args.method in {"lora", "qlora"}:
        del model, trainer
        torch.cuda.empty_cache()
        merge_adapter(args.model, save_dir, out_dir, args.trust_remote_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

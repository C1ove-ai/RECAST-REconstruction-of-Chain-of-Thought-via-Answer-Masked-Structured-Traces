#!/usr/bin/env python3
"""End-to-end RECAST pipeline:

    target model answers (Claude / OpenAI-compatible relay)
        -> decoder reconstructs CoT (DeepSeek)
        -> LLM cleaning
        -> final answers
        -> SFT-ready dataset + validation
        -> a self-contained, timestamped job directory with manifest

One command:

    python recast_pipeline.py --dataset math500 --target-model claude-sonnet-5-5 --limit 5

The job directory is created under jobs/<UTC-ish timestamp>_<dataset>_<model>_<pattern>/
and contains raw/, decoded/, cleaned/, answers/, sft/, logs/, config/, manifest.json,
summary.md. All paths are absolute; nothing is mixed with other jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "eval"
TESTSET_DIR = ROOT / "testset"
DEFAULT_DECODER_TEMPLATE = ROOT / "templates" / "decoder_template_solution.txt"
DEFAULT_TARGET_CONFIG = ROOT / "config.relay.json"
DEFAULT_DECODER_API = ROOT / "api.txt"
DEFAULT_CLEANER_API = ROOT / "api_clean.txt"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTSET_DIR))

from template_router import route  # noqa: E402
from download_testset import download_bench  # noqa: E402

BENCH_ALIASES = {
    "math500": "math500", "math_500": "math500", "math-500": "math500",
    "gsm8k": "gsm8k", "gsm_8k": "gsm8k", "gsm-8k": "gsm8k",
}


class PipelineError(RuntimeError):
    pass


def slugify(text: str, limit: int = 48) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-._")
    return (text or "x")[:limit]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def tail_log(path: Path, lines: int = 25) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def resolve_dataset(name: str) -> Path:
    path = Path(name)
    if path.is_file():
        return path.resolve()
    key = slugify(name).lower().replace("-", "_")
    candidates = [
        TESTSET_DIR / f"{key}_test.jsonl",
        TESTSET_DIR / f"{key}.jsonl",
        EVAL_DIR / "data" / "benchmarks" / f"{key}.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    bench = BENCH_ALIASES.get(key.replace("_", "")) or BENCH_ALIASES.get(key)
    if bench:
        return download_bench(bench, TESTSET_DIR).resolve()
    raise PipelineError(
        f"dataset not found: {name}. Use a local JSONL path (--dataset path.jsonl) "
        f"or put it under testset/ or eval/data/benchmarks/.")


def resolve_target(args: argparse.Namespace) -> dict:
    candidates = [Path(args.target_config)] if args.target_config else [DEFAULT_TARGET_CONFIG]
    cfg_path: Path | None = None
    cfg: dict = {}
    for candidate in candidates:
        if candidate and candidate.exists():
            cfg = json.loads(candidate.read_text(encoding="utf-8-sig"))
            cfg_path = candidate
            break
    if cfg_path is None:
        raise PipelineError(
            "target relay config not found. Put config.relay.json in the repo root or pass "
            "--target-config <path> (must contain base_url/model/api_key).")
    if not cfg.get("api_key"):
        raise PipelineError(f"{cfg_path} has no api_key")
    return {
        "model": args.target_model or cfg.get("model") or "claude-sonnet-5-5",
        "base_url": args.target_url or cfg.get("base_url") or "",
        "config_path": str(cfg_path),
    }


def resolve_pattern(rows: list[dict], source: str, args: argparse.Namespace) -> dict:
    if args.prompt_file:
        return {"pattern": "custom", "profile": None, "secondary": None,
                "reason": "custom_prompt", "source": source}
    if args.pattern == "none":
        return {"pattern": "none", "profile": None, "secondary": None,
                "reason": "disabled", "source": source}
    info = dict(route(rows, source, args.profile, args.pattern))
    info["reason"] = info.get("profile_reason") or info.get("reason")
    return info


def make_job_dir(args: argparse.Namespace, dataset_path: Path, model: str, pattern: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_slug = slugify(dataset_path.stem.replace("_test", "").replace("-test", ""))
    model_slug = slugify(model)
    pattern_slug = slugify(pattern or "none")
    if args.job_name:
        base = f"{slugify(args.job_name)}_{stamp}"
    else:
        base = f"{stamp}_{dataset_slug}_{model_slug}_{pattern_slug}"
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    job_dir = root / base
    suffix = 2
    while job_dir.exists():
        job_dir = root / f"{base}-{suffix}"
        suffix += 1
    return job_dir


def python_cmd(*parts: str) -> list[str]:
    return [sys.executable, *parts]


def run_step(name: str, cmd: list[str], cwd: Path, log_path: Path,
             allow_failure: bool = False) -> dict:
    print(f"[step] {name}: " + " ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.run(
            cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
    elapsed = round(time.perf_counter() - started, 1)
    result = {"returncode": proc.returncode, "elapsed_s": elapsed, "log": str(log_path)}
    if proc.returncode != 0 and not allow_failure:
        raise PipelineError(f"{name} failed (exit {proc.returncode}). Log: {log_path}\n"
                            f"{tail_log(log_path)}")
    print(f"[{'ok' if proc.returncode == 0 else 'warn'}] {name} ({elapsed}s)")
    return result


def collect_cmd(args, dataset_path: Path, job: Path, target: dict) -> list[str]:
    raw = job / "raw" / "collected.jsonl"
    cmd = python_cmd(
        str(ROOT / "collect_claude.py"),
        "--data", str(dataset_path),
        "--out", str(raw),
        "--pattern", args.pattern,
        "--model", target["model"],
        "--config", target["config_path"],
        "--workers", str(args.workers),
        "--temperature", str(args.temperature),
        "--max-tokens", str(args.max_tokens),
    )
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]
    if args.profile:
        cmd += ["--profile", args.profile]
    if args.prompt_file:
        cmd += ["--prompt-file", str(Path(args.prompt_file).resolve())]
    if target.get("base_url"):
        cmd += ["--url", target["base_url"]]
    return cmd


def decode_cmd(args, job: Path) -> list[str]:
    cmd = python_cmd(
        str(ROOT / "recast_decode.py"),
        "--dataset", str(job / "raw" / "successful.jsonl"),
        "--template", str(Path(args.decoder_template).resolve()),
        "--out", str(job / "decoded"),
        "--model", args.decoder_model,
        "--temperature", str(args.decoder_temperature),
        "--max-tokens", str(args.decoder_max_tokens),
    )
    if args.decoder_url:
        cmd += ["--base-url", args.decoder_url]
    if args.decoder_api:
        cmd += ["--api", str(Path(args.decoder_api).resolve())]
    return cmd


def clean_cmd(args, decoded_path: Path, job: Path) -> list[str]:
    cmd = python_cmd(
        str(EVAL_DIR / "data" / "clean_with_llm.py"),
        "--decoded", str(decoded_path),
        "--out", str(job / "cleaned" / "cleaned_cots.json"),
        "--workers", str(args.cleaner_workers),
        "--model", args.cleaner_model,
    )
    if args.cleaner_url:
        cmd += ["--base-url", args.cleaner_url]
    if args.cleaner_api:
        cmd += ["--api", str(Path(args.cleaner_api).resolve())]
    return cmd


def extract_cmd(args, decoded_path: Path, job: Path, has_reference: bool) -> list[str]:
    cmd = python_cmd(
        str(EVAL_DIR / "data" / "extract_answers.py"),
        "--decoded", str(decoded_path),
        "--out", str(job / "answers" / "answers.json"),
        "--model", args.decoder_model,
        "--mode", "api" if has_reference else "hybrid",
    )
    if has_reference:
        cmd += ["--reference-jsonl", str(job / "raw" / "successful.jsonl")]
    if args.decoder_url:
        cmd += ["--base-url", args.decoder_url]
    if args.decoder_api:
        cmd += ["--api", str(Path(args.decoder_api).resolve())]
    return cmd


def build_cmd(args, decoded_path: Path, job: Path) -> list[str]:
    return python_cmd(
        str(EVAL_DIR / "data" / "build_dataset.py"),
        "--decoded", str(decoded_path),
        "--answers", str(job / "answers" / "answers.json"),
        "--cleaned", str(job / "cleaned" / "cleaned_cots.json"),
        "--out-dir", str(job / "sft" / "processed"),
        "--target", "cot_answer",
    )


def validate_cmd(job: Path) -> list[str]:
    return python_cmd(
        str(EVAL_DIR / "data" / "validate_dataset.py"),
        "--data-dir", str(job / "sft" / "processed"),
    )


def load_decoded(decoded_dir: Path) -> Path:
    files = sorted(decoded_dir.glob("decoded_*.json"))
    if not files:
        raise PipelineError(f"decoder produced no decoded_*.json in {decoded_dir}")
    return files[-1]


def write_summary(job: Path, manifest: dict) -> None:
    d = manifest.get("dataset", {})
    t = manifest.get("target", {})
    p = manifest.get("pattern", {})
    stages = manifest.get("stages", {})
    lines = [
        f"# RECAST job `{manifest.get('job_id')}`",
        "",
        f"- status: **{manifest.get('status')}**",
        f"- dataset: `{d.get('path')}` (source: `{d.get('source')}`, reference answers: {d.get('has_reference')})",
        f"- target model: `{t.get('model')}` @ `{t.get('base_url')}`",
        f"- pattern: `{p.get('pattern')}` (profile: `{p.get('profile')}`, reason: `{p.get('reason')}`)",
        f"- decoder: `{manifest.get('decoder', {}).get('model')}` template `{Path(manifest.get('decoder', {}).get('template', '')).name}`",
        f"- limit: {manifest.get('limit')}",
        "",
        "| stage | status | seconds |",
        "|---|---|---|",
    ]
    for name, info in stages.items():
        lines.append(f"| {name} | {info.get('status')} | {info.get('elapsed_s', '-')} |")
    lines += ["", "## Artifacts", ""]
    for name, info in manifest.get("artifacts", {}).items():
        lines.append(f"- **{name}**: `{info}`")
    stats = manifest.get("counts", {})
    if stats:
        lines += ["", "## Counts", "", "```json",
                  json.dumps(stats, ensure_ascii=False, indent=2), "```"]
    lines += [
        "", "## Training", "",
        "```bash",
        f"python eval/train/train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --method qlora \\",
        f"    --data-dir {job / 'sft' / 'processed'} \\",
        f"    --output-dir runs/{slugify(manifest.get('job_id', 'job'))}_sft",
        "```",
        "",
        "## Logs", "",
        f"- `{job / 'logs'}`",
    ]
    (job / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Run the full RECAST CoT-reversal pipeline")
    parser.add_argument("--dataset", required=True,
                        help="math500 | gsm8k | local JSONL path (question/answer rows)")
    parser.add_argument("--target-model", default="",
                        help="target model id in the relay config (e.g. claude-sonnet-5-5)")
    parser.add_argument("--target-config", default="",
                        help="relay config JSON (default: config.relay.json)")
    parser.add_argument("--target-url", default="", help="relay base URL override")
    parser.add_argument("--pattern", default="auto",
                        help="auto|none|math_solver|socratic|expert_panel|pair_programming|red_blue|retrospective")
    parser.add_argument("--profile", default="", help="override dataset profile")
    parser.add_argument("--prompt-file", default="", help="single custom elicitation prompt")
    parser.add_argument("--limit", type=int, default=0, help="questions to run (0 = all)")
    parser.add_argument("--workers", type=int, default=1, help="collection concurrency")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--decoder-model", default="deepseek-chat")
    parser.add_argument("--decoder-url", default="", help="decoder base URL (default: DeepSeek official)")
    parser.add_argument("--decoder-api", default="", help="decoder key file (default: api.txt)")
    parser.add_argument("--decoder-template", default=str(DEFAULT_DECODER_TEMPLATE))
    parser.add_argument("--decoder-temperature", type=float, default=0.7)
    parser.add_argument("--decoder-max-tokens", type=int, default=8192)
    parser.add_argument("--cleaner-model", default="deepseek-chat")
    parser.add_argument("--cleaner-url", default="", help="cleaner base URL (default: DeepSeek official)")
    parser.add_argument("--cleaner-api", default="", help="cleaner key file (default: api_clean.txt)")
    parser.add_argument("--cleaner-workers", type=int, default=2)
    parser.add_argument("--out-root", default=str(ROOT / "jobs"))
    parser.add_argument("--job-name", default="", help="optional human-readable job name prefix")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()

    try:
        dataset_path = resolve_dataset(args.dataset)
        rows = read_jsonl(dataset_path)
        if not rows:
            raise PipelineError(f"empty dataset: {dataset_path}")
        source = str(rows[0].get("source") or dataset_path.stem)
        target = resolve_target(args)
        pattern_info = resolve_pattern(rows, source, args)
        pattern_name = pattern_info.get("pattern") or "none"
        has_reference = any(str(row.get("answer") or "").strip() for row in rows)
        job_dir = make_job_dir(args, dataset_path, target["model"], pattern_name)
    except PipelineError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    print(f"[plan] dataset={dataset_path} rows={len(rows)} reference_answers={has_reference}")
    print(f"[plan] target={target['model']} pattern={pattern_name} "
          f"profile={pattern_info.get('profile')} limit={args.limit or 'all'}")
    print(f"[plan] job={job_dir}")

    if args.dry_run:
        print(f"[dry-run] collect: {' '.join(collect_cmd(args, dataset_path, job_dir, target))}")
        print(f"[dry-run] decode : {' '.join(decode_cmd(args, job_dir))}")
        print(f"[dry-run] clean  : {' '.join(clean_cmd(args, job_dir / 'decoded' / 'decoded_<stamp>.json', job_dir))}")
        print(f"[dry-run] extract: {' '.join(extract_cmd(args, job_dir / 'decoded' / 'decoded_<stamp>.json', job_dir, has_reference))}")
        print(f"[dry-run] build  : {' '.join(build_cmd(args, job_dir / 'decoded' / 'decoded_<stamp>.json', job_dir))}")
        print(f"[dry-run] validate: {' '.join(validate_cmd(job_dir))}")
        return 0

    for subdir in ("raw", "decoded", "cleaned", "answers", "sft", "logs", "config"):
        (job_dir / subdir).mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "job_id": job_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "dataset": {"path": str(dataset_path), "source": source,
                    "rows": len(rows), "has_reference": has_reference},
        "target": {"model": target["model"], "base_url": target["base_url"],
                   "config_path": target["config_path"]},
        "pattern": pattern_info,
        "decoder": {"model": args.decoder_model,
                    "template": str(Path(args.decoder_template).resolve())},
        "cleaner": {"model": args.cleaner_model},
        "limit": args.limit,
        "stages": {},
        "artifacts": {},
        "counts": {},
    }
    write_json(job_dir / "manifest.json", manifest)
    write_json(job_dir / "config" / "run_config.json",
               {k: v for k, v in vars(args).items() if "key" not in k.lower()})
    for src, dst in ((Path(args.decoder_template), job_dir / "config" / "decoder_template.txt"),
                     (ROOT / "system_prompt.txt", job_dir / "config" / "system_prompt.txt")):
        if src.exists():
            (job_dir / "config" / dst.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    try:
        # 1) collect target-model answers
        res = run_step("collect", collect_cmd(args, dataset_path, job_dir, target),
                       ROOT, job_dir / "logs" / "01_collect.log", allow_failure=True)
        raw_rows = read_jsonl(job_dir / "raw" / "collected.jsonl") if (job_dir / "raw" / "collected.jsonl").exists() else []
        success_rows = [row for row in raw_rows if not row.get("error") and row.get("response")]
        if not success_rows:
            raise PipelineError(f"no successful target-model answers. See {job_dir / 'logs' / '01_collect.log'}")
        write_jsonl(job_dir / "raw" / "successful.jsonl", success_rows)
        summary = read_json(job_dir / "raw" / "collected.summary.json")
        if summary:
            write_json(job_dir / "config" / "target_prompts.json", summary)
        pattern_counts: dict[str, int] = {}
        for row in success_rows:
            key = f"{row.get('prompt_variant')}/{row.get('question_variant')}"
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
        manifest["stages"]["collect"] = {"status": "ok" if res["returncode"] == 0 else "partial",
                                         "elapsed_s": res["elapsed_s"], "log": res["log"]}
        manifest["counts"]["collected_ok"] = len(success_rows)
        manifest["counts"]["collected_errors"] = len(raw_rows) - len(success_rows)
        manifest["counts"]["prompt_variants"] = pattern_counts
        manifest["artifacts"]["collected"] = str(job_dir / "raw" / "collected.jsonl")
        manifest["artifacts"]["successful"] = str(job_dir / "raw" / "successful.jsonl")
        write_json(job_dir / "manifest.json", manifest)

        # 2) decode CoT with DeepSeek
        res = run_step("decode", decode_cmd(args, job_dir), ROOT,
                       job_dir / "logs" / "02_decode.log")
        decoded_path = load_decoded(job_dir / "decoded")
        decoded = read_json(decoded_path)
        decoded_records = decoded.get("records", [])
        if not decoded_records:
            raise PipelineError(f"decoder output has no records: {decoded_path}")
        manifest["stages"]["decode"] = {"status": "ok", "elapsed_s": res["elapsed_s"], "log": res["log"]}
        manifest["counts"]["decoded_records"] = len(decoded_records)
        manifest["artifacts"]["decoded"] = str(decoded_path)
        write_json(job_dir / "manifest.json", manifest)

        # 3) clean the reconstructed CoTs
        res = run_step("clean", clean_cmd(args, decoded_path, job_dir), EVAL_DIR,
                       job_dir / "logs" / "03_clean.log")
        cleaned = read_json(job_dir / "cleaned" / "cleaned_cots.json")
        status_counts: dict[str, int] = {}
        for entry in cleaned.get("cleaned", {}).values():
            status = entry.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        manifest["stages"]["clean"] = {"status": "ok", "elapsed_s": res["elapsed_s"], "log": res["log"]}
        manifest["counts"]["cleaned_status"] = status_counts
        manifest["artifacts"]["cleaned"] = str(job_dir / "cleaned" / "cleaned_cots.json")
        write_json(job_dir / "manifest.json", manifest)

        # 4) final answers (benchmark reference answers when available)
        res = run_step("extract_answers", extract_cmd(args, decoded_path, job_dir, has_reference),
                       EVAL_DIR, job_dir / "logs" / "04_extract_answers.log")
        answers = read_json(job_dir / "answers" / "answers.json")
        method_counts: dict[str, int] = {}
        for entry in answers.get("answers", {}).values():
            method = entry.get("method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1
        manifest["stages"]["extract_answers"] = {"status": "ok",
                                                 "elapsed_s": res["elapsed_s"], "log": res["log"]}
        manifest["counts"]["answer_methods"] = method_counts
        manifest["artifacts"]["answers"] = str(job_dir / "answers" / "answers.json")
        write_json(job_dir / "manifest.json", manifest)

        # 5) build SFT-ready dataset
        res = run_step("build_dataset", build_cmd(args, decoded_path, job_dir), EVAL_DIR,
                       job_dir / "logs" / "05_build_dataset.log")
        dataset_stats = read_json(job_dir / "sft" / "processed" / "dataset_stats.json")
        manifest["stages"]["build_dataset"] = {"status": "ok",
                                               "elapsed_s": res["elapsed_s"], "log": res["log"]}
        manifest["counts"]["dataset"] = {k: dataset_stats.get(k) for k in (
            "input_records", "kept_samples", "splits", "dropped", "cot_sources", "used_cleaned")}
        manifest["artifacts"]["sft_dataset"] = str(job_dir / "sft" / "processed")
        write_json(job_dir / "manifest.json", manifest)

        # 6) validate
        res = run_step("validate", validate_cmd(job_dir), EVAL_DIR,
                       job_dir / "logs" / "06_validate.log")
        manifest["stages"]["validate"] = {"status": "ok", "elapsed_s": res["elapsed_s"], "log": res["log"]}
        manifest["status"] = "ok" if all(
            stage.get("status") == "ok" for stage in manifest["stages"].values()) else "partial"
        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(job_dir / "manifest.json", manifest)
        write_summary(job_dir, manifest)

        print(f"\n[done] job={job_dir}")
        print(f"[summary] {job_dir / 'summary.md'}")
        print(f"[manifest] {job_dir / 'manifest.json'}")
        print(f"[sft] {job_dir / 'sft' / 'processed'}")
        print(f"[next] python eval/train/train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct "
              f"--method qlora --data-dir {job_dir / 'sft' / 'processed'} "
              f"--output-dir runs/{slugify(job_dir.name)}_sft")
        return 0
    except Exception as exc:  # noqa: BLE001 - record the failure in the manifest
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(job_dir / "manifest.json", manifest)
        write_summary(job_dir, manifest)
        print(f"\n[error] {exc}", file=sys.stderr)
        print(f"[manifest] {job_dir / 'manifest.json'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

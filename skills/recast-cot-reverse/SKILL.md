---
name: recast-cot-reverse
description: "Run RECAST end-to-end for a dataset and target model: collect the target model's answers, reconstruct its chain-of-thought with a decoder, clean the traces, extract final answers, and build SFT-ready JSONL plus a timestamped job directory. Use when the user asks to run CoT reversal / 反推思维链 / 逆向 CoT on a dataset with a model."
whenToUse: The user says "用某数据集跑一下某模型的 CoT 逆向 / 反推思维链 / 还原 CoT"，or wants SFT data derived from a black-box model's answers.
metadata:
  short-description: Run target-model CoT reversal end-to-end
---

# RECAST CoT reversal

Given a dataset and a target model, run the whole RECAST loop and leave a
self-contained, timestamped job directory. Do not re-implement the pipeline;
run the project script.

## 1. Locate the project

The pipeline lives at `recast_pipeline.py` in the repository root. In the
original workspace layout it may instead be at `<workspace>/RECAST/recast_pipeline.py`.
Resolve the repo root in this order:

1. `$env:RECAST_ROOT` if set;
2. a `recast_pipeline.py` in the current directory or an ancestor;
3. `<workspace>/RECAST/recast_pipeline.py` if present.

Then run from anywhere with absolute paths.

## 2. Check prerequisites (never print key values)

- `config.relay.json` — target model relay: `base_url`, `model`, `api_key`;
- `api.txt` — decoder key;
- `api_clean.txt` — cleaning-model key.

If a file is missing, ask the user for it before running. Never echo API keys.

**Model roles are swappable.** Defaults reflect the authors' setup, but every
role only needs an OpenAI-compatible URL plus a key:

| role | default | change with |
|---|---|---|
| target model (answers) | configured relay model | `--target-config/--target-model/--target-url` |
| decoder (reconstructs CoT) | `deepseek-chat` via DeepSeek | `--decoder-model/--decoder-url/--decoder-api` |
| cleaner (cleans CoT) | `deepseek-chat` via DeepSeek | `--cleaner-model/--cleaner-url/--cleaner-api` |
| answer extractor | same as decoder | inherits decoder settings |

Example for a different decoder endpoint:

```bash
python recast_pipeline.py --dataset math500 --target-model my-teacher \
    --decoder-model my-decoder --decoder-url https://my-endpoint/v1 \
    --decoder-api path/to/decoder_key.txt
```

## 3. Pick dataset and model

- Dataset: `math500` and `gsm8k` are auto-downloaded to `testset/`.
  For another dataset use a local JSONL path with `{question, answer?}` rows:
  `--dataset path/to/data.jsonl`.
- Target model: `--target-model <relay model id>` (default: the model in
  `config.relay.json`). The decoder and cleaner are DeepSeek by default.
- Pattern: automatic by default (`--pattern auto`):
  math → `math_solver`, code → `pair_programming`, science → `expert_panel`,
  logic/planning → `socratic`, open-ended → `retrospective`.
  Override with `--pattern <id>` or `--pattern none` for the old long chain.

If the user did not specify a dataset or model, default to `math500` + the
configured target model, and say what you chose.

## 4. Run

Smoke test first when the job is non-trivial (more than a few questions):

```bash
python <repo>/recast_pipeline.py --dataset math500 --target-model claude-sonnet-5-5 --limit 2
```

Then run the requested scale:

```bash
python <repo>/recast_pipeline.py --dataset math500 --target-model claude-sonnet-5-5
```

If the skill is installed outside the repo, its bundled wrapper locates RECAST
automatically:

```bash
python <skill-dir>/scripts/run_recast.py --dataset math500 --target-model claude-sonnet-5-5
```

Useful flags: `--limit N`, `--workers N`, `--pattern <id>`, `--profile <name>`,
`--target-config <path>`, `--target-url`, `--decoder-template <path>`,
`--out-root <dir>`, `--job-name <name>`, `--dry-run`.

## 5. Report

The pipeline creates `jobs/<timestamp>_<dataset>_<model>_<pattern>/` with:

```
raw/        target-model answers (collected.jsonl + successful.jsonl)
decoded/    reconstructed CoT run
cleaned/    cleaned CoT overlay
answers/    final answer sidecar
sft/processed/{sft_*.jsonl, eval_*.jsonl, dataset_stats.json}
logs/       per-stage logs
config/     prompts, decoder template, run config
manifest.json / summary.md
```

Report: job path, collected/success counts, pattern used, decoded/cleaned counts,
SFT sample counts, and the one-line training command from `summary.md`.
Link the job's `summary.md` and `sft/processed/`.

## 6. Failure handling

- `content_filter`: the script automatically falls back through patterns and
  medium/short prompts. If all fail, lower `--limit`, try `--pattern none`, or
  check the relay's model/policy. Do not silently change the user's model.
- missing keys/config: ask the user; never print the key.
- no SFT samples: inspect `logs/05_build_dataset.log` and
  `cleaned/cleaned_cots.json`; report which stage dropped samples.
- relay errors/timeouts: inspect `logs/01_collect.log`; retry with lower
  `--workers` or a later run.
- Do not rerun the whole job just to inspect results; the job directory is
  self-contained and already named.

Detailed flags, stage mapping and naming are in
[references/pipeline.md](references/pipeline.md).

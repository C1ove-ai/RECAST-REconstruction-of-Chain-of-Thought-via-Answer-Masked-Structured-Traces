# RECAST: Reconstruct, Don't Just Expose

**Answer-masked auditable reasoning distillation from black-box models.**

RECAST takes a dataset of questions, collects a target model's answers, reconstructs an
auditable first-person chain-of-thought (CoT) with a decoder model, cleans the trace,
extracts final answers, and emits SFT-ready chat data. It ships with a one-command
pipeline and a portable agent skill for Codex / Claude Code / DeepSeek Harness.

The reconstructed rationale is a **proxy** for the target model's hidden reasoning, not
ground truth — evaluate it downstream (student accuracy, faithfulness probes, human/LLM
judging) rather than assuming it is the "real" CoT.

Research use only; users are responsible for complying with their model provider's terms！！！
---

## Pipeline

```text
dataset (math500 / gsm8k / custom JSONL)
        │
        ▼
[1] target model (black-box)  ──►  answers (teacher-student dialogue, configurable pattern)
        │
        ▼
[2] decoder model             ──►  reconstructed CoT (first-person solution trace)
        │
        ▼
[3] cleaner model             ──►  removes dialogue/meta noise, normalizes format
        │
        ▼
[4] answer extraction         ──►  final answers (ground-truth reference when available)
        │
        ▼
[5] build + validate          ──►  SFT-ready JSONL + job manifest/summary
```

Every run creates a self-contained job directory under `jobs/`:

```text
jobs/<YYYYMMDD-HHMMSS>_<dataset>_<target-model>_<pattern>/
├── raw/            target answers (collected.jsonl, successful.jsonl, summary)
├── decoded/        reconstructed CoT run (json + md)
├── cleaned/        cleaned CoT overlay
├── answers/        final answer sidecar
├── sft/processed/  sft_train/val/test.jsonl, eval_*.jsonl, dataset_stats.json
├── logs/           01_collect.log ... 06_validate.log
├── config/         prompts, decoder template, run config
├── manifest.json   machine-readable stages / counts / paths
└── summary.md      human-readable report + training command
```

## Highlights

- **One command end-to-end** — collect → decode → clean → answer → SFT data → validate.
- **Automatic dataset → pattern routing** — math, code, science, logic/planning and
  open-ended datasets get different elicitation patterns.
- **Filter resilient** — long → medium → short prompt fallbacks and LaTeX-light question
  variants when a relay returns `content_filter`.
- **Self-contained results** — timestamped job dirs, per-stage logs, manifest + summary.
- **Swappable model roles** — target, decoder and cleaner each only need an
  OpenAI-compatible endpoint + key; defaults are the authors' reference setup.
- **SFT-ready output** — chat JSONL with assistant-only loss, plus optional LoRA/QLoRA
  training and evaluation harness under `eval/`.
- **Agent skill** — say "help me run math500 in claude for CoT reverse" in Codex / Claude Code /
  DeepSeek Harness after installing the skill.

---

## Quickstart

Requirements: Python 3.10+ (Windows/Linux). The core pipeline only needs `openai`; the
optional training/eval stack is described in [`eval/requirements.txt`](eval/requirements.txt).

```bash
git clone <your-repo-url> recast-cot-reverse
cd recast-cot-reverse
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

cp config.relay.example.json config.relay.json   # then edit
cp api.example.txt api.txt                       # decoder key
cp api_clean.example.txt api_clean.txt           # cleaner key
```

Fill in the three files:

| File | Content |
|---|---|
| `config.relay.json` | target model relay: `{"base_url": "...", "model": "...", "api_key": "..."}` |
| `api.txt` | decoder key, e.g. `sk-...` |
| `api_clean.txt` | cleaner key, e.g. `sk-...` |

Smoke test (2 questions) and inspect the job:

```bash
python recast_pipeline.py --dataset math500 --target-model <model-id> --limit 2
```

Run the requested scale:

```bash
python recast_pipeline.py --dataset math500 --target-model <model-id>
python recast_pipeline.py --dataset gsm8k --target-model <model-id> --limit 100
```

Use `--dry-run` to print the resolved dataset/model/pattern and stage commands without
making API calls.

---

## Model configuration: every role is swappable

Defaults are the authors' reference setup, but **no role is hard-wired to a vendor**.
Any OpenAI-compatible `/chat/completions` endpoint works; you only need a URL and a key.

| Role | What it does | Default reference setup | How to change |
|---|---|---|---|
| target model | answers the dataset questions | model from `config.relay.json` | edit the config, or `--target-config`, `--target-model`, `--target-url` |
| decoder | reconstructs the first-person CoT | `deepseek-chat` via DeepSeek official | `--decoder-model`, `--decoder-url`, `--decoder-api` |
| cleaner | cleans the reconstructed CoT | `deepseek-chat` via DeepSeek official | `--cleaner-model`, `--cleaner-url`, `--cleaner-api` |
| answer extraction | extracts final answers | inherits decoder settings | decoder flags |

Example with three different endpoints:

```bash
python recast_pipeline.py --dataset math500 \
  --target-config my_teacher_relay.json --target-model teacher-x \
  --decoder-model decoder-y --decoder-url https://decoder.example/v1 \
  --decoder-api my_decoder_key.txt \
  --cleaner-model cleaner-z --cleaner-url https://cleaner.example/v1 \
  --cleaner-api my_cleaner_key.txt
```

The decoder prompt itself is also swappable: `--decoder-template` accepts any template
consuming `{问题}` and `{完整回答}`. `templates/decoder_template_solution.txt` is the
recommended default; `decoder_template.txt` is the legacy 7-stage audit template.

Key files are gitignored and never printed by the pipeline.

---

## Datasets

| Dataset | Source | Notes |
|---|---|---|
| `math500` | MATH-500 test split | auto-downloaded to `testset/` |
| `gsm8k` | GSM8K test split | auto-downloaded to `testset/` |
| custom | local JSONL | one row per line |

You can add more datasets！
Custom dataset format:

```json
{"id": "q-001", "question": "……", "answer": "optional final answer"}
```

- `question` is required.
- `answer` is optional. When present, it is used as the reference answer
  (`testset_reference`) instead of asking the extraction model; when absent, the final
  answer is extracted from the target model's answer.
- Files produced by `eval/data/download_benchmarks.py` under
  `eval/data/benchmarks/<name>.jsonl` are also accepted by name.

---

## Elicitation patterns

The target model is prompted with a dataset-appropriate teacher-student pattern. `--pattern
auto` (default) uses `template_router.py`:

| Profile | Primary pattern | Secondary |
|---|---|---|
| math | `math_solver` | `red_blue` |
| code | `pair_programming` | `red_blue` |
| science | `expert_panel` | `red_blue` |
| logic | `socratic` | `expert_panel` |
| planning | `socratic` | `retrospective` |
| open-ended | `retrospective` | `socratic` |
| general | `socratic` | `math_solver` |

Routing order: known dataset name → keyword scoring → `general`. Override with
`--profile` / `--pattern <id>`, or disable pattern routing with `--pattern none`
(old long → medium → short prompt chain). If the relay filters a prompt, the collector
automatically tries the secondary pattern, then medium/short variants, and a LaTeX-light
question variant.

See [`templates/patterns/README.md`](templates/patterns/README.md) for the design
rationale and how each pattern was optimized.

---

## Agent skill

The skill lives in [`skills/recast-cot-reverse/`](skills/recast-cot-reverse/SKILL.md) and
works in Codex, Claude Code, DeepSeek Harness and other hosts that support the standard
`SKILL.md` format.

```powershell
# From the repository root (PowerShell)
powershell -NoProfile -ExecutionPolicy Bypass -File skills\install_skill.ps1
```

Defaults install to project-level DSH (`.dsh/skills`) and Claude Code
(`.claude/skills`). Install user-level targets too:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File skills\install_skill.ps1 -Targets all
```

Start a new agent session (or reload skills), then say:

> 帮我用 math500 跑一下 claude-sonnet-5-5 的 CoT 逆向

The agent will load the skill, locate `recast_pipeline.py`, pick dataset/model/pattern,
run the pipeline, and report the job path + summary. The skill also bundles
`scripts/run_recast.py`, which locates the repo from any working directory.

See [`skills/README.md`](skills/README.md) for install targets and
[`references/pipeline.md`](skills/recast-cot-reverse/references/pipeline.md) for full flags.

---

## CLI essentials

```text
--dataset PATH|NAME           math500 | gsm8k | local JSONL
--target-model ID             relay model id
--target-config PATH          relay config JSON
--target-url URL              relay base URL override
--pattern auto|none|ID        elicitation pattern
--profile NAME                force dataset profile
--prompt-file PATH            single custom prompt with {user_query}
--limit N                     number of questions (0 = all)
--workers N                   collection concurrency
--temperature F               target sampling temperature
--max-tokens N                target max output tokens
--decoder-model / --decoder-url / --decoder-api / --decoder-template
--cleaner-model / --cleaner-url / --cleaner-api
--out-root DIR                default: jobs/
--job-name NAME               optional job prefix
--dry-run                     print plan, no API calls
```

Full flag reference: [`skills/recast-cot-reverse/references/pipeline.md`](skills/recast-cot-reverse/references/pipeline.md).
`python recast_pipeline.py --help` is always authoritative.

---

## What each stage does

| Stage | Script | Output |
|---|---|---|
| collect | `collect_claude.py` | target answers + per-question pattern/template metadata |
| decode | `recast_decode.py` | first-person CoT reconstructed from each answer |
| clean | `eval/data/clean_with_llm.py` | removes dialogue/meta noise, length sanity, retry |
| extract answers | `eval/data/extract_answers.py` | benchmark reference answers or LLM extraction |
| build SFT data | `eval/data/build_dataset.py` | `sft_train/val/test.jsonl`, `eval_*.jsonl`, stats |
| validate | `eval/data/validate_dataset.py` | schema/duplication/leakage checks |

The decoder enforces a minimum output length (relative to the source answer); short
outputs are retried with an expansion instruction. The cleaner runs a targeted
JSON-contract prompt with length/leakage guards; failures fall back to the raw CoT and
the rule-based safety net.

---

## Optional training and evaluation

The pipeline stops at SFT-ready data; training is a separate step:

```bash
python eval/train/train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --method qlora \
  --data-dir jobs/<job>/sft/processed \
  --output-dir runs/<job>_sft --merge
```

The repo also ships evaluation harnesses (`eval/harness/`), benchmark downloaders, a
faithfulness probe, and metric docs under `eval/`. See [`eval/README.md`](eval/README.md).

---

## Troubleshooting

| Symptom | Likely cause / action |
|---|---|
| `401/402` during collect/decode/clean | wrong key, expired key, or balance exhausted; check the corresponding log |
| `content_filter` keeps appearing | relay policy/model version changed; try `--pattern none`, smaller `--limit`, another relay/model |
| no SFT samples | inspect `cleaned/cleaned_cots.json` statuses and `sft/processed/dataset_stats.json` (`dropped` reasons) |
| dataset not found | use `--dataset path.jsonl`, or place the file under `testset/` or `eval/data/benchmarks/` |
| HF download fails | put the dataset JSONL locally and pass `--dataset`; the pipeline itself does not require `datasets` |
| Windows cannot run PowerShell installer | run with `powershell -NoProfile -ExecutionPolicy Bypass -File ...` |

Cost note: each question uses one target-model call, one decoder call and one cleaner
call (plus answer extraction when no reference answer exists). Start with `--limit 2`,
inspect the job, then scale.

---

## Limitations

- The reconstructed rationale is a **proxy**, not the model's true hidden reasoning;
  validate it downstream rather than treating it as ground truth.
- Relay providers can change model versions, routing and safety filters between runs;
  record the job timestamp/config when comparing experiments.
- Answers may leak into rationales; the pipeline includes leakage checks and cleaning,
  but human spot checks are recommended.
- Reconstructed traces can be unfaithful (post-hoc rationalization); see
  `eval/docs/metrics.md` for faithfulness probes.
- The optional training scripts require a CUDA GPU and were not executed by this
  pipeline; treat GPU results as environment-dependent.
- Dataset licenses follow their upstream sources (MATH: MIT, GSM8K: MIT); verify before
  redistribution.

---

## Repository layout

```text
recast-cot-reverse/
├── recast_pipeline.py        # one-command end-to-end pipeline
├── collect_claude.py         # target-model collection + pattern routing
├── recast_decode.py          # decoder runner (DeepSeek by default)
├── ask_claude.py             # one-shot target-model call
├── template_router.py        # dataset -> elicitation pattern
├── decoder_template.txt      # legacy 7-stage audit template
├── system_prompt.txt         # decoder persona
├── templates/                # decoder templates + elicitation patterns
├── skills/                   # portable agent skill + installer
├── testset/                  # benchmark downloader (data not committed)
├── eval/                     # cleaning, answers, build, validation, train, eval harness
├── pattern_extension.txt     # original pattern notes
├── config.relay.example.json
├── api.example.txt
├── api_clean.example.txt
├── requirements.txt
└── .gitignore
```

---

## License

MIT. See [LICENSE](LICENSE).

## Citation and attribution

RECAST is released under the MIT License. If you use RECAST in academic work,
please cite the repository below while the paper is in preparation:

@misc{recast2026,
  title        = {RECAST: REconstruction of Chain-of-Thought via Answer-Masked Structured Traces},
  author       = {Qi Wang},
  year         = {2026},
  howpublished = {\url{https://github.com/C1ove-ai/RECAST-REconstruction-of-Chain-of-Thought-via-Answer-Masked-Structured-Traces}},
  note         = {Paper in preparation; GitHub repository}
}
If you use RECAST in a product, demo, blog post, course, or derivative tool, please:

mention that it is built with RECAST and link back to this repository;
keep this citation/attribution section when redistributing a modified version;
optionally open a GitHub issue or Discussion to share your use case — it helps the project and is always appreciated.
When the paper is published, replace the BibTeX entry above with the official citation (venue, DOI, etc.).

Note: the MIT License does not legally require citation; this section is an academic/community attribution request, not an additional license condition.

## Acknowledgements

This project builds on ideas from step-by-step distillation, self-training and
faithfulness research, including LIMO, DeepSeek-R1, Distilling Step-by-Step, STaR,
Self-Consistency and related work. See `eval/docs/metrics.md` and
`eval/docs/training_methods.md` for references.

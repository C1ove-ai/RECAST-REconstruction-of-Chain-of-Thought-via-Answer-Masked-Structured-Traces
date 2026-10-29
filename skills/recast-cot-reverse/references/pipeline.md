# RECAST pipeline reference

## Stages and artifacts

| Stage | Script | Artifact in job dir |
|---|---|---|
| collect | `collect_claude.py` | `raw/collected.jsonl`, `raw/collected.summary.json`, `raw/successful.jsonl` |
| decode | `recast_decode.py` (DeepSeek) | `decoded/decoded_<stamp>.json` |
| clean | `eval/data/clean_with_llm.py` | `cleaned/cleaned_cots.json` |
| extract answers | `eval/data/extract_answers.py` | `answers/answers.json` |
| build SFT data | `eval/data/build_dataset.py` | `sft/processed/sft_{train,val,test}.jsonl`, `eval_*.jsonl`, `dataset_stats.json` |
| validate | `eval/data/validate_dataset.py` | nothing (exit code; output in `logs/06_validate.log`) |

The job also contains `manifest.json` (machine-readable stage/count/path record) and
`summary.md` (human-readable report + training command).

## Job naming

```
jobs/<YYYYMMDD-HHMMSS>_<dataset>_<target-model>_<pattern>/
```

Each run gets a new directory; if a name collides, a `-2`, `-3` suffix is added.
`--job-name foo` makes the prefix `foo_<timestamp>`. Dataset/model/pattern are
slugified (spaces and unsafe characters become `-`).

The embedded timestamp is local time; the manifest `created_at` uses ISO local time.

## Dataset selection

`--dataset` accepts:

1. a local JSONL path;
2. `math500` / `gsm8k` (auto-downloaded via `testset/download_testset.py`);
3. a name matching `testset/<name>_test.jsonl`, `testset/<name>.jsonl`,
   or `eval/data/benchmarks/<name>.jsonl` (e.g. files produced by
   `eval/data/download_benchmarks.py`).

Every row needs `question`; `answer` is optional. When the dataset has `answer`,
the pipeline uses it as the reference answer (`testset_reference`) instead of
extracting answers from the teacher response.

## Target model

The target model is whatever OpenAI-compatible relay model is configured in
`config.relay.json` (or `--target-config <path>`):

```json
{"base_url": "https://...", "model": "claude-sonnet-5-5", "api_key": "sk-..."}
```

`--target-model` overrides the model id; `--target-url` overrides the base URL.
The decoder (`--decoder-model`, default `deepseek-chat`) and cleaner
(`--cleaner-model`, default `deepseek-chat`) use `api.txt` / `api_clean.txt`
unless overridden.

## Model roles are swappable

Defaults are the authors' setup; no role is hard-wired to a vendor. Each role
only needs an OpenAI-compatible URL and a key:

| role | default | change with |
|---|---|---|
| target model | model from `config.relay.json` | `--target-config`, `--target-model`, `--target-url` |
| decoder | `deepseek-chat` @ DeepSeek | `--decoder-model`, `--decoder-url`, `--decoder-api` |
| cleaner | `deepseek-chat` @ DeepSeek | `--cleaner-model`, `--cleaner-url`, `--cleaner-api` |
| answer extraction | inherits decoder | decoder flags |

The decoder is also swappable at the template level: `--decoder-template` can
point at any prompt that consumes `{问题}` + `{完整回答}` (see `templates/`).

Example with different endpoints for all three roles:

```bash
python recast_pipeline.py --dataset math500 \
  --target-config my_teacher_relay.json --target-model teacher-x \
  --decoder-model decoder-y --decoder-url https://decoder.example/v1 \
  --decoder-api my_decoder_key.txt \
  --cleaner-model cleaner-z --cleaner-url https://cleaner.example/v1 \
  --cleaner-api my_cleaner_key.txt
```

## Pattern selection

`--pattern auto` uses `template_router.py`:

- known dataset name → profile from `templates/patterns/registry.json`;
- unknown dataset → keyword scoring over the first 20 questions;
- fallback profile `general`.

Profiles and primary patterns:

| profile | primary | secondary |
|---|---|---|
| math | math_solver | red_blue |
| code | pair_programming | red_blue |
| science | expert_panel | red_blue |
| logic | socratic | expert_panel |
| planning | socratic | retrospective |
| open_ended | retrospective | socratic |
| general | socratic | math_solver |

The collection prompt chain is `pattern → secondary pattern → medium → short`;
each prompt is tried with the original question and then a LaTeX-light variant.
`--pattern none` restores `long → medium → short`. `--profile` forces the
profile; `--pattern <id>` forces a pattern; `--prompt-file` uses one custom
prompt (disables the chain).

## Key flags

```text
--dataset PATH|NAME          dataset JSONL or known benchmark name
--target-model ID            relay model id
--target-config PATH         relay config JSON
--target-url URL             relay base URL override
--pattern auto|none|ID       elicitation pattern
--profile NAME               force dataset profile
--prompt-file PATH           single custom prompt with {user_query}
--limit N                    questions to run (0 = all)
--workers N                  collection concurrency
--temperature F              target sampling temperature (default 0.7)
--max-tokens N               target max output tokens (default 8192)
--decoder-model ID           default deepseek-chat
--decoder-url URL            decoder base URL override
--decoder-api PATH           default api.txt
--decoder-template PATH      default templates/decoder_template_solution.txt
--decoder-temperature F      default 0.7
--decoder-max-tokens N       default 8192
--cleaner-model ID           default deepseek-chat
--cleaner-url URL            cleaner base URL override
--cleaner-api PATH           default api_clean.txt
--cleaner-workers N          default 2
--out-root DIR               default jobs
--job-name NAME              optional job prefix
--dry-run                    print resolved dataset/model/pattern/commands and exit
```

## Scaling and cost

Each question costs one target-model call (`collect`), one decoder call, one
cleaner call, and (when no reference answers exist) one answer-extraction call.
A 500-question MATH-500 run is therefore ~500 target calls plus ~1000-1500
DeepSeek calls. Run `--limit 2` first, inspect the job, then run the full set.

## Failure handling

- **content_filter**: the collector tries pattern → secondary → medium → short,
  and original → plain-math question variants. If everything fails, reduce the
  batch, try `--pattern none`, or change relay/model.
- **auth/balance errors**: check `logs/01_collect.log`, `logs/02_decode.log`,
  `logs/03_clean.log`; fix the corresponding key file.
- **empty SFT dataset**: inspect `cleaned/cleaned_cots.json` (status values),
  `dataset_stats.json` (`dropped` reasons), and `logs/05_build_dataset.log`.
- **validation failure**: inspect `logs/06_validate.log`; build still leaves the
  raw/decoded/cleaned artifacts for debugging.

## After the job

The summary includes the training command, e.g.:

```bash
python eval/train/train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --method qlora \
  --data-dir jobs/<job>/sft/processed \
  --output-dir runs/<job>_sft
```

Do not run training unless the user asks; the pipeline stops after building and
validating the SFT-ready data.

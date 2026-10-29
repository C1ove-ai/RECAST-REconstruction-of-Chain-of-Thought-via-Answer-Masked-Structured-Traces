# RECAST agent instructions

This repository contains **RECAST**, a black-box chain-of-thought reconstruction
pipeline and its SFT-data tooling.

When the user asks to "跑一下某某模型的 CoT 逆向 / 反推思维链 / 还原 CoT" on a
dataset, do not rebuild the pipeline by hand. Use the `recast-cot-reverse` skill
if it is available, or run the one-command pipeline from the repository root:

```bash
python recast_pipeline.py --dataset <math500|gsm8k|path.jsonl> \
    --target-model <relay-model-id> [--limit N]
```

Key files:

- `recast_pipeline.py` — full pipeline: collect target answers → decoder CoT →
  clean → extract answers → build/validate SFT data, with a timestamped job dir;
- `collect_claude.py` — target-model collection with auto pattern routing and
  content-filter fallbacks;
- `recast_decode.py` — CoT decoder (DeepSeek by default, swappable; accepts `.jsonl` datasets directly);
- `eval/data/` — cleaning, answer extraction, dataset build, validation;
- `templates/patterns/` — per-dataset elicitation patterns;
- `skills/recast-cot-reverse/` — portable agent skill (SKILL.md).

Never print API keys from `api.txt`, `api_clean.txt`, or `config.relay.json`.
The pipeline writes self-contained jobs under `jobs/`; report the job
path and `summary.md` rather than scattering outputs elsewhere.

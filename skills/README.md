# Installing the `recast-cot-reverse` skill

Canonical skill source: [`recast-cot-reverse/`](recast-cot-reverse/SKILL.md).

The skill is a standard `SKILL.md` bundle (works in Codex, Claude Code, DeepSeek
Harness and other hosts that support the skill format). It only orchestrates the
project pipeline; the actual code stays in this repo.

## Install targets

| Target | Path | Notes |
|---|---|---|
| DSH, project | `<repo>/.dsh/skills/recast-cot-reverse` | scanned at rank 100 |
| Claude Code, project | `<repo>/.claude/skills/recast-cot-reverse` | project-local skill |
| shared agents, project | `<repo>/.agents/skills/recast-cot-reverse` | scanned by DSH rank 200 |
| Codex, user | `~/.codex/skills/recast-cot-reverse` | |
| Claude Code, user | `~/.claude/skills/recast-cot-reverse` | |
| DSH, user | `~/.dsh/skills/recast-cot-reverse` | |
| shared agents, user | `~/.agents/skills/recast-cot-reverse` | |

## Install

From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File skills\install_skill.ps1
```

Defaults install to the **project** DSH and Claude Code folders. Install all
known targets:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File skills\install_skill.ps1 -Targets all
```

Specific targets:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File skills\install_skill.ps1 -Targets codex,claude
```

After installing, start a new agent session (or reload skills) and say:

> 帮我用 math500 跑一下 claude-sonnet-5-5 的 CoT 逆向

## Prerequisites

The skill calls `recast_pipeline.py`, which needs:

- `config.relay.json` — target model relay (`base_url`, `model`, `api_key`);
- `api.txt` — decoder key (DeepSeek);
- `api_clean.txt` — cleaning-model key.

Keys are never printed by the pipeline or the skill.

# 数学测试集（供目标模型响应收集）

本目录由 `download_testset.py` 生成，包含两个公开推理测试集，统一成同一 schema，
供 `collect_claude.py` 批量收集目标模型（老师-学生模板）的响应。

## 文件

| 文件 | 条数 | 来源 | 许可 |
|---|---|---|---|
| `math500_test.jsonl` | 500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)（MATH 测试集精选） | 跟随上游 MATH 数据集（MIT） |
| `gsm8k_test.jsonl` | 1319 | [openai/grade-school-math](https://github.com/openai/grade-school-math) test split | MIT |
| `manifest.json` | — | 下载时间、源 URL、条数、sha256 | — |

重新下载：

```bash
cd recast-cot-reverse
python testset/download_testset.py                 # 两个都下
python testset/download_testset.py --bench math500
```

## Schema（每行一个 JSON）

```json
{"id": "math500-0001", "source": "math500",
 "question": "Convert the point $(0,3)$ ...",
 "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
 "solution": "...", "subject": "Precalculus", "level": 2, "unique_id": "test/precalculus/807.json"}
```

GSM8K 的 `answer` 已从 `####` 之后抽取为最终答案，`solution` 保留完整解答。

## 收集目标模型响应

`collect_claude.py` 默认 `--pattern auto`：先由 `template_router.py` 根据数据集
选择外显 CoT 的模式（见 [templates/patterns/README.md](../templates/patterns/README.md)），
再把 `{user_query}` 替换成题目、逐题调用 relay，输出
`runs/claude_<bench>_<stamp>.jsonl`（可断点续跑）。

如果要做完整流程（收集 → 解码还原 CoT → 清洗 → 答案 → SFT 数据），直接用
`recast_pipeline.py`，一次命令生成带时间戳的 job 目录：

```bash
cd recast-cot-reverse
python recast_pipeline.py --dataset math500 --target-model claude-sonnet-5-5 --limit 5
```

产物在 `jobs/<时间戳>_<数据集>_<模型>_<pattern>/`，包含 `raw/ decoded/
cleaned/ answers/ sft/ logs/ config/ manifest.json summary.md`。详见
[recast_pipeline.py](../recast_pipeline.py) 和
[skills/recast-cot-reverse/SKILL.md](../skills/recast-cot-reverse/SKILL.md)。

| 数据集类型 | 例子 | 主模式 |
|---|---|---|
| 数学 | GSM8K、MATH-500、AIME | `math_solver` |
| 代码 | HumanEval、MBPP、LiveCodeBench | `pair_programming` |
| 科学/STEM | GPQA、ScienceQA | `expert_panel` |
| 逻辑/规划 | BBH、LogiQA、PlanBench | `socratic` |
| 开放写作/需求 | 产品/需求分析 | `retrospective` |

```bash
cd recast-cot-reverse

# 自动路由 + 自动 filter 回退
python collect_claude.py --bench math500 --limit 5
python collect_claude.py --bench gsm8k --limit 100
python collect_claude.py --bench math500 --workers 4

# 查看某数据集会选哪个模式
python template_router.py --data testset/math500_test.jsonl

# 手动覆盖
python collect_claude.py --bench math500 --pattern math_solver
python collect_claude.py --bench math500 --profile code
python collect_claude.py --bench math500 --pattern none      # 旧链 long→medium→short
```

Windows 一键包装：

```powershell
.\run_collect_claude.ps1 -Bench math500 -Limit 5 -Pattern auto
```

Relay 配置默认读 `config.relay.json`；也可用 `--config`、`--url`、`--model`、
`--api-key` 覆盖。key 不会打印。

默认链：**pattern（按数据集自动选）→ secondary pattern → medium → short**。
每题先试原始题面，再试 LaTeX 较轻的 `plain_math` 题面；成功即停，失败原因记录在
`attempts` 里。`--pattern none` 回到 `long → medium → short`；`--no-fallback` 只跑
第一个模板；`--prompt-file` 使用单模板、不启用链。

## 输出字段

```json
{"id": "math500-0001", "source": "math500", "question": "...",
 "dataset_profile": "math", "pattern": "math_solver",
 "prompt_variant": "pattern:math_solver|medium|short", "question_variant": "original|plain_math",
 "attempts": [{"prompt_variant": "pattern:math_solver", "question_variant": "original", "error": "content_filter"}],
 "answer": "...", "reference_answer": "...",
 "model": "claude-sonnet-5-5", "response": "[老师]：……",
 "usage": {"prompt_tokens": 1024, "completion_tokens": 830, "total_tokens": 1854},
 "finish_reason": "stop", "elapsed_s": 7.9, "error": null,
 "created_at": "2026-10-28T..."}
```

- 断点续跑：默认跳过输出文件里已有的 `id`；`--force` 重跑覆盖。
- 失败记录会带 `error`，不会中断整批；加 `--retry-errors` 会先删掉失败行再补跑。
- 内容过滤：relay 可能对某个模式 prompt 或题目 LaTeX 误判 `content_filter`；脚本会自动
  按 pattern → secondary → medium → short、original → plain_math 逐级回退，并记录最终
  使用的 `prompt_variant` / `question_variant` 以及失败过的 `attempts`。
- 每条响应是一个 JSONL 记录，另有一个同名 `.summary.json` 记录模型/配置/用量/耗时、
  路由结果（`dataset_profile` / `pattern` / `routing_reason`）以及完整 prompt 链。

## 建议

- 先用 `--limit 5` 验证 relay、prompt 和输出格式，再跑全量。
- MATH-500 是竞赛题，和 RECAST 的 IMO 级数据分布更接近；GSM8K 更基础，适合做"不退化"对照。
- 后续可以把 `response` 作为教师回答喂给 `eval/data/clean_with_llm.py` 与 `recast_decode.py`，
  做"Claude 回答 → 还原 CoT → 训练学生"的闭环。

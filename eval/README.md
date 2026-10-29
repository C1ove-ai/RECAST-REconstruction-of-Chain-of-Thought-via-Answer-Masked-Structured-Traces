# RECAST-Eval：用还原的 CoT 训练学生模型，并与教师 / base 对比

这一层回答三个问题：

1. 还原出的 CoT 作为训练数据，能不能让小模型在数学推理上变强？
2. 学生和教师（闭源模型）差多少？和训练前的自己（base）比提升了多少？
3. 还原 CoT 与"教师原始回答"作为监督信号，哪个更有效？（消融）

> 初始仓库不附带 decoded 输出；先用 `recast_pipeline.py` 或 `recast_decode.py --dataset` 生成数据。
> 管线本身按 N 条设计：先准备足够多且不重复的问题并完成解码，再执行数据管线即可。

---

## 目录结构

```
eval/
├── README.md                  # 本文件
├── requirements.txt
├── system_prompt_student.txt  # 学生模型 system prompt（数据构建与评测共用）
├── data/
│   ├── text_clean.py          # 规则清洗/归一化/正则答案抽取
│   ├── rec_io.py              # 读取 decoded runs，生成稳定 record_key
│   ├── clean_with_llm.py      # 专用清洗模型（DeepSeek）语义清洗 -> data/cleaned_cots.json
│   ├── extract_answers.py     # 从教师回答抽取最终答案 -> data/answers.json
│   ├── build_dataset.py       # decoded runs -> SFT 数据 + 评测集
│   ├── validate_dataset.py    # 上传服务器前检查数据
│   ├── download_benchmarks.py # GSM8K / MATH-500 / AMC23 / AIME24 -> JSONL
│   ├── answers.json           # 生成的答案 sidecar
│   ├── cleaned_cots.json      # LLM 清洗后的 CoT 覆盖层（原始 decoded_cot 不动）
│   ├── processed/             # 生成的 SFT/评测数据
│   └── benchmarks/            # 生成的公开评测集
├── train/
│   ├── train_sft.py           # SFT / LoRA / QLoRA（transformers.Trainer + PEFT）
│   └── sample_and_filter.py   # 阶段2 RFT：自采样、按答案正确性过滤
├── harness/
│   ├── math_verify.py         # 答案抽取 + 等价判断（分数/LaTeX/集合/元组/sympy）
│   ├── metrics.py             # pass@1 / avg@k / maj@k / invalid / bootstrap CI
│   ├── run_eval.py            # 本地模型/教师 API 批量生成 + 打分
│   ├── compare_results.py     # 多份结果汇总成对比表
│   ├── faithfulness.py        # early-answering 忠实度探针
│   └── api_client.py          # OpenAI 兼容 API 客户端
├── docs/
│   ├── metrics.md             # 论文/工具里的评测指标 + 本项目怎么测
│   ├── training_methods.md    # SFT vs LoRA/QLoRA vs DPO/RFT 选型
│   └── data_pipeline.md       # 数据 schema、清洗规则、target 变体
├── scripts/                   # Linux 一键脚本
├── runs/                      # 训练产物（adapter / merged）
└── results/                   # 评测结果（summary + generations + 表格）
```

---

## 快速开始（服务器，Linux）

```bash
cd eval
pip install -r requirements.txt

# 1) 数据：清洗 CoT -> 抽答案 -> 构建 SFT -> 校验
python data/clean_with_llm.py          # 专用清洗模型，key 在 api_clean.txt
python data/extract_answers.py
python data/build_dataset.py
python data/validate_dataset.py

# 2) 公开评测集（需要联网；也可以用自己准备好的 JSONL）
python data/download_benchmarks.py --bench gsm8k math500 amc23

# 3) 训练（先 QLoRA；显存充足再考虑全参）
python train/train_sft.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --method qlora \
    --data-dir data/processed \
    --output-dir runs/sft_qwen1.5b

# 4) 评测：base / 训练后学生 / 教师
python harness/run_eval.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --data data/benchmarks/gsm8k.jsonl --name base_qwen_gsm8k
python harness/run_eval.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter runs/sft_qwen1.5b/final_adapter \
    --data data/benchmarks/gsm8k.jsonl --name sft_qwen_gsm8k
python harness/run_eval.py --provider teacher --model deepseek-chat \
    --data data/benchmarks/gsm8k.jsonl --name teacher_gsm8k --limit 200

# 5) 汇总对比
python harness/compare_results.py results/*.json --baseline base --out results/summary.md
```

更接近论文协议的设置：每个问题采样 8 次（`--k 8 --temperature 0.7`），同时报告 pass@1、avg@8、maj@8；
AIME 一套只有 30 题，务必带 bootstrap 置信区间（`metrics.py` 已内置）。

---

## 数据设计：为什么不是简单拼 [问题 + 还原 CoT + 教师原回答]

SFT 的输入是 system+user+assistant 三段式，训练损失只算 assistant 部分。主实验推荐：

| target | assistant 内容 | 用途 |
|---|---|---|
| `cot_answer`（默认） | 清洗后的还原 CoT + `最终答案：<抽取的答案>` | 主实验：验证还原 CoT 的训练价值 |
| `source_only` | 教师原始回答（当前是师生对话体） | 消融：直接蒸馏教师回答 vs 蒸馏还原 CoT |
| `cot_only` | 只有还原 CoT | 消融：是否显式教最终答案格式 |
| `cot_plus_source` | 还原 CoT + 教师完整回答 | 你原本的设想；**不推荐作为主实验**，见下 |

`cot_plus_source` 的问题：教师回答本身是对话体，会教模型输出"老师/学生"的呈现形式；内容与 CoT 重复、序列更长；而且把"教师答案的信息量"和"还原 CoT 的信息量"混在一起，无法回答"还原 CoT 到底有没有用"。
如果确实想对比，它应该是一个单独的实验组，而不是默认格式。

数据清洗分两层：

**第一层：专用清洗模型（`clean_with_llm.py`，DeepSeek）**。它把还原 CoT 里的课堂/对话壳子和"如何还原/如何组织回答"的元叙述整段删掉（包括把角色 A/B 之类的换说法一起清掉），只保留对题目本身的推理；结果写入覆盖层 `data/cleaned_cots.json`，原始 `decoded_cot` 不动。`build_dataset.py` 默认优先用 `status=="ok"` 的清洗结果，失败时回退原始 CoT。
- key：`api_clean.txt`（或 `DEEPSEEK_CLEAN_API_KEY`）；`--workers` 并发、`--force` 重洗、`--only-leaky` 只洗规则命中的；
- 成本：每条约 3-4k tokens（当前 5 条约 17k tokens，2 workers 约 15 秒）；
- 想对比"不洗"的效果：`python data/build_dataset.py --no-cleaned`。

**第二层：规则安全网（`build_dataset.py` + `text_clean.py`）**，防止清洗模型漏删或编造：

- 去掉解码器偶尔加的 `# 思维链还原` 标题、归一化空行；
- 丢弃清洗后仍含对话/元叙述泄漏的样本（老师/学生/对话/模板/还原/解码/角色/呈现…），可用 `--keep-leak` 关闭；
- 丢弃过短（<200 字符）或超长（>40000 字符）的 CoT；
- 按归一化问题去重（同一个问题多次解码只保留最新一次 run 的样本，可用 `--decoded` 指定 run）；
- 按 seed 划分 train/val/test（≥5 条起有 val/test；小数据时全部进 train 并给出警告）；
- 从教师回答抽取最终答案（`extract_answers.py`，正则 + LLM 兜底），写入 `data/answers.json`。

样本 `meta.cot_source` 会标明这条用的是 `llm_clean` 还是 `raw`；`dataset_stats.json` 里有 `cleaned_available / used_cleaned / cot_sources` 统计。

详细字段和规则见 [docs/data_pipeline.md](docs/data_pipeline.md)。

---

## 训练选型（结论）

完整论证见 [docs/training_methods.md](docs/training_methods.md)，结论：

1. **第一优先：QLoRA SFT**（4-bit 基座 + LoRA，`r=16, alpha=32, lr=2e-4, 2-3 epochs`）。1.5B 在 24GB 上很轻松，8GB 也能跑（seq 1024、grad checkpointing）。
2. **数据量足够（≥1-2k）且显存 ≥40GB**：全参 SFT 效果通常更好。
3. **阶段 2 可选 RFT**：用训练好的学生自采样，只保留答案正确的轨迹继续 SFT（`train/sample_and_filter.py`），比 DPO 更适合数学题，因为答案可自动验证。
4. **不推荐先上 DPO/GRPO**：DPO 需要 chosen/rejected 偏好对，本项目只有正样本；数学推理更适合 rejection sampling。等 SFT 稳定后再考虑。

训练脚本不依赖 TRL，只用 `transformers.Trainer + PEFT + bitsandbytes`，避免版本漂移；兼容 transformers 4.46+ 与 5.x。

---

## 评测矩阵

| 目的 | 数据 | base | SFT(cot) | SFT(source) | 教师 |
|---|---|---|---|---|---|
| 在域保真（学生是否学到还原 CoT 的结论） | `data/processed/eval_test.jsonl` | ✓ | ✓ | ✓ | 参考（按构造为 100%） |
| 分布外泛化 | GSM8K / MATH-500 | ✓ | ✓ | ✓ | ✓（API） |
| 难题能力（可选） | AMC23 / AIME24 | ✓ | ✓ | ✓ | ✓（API） |
| 忠实度 | 学生自己生成的 CoT | early-answering 探针 | ✓ | ✓ | — |

指标与文献依据见 [docs/metrics.md](docs/metrics.md)：pass@1、avg@k、maj@k、invalid rate、bootstrap CI、early-answering 忠实度、LLM-as-judge（可选）。

注意：在域测试集的参考答案来自教师回答，所以教师的"在域分数"按构造是 100%，这一列只用于看学生离教师还有多远，不是能力对比。

---

## 资源估算（粗略，按 seq 2048、batch=1、grad accum=8）

| 方法 | 模型 | 显存 | 说明 |
|---|---|---|---|
| QLoRA 4bit | 1.5B | ~8-12 GB | 24GB 推荐；8GB 需降 seq_len |
| QLoRA 4bit | 3B | ~14-18 GB | 24GB 可跑 |
| 全参 SFT | 1.5B | ~20-28 GB | 40GB 更稳 |
| 全参 SFT | 3B | ~36-48 GB | 48GB 起步 |

数据量建议：100 条→只能验证管线；500-1k 条→能看到趋势；≥2k 条→结论较稳。LIMO 说明少量高质量数据也能工作，但那是强基座 + 精选数据，小基座通常需要更多样本。

---

## 关键提醒

- 先准备足够多且不重复的问题并完成解码，否则训练和评测都没有统计意义。
- 公开评测集可能变化/污染：训练前检查训练问题与测试集是否重复（`validate_dataset.py` 已做问题去重提示）。
- 教师基线请用你真正的"闭源教师模型"API；DeepSeek 只是解码器，若没有教师 API，用 `--provider teacher --model deepseek-chat` 只能当代理基线。
- 答案抽取有误差：数学答案建议先人工抽查 20-50 条 `answers.json`，必要时重跑 `extract_answers.py --force`。
- `docs/metrics.md` 里的 DFI 指标需要 **gold-CoT 对照**（人工解法），当前数据集没有，暂不可算；先用 `source_only` 对照 + base/学生/教师三方对比。
- 专用清洗模型的 key 在 `api_clean.txt`；`build_dataset.py --no-cleaned` 可关掉 LLM 清洗做对照。
- 训练脚本需要在带 CUDA 的机器上首次跑通；建议先用 `--method qlora` + 100 条数据跑 20 step。
- 训练脚本兼容 transformers 4.46+ 与 5.x；GPU 训练路径未在 CI 中执行，请先在目标机器上做 smoke test。

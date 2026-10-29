# 数据管线说明

## 1. 数据来源

```
output/decoded_*.json      # 解码器输出（一次运行一个文件）
  records[]:
    id, question, source_answer, decoded_cot,
    system_prompt, dialogue_leak_terms, usage, ...
```

`build_dataset.py` 会把同一问题在多个 run 中的记录按问题去重（保留最新 run），
因此可以放心地多次重跑解码器。

## 2. 数据流

```
decoded_*.json
   │
   ├─ clean_with_llm.py   专用清洗模型（DeepSeek）语义清洗 → data/cleaned_cots.json
   │                        删课堂/对话痕迹、元叙述、呈现形式；只做删除/改写，不改数学内容
   ├─ extract_answers.py  从教师回答抽最终答案 → data/answers.json
   │
   ▼
build_dataset.py
   │  优先用 cleaned_cots.json 里 status=="ok" 的 CoT，否则回退原始 decoded_cot
   │  再做规则安全网（泄漏词/长度/去重/切分）
   ▼
data/processed/{sft_*, eval_*, dataset_stats.json}
   │
   ▼
validate_dataset.py
```

## 3. 中间产物

| 文件                                          | 由谁生成                 | 内容                                  |
| ------------------------------------------- | -------------------- | ----------------------------------- |
| `data/cleaned_cots.json`                    | `clean_with_llm.py`  | 每条记录的清洗后 CoT（overlay，键为 record_key） |
| `data/answers.json`                         | `extract_answers.py` | 每条记录从教师回答抽取的最终答案                    |
| `data/processed/sft_{train,val,test}.jsonl` | `build_dataset.py`   | 训练用 chat 样本                         |
| `data/processed/eval_{val,test}.jsonl`      | `build_dataset.py`   | 评测用 question+answer（不含 CoT，防评测泄漏）   |
| `data/processed/dataset_stats.json`         | `build_dataset.py`   | 丢弃原因、split 大小、token 分布、cleaned 使用情况 |

## 4. LLM 清洗（`data/clean_with_llm.py`）

### 4.1 为什么需要

规则过滤只能**丢弃**带对话/元叙述的还原 CoT；清洗模型可以**修复**它：
删掉课堂壳子和"如何还原/如何组织回答"的元叙述，保留对题目本身的推理。
这样原本会被丢掉的样本可以重新进入训练集。

### 4.2 输入 / 输出契约

- 输入：`question` + 原始 `decoded_cot`（经基础格式化）；
- 输出严格 JSON：

```json
{
  "cleaned_cot": "清洗后的完整思维链，保留阶段小标题",
  "removed": ["dialogue", "shell", "task_meta", "answer_line"],
  "notes": "一句话说明改了什么"
}
```

- `build_dataset.py` 只使用 `status == "ok"` 且 `cleaned_cot` 非空的条目；
  其余情况回退原始 CoT，再由规则过滤决定去留。

### 4.3 清洗规则（写进 prompt）

- 删除/改写老师、学生、同学、课堂、对话等痕迹，只保留"我"的思考；
- 凡是讨论"这份回答如何呈现、如何被还原/解码/模拟、任务/要求/模板/输入"的整句整段，**直接删除**，不允许换词保留；
- 不得出现"角色 / 分角色 / 呈现 / 模拟 / 还原 / 解码"等词；
- 保留阶段小标题与顺序；某阶段删空可保留标题或删掉，但不用空话填充；
- 删掉显式"最终答案：…"/"答案是…"行（最终答案由 `build_dataset.py` 统一追加）；
- 不修正数学错误、不补新内容；
- 只输出 JSON。

### 4.4 状态与重试

每次调用最多 `--attempts 3` 次（默认 3）；以下情况会用反馈重试，反馈里会带上命中禁用词的句子：

| 状态                     | 触发条件            | build 时的处理           |
| ---------------------- | --------------- | -------------------- |
| `ok`                   | 清洗成功且通过长度/泄漏检查  | 使用清洗后的 CoT           |
| `still_leaky`          | 清洗后仍含禁用词        | 回退原始 CoT（大概率被规则过滤丢掉） |
| `rejected_short`       | 清洗后 < 原文 40%    | 回退原始 CoT             |
| `rejected_long`        | 清洗后 > 原文 250%   | 回退原始 CoT             |
| `empty_raw`            | 原始 CoT 为空       | 跳过                   |
| `api_error` / `failed` | API 或 JSON 解析失败 | 回退原始 CoT             |

长度范围可以用 `--min-length-ratio/--max-length-ratio` 调整；这些是"防误删"的护栏，
不是硬指标。

### 4.5 Key 与成本

- 清洗模型 key 放在 `api_clean.txt`（也可用环境变量 `DEEPSEEK_CLEAN_API_KEY`，或 `--api` 指定）；
- 默认 `deepseek-chat`、temperature=0、JSON 输出；
- 实测每条约 3–4k tokens（输入约 2–3k 字符 + 输出约 2.5k 字符）；小样本（如 5 条）用 2 workers 约十几秒；
- 支持 `--workers N` 并发、`--sleep` 限速、`--force` 重洗、`--only-leaky` 只洗规则命中的样本；
- 结果幂等缓存：已 `ok` 的记录默认跳过。

```bash
python data/clean_with_llm.py --workers 4          # 清洗新记录
python data/clean_with_llm.py --only-leaky         # 只修规则命中的
python data/clean_with_llm.py --force              # 全部重新清洗
python data/clean_with_llm.py --force --record-key decoded_20261028-143620:1  # 只重洗一条
```

## 5. 规则安全网（`data/text_clean.py`）

即使经过 LLM 清洗，`build_dataset.py` 仍会做廉价规则检查（防止清洗模型漏掉/编造）：

1. 去 BOM/零宽字符，统一换行，压缩连续空行；删掉开头 `# 思维链还原` 之类标题；
2. 泄漏过滤：
   - 对话词：老师 / 学生 / 同学 / 课堂 / 师生 / 对话
   - 元叙述：用户提供了 / 我被要求 / 思维链还原 / 还原工程师 / 解码模型 /
     闭源模型的完整回答 / 本次任务 / 长度要求 / 角色 / 分角色 / 呈现 / 模拟 / 还原 / 解码
3. 长度过滤：`--min-cot-chars 200`、`--max-cot-chars 40000`；
4. 问题去重（NFKC + 去标点/空白后比较）；
5. 同一问题保留最新 run 的样本。

注意：`还原 / 解码 / 呈现 / 角色` 等词被加进元叙述黑名单，是因为它们几乎只出现在
"描述还原过程/呈现形式"的句子里；如果后续数据里出现合法用法，按需从 `META_TERMS` 移除。

若想保留泄漏样本（不建议），加 `--keep-leak`；若想绕过 LLM 清洗做对照，
加 `--no-cleaned`。

## 6. SFT 样本 schema

```json
{
  "id": "decoded_20261028-144909:1",
  "source": "decoded_20261028-144909",
  "messages": [
    {"role": "system", "content": "你是一个严谨的数学推理助手……"},
    {"role": "user", "content": "设 a,b 为正整数……"},
    {"role": "assistant", "content": "## 任务识别……\n\n最终答案：……"}
  ],
  "meta": {
    "record_key": "decoded_20261028-144909:1",
    "question": "...",
    "answer": "...",
    "answer_type": "proof|numeric|text",
    "answer_method": "regex:boxed|api|...",
    "cot_source": "llm_clean|raw",
    "cot_chars": 2532,
    "cot_tokens_est": 1608,
    "assistant_chars": 2561,
    "assistant_tokens_est": 1626,
    "leak_terms": []
  }
}
```

训练时 `train_sft.py` 用 tokenizer 的 chat template 生成 token 序列，
并**只对 assistant 部分计算 loss**（system/user token 的 label 置 -100）。

## 7. target 变体

| target            | assistant      | 用途     | 说明                    |
| ----------------- | -------------- | ------ | --------------------- |
| `cot_answer`      | CoT + `最终答案：X` | 主实验    | 推荐：显式教会模型输出可抽取的答案     |
| `cot_only`        | CoT            | 消融     | 模型可能不会单独输出答案，评测抽取会变差  |
| `source_only`     | 教师原始回答         | 消融     | 直接蒸馏教师格式；含对话体，会学到呈现风格 |
| `cot_plus_source` | CoT + 教师原始回答   | 你原本的设想 | 不推荐做主实验：重复、超长、混淆变量    |

## 8. 最终答案抽取（`extract_answers.py`）

教师回答目前是师生对话体，答案往往散在最后。三种模式：

- `regex`：`\boxed{}`、`答案：`、`最终答案`、结论句、最后一行；
- `api`：让 DeepSeek 用严格提示抽取（temperature=0，每条约 1 次小请求）；
- `hybrid`（默认）：正则置信度 ≥0.5 直接用，否则调 API。

输出 `data/answers.json`；默认跳过已有记录，新增数据不需要 `--force`。

## 9. 划分

- 样本数 <5：全部 train，无 val/test（会警告）；
- 5-9：val=1、test=1；
- ≥10：按 `--val-ratio 0.1 --test-ratio 0.1`，且保证训练集不少于一半；
- `eval_test.jsonl` 不含 CoT/教师回答，只含 question + 参考答案，避免评测泄漏。

## 10. 扩充数据

```bash
# 1. 在 decode_dataset.txt 增加题对，重跑解码
cd recast-cot-reverse && python recast_decode.py

# 2. 回到 eval，重建数据
cd eval
python data/clean_with_llm.py              # 只处理未清洗/失败的新记录
python data/extract_answers.py             # 只处理新记录（已有的会跳过）
python data/build_dataset.py
python data/validate_dataset.py
```

## 11. 质量检查清单

- [ ] `cleaned_cots.json` 的 status 分布：`ok` 比例是否合理，`still_leaky/rejected_*` 是否集中；
- [ ] 抽 10-20 条对比 `decoded_cot` 与 `cleaned_cot`，确认只删了壳子、没删有效推理；
- [ ] `answers.json` 人工抽检 20-50 条，证明题答案是否准确；
- [ ] `dataset_stats.json` 的 `dropped`、`cot_sources` 是否符合预期（llm_clean 应占多数）；
- [ ] `validate_dataset.py` 的重复问题数是否为 0（跨 split 泄漏）；
- [ ] 训练集/测试集与公开 benchmark 之间没有重复题目（尤其 GSM8K）；
- [ ] `sft_train.jsonl` 的 assistant 是否都以 `最终答案：` 结尾（cot_answer 模式）。

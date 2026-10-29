# RECAST 学生模型训练方法：SFT / QLoRA / RFT

> **目标**：用 RECAST 还原出的 CoT 训练 1.5B–3B 学生模型，验证"还原 CoT"本身是否有训练价值。
> **硬件**：显存有限时先做 smoke test；正式训练建议按 24GB / 48GB 两档讨论。
> **框架**：纯 `transformers` + `peft` + `bitsandbytes`，不用 TRL。**读者假设**：没训练过模型，但能跑脚本、看懂 JSONL。
> **重要**：文中显存、吞吐、时间均为**量级估算**，不是实测；会因框架版本、序列长度、gradient checkpointing 有 30%–2x 偏差，请再留 20% 余量。
## 0. 结论先行
### 0.1 推荐路线

| 阶段 | 做法 | 关键配置 | 何时进入 |
|---|---|---|---|
| 阶段 1（必做） | **QLoRA SFT**：4bit 基座 + LoRA，目标 = 还原 CoT + `最终答案：` | r=16, alpha=32, lr=2e-4, 2–3 epochs, max_seq=2048 | 现在 |
| 阶段 2（可选） | **RFT / 拒绝采样**：采样 → 验证器筛正确 → 再 SFT 一轮 | 每题 4–8 条，留正确且合规的 1 条，lr 减半，1 epoch | 阶段 1 稳定提升、答案可自动验证 |
| 阶段 3（暂缓） | DPO / GRPO | 见 §2 | 有偏好数据或明确要冲上限 |
**数据 ≥3k–5k 且单卡显存 ≥40GB 时，阶段 1 改全参 SFT**；24GB 优先 QLoRA，不要硬上全参。
### 0.2 先别做

- 不要一上来 DPO/GRPO：只有正样本、没有偏好对；数学题更适合"采样 + 验证 + 再 SFT"。
- 不要在 4bit 权重上 merge LoRA，也不要用 4bit 模型做最终评测（慢且不准）。
- 不要只训最终答案（answer-only）：推理时没有 CoT 可条件，等于白训。
- 不要用 test 挑 checkpoint，也不要在 8GB 笔记本跑正式训练；先把评测跑通，没有稳定评测，loss 曲线说明不了任何事。
## 1. 全参 SFT vs LoRA vs QLoRA
### 1.1 原理差异

| 维度 | 全参 SFT | LoRA | QLoRA |
|---|---|---|---|
| 更新什么 | 全部权重 | 冻结基座，只训低秩矩阵 `ΔW = BA` | 同 LoRA，但基座量化为 4bit |
| 基座精度 | bf16 | bf16 | 4bit NF4 + double quant，计算时反量化到 bf16 |
| 可训练参数（1.5B/3B） | 100% | 约 1%（r=16 全 linear 约 20M/32M） | 同 LoRA |
| 优化器状态 | 每参数 8 字节（fp32 m+v） | 仅 adapter，几乎可忽略 | 同 LoRA，可用 paged 8bit |
| 显存大头 | 权重 + 梯度 + 优化器 | 基座 + 激活 | 量化基座 + 激活 |
| 效果（同数据） | 最强 | 接近全参，小数据常持平 | 略低，通常 0–2 分 |
| 遗忘风险 | 最高 | 较低 | 最低（基座冻结） |
一句话：QLoRA = LoRA + 4bit 基座，省显存但多一次反量化；LoRA 是"只学补丁"，全参是"整模型重写"。LoRA/QLoRA 训完都要 merge 才能当普通模型部署。
### 1.2 显存估算（bf16 单卡，估算）

全参 ≈ 权重 2B + 梯度 2B + AdamW 状态 8B = **12 字节/参数**；若保留 fp32 master 再加 4 字节。LoRA/QLoRA 只需基座 + 激活。

| 配置 | 权重 | 梯度 | 优化器 | 激活（bs=1, seq=2048, ckpt） | 合计（估） |
|---|---|---|---|---|---|
| 1.5B 全参 | 3.1 GB | 3.1 GB | 12.3 GB | 1–2 GB | **19–21 GB**（+master 25） |
| 3B 全参 | 6.2 GB | 6.2 GB | 24.6 GB | 2–3 GB | **38–40 GB**（+master 50） |
| 1.5B LoRA | 3.1 GB | ~0.02 GB | ~0.05 GB | 1.5–3 GB | **5–7 GB** |
| 3B LoRA | 6.2 GB | ~0.03 GB | ~0.1 GB | 2–4 GB | **9–11 GB** |
| 1.5B QLoRA | 1.1–1.3 GB | ~0.02 GB | ~0.05 GB | 1.5–3 GB | **4–5.5 GB** |
| 3B QLoRA | 2.2–2.6 GB | ~0.03 GB | ~0.1 GB | 2–4 GB | **6–8 GB** |
另加 0.6–1.2 GB CUDA context 与碎片；激活随 batch 线性涨。降显存三板斧：gradient checkpointing（省 50–70% 激活，慢 20–30%）、8bit AdamW（优化器除以 4）、缩短 max_seq（最有效）。QLoRA 省的是显存不是时间，价值在于让小卡跑更大的模型或 batch。
### 1.3 1.5B / 3B × 8 / 24 / 48GB 可行性

✅ 舒适；⚠️ 可行但紧（要调 batch/seq）；❌ 不可行。均为估算。

| 配置 | 8GB | 24GB | 48GB |
|---|---|---|---|
| 1.5B 全参 | ❌ | ⚠️ 8bit AdamW + ckpt，bs=1–2，seq≤1024–2048 | ✅ bs=4–8 |
| 1.5B LoRA | ⚠️ bs=1，seq≤1024，易 OOM，仅 smoke | ✅ bs=4–8 | ✅（非必要） |
| 1.5B QLoRA | ✅ bs=1–2，seq=2048 | ✅ bs=4–8 | ✅ |
| 3B 全参 | ❌ | ❌（37GB+ 起步） | ⚠️ 8bit AdamW + ckpt，bs=2；fp32 AdamW 仅 bs=1、seq≤1024 |
| 3B LoRA | ❌（权重 6GB + 激活必爆） | ✅ bs=2–4 | ✅ bs=4–8 |
| 3B QLoRA | ⚠️ bs=1，seq=512–1024，可能 OOM | ✅ bs=2–4 | ✅ |
结论：8GB 只验证代码；24GB 甜点是 1.5B/3B QLoRA；48GB 才考虑 1.5B/3B 全参。
### 1.4 速度（同模型内相对吞吐，估算）

| 方法 | 1.5B | 3B | 说明 |
|---|---|---|---|
| 全参 | 1.0x | 1.0x | 计算最重，但无额外反量化 |
| LoRA | 1.2–1.6x | 1.2–1.5x | 反向只到 adapter |
| QLoRA | 0.7–1.0x | 0.6–0.9x | 4bit 反量化开销，小模型尤其明显 |
### 1.5 QLoRA 的坑

| 坑 | 现象 | 正确处理 |
|---|---|---|
| **merge** | 4bit 上 `merge_and_unload()` 报错或精度异常 | 用 **bf16 重载基座**（CPU）→ 加载 adapter → merge → 存 16bit |
| **4bit 推理慢** | 1.5B/3B 反量化开销占比高，可能比 bf16 还慢 | 评测一律用 merge 后 bf16，4bit 只用于训练 |
| **bf16 计算** | fp16 跑 Qwen2.5 出现 NaN / 溢出 | `compute_dtype=bfloat16`、`quant_type="nf4"`、`use_double_quant=True`；老卡不支持 bf16 才退 fp16 |
| **gradient ckpt** | 梯度为 None 或显存没降 | 先 `prepare_model_for_kbit_training`，再 `gradient_checkpointing_enable` + `enable_input_require_grads` + `use_cache=False` |
| **packing** | 样本互相"看见"，loss 异常低、生成串题 | 首轮不开；要开必须逐样本重置 position_ids/attention |
| **保存** | 把量化模型当最终产物，之后加载混乱 | 只存 adapter + tokenizer，记录 base 名与 revision |
另外：`pad_token` 为 None 时设成 `eos_token`；Qwen2.5 ChatML 模板在 system 为空时会插入默认 system，训练与评测必须用**完全一致**的 system。
## 2. 为什么 DPO 不是首选
### 2.1 四种方法的数据前提

| 方法 | 需要什么数据 | 优化信号 | 额外成本 | 适配度 |
|---|---|---|---|---|
| SFT | (prompt, 正样本) | 逐 token 模仿 | 1x | ✅ 首选 |
| DPO | (prompt, chosen, rejected) | 序列级相对偏好 | 参考模型 + 采样/标注 | ❌ 无 pairs |
| RFT / 拒绝采样（STaR 式） | prompt + 可验证答案 + 采样 | 正确/错误（0/1） | 每题采样 N 条 | ✅ 阶段 2 |
| GRPO | prompt + 奖励函数 + 组采样 | 组内相对奖励 | 采样 + RL 基建 | ⏳ 以后 |
DPO（[DPO](https://arxiv.org/abs/2305.18290)）要求每个 prompt 同时有更好和更差的回答。RECAST 每题只有一条还原 CoT，全是正样本；直接套 DPO 会退化成"一半数据做 SFT + 常数 margin"，没有意义。
### 2.2 数学推理更适合 RFT

答案可验证，这是拒绝采样最舒服的场景：**STaR** 自举正确 rationale（[STaR](https://arxiv.org/abs/2203.14465)）；**DeepSeek-R1** 用冷启动 SFT + 拒绝采样 SFT + RL（[DeepSeek-R1](https://arxiv.org/abs/2501.12948)）；**GRPO** 用组内相对优势省掉 value 网络（[DeepSeekMath](https://arxiv.org/abs/2402.03300)）；**LIMO / s1** 说明少量高质量长 CoT + SFT 就能激发推理（[LIMO](https://arxiv.org/abs/2502.03387)、[s1](https://arxiv.org/abs/2501.19393)）。

本项目阶段 2 的最小闭环 = 采样 → `eval/harness/math_verify.py` 判分 → 留正确 → 再 SFT。复用现有 pipeline，不需要偏好标注和 reference model。

**注意答案类型**：`build_dataset.py` 把答案分为 `numeric / proof / text`。只有 numeric（及部分可归一化 text）能自动判对错；proof 类（如"证明商是完全平方数"）没有唯一答案串，需要 LLM 判题或人工抽检。**先只在 numeric 子集做 RFT**。
### 2.3 DPO 的潜在用途与最小方案

| 用途 | chosen | rejected | 数据来源 |
|---|---|---|---|
| 去冗长 | 短而完整的 CoT | 同题超长/重复 CoT | 同题不同长度采样，按长度+正确性配对 |
| 去不忠实 | 推理与答案一致 | 中途改口、答案矛盾 | 规则检测 + 人工抽检 |
| 去对话体残留 | 干净还原 CoT | 带"老师/学生"字样 | 用现有 leak 过滤器反向构造 |
最小方案：SFT 模型对 1000 道**数值题**各采 4 条（T=0.8, top_p=0.95）→ 按正确性配对 800–1500 对 → LoRA DPO（beta=0.1, lr=5e-6, 1 epoch, seq=2048，reference 冻结 SFT 权重，显存约 ×2）→ 同时看 accuracy 与平均长度（DPO 容易靠变短骗偏好）。成本 ≈ 4 倍采样 + 一次 DPO，收益通常小于同算力投入 RFT。顺序：SFT → RFT →（有明确偏好需求）DPO → GRPO。
## 3. 数据角度
### 3.1 格式（与 `eval/data/build_dataset.py` 对齐）

处理后落在 `eval/data/processed/`：`sft_{train,val,test}.jsonl`（训练）、`eval_{val,test}.jsonl`（`{id, source, question, answer}`，给 `run_eval.py`）、`dataset_stats.json`（条数、丢弃原因、token 分位）。单条样本：

```json
{"id": "decoded_20261028-144909:1", "source": "decoded_20261028-144909",
 "messages": [
   {"role": "system", "content": "你是一个严谨的数学推理助手。解答时先写出清晰的推理步骤，最后单独给出最终答案。"},
   {"role": "user", "content": "设 a,b 为正整数且 ..."},
   {"role": "assistant", "content": "## 任务识别\n...\n\n最终答案：完全平方数"}],
 "meta": {"answer": "完全平方数", "answer_type": "proof", "cot_tokens_est": 1089}}
```

| 约定 | 值 | 原因 |
|---|---|---|
| system | 固定一句，训练/评测一致 | 换 system 等于换任务 |
| 答案标记 | 空行 + `最终答案：` | `math_verify.py` 的抽取正则认它 |
| 特殊 token | assistant 末端的 `im_end` 参与 loss | 不学 EOS 就不会停 |
| 编码 | UTF-8 无 BOM，一行一 JSON | BOM 会让首行 `json.loads` 失败 |
`build_dataset.py` 的四种 target 可直接做消融：`cot_answer`（默认，主实验）、`source_only`（教师原答，基线）、`cot_only`（只有 CoT，负面对照，预期崩）、`cot_plus_source`（CoT + 教师全文，不推荐：信息重复且教进对话体风格）。当前版本默认优先使用 `clean_with_llm.py` 的清洗结果（`meta.cot_source=llm_clean`），清洗失败时回退原始 `decoded_cot`；想对比清洗前后用 `--no-cleaned`。
### 3.2 数据量预期

**经验判断，不是实测**；前提是 CoT 质量合格、答案正确。上限受教师水平和学生容量限制。

| 数据量 | 能学到什么 | 相对 base | 主要风险 | 用途 |
|---|---|---|---|---|
| 100 | 格式、`最终答案` 收尾 | 格式大涨，accuracy ±3 | 背题 | pipeline smoke |
| 500 | 推理骨架、分步习惯 | +2～6 | 覆盖窄 | 最小可报告实验 |
| 1000 | 稳定 CoT 风格 + 初步泛化 | +4～10 | 依赖过滤 | 主实验起点（s1/LIMO 量级） |
| 5000 | 更稳、题型更全 | +6～15，边际递减 | 教师噪声放大 | 数据够就上 |
若 5k 比 1k 只涨 1–2 分，瓶颈在质量/多样性，不在数量。
### 3.3 质量过滤（训练前必做）

| 检查 | 规则 | 处理 |
|---|---|---|
| 去重 | 问题 NFKC 归一化去标点做 key；CoT 可加 SimHash | 保留第一条并计数 |
| 泄漏词 | `老师/学生/同学/课堂/师生/对话`；`思维链还原/还原工程师/闭源模型/本次任务/长度要求` | 直接丢（`--keep-leak` 仅作对比） |
| 长度 | CoT 200–40000 字符，token 化后 ≤ max_seq | 太长丢样本，不截断 |
| 答案正确 | `最终答案` 与 `answers.json` 的 gold 归一化比较 | 不一致的丢；单独统计 `missing_answer` |
| 一致性 | CoT 结论不能与最终答案矛盾 | 抽检 + "因此/所以"后接结论的规则 |
| 格式/退化 | 必须有 `最终答案：` 且非空；重复 n-gram、乱码、特殊 token 残留 | 丢弃或回炉重解 |
过滤后跑 `python eval/data/validate_dataset.py --tokenizer Qwen/Qwen2.5-1.5B-Instruct --max-seq-len 2048`，确认无空 assistant、跨 split 重复题、超长样本。
### 3.4 划分

| split | 比例 | 用途 | 注意 |
|---|---|---|---|
| train | 80–90% | 训练 | 固定 seed（默认 42） |
| val | 5–10% | 选 checkpoint、调参 | 可反复看，别过拟合 |
| test | 5–10% | 最终报告 | 训练前冻结，只跑一次 |
要**按题目来源/题型分组**划分：同题变体、同竞赛近似题跨 split 就是泄漏。`build_dataset.py` 已按问题 key 全局去重，近似题仍需人工看。
### 3.5 顺序、epoch 与过拟合

- 顺序：固定 seed 随机打乱。按长度排序能省 padding 但让 batch 同质，**不推荐**；课程学习收益不稳，第一次别加。
- epoch：1k–5k 用 2–3；<500 用 1–2；>10k 用 1–2。小数据多 epoch = 背题。
- 过拟合信号：train loss 持续降、val loss 先降后升、生成逐字复述训练样本；早停看**下游 accuracy**（§7.3），val loss 只作辅助；数据增强优先级低于"多采新题"。
### 3.6 token 预算与序列长度

`总 token ≈ 样本数 × 平均 assistant token × epoch`，按 1200 token/条估：

| 数据量 | 1 epoch | 3 epochs | 24GB 1.5B QLoRA（估） | 8GB GPU（估） |
|---|---|---|---|---|
| 100 | 0.12M | 0.36M | <5 分钟 | 10–20 分钟 |
| 500 | 0.6M | 1.8M | 10–20 分钟 | 40–90 分钟 |
| 1000 | 1.2M | 3.6M | 20–40 分钟 | 1–2.5 小时 |
| 5000 | 6M | 18M | 2–4 小时 | 8–15 小时（不建议） |
Qwen2.5 中文约 1 token / 1.5 字符；1500–2500 汉字 CoT ≈ 1000–1700 token，加 prompt/答案约 1500–2500 token。`max_seq_len=2048` 覆盖多数样本，p90 超了用 3072。训练时间不是瓶颈，**评测和过滤**才是；先按 1k 跑通全流程。

**超长丢样本，不要截断**：截断会切掉 `最终答案`，直接教出"不结尾"的模型。packing 理论提速 15–35%，但必须逐样本隔离 attention，配错就串题；第一轮不开。
### 3.7 answer-only 与 CoT 目标

| 目标 | labels | 学到 | 推理时 | 结论 |
|---|---|---|---|---|
| answer-only | 只对 `最终答案：X` 算 loss，CoT 当输入 | "看着 CoT 写答案" | 没有 CoT 可条件 | 只作对照 |
| CoT-only | 只对 CoT 算 loss | 会推理，不会收尾 | 可能永不输出答案 | 负面对照，预期崩 |
| **CoT + 答案** | assistant 全部 token（含 EOS） | 先推理后作答 | 可直接用 | ✅ 主实验 |
术语歧义：这里"answer-only"指**只在答案段算 loss**，不是"训练数据只有答案"。
## 4. 超参数
### 4.1 QLoRA 推荐表

通用：4bit NF4 + double quant，compute dtype bf16，target modules = 全部 linear（`q,k,v,o,gate,up,down`），dropout 0.05，optim=`paged_adamw_8bit`，cosine，warmup 3–5%，wd 0.01，grad clip 1.0。

| 参数 | 1.5B（24GB） | 3B（24GB） | 1.5B（48GB） | 3B（48GB） |
|---|---|---|---|---|
| per-device batch | 4 | 2 | 8 | 4 |
| grad accum | 4 | 8 | 2 | 4 |
| 有效 batch | 16 | 16 | 16 | 16 |
| lr | 2e-4 | 2e-4（不稳 1.5e-4） | 2e-4 | 2e-4 |
| epochs | 2–3 | 2–3 | 2–3 | 2–3 |
| max_seq_len | 2048 | 2048 | 2048 | 2048 |
| eval / save | 每 20–30 step / 每 epoch + best | 同左 | 同左 | 同左 |
| LoRA r / alpha | 16 / 32 | 16 / 32 | 16 / 32 | 16 / 32 |
有效 batch 16 是 1k–5k 的稳妥值；数据更少可降到 8。
### 4.2 全参推荐表

| 参数 | 1.5B（24GB，8bit AdamW） | 1.5B（48GB） | 3B（48GB，8bit AdamW） |
|---|---|---|---|
| per-device batch | 1–2 | 8 | 2 |
| grad accum | 8–16 | 2 | 8 |
| lr | 2e-5 | 2e-5 | 1e-5～2e-5 |
| epochs | 2–3 | 2–3 | 2–3 |
| max_seq_len | 1024 | 2048 | 2048 |
| gradient ckpt | 开 | 可关 | 开 |
| 备注 | 很紧，优先 QLoRA | 舒适 | 不要 fp32 AdamW + bs>1 |
全参 lr 比 LoRA 低一个数量级：LoRA 只学补丁可以大步走，全参大步走会破坏基座。
### 4.3 8GB 极端设置（仅 smoke test）

| 项 | 1.5B QLoRA | 3B QLoRA |
|---|---|---|
| max_seq_len | 768 | 512 |
| per-device batch | 1 | 1 |
| grad accum | 16 | 32 |
| LoRA r / alpha | 8 / 16 | 8 / 16 |
| lr / epochs | 1e-4 / 1 | 1e-4 / 1 |
| 预期 | 能跑通、loss 缓降 | 很可能 OOM，能跑也只是验证 shape |
目标是确认数据管线、mask、chat template、保存/合并正确，不是看效果。loss 从 ~2.5 降到 ~1.5、能生成带 `最终答案：` 的样本，就可以上服务器。
## 5. 实验矩阵
### 5.1 主实验（同一 test 集、同一解码参数）

| # | 实验 | 训练数据 | 回答的问题 | 成本 |
|---|---|---|---|---|
| 1 | Base 0-shot | 无 | 下限 | 0 |
| 2 | **SFT(还原 CoT)** | `cot_answer` | 还原 CoT 是否可训 | 1x |
| 3 | SFT(教师原始回答) | `source_only` | "还原"是否比原样喂更好 | 1x |
| 4 | SFT(CoT，无答案) | `cot_only` | 答案段是否必要 | 1x |
| 5 | 教师（闭源模型） | 无 | 上限参照 | API 费 |
| 6 | SFT(answer-only) | 只算答案 loss | 排除"只是多训了答案" | 1x |
关键对比 **2 vs 3**：还原 CoT 训出的学生接近甚至超过"直接学教师原答"，才说明还原 pipeline 有独立价值；**2 vs 5** 决定还剩多少差距。
### 5.2 消融

| 变量 | 取值 | 观察 | 优先级 |
|---|---|---|---|
| 数据量 | 100 / 500 / 1000 / 5000 | accuracy 拐点 | 高 |
| epoch | 1 / 2 / 3 / 5 | val accuracy vs train loss | 高 |
| 随机种子 | 2–3 个 | 结论是否稳 | 高（主实验必做） |
| LoRA rank | 8 / 16 / 32 / 64 | 是否需要更大容量 | 中 |
| target modules | 仅 q,v / 全部 linear | 效果 vs 显存 | 中 |
| 含最终答案 | `cot_answer` / `cot_only` | 格式崩溃率 | 中 |
执行顺序：100 条跑通全链路 → 1k 主实验（2 seeds，含 `source_only` 与 base）→ 若 2 不占优先查数据，别换模型 → 放大 3k–5k 或换 3B → 48GB 补 1.5B 全参 → 最后才 RFT/DPO。
## 6. 常见失败模式

| 现象 | 可能原因 | 先查什么 | 处理 |
|---|---|---|---|
| **格式崩溃**：不输出 `最终答案` | 目标缺答案段 / 截断切掉结尾 / EOS 未参与 loss / `cot_only` | 抽样看 target 原文与 `format_rate` | 用 `cot_answer`；丢超长样本；确认 labels 含 `im_end` |
| **复读/截断** | 数据重复段落；lr 太大；`max_new_tokens` 或 max_seq 不够 | 生成是否逐字重复、长度分布 | 去重、降 lr、repetition_penalty=1.05、评测 max_new_tokens 2048–3072 |
| **CoT 泄漏**：推理中途写出答案 | 教师答案混在 CoT 里；拼接位置不当 | 搜 CoT 里的 `最终答案`/`答案` | 训练前删掉 CoT 内答案句，答案只放标记后 |
| **过拟合** | 数据少、epoch 多、r 大 | train/val loss 分叉；生成背题 | 减 epoch、加 dropout、加数据、早停 |
| **灾难性遗忘** | 全参 lr 大、数据分布窄 | 通用指令集分数掉 | 降 lr、混 5–10% 通用数据、优先 LoRA（[Tulu 3](https://arxiv.org/abs/2411.15124)） |
| **评测污染** | 教师见过 benchmark；近似题跨 split | test 题是否在训练集；n-gram 重叠 | 冻结 test、按来源分组、换私有题 |
| **loss 不降/NaN** | fp16 溢出、lr 过大、mask 全 -100 | 打印一个 batch 的 labels | bf16、降 lr、检查 padding mask |
排障顺序：**先看一条真实样本的 token/labels，再看 loss，再看生成**。多数"模型不听话"是数据或 mask 问题。
## 7. 与评测的衔接
### 7.1 保存与 merge

```python
# 训练：只存 adapter，不要存 4bit 基座
trainer.save_model("outputs/sft_cot_r16")
tokenizer.save_pretrained("outputs/sft_cot_r16")   # 另写 base.txt 记录 base 名 + revision
# 评测前：在 bf16 基座上 merge 成 16bit
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
base, adapter = "Qwen/Qwen2.5-1.5B-Instruct", "outputs/sft_cot_r16"
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained("eval/merged/qwen2.5-1.5b-recast-cot", safe_serialization=True)
AutoTokenizer.from_pretrained(adapter).save_pretrained("eval/merged/qwen2.5-1.5b-recast-cot")
```

CPU merge 需约 2 倍模型内存（3B 约 12–14GB RAM），一般机器够。merged 目录是评测和交付的最终模型。本仓库也提供了一键合并：`python train/train_sft.py ... --merge`，会按 transformers 主版本自动选择 `torch_dtype`/`dtype` 并合并成 16bit。
### 7.2 用 run_eval.py 评测

入口是 `eval/harness/run_eval.py`（以下命令都在 `eval/` 下执行）：

```bash
# 本地学生（base 或 adapter）
python harness/run_eval.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter runs/sft-s0/final_adapter \
  --data data/processed/eval_test.jsonl \
  --name sft_eval_test \
  --batch-size 8 --max-new-tokens 2048 --temperature 0

# 教师（OpenAI 兼容 API，默认 DeepSeek；换 --base-url/--model/--api）
python harness/run_eval.py --provider teacher --model deepseek-chat \
  --data data/processed/eval_test.jsonl --name teacher_eval_test --k 1
```

产物是 `results/<name>.json`（汇总指标）和 `results/<name>.generations.jsonl`（逐题 pred/correct/tokens，供配对检验）。所有对比组用**完全一致**的解码参数（greedy、同 max_new_tokens、同 system、同 chat template）。答案抽取失败计入 `invalid_rate`，不要混进答错。test 只用于最终表格。
### 7.3 checkpoint 选择

| 信号 | 成本 | 优点 | 缺点 | 角色 |
|---|---|---|---|---|
| val loss | 低 | 连续、稳 | 与下游 accuracy 可能背离 | 辅助、早停参考 |
| val accuracy | 中 | 直接对应目标 | 方差大、proof 题难自动判 | **主指标** |
| format rate | 低 | 快速发现崩格式 | 只说明格式对 | 门禁（<90% 不选） |
| train loss | 低 | 诊断过拟合 | 不代表泛化 | 只看趋势 |
做法：每 epoch 存一次，在 val 上算 accuracy + format rate 选 top-2，再对比 val loss；确认后 test 只跑一次。**不要按 test 调 checkpoint。**
## 8. 服务器落地清单
### 8.1 依赖与自检

```text
# Python 3.10/3.11；先按服务器 CUDA 选 torch wheel，再装其余
torch>=2.4
transformers>=4.46      # 4.5x 或 5.x 均可；脚本按主版本自动选择 torch_dtype/dtype
peft>=0.12
bitsandbytes>=0.43
accelerate>=1.0
datasets>=2.20
safetensors
sentencepiece
einops
sympy            # math_verify 的可选增强
```

`python -c "import torch,bitsandbytes;print(torch.cuda.is_available(),torch.cuda.is_bf16_supported())"`；再跑 `python -m bitsandbytes` 看 CUDA 是否匹配。低显存/Windows 上别折腾 FlashAttention，`attn_implementation="sdpa"` 足够。
### 8.2 显存监控

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 1
```

训练脚本定期打印 `torch.cuda.max_memory_allocated()/2**30`；用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片；日志同时记 tokens/s 和 step time，吞吐突然减半通常是数据加载或 swap。
### 8.3 常见报错

| 报错 | 原因 | 处理 |
|---|---|---|
| `CUDA Setup failed despite GPU being available` | bitsandbytes 与 torch/CUDA 不匹配 | 按 CUDA 版本重装 bnb，`python -m bitsandbytes` 验证；Linux only |
| `undefined symbol: libcudart...` | 装了 CPU 版/错版本 torch | 重装匹配 wheel，别混用 conda/pip |
| `TemplateError: chat template` | tokenizer 旧或缺 chat_template | 升 transformers，或手写 ChatML |
| `CUDA out of memory` | batch/seq 大、未开 ckpt、碎片、别人占卡 | 降 bs → 降 seq → 开 ckpt → 4bit → `expandable_segments` |
| loss 变 NaN | fp16 溢出 / lr 大 / 空 label | bf16、降 lr、检查 mask 是否全 -100 |
| adapter 输出乱码 | base 与 adapter 不匹配 | 校验 base 名与 revision，merge 用同一 tokenizer |
| 训练极慢 | 未开 ckpt 换页、dataloader 瓶颈、seq 过长 | 开 ckpt、`num_workers=4`、按长度分桶 |
### 8.4 上传前检查

1. `validate_dataset.py` 全绿；`dataset_stats.json` 的 `leak`、`duplicate_question` 数量可解释。
2. 抽 5 条肉眼确认：assistant 以 `最终答案：` 收尾，CoT 无"老师/学生"。
3. 打印 token 长度 p50/p90/max，确认 p90 < max_seq_len。
4. 用 100 条在目标卡跑 20 step：显存峰值 < 容量 85%、loss 在降。
5. 记录 seed、数据 hash、base revision 到 `runs/`。
## 9. 参考资料

- QLoRA：[arXiv:2305.14314](https://arxiv.org/abs/2305.14314)；LoRA：[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- DPO：[arXiv:2305.18290](https://arxiv.org/abs/2305.18290)；STaR：[arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
- DeepSeekMath（GRPO）：[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)；DeepSeek-R1：[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- LIMO：[arXiv:2502.03387](https://arxiv.org/abs/2502.03387)；s1：[arXiv:2501.19393](https://arxiv.org/abs/2501.19393)
- Tulu 3：[arXiv:2411.15124](https://arxiv.org/abs/2411.15124)；Distilling Step-by-Step：[arXiv:2305.02301](https://arxiv.org/abs/2305.02301)
- PEFT：[github.com/huggingface/peft](https://github.com/huggingface/peft)；TRL（仅参考 RFT/DPO 实现，不引入）：[github.com/huggingface/trl](https://github.com/huggingface/trl)
- LLaMA-Factory（QLoRA 配置对照）：[github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

> 注：显存/吞吐/耗时均为估算，尚未实测；论文只用于方法定位，不复述其数字。DeepSeek-R1-Distill-Qwen-1.5B 自带 ` thinking` 模板，与本项目 `最终答案：` 格式如何统一**待确认**。

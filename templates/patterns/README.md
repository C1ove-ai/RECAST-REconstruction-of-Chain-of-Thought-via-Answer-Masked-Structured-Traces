# 外显 CoT 的提问模式（pattern）

这些模板不是“解码器模板”，而是**让目标模型（Claude 等）在被提问时主动外显推理过程**的
老师-学生对话模板，每一个都包含 `{user_query}` 占位符，供 `collect_claude.py` 使用。

原始想法来自 [pattern_extension.txt](../../pattern_extension.txt) 的五种模式。逐条检查后：
**其中 4 个适合特定类型的数据集，1 个（复盘工作坊）只适合开放型/需求型任务；
数学类缺少一个专门模板，因此新增 `math_solver`。**

## 各模式的适用性检查与优化

| 原始模式 | 适合的数据集/任务 | 判断 | 优化点（已落在文件里） |
|---|---|---|---|
| 苏格拉底追问分析 | 逻辑谜题、条件推理、规划、需求不清的开放题 | 部分适合：容易一直追问不给答案 | 强制“追问到足够清楚后必须给出方案、执行、验证、最终答案” |
| 专家会诊拆分 | 物理/化学/生物、跨学科 STEM、系统设计 | 适合：多视角拆解是这类题的核心 | 限制角色数量，要求主持人比较、收敛、给出可执行结论，避免只罗列观点 |
| 结对编程解决 | 代码生成、调试、算法实现 | 适合：driver-navigator + 测试 | 增加单元测试/反例/复现步骤，明确最后给可用方案和验证结果 |
| 红蓝对抗验证 | 证明题、安全/对抗推理、方案验证 | 更适合做**主模式或第二阶段**，单独使用容易只挑错不给解 | 明确红队只负责攻击、裁判必须收敛出修正后的方案/最终答案 |
| 复盘工作坊扩展需求 | 产品/需求分析、项目复盘、开放写作 | 不适合数学/代码题：会引入无关“扩展需求” | 改成“先完整解决，再复盘/扩展”，并保留最终结论 |
| **math_solver（新增）** | 算术、代数、几何、数论、组合、竞赛/证明 | 原始五种都缺少“逐步推导 + 具体验证”的数学主模式 | 老师读题 + 学生推导 + 学生验证；证明题让学生 C 攻击漏洞 |

结论：原始五条不是“每条都通用”，它们各自服务的推理形态不同；直接混用会让数学题变成
“需求讨论”，让代码题变成“角色辩论”。**按数据集选模式**比固定一个长模板更合理。

## 数据集 → 模式映射

| 数据集/任务类型 | 例子 | 主模式 | 备用模式 |
|---|---|---|---|
| 数学（算术/竞赛/证明） | GSM8K、MATH-500、AMC23、AIME24、IMO | `math_solver` | `red_blue` |
| 代码/调试/算法 | HumanEval、MBPP、LiveCodeBench、Codeforces | `pair_programming` | `red_blue` |
| 科学/跨学科 STEM | GPQA、ScienceQA、SciBench | `expert_panel` | `red_blue` |
| 逻辑/条件推理 | BBH、LogiQA、LSAT | `socratic` | `expert_panel` |
| 规划/agent | PlanBench、Tau-bench、AgentBench | `socratic` | `retrospective` |
| 开放写作/产品需求 | 需求分析、方案评审、复盘 | `retrospective` | `socratic` |
| 未知/混合 | 自定义数据 | `socratic` | `math_solver` |

自动选择规则（`template_router.py`）：

1. 数据集名命中 `registry.json` 的 `datasets`（支持 `_test`/`-test` 后缀归一化）→ 直接用该 profile；
2. 否则对前 20 条问题做关键词打分（数学符号、代码、科学、逻辑、规划、开放题）；
3. 仍无信号 → `general`；
4. 可以用 `--profile` 覆盖 profile，用 `--pattern` 覆盖具体模式。

## 文件

```
templates/patterns/
├── registry.json          # profile → pattern、数据集映射、关键词
├── math_solver.txt
├── socratic.txt
├── expert_panel.txt
├── pair_programming.txt
├── red_blue.txt
└── retrospective.txt
```

## 用法

```bash
# 查看某数据集会选哪个模式
python template_router.py --data testset/math500_test.jsonl
python template_router.py --question "Write a Python function to ..." --source humaneval

# 收集时自动选择（默认 --pattern auto）
python collect_claude.py --bench math500 --limit 5

# 强制某个模式 / 关闭自动选择
python collect_claude.py --bench math500 --pattern math_solver
python collect_claude.py --bench math500 --pattern none      # 回到 long→medium→short
python collect_claude.py --bench math500 --profile code     # 手动指定 profile
```

每个收集记录会带 `dataset_profile`、`pattern`、`prompt_variant`、`question_variant`；
summary 里保存完整 prompt 链和路由结果，方便复现。

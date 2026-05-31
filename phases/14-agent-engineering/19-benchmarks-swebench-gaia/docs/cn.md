# 基准测试：SWE-bench、GAIA、AgentBench（Benchmarks: SWE-bench, GAIA, AgentBench）

> 三个基准测试是 2026 年智能体评估的锚点。SWE-bench 测试代码补丁。GAIA 测试通用工具使用。AgentBench 测试多环境推理。了解它们的组成、污染情况，以及它们未能测量的内容。

**类型：** 学习  
**语言：** Python（标准库）  
**前置知识：** Phase 14 · 06（工具使用）  
**预计时间：** 约 60 分钟

## 学习目标

- 说出 SWE-bench 的测试框架（FAIL_TO_PASS）并解释为什么它以单元测试为门控。
- 解释 SWE-bench Verified（OpenAI，500 个任务）的存在原因及其移除了什么。
- 描述 GAIA 的设计：对人类简单，对 AI 困难；三个难度级别。
- 说出 AgentBench 的八个环境及其对开源 LLM 的主要阻碍。
- 总结 SWE-bench+ 的污染发现及其含义。

## 问题所在

排行榜告诉你哪个模型在某个基准测试中胜出。它们不能告诉你：

- 基准测试是否被污染（训练数据中的解决方案、测试集泄露）。
- 基准测试是否测量了你关心的内容（代码 vs 浏览 vs 通用）。
- 评估器是否健壮（AST 匹配、状态检查、人工审查）。

在引用任何数字之前，先了解这三个锚点基准测试及其失败模式。

## 核心概念

### SWE-bench（Jimenez 等人，ICLR 2024 口头报告）

- 来自 12 个流行 Python 仓库的 2,294 个真实 GitHub issue。
- 智能体获得：修复前提交时的代码库 + 自然语言 issue 描述。
- 智能体产出：一个补丁。
- 评估器：应用补丁，运行仓库的测试套件。补丁必须使 FAIL_TO_PASS 测试翻转（之前失败，现在通过），同时不破坏 PASS_TO_PASS 测试。

SWE-agent（Yang 等人，2024）在发布时通过强调智能体-计算机接口（模型能理解的文件编辑命令、搜索语法）达到 12.5%。

### SWE-bench Verified

OpenAI，2024 年 8 月。人工精选的 500 任务子集。移除了模糊的 issue、不可靠的测试，以及修复不明确的任务。"你的智能体是否能提交真实补丁？"的主要基准测试。

### 污染

- 超过 94% 的 SWE-bench issue 早于大多数模型的训练截止日期。
- **SWE-bench+** 发现 32.67% 的成功补丁在 issue 文本中泄露了解决方案（模型在描述中看到了修复），31.08% 由于测试覆盖率弱而可疑。
- Verified 更干净，但并非无污染。

实际含义：在 SWE-bench 上得分 50% 的模型在 SWE-bench+ 上可能得分 35%。如果你声称 SWE-bench 性能，请同时报告两者。

### GAIA（Mialon 等人，2023 年 11 月）

- 466 个问题；300 个保留用于 huggingface.co/gaia-benchmark 的私有排行榜。
- 设计理念："对人类（92%）概念上简单，但对 AI 困难（GPT-4 with plugins：15%）。"
- 测试推理、多模态、网页、工具使用。
- 三个难度级别；第 3 级需要跨模态的长工具链。

GAIA 是你用来衡量"通用能力"时运行的测试。不要与代码特定的基准测试混淆。

### AgentBench（Liu 等人，ICLR 2024）

- 横跨代码（Bash、DB、KG）、游戏（Alfworld、LTP）、网页（WebShop、Mind2Web）和开放式生成的 8 个环境。
- 多轮，每个分割约 4k-13k 轮次。
- 主要发现：长期推理、决策制定和遵循指令是开源 LLM 追赶商业产品的阻碍。

### 这些测试未能测量的内容

- 实际运营成本（token、实际时间）。
- 对抗性条件下的安全行为。
- 在你的领域中的性能（使用你自己的评估，第 30 课）。
- 尾部失败（基准测试取平均值；生产运营者关心最差的 1%）。

### 基准测试在哪里出错

- **单一数字迷恋。** SWE-bench 50% 告诉你的信息远少于 P50/P75/P95 成本 + 步骤分布。
- **污染声明。** 报告 SWE-bench 时不提 Verified 或 SWE-bench+ 是有误导性的。
- **以基准测试为开发目标。** 针对基准测试优化会偏离生产实用性。

## 构建它

`code/main.py` 实现了一个玩具 SWE-bench 风格框架：

- 合成的 bug 修复任务（3 个任务）。
- 一个提议补丁的脚本化"智能体"。
- 一个检查 FAIL_TO_PASS（bug 现已修复）和 PASS_TO_PASS（没有任何东西损坏）的测试运行器。
- 一个基于问题分解深度的 GAIA 风格难度分类器。

运行：

```
python3 code/main.py
```

输出显示每个任务和每个难度级别的解决率，并使评估器规则具体化。

## 使用它

- **SWE-bench Verified** 用于代码智能体。始终报告 Verified 分数。
- **GAIA** 用于通用智能体。使用私有排行榜分割。
- **AgentBench** 用于多环境对比。
- **自定义评估**（第 30 课）用于你产品的实际形态。

## 交付它

`outputs/skill-benchmark-harness.md` 为任何代码库-任务对构建一个带有 FAIL_TO_PASS / PASS_TO_PASS 门控的 SWE-bench 风格框架。

## 练习

1. 将玩具框架移植到真实仓库（选一个你的）。为已知 bug 编写 3 个 FAIL_TO_PASS 测试。
2. 添加步骤计数指标。在你的 3 个任务上，每次解决需要多少智能体步骤？
3. 阅读 SWE-bench+ 论文。实现一个解决方案泄露检查（对 issue 文本与 diff 进行模式匹配）。
4. 从公共分割下载一个 GAIA 问题。追踪 GPT-4 级智能体会做什么。它需要哪些工具？
5. 阅读 AgentBench 的按环境细分。哪个环境与你的产品表面最相似？"最先进"在那里是什么样子？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| SWE-bench | "代码智能体基准测试" | 2,294 个 GitHub issue；补丁必须使 FAIL_TO_PASS 测试翻转 |
| SWE-bench Verified | "干净的 SWE-bench" | 500 个人工精选任务，OpenAI |
| FAIL_TO_PASS | "修复门控" | 之前失败、补丁后必须通过的测试 |
| PASS_TO_PASS | "无回归门控" | 之前通过、现在仍必须通过的测试 |
| GAIA | "通用基准测试" | 466 个人类易/AI 难的多工具问题 |
| AgentBench | "多环境基准测试" | 8 个环境；长期多轮 |
| 污染（Contamination） | "训练集泄露" | 基准测试任务出现在模型训练中 |
| SWE-bench+ | "污染审计" | 在成功的 SWE-bench 补丁中发现 32.67% 的解决方案泄露 |

## 延伸阅读

- [Jimenez 等人，SWE-bench（arXiv:2310.06770）](https://arxiv.org/abs/2310.06770) — 原始基准测试
- [OpenAI，SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 精选子集
- [Mialon 等人，GAIA（arXiv:2311.12983）](https://arxiv.org/abs/2311.12983) — 通用基准测试
- [Liu 等人，AgentBench（arXiv:2308.03688）](https://arxiv.org/abs/2308.03688) — 多环境套件

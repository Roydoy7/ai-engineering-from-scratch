# Self-Refine 与 CRITIC：迭代式输出改进（Self-Refine and CRITIC: Iterative Output Improvement）

> Self-Refine（Madaan 等人，2023）让一个 LLM 扮演三个角色——生成、反馈、精炼——在循环中运行。7 项任务的平均绝对增益：+20 个百分点。CRITIC（Gou 等人，2023）通过将验证步骤路由到外部工具来强化反馈环节。2026 年，这一模式在每个框架中都以"评估器-优化器"（Anthropic）或护栏循环（OpenAI Agents SDK）的形式落地。

**类型：** 构建  
**语言：** Python（标准库）  
**前置知识：** Phase 14 · 01（智能体循环）、Phase 14 · 03（Reflexion）  
**预计时间：** 约 60 分钟

## 学习目标

- 说出 Self-Refine 的三个提示（生成、反馈、精炼），并解释为什么历史记录对精炼提示很重要。
- 解释 CRITIC 的核心洞察：没有外部接地，LLM 在自我验证方面不可靠。
- 用标准库实现一个带历史记录和可选外部验证器的 Self-Refine 循环。
- 将该模式映射到 Anthropic 的"评估器-优化器"工作流和 OpenAI Agents SDK 的输出护栏。

## 问题所在

智能体产生了一个几乎正确的答案。也许一行代码有语法错误，也许摘要太长，也许计划遗漏了边界情况。你想要的是：智能体批评自己的输出，然后修正它。

Self-Refine 表明这在单个模型下就能实现，无需训练数据，无需强化学习。但有一个问题：LLM 在对硬事实进行自我验证时表现很差。CRITIC 给出了解决方案——将验证步骤路由到外部工具（搜索、代码解释器、计算器、测试运行器）。

这两篇论文共同定义了 2026 年迭代改进的默认范式：生成、验证（尽可能通过外部工具）、精炼，直到验证器通过为止。

## 核心概念

### Self-Refine（Madaan 等人，NeurIPS 2023）

一个 LLM，三个角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
当反馈说"没有问题"或预算耗尽时停止。
```

关键细节：`refine` 能看到完整的历史记录——所有先前的输出和批评——因此不会重复错误。论文对此进行了消融：删去历史记录，质量急剧下降。

主要结果：在 7 项任务（数学、代码、缩写、对话）上平均绝对改进 +20 个百分点，包括 GPT-4。无训练，无外部工具，单一模型。

### CRITIC（Gou 等人，arXiv:2305.11738，v4 2024 年 2 月）

Self-Refine 的弱点：反馈步骤是 LLM 对自身进行评分。对于事实性声明，这不可靠（幻觉对产生它的模型来说往往看起来很有说服力）。CRITIC 将 `feedback(task, output)` 替换为 `verify(task, output, tools)`，其中 `tools` 包括：

- 用于事实性声明的搜索引擎。
- 用于代码正确性的代码解释器。
- 用于算术的计算器。
- 领域专用验证器（单元测试、类型检查器、代码检查器）。

验证器生成基于工具结果的结构化批评。精炼器然后以此批评为条件。

主要结果：CRITIC 在事实性任务上优于 Self-Refine，因为批评是有接地的。在没有外部验证器的任务（创意写作、格式化）上，CRITIC 退化为 Self-Refine。

### 停止条件

两种常见形式：

1. **验证器通过。** 外部测试返回成功。有此选项时优先使用（单元测试、类型检查器、护栏断言）。
2. **没有反馈发出。** 模型说"输出没问题"。更便宜但不可靠；与最大迭代次数上限配合使用。

2026 年的默认做法：组合使用。"如果验证器通过 OR 模型说没问题 AND 迭代次数 >= 2 OR 迭代次数 >= 最大迭代次数，则停止。"

### 评估器-优化器（Anthropic，2024）

Anthropic 2024 年 12 月的帖子将此命名为五种工作流模式之一。两个角色：

- 评估器：为输出评分并产生批评。
- 优化器：根据批评修改输出。

循环直到评估器通过。这是 Anthropic 框架中的 Self-Refine/CRITIC。Anthropic 补充的关键工程细节：评估器和优化器的提示应该有实质性差异，这样模型不会只是橡皮图章式地盖章。

### OpenAI Agents SDK 输出护栏

OpenAI Agents SDK 将此模式作为"输出护栏"发布。护栏是在智能体最终输出上运行的验证器。如果护栏触发（引发 `OutputGuardrailTripwireTriggered`），输出被拒绝，智能体可以重试。护栏可以调用工具（CRITIC 风格）或是纯函数（Self-Refine 风格）。

### 2026 年的常见陷阱

- **橡皮图章循环。** 同一模型用相同提示风格进行生成和批评，会收敛到"看起来不错"。使用结构上不同的提示，或用更小更便宜的模型进行批评。
- **过度精炼。** 每次精炼都增加延迟和 token。预算 1-3 次；之后升级到人工审查。
- **在简单任务上用 CRITIC。** 如果没有外部验证器，CRITIC 退化为 Self-Refine；不要为存根验证器付出延迟代价。

## 构建它

`code/main.py` 在一个玩具任务上实现 Self-Refine 和 CRITIC：给定主题生成一个简短的项目列表。验证器检查格式（3 个要点，每个不超过 60 个字符）。CRITIC 添加一个对已知幻觉进行惩罚的外部"事实验证器"。

组件：

- `generate` — 脚本化的生产者。
- `feedback` — LLM 风格的自我批评。
- `verify_external` — CRITIC 风格的接地验证器。
- `refine` — 根据历史记录重写输出。
- 停止条件 — 验证器通过或最多 4 次迭代。

运行：

```
python3 code/main.py
```

比较 Self-Refine 和 CRITIC 的运行。CRITIC 捕获了 Self-Refine 遗漏的一个事实性错误，因为外部验证器具有自我批评所没有的接地。

## 使用它

Anthropic 的评估器-优化器是这种模式的 Claude 友好表述。OpenAI Agents SDK 的输出护栏是 CRITIC 形态的（护栏可以调用工具）。LangGraph 提供了一个看起来像 Self-Refine 的反思节点。Google 的 Gemini 2.5 Computer Use 添加了每步安全评估器，它是 CRITIC 的变体：每个动作在提交前都经过验证。

## 交付它

`outputs/skill-refine-loop.md` 根据任务形态、验证器可用性和迭代预算配置一个评估器-优化器循环。输出生成器、评估器/验证器和优化器的提示，以及停止策略。

## 练习

1. 用 max_iterations=1 运行玩具。CRITIC 还有帮助吗？
2. 将外部验证器替换为噪声较大的验证器（随机 30% 假阳性）。循环会怎么做？这是 2026 年大多数护栏技术栈的现实。
3. 实现"不同模型的生成器-批评者"变体：大模型生成，小模型批评。它比同一模型更好吗？
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。说出三类验证工具并各举一例。
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的验证器角色。SDK 做对了什么，做错了什么？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Self-Refine | "自我修正的 LLM" | 在单一模型中运行生成 -> 反馈 -> 精炼循环，带历史记录。 |
| CRITIC | "工具接地验证" | 将反馈替换为外部验证器（搜索、代码、计算器、测试）。 |
| 评估器-优化器（Evaluator-Optimizer） | "Anthropic 工作流模式" | 两个角色——评估器评分，优化器修改——循环至收敛。 |
| 输出护栏（Output guardrail） | "事后检查" | OpenAI Agents SDK 的验证器，在智能体产生输出后运行。 |
| 验证步骤（Verify step） | "批评阶段" | 承重的决策：接地验证还是自我评估。 |
| 精炼历史（Refine history） | "模型已尝试的内容" | 前置到精炼提示的先前输出 + 批评；删除则质量崩溃。 |
| 橡皮图章循环（Rubber-stamp loop） | "自我认同失败" | 相同提示的批评返回"看起来不错"；用结构上不同的提示修复。 |
| 停止条件（Stop condition） | "收敛测试" | 验证器通过 OR 没有反馈 AND 迭代次数上限；永远不要只用单一条件。 |

## 延伸阅读

- [Madaan 等人，Self-Refine（arXiv:2303.17651）](https://arxiv.org/abs/2303.17651) — 规范论文
- [Gou 等人，CRITIC（arXiv:2305.11738）](https://arxiv.org/abs/2305.11738) — 工具接地验证
- [Anthropic，构建有效智能体](https://www.anthropic.com/research/building-effective-agents) — 评估器-优化器工作流模式
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 作为 CRITIC 形态验证器的输出护栏

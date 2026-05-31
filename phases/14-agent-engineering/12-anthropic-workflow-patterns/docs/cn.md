# Anthropic 的工作流模式：简单优于复杂（Anthropic's Workflow Patterns: Simple Over Complex）

> Schluntz 和 Zhang（Anthropic，2024 年 12 月）将工作流（预定义路径）与智能体（动态工具使用）区分开来。五种工作流模式涵盖了大多数情况。从直接 API 调用开始。只有在步骤无法预测时才添加智能体。

**类型：** 学习 + 构建  
**语言：** Python（标准库）  
**前置知识：** Phase 14 · 01（智能体循环）  
**预计时间：** 约 60 分钟

## 学习目标

- 说出 Anthropic 的五种工作流模式：提示链、路由、并行化、编排者-工作者、评估器-优化器。
- 解释智能体与工作流的区别以及各自的工程成本。
- 识别何时选择工作流而非智能体（以及反之）。
- 在标准库中针对脚本化 LLM 实现全部五种模式。

## 问题所在

团队为本来只需要单个函数调用的问题引入多智能体框架。代价是真实的：框架添加了使提示模糊、隐藏控制流并引入过早复杂性的层次。Schluntz 和 Zhang 2024 年 12 月的帖子是业界最常引用的反驳：从简单开始，只有在复杂性能赚回其成本时才添加。

## 核心概念

### 工作流 vs 智能体

- **工作流。** 通过预定义代码路径编排的 LLM 和工具。工程师拥有图结构。
- **智能体。** LLM 动态指导自己的工具并采取自己的步骤。模型拥有图结构。

两者各有其用。工作流更便宜、更快、更容易调试。智能体解锁开放式问题，但使失败模式更难推理。

### 增强型 LLM

所有五种模式的基础：一个 LLM 加上三种能力——搜索（检索）、工具（动作）、记忆（持久化）。任何 API 调用都可以使用这些。

### 五种模式

1. **提示链（Prompt chaining）。** 调用 1 的输出是调用 2 的输入。在任务有清晰的线性分解时使用。步骤之间可选的程序化门控。

2. **路由（Routing）。** 分类器 LLM 选择调用哪个下游 LLM 或工具。当类型截然不同的输入需要不同处理时使用（一线支持 vs 退款 vs 缺陷 vs 销售）。

3. **并行化（Parallelization）。** 并发运行 N 个 LLM 调用，聚合结果。两种形态：分段（不同块）和投票（相同提示，N 次运行，多数/综合）。

4. **编排者-工作者（Orchestrator-workers）。** 编排者 LLM 动态决定运行哪些工作者（也是 LLM）并综合其输出。类似于智能体循环，但编排者不会无限循环。

5. **评估器-优化器（Evaluator-optimizer）。** 一个 LLM 提出答案，另一个 LLM 评估它。迭代直到评估器通过。这是 Self-Refine（第 05 课）的推广。

### 工作流胜过智能体的场景

- **可预测的任务。** 如果你能枚举步骤，你应该这样做。
- **成本受限的任务。** 工作流有有界的步骤计数；智能体可能螺旋上升。
- **合规受限的任务。** 审计员想要阅读图结构，而非从轨迹中推断它。

### 智能体胜过工作流的场景

- **开放式研究。** 当下一步取决于上一步返回了什么。
- **可变长度任务。** 数分钟到数小时的工作，步骤数未知。
- **新颖领域。** 当你还不知道正确的工作流时——先探索，后编码。

### 上下文工程的配套资料

"AI 智能体的有效上下文工程"（Anthropic 2025）将相邻学科正式化：200k 的窗口是预算，不是容器。包含什么、何时压缩、何时让上下文增长。在 Phase 14 上下文压缩课程中详细介绍。

## 构建它

`code/main.py` 针对 `ScriptedLLM` 实现了全部五种工作流模式：

- `prompt_chain(input, steps)` — 顺序执行。
- `route(input, classifier, handlers)` — 分类 + 分发。
- `parallel_vote(prompt, n, aggregator)` — N 次运行，聚合。
- `orchestrator_workers(task, workers)` — 编排者选择工作者。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — 循环直到通过。

运行：

```
python3 code/main.py
```

每种模式打印其追踪。每种模式约 10-15 行代码；框架的成本以千行计。

## 使用它

- 大多数任务直接使用 API 调用。
- 只有当模式真正需要持久状态（LangGraph）、行为者模型并发（AutoGen v0.4）或角色模板（CrewAI）时才使用框架。
- 当你想要 Claude Code 框架形态而无需重建时，使用 Claude Agent SDK。

## 交付它

`outputs/skill-workflow-picker.md` 为给定的任务描述选择正确的模式，包括决策理由以及在工作流不足时到智能体的重构路径。

## 练习

1. 实现带置信度阈值的路由。低于阈值时 -> 升级到人工处理。对于一线支持用例，阈值应该在哪里？
2. 为 `parallel_vote` 添加超时。当一个调用挂起时会发生什么？如何在缺失投票的情况下聚合？
3. 将 `evaluator_optimizer` 变成老虎机：跨迭代保留 top-2 输出，这样后期的好结果不会被后期的坏结果覆盖。
4. 将提示链与路由结合：路由器选择三条链中的一条。测量 token 成本与单个大提示替代方案的对比。
5. 选择你的一个生产功能。画出工作流图。计算步骤数。智能体在这里真的更好吗？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 工作流（Workflow） | "预定义流程" | 工程师拥有的 LLM 和工具调用图。 |
| 智能体（Agent） | "自主 AI" | 模型拥有的图；动态工具指导。 |
| 增强型 LLM（Augmented LLM） | "带工具的 LLM" | LLM + 搜索 + 工具 + 记忆；原子单元。 |
| 提示链（Prompt chaining） | "顺序调用" | 调用 N 的输出是调用 N+1 的输入。 |
| 路由（Routing） | "分类器分发" | 选择哪条链/模型处理输入。 |
| 并行化（Parallelization） | "扇出" | N 个并发调用；通过分段或投票聚合。 |
| 编排者-工作者（Orchestrator-workers） | "分发器智能体" | 编排者 LLM 动态选择专家 LLM。 |
| 评估器-优化器（Evaluator-optimizer） | "提议者 + 评判者" | 迭代直到评估器通过；Self-Refine 的推广。 |

## 延伸阅读

- [Anthropic，构建有效智能体（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents) — 五种工作流模式
- [Anthropic，AI 智能体的有效上下文工程](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 配套学科
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 有状态图赚回其成本的时机
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 编排者-工作者模式的产品化

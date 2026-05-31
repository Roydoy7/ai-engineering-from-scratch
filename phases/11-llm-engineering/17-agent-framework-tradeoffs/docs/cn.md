# 智能体框架权衡——LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架都在演示同一个 Demo（研究型智能体生成报告），也都藏着同一个 Bug（状态 Schema 与编排层打架）。选一个抽象符合你问题形态的框架；其余的都是你会写两遍的胶水代码。

**类型：** 学习  
**语言：** Python  
**前置知识：** Phase 11 · 09（函数调用）、Phase 11 · 16（LangGraph）  
**预计时间：** 约 45 分钟

## 问题所在

你有一个需要不止一次 LLM 调用的任务。也许是研究工作流（规划、搜索、总结、引用）。也许是代码评审流水线（解析 diff、批评、修补、验证）。也许是一个多轮助理，需要预订机票、写邮件、提交报销单。你选了一个框架。

三天后，你发现这个框架的抽象在漏水。CrewAI 给了你角色，但当"研究员"需要把结构化计划交给"作者"时它跟你打架。AutoGen 给了你智能体之间的对话，但没有一等状态，所以你的检查点是一个对话日志的 pickle。LangGraph 给了你一个状态图，但强迫你在还不知道智能体会做什么之前就命名每一个转移。Agno 给了你一个单智能体抽象，当你尝试扇出到三个并发工作器时它就崩溃了。

解决方案不是"选最好的框架"。而是将框架的核心抽象与你问题的形态匹配起来。本章绘制这张映射图。

## 核心概念

![智能体框架矩阵：核心抽象 vs 问题形态](../assets/framework-matrix.svg)

四个框架主导着 2026 年的格局。它们的核心抽象并不相同。

| 框架 | 核心抽象 | 最适合 | 最不适合 |
|------|----------|--------|----------|
| **LangGraph** | `StateGraph`——类型化状态、节点、条件边、检查点器。 | 具有显式状态和人机协作中断的工作流；需要时间旅行调试的生产智能体。 | 拓扑结构未知的松散角色驱动头脑风暴。 |
| **CrewAI** | `Crew`——角色（目标、背景故事）、任务、流程（顺序或层级）。 | 具有简短线性/层级计划的角色扮演或人设驱动工作流。 | 超出团队轮次历史的任何有状态场景；复杂分支。 |
| **AutoGen** | `ConversableAgent` 配对——两个或更多智能体轮流发言直到退出条件满足。 | 多智能体*对话*（师生、提议者-批评者、演员-审阅者），思考从聊天中涌现。 | 具有已知 DAG 的确定性工作流；需要跨重启持久状态的任何场景。 |
| **Agno** | `Agent`——单个 LLM + 工具 + 记忆，可组合为团队。 | 快速构建单智能体和轻量级团队；强多模态支持和内置存储驱动。 | 具有自定义归约器的深度显式分支图。 |

### "抽象"的实际含义

框架的核心抽象是你在白板上画架构时所画的那个东西。

- **LangGraph** → 你画一张图。节点是步骤，边是转移，每个点的状态对象都是类型化的。心智模型是状态机。
- **CrewAI** → 你画一张组织架构图。每个角色有职责描述，管理者负责路由任务。心智模型是一小队专家。
- **AutoGen** → 你画一个 Slack 私信。两个智能体互相发消息；如果需要主持人，第三个加入。心智模型是聊天。
- **Agno** → 你画一个单独的方框，工具挂在上面。并排放几个方框就组成一个团队。心智模型是"开箱即用的智能体"。

### 状态问题

状态是大多数框架选择在生产中崩溃的地方。

- **LangGraph。** 类型化状态（`TypedDict` 或 Pydantic 模型），每字段归约器，一等检查点器（SQLite/Postgres/Redis）。恢复、中断和时间旅行是免费的。*(参见 Phase 11 · 16。)*
- **CrewAI。** 状态通过 `context` 字段以字符串在任务间流动，或通过 `output_pydantic` 以结构化方式传递。没有开箱即用的持久化团队存储；如果团队必须在重启后存活，你需要自己接入。
- **AutoGen。** 状态是聊天历史和任何用户定义的 `context`。对话记录持久化；任意工作流状态不持久化，除非你自己写适配器。
- **Agno。** 通过 `storage=` 附加到 `Agent` 的内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB）——对话会话和用户记忆自动持久化。不是完整的图检查点器；是会话存储。

### 分支问题

每个非平凡的智能体都会分支。由谁决定分支很重要。

- **LangGraph** — 你决定，通过条件边。路由是一个带命名分支的 Python 函数。分支在编译后的图中是一等概念；检查点器记录走了哪条分支。
- **CrewAI** — 在层级模式下由管理者决定；在顺序模式下你在构建时决定。路由隐含在任务列表中；除了管理者的提示词之外没有一等"if"。
- **AutoGen** — 智能体通过聊天决定。分支从谁下一个发言中涌现。`GroupChatManager` 选择下一个发言者；你可以手写 `speaker_selection_method`，但默认是 LLM 驱动的。
- **Agno** — 智能体通过调用哪个工具来决定。团队有协调者/路由器/协作者模式；超出此范围的分支是开发者的责任。

### 可观测性问题

- **LangGraph** — 通过 LangSmith 或任何 OTel 导出器实现 OpenTelemetry。每次节点转移都是一个 trace span；检查点同时充当可重放的 trace。LangSmith 是官方选项；Langfuse/Phoenix 也有适配器。
- **CrewAI** — 自 2025 年底起支持一等 OpenTelemetry；集成 Langfuse、Phoenix、Opik、AgentOps。
- **AutoGen** — 通过 `autogen-core` 集成 OpenTelemetry；AgentOps 和 Opik 有连接器。Trace 粒度是每智能体消息，而非每节点。
- **Agno** — 内置 `monitoring=True` 标志加 OpenTelemetry 导出器；与 Langfuse 紧密集成用于会话 trace。

### 成本与延迟

四个框架都增加了每次调用的开销（框架逻辑、验证、序列化）。开销从小到大的大致顺序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要取决于框架额外做了多少 LLM 路由。CrewAI 的层级管理者要花 token 决定谁下一步执行；AutoGen 的 `GroupChatManager` 同理。LangGraph 只在你写了 `llm.invoke` 的地方花 token。Agno 的单智能体路径很薄。

当每次运行的成本很重要时，优先使用显式路由（LangGraph 的边、AutoGen 的 `speaker_selection_method`），而不是 LLM 选择的路由。

### 互操作性

- **LangGraph** ↔ LangChain 工具、检索器、LLM。一等 MCP 适配器（工具以 MCP 服务器形式导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具和 MCP 工具均可适配。通过 `allow_delegation=True` 实现 Crew 间委托。
- **AutoGen** → `FunctionTool` 包装任意 Python 可调用对象；MCP 适配器可用。与 AG2 生态系统紧密耦合用于智能体间模式。
- **Agno** → `@tool` 装饰器或 BaseTool 子类；MCP 适配器；工具可在智能体和团队间共享。

## 技能指南

> 你能用一句话解释，为什么某个框架适合某个智能体问题。

构建前检查清单：

1. **画出形态。** 这是一张图（类型化状态、命名转移）？一个角色扮演（专家们传递工作）？一个聊天（智能体对话直到完成）？还是一个带工具的单智能体？
2. **决定由谁分支。** 开发者决定的分支 → LangGraph。管理者智能体决定 → CrewAI 层级模式。聊天涌现 → AutoGen。工具调用决定 → Agno。
3. **检查状态预算。** 你需要从检查点恢复吗？时间旅行？运行中途的人工中断？如果是，LangGraph 是默认选择；Agno 会话覆盖对话范围的状态。
4. **检查成本预算。** LLM 选择的路由每轮额外花费 token。如果智能体每天运行数千次，优先显式路由。
5. **计算框架开销。** 每个框架都是一个额外的依赖项。如果任务就是两次 LLM 调用加一个工具，写 30 行纯 Python；没有框架比没有框架更便宜。

在你能画出图、组织架构图、聊天界面或智能体方框之前，拒绝伸手去拿框架。拒绝选一个会迫使你与它的状态模型搏斗来实现你真正需要的东西的框架。

## 决策矩阵

| 问题形态 | 首选框架 | 理由 |
|----------|----------|------|
| 具有类型化状态、人工审批、长时运行的工作流 DAG | LangGraph | 一等状态、检查点器、中断、时间旅行。 |
| 具有清晰角色的研究/写作流水线 | CrewAI（顺序）或 LangGraph 子图 | 角色-任务在 CrewAI 中表达成本低；分支变复杂时用 LangGraph 升级。 |
| 提议者-批评者或师生对话 | AutoGen | 双智能体聊天是它的原生形态。 |
| 带工具、会话、记忆的单智能体 | Agno | 最简设置，内置存储和记忆。 |
| 带归约器的数千个并行扇出 | LangGraph + `Send` | 唯一拥有一等并行分派 API 的框架。 |
| 快速原型，不绑定框架 | 纯 Python + 服务商 SDK | 没有框架是最快的框架。 |

## 练习

1. **简单。** 用同一个任务——"研究 Anthropic 总部，写一份 200 字简报，附引用来源"——分别用 LangGraph（四个节点：规划、搜索、写作、引用）和 CrewAI（三个角色：研究员、作者、编辑）实现。报告每次运行的 token 成本和代码行数。
2. **中等。** 用 AutoGen（研究员 ↔ 作者聊天，编辑通过 `GroupChat` 加入）和 Agno（单智能体带 `search_tools` 和 `write_tools`，加会话存储）实现同一任务。按（a）每次运行成本、（b）崩溃后恢复能力、（c）在写作步骤前注入人工审批的能力，对四种实现排名。
3. **困难。** 构建一个决策树脚本 `pick_framework.py`，接受简短问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`），返回带一句话理由的推荐结果。用你自己设计的六个案例验证。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 编排（Orchestration） | "智能体如何协调" | 决定哪个节点/角色/智能体下一步运行的层。 |
| 持久状态（Durable state） | "重启后可恢复" | 能在进程死亡后存活、附加到检查点或会话存储的状态。 |
| LLM 选择路由（LLM-selected routing） | "让模型决定" | 规划 LLM 每轮选择下一步；灵活但每次决策都要花 token。 |
| 显式路由（Explicit routing） | "开发者决定" | Python 函数或静态边选择下一步；成本低且可审计。 |
| Crew | "CrewAI 团队" | 角色 + 任务 + 流程（顺序或层级）绑定为单个可运行对象。 |
| GroupChat | "AutoGen 的多智能体聊天" | N 个智能体之间有发言人选择器的托管对话。 |
| Team（Agno） | "多智能体 Agno" | 一组智能体上的路由/协调/协作模式。 |
| StateGraph | "LangGraph 的图" | 类型化状态、节点、条件边、检查点器的抽象。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — StateGraph、检查点器、中断、时间旅行。
- [CrewAI 文档](https://docs.crewai.com/) — Crews、Flows、Agents、Tasks、Processes。
- [AutoGen 文档](https://microsoft.github.io/autogen/) — ConversableAgent、GroupChat、团队、工具。
- [Agno 文档](https://docs.agno.com/) — Agent、Team、Workflow、存储、记忆。
- [Anthropic — 构建高效智能体（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents) — 框架无关的模式库（提示词链、路由、并行化、编排器-工作器、评估器-优化器）。
- [Yao 等，"ReAct: Synergizing Reasoning and Acting"（ICLR 2023）](https://arxiv.org/abs/2210.03629) — 每个框架都在包装的那个循环。
- [Wu 等，"AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation"（2023）](https://arxiv.org/abs/2308.08155) — AutoGen 的设计论文。
- [Park 等，"Generative Agents: Interactive Simulacra of Human Behavior"（UIST 2023）](https://arxiv.org/abs/2304.03442) — CrewAI 风格人设栈构建的角色扮演基础。
- Phase 11 · 16（LangGraph） — 本章与之对比的框架。
- Phase 11 · 19（Reflexion） — 一种能自然映射到 LangGraph 但与 CrewAI 格格不入的模式。
- Phase 11 · 22（生产可观测性） — 如何对你选择的任何框架进行检测。

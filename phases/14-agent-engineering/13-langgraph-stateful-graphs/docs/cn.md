# LangGraph：有状态图与持久执行（LangGraph: Stateful Graphs and Durable Execution）

> LangGraph 是 2026 年低级有状态编排的参考标准。智能体是状态机；节点是函数；边是转换；状态是不可变的，并在每一步之后进行检查点保存。可以从任何失败点精确恢复。

**类型：** 学习 + 构建  
**语言：** Python（标准库）  
**前置知识：** Phase 14 · 01（智能体循环）、Phase 14 · 12（工作流模式）  
**预计时间：** 约 75 分钟

## 学习目标

- 描述 LangGraph 的核心模型：带不可变状态、函数节点、条件边和步后检查点的状态机。
- 说出文档重点介绍的四种能力：持久执行、流式传输、人在循环中、全面的记忆。
- 解释 LangGraph 支持的三种编排拓扑：监督者、对等（群）、层次（嵌套子图）。
- 用标准库实现一个带不可变状态、条件边和检查点/恢复循环的状态图。

## 问题所在

智能体和工作流共享一个问题：当 40 步运行在第 38 步失败时，你希望从第 38 步恢复，而不是重新开始。二等状态模型让操作员不得不围绕假设全新运行的库手动实现重试。

LangGraph 的设计答案：状态是一等类型对象，变异是显式的，检查点在每个节点之后持久化。恢复就是一个 `load_state(session_id)` 调用。

## 核心概念

### 图

图由以下部分定义：

- **状态类型。** 一个类型化字典（或 Pydantic 模型），每个节点都读取和变异它。
- **节点。** 纯函数 `(state) -> state_update`。返回后更新被合并到状态中。
- **边。** 节点之间的条件或直接转换。
- **入口和出口。** `START` 和 `END` 哨兵节点标记边界。

示例：一个包含 `classify`、`refund`、`bug`、`sales`、`done` 节点的智能体——一个作为图的路由工作流。

### 持久执行

每个节点返回后，运行时将状态序列化并写入检查点（SQLite、Postgres、Redis、自定义）。在第 N 步失败时，运行时可以 `resume(session_id)` 并从第 N+1 步开始，带有精确的状态。

LangGraph 文档明确指出了这很重要的生产用户：Klarna、Uber、J.P. Morgan。主张的不是图的形状；而是图的形状加上检查点使恢复变得廉价。

### 流式传输

每个节点可以生成部分输出。图将每节点增量事件流式传输给调用者，使 UI 在图运行时即时更新。

### 人在循环中

在节点之间检查和修改状态。实现：在关键节点之前暂停，向人展示状态，接受修改，恢复。检查点使这变得容易，因为状态已经被序列化。

### 记忆

短期（运行内——状态中的对话历史）和长期（跨运行——通过检查点加上单独的长期存储持久化）。LangGraph 通过工具与外部记忆系统（Mem0、自定义）集成。

### 三种拓扑

1. **监督者（Supervisor）。** 中央路由 LLM 分发给专业子智能体。`langgraph-supervisor` 中的 `create_supervisor()`（尽管 LangChain 团队在 2026 年建议直接通过工具调用来实现，以获得更好的上下文控制）。
2. **群/对等（Swarm / peer-to-peer）。** 智能体通过共享工具接口直接切换。没有中央路由。
3. **层次（Hierarchical）。** 监督者管理子监督者，实现为嵌套子图。

### 这个模式在哪里出错

- **检查点过小。** 只检查点对话轮次会使工具状态和记忆写入无法恢复。完整状态必须序列化。
- **非确定性节点。** 恢复假设节点输入产生相同的状态更新。随机种子、挂钟时间、外部 API 必须被捕获。
- **过度使用条件边。** 每条边都是条件的图是一个无法推理的状态机。优先使用线性链，偶尔分支。

## 构建它

`code/main.py` 实现了一个标准库有状态图：

- `State` — 带 `messages`、`step`、`route`、`output`、`human_approval` 的类型化字典。
- `Node` — 接受状态并返回更新字典的可调用对象。
- `StateGraph` — 节点 + 边 + 条件边 + 运行 + 恢复。
- `SQLiteCheckpointer`（内存中的仿制品）— 在每个节点之后序列化状态；`load(session_id)` 恢复。
- 演示图：classify -> branch(refund / bug / sales) -> 人工审核门 -> send。

运行：

```
python3 code/main.py
```

追踪显示第一次运行在人工审核门失败、持久化，然后恢复产生最终输出。

## 使用它

- **LangGraph** — 参考标准，生产就绪。使用 `create_react_agent`、`create_supervisor` 或构建自己的图。
- **AutoGen v0.4**（第 14 课）— 高并发场景的行为者模型替代方案。
- **Claude Agent SDK**（第 17 课）— 内置会话存储和子智能体的托管框架。
- **自定义** — 当你需要精确控制状态形状或检查点后端时。

## 交付它

`outputs/skill-state-graph.md` 在任何目标运行时生成一个 LangGraph 形态的状态图，内置检查点和恢复。

## 练习

1. 从 `classify` 到 `end` 添加一个条件边，当分类置信度低于阈值时触发。在人工手动设置 `route` 后恢复运行。
2. 将类 SQLite 的仿制品替换为真实的 SQLite 检查点。测量每步序列化开销。
3. 实现并行边：两个节点并发运行，通过自定义归约器合并。不可变状态在这里有什么价值？
4. 阅读 `langgraph-supervisor` 参考文档。将玩具移植到 `create_supervisor`。比较追踪形态。
5. 添加流式传输：每个节点在运行时生成部分状态。在增量到达时打印它们。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 状态图（State graph） | "智能体作为状态机" | 类型化状态 + 节点 + 边 + 归约器。 |
| 检查点（Checkpointer） | "持久化后端" | 在每个节点之后序列化状态；支持恢复。 |
| 归约器（Reducer） | "状态合并器" | 将当前状态与节点更新合并的函数。 |
| 条件边（Conditional edge） | "分支" | 由状态函数选择的边。 |
| 子图（Subgraph） | "嵌套图" | 在另一个图中用作节点的图。 |
| 持久执行（Durable execution） | "从失败中恢复" | 以精确状态从最后成功的节点重启。 |
| 监督者（Supervisor） | "路由 LLM" | 专业子智能体的中央分发器。 |
| 群（Swarm） | "对等智能体" | 智能体通过共享工具切换；没有中央路由。 |

## 延伸阅读

- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 参考文档
- [langgraph-supervisor 参考](https://reference.langchain.com/python/langgraph/supervisor/) — 监督者模式 API
- [AutoGen v0.4，微软研究院](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 行为者模型替代方案
- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) — 会话存储和子智能体

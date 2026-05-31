# LangGraph——智能体的状态机（LangGraph — State Machines for Agents）

> 手写的 ReAct 循环是一个 `while True`。用 LangGraph 写的 ReAct 循环是一张图，你可以对它做检查点、中断、分支和时间旅行。智能体本身没变，变的是它外面的那层框架。

**类型：** 构建  
**语言：** Python  
**前置知识：** Phase 11 · 09（函数调用）、Phase 11 · 14（模型上下文协议）  
**预计时间：** 约 75 分钟

## 问题所在

你上线了一个函数调用智能体。它跑了三轮还正常，然后出了问题：模型调用了一个返回 500 的工具，用户在任务中途改变了主意，或者智能体决定退款却没有人工确认。`while True:` 循环没有任何钩子。你无法暂停它，无法倒回它，也无法分叉出"如果模型选了另一个工具会怎样"的场景。一旦把它推出演示阶段，这个智能体就变成了一个黑盒——要么成功，要么失败。

一旦你看清它，下一步就显而易见了。智能体本来就是一个状态机——系统提示词加上消息历史，加上待处理的工具调用，加上下一步动作。让这个状态机变得显式：为"模型思考"、"工具运行"、"人工审批"设置节点，为它们之间的条件转移设置边。一旦图变得显式，框架就免费获得了四种能力：检查点（在每步之间保存状态）、中断（为人工暂停）、流式传输（流式传输 token 和中间事件）以及时间旅行（回到先前状态并尝试不同分支）。

LangGraph 就是实现这一抽象的库。它不是 LangChain 意义上的智能体框架（"给你一个 AgentExecutor，祝你好运"）。它是一个具备一等状态、一等持久化和一等中断的图运行时。智能体循环是你画出来的，不是手写出来的。

## 核心概念

![LangGraph StateGraph：节点、边与检查点器](../assets/langgraph-stategraph.svg)

一个 `StateGraph` 有三个要素。

1. **状态（State）。** 流经整张图的类型化字典（TypedDict 或 Pydantic 模型）。每个节点接收完整状态并返回部分更新，LangGraph 通过每个字段的*归约器（reducer）*合并更新——列表字段用 `operator.add` 累加，其他字段默认覆盖。
2. **节点（Nodes）。** Python 函数 `state -> partial_state`，每个都是一个离散步骤："调用模型"、"运行工具"、"生成摘要"。
3. **边（Edges）。** 节点之间的转移。静态边指向固定位置。条件边接受一个路由函数 `state -> next_node_name`，使图可以根据模型输出进行分支。

你需要编译图。编译过程绑定拓扑结构、附加检查点器（可选但对生产至关重要），并返回一个可运行对象。你用初始状态和 `thread_id` 来调用它。执行的每一步都会以 `(thread_id, checkpoint_id)` 为键持久化一个检查点。

### 四大超能力

**检查点（Checkpointing）。** 每次节点转移后，新状态会被写入存储（测试用内存，生产用 Postgres/Redis/SQLite）。用相同的 `thread_id` 再次调用图即可恢复。图从暂停处继续。

**中断（Interrupts）。** 将节点标记为 `interrupt_before=["human_review"]`，执行会在该节点运行前停止。状态持久化。你的 API 向用户返回"等待审批"。稍后对同一 `thread_id` 发送带有 `Command(resume=...)` 的请求即可恢复执行。

**流式传输（Streaming）。** `graph.stream(state, mode="updates")` 会在事件发生时实时产出状态增量。`mode="messages"` 在模型节点内部流式传输 LLM token。`mode="values"` 产出完整快照。你选择在 UI 中呈现哪种。

**时间旅行（Time-travel）。** `graph.get_state_history(thread_id)` 返回完整的检查点日志。将任意历史 `checkpoint_id` 传给 `graph.invoke`，就从那个点分叉。非常适合调试（"如果模型选了工具 B 会怎样？"）以及重放生产 trace 的回归测试。

### 归约器才是关键

每个状态字段都有一个归约器。大多数默认值没问题——新值覆盖旧值。但消息列表需要 `operator.add`，这样新消息才会追加而不是替换。并行边通过归约器合并各自的更新。如果两个节点都更新了 `messages`，而你忘记了 `Annotated[list, add_messages]`，第二个节点会静默地赢，你会丢失半个回合的内容。归约器是这个库中唯一细微的地方；搞对它，其余的自然组合起来。

### 用四个节点构建 ReAct 图

一个生产级 ReAct 智能体由四个节点和两条边组成：

1. `agent`——用当前消息历史调用 LLM，返回助理消息（可能包含 tool_calls）。
2. `tools`——执行最后一条助理消息中的所有 tool_calls，将工具结果作为 tool 消息追加进去。
3. 从 `agent` 出发的条件边：若最后一条消息包含 tool_calls，路由到 `tools`，否则路由到 `END`。
4. 从 `tools` 回到 `agent` 的静态边。

就这些。你用大约 40 行代码就获得了完整的 ReAct 循环（思考 → 行动 → 观察 → 思考 → ……），带检查点、中断和流式传输。

### StateGraph 与 Send（扇出）

`Send(node_name, state)` 允许一个节点分派并行子图。例如：智能体决定同时查询三个检索器。每个 `Send` 会生成目标节点的一次并行执行；它们的输出通过状态归约器合并。这是 LangGraph 在不使用线程原语的情况下表达编排器-工作器模式的方式。

### 子图（Subgraphs）

一个编译后的图可以成为另一张图中的节点。外部图看到的是单个节点；内部图有自己的状态和检查点。团队正是用这种方式构建主管-工作器智能体：主管图将用户意图路由到各领域的工作器子图。

## 动手构建

### 第一步：状态与节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是使消息列表累加而非覆盖的归约器。忘记它是 LangGraph 中最常见的 bug。

### 第二步：带线程地运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每次更新都是一个字典 `{node_name: state_delta}`。你的前端可以将这些实时推送到 UI，让用户看到"智能体正在思考……调用 search_web……获得结果……正在回答"。

### 第三步：添加人机协作中断

标记一个节点，使执行在它运行前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # 在每次工具调用前暂停
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] 已被设置。检查提议的工具调用。
# 若批准：
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# 若拒绝：写入一条拒绝消息并恢复
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状态、检查点和线程在中断期间全部持久化。只有在执行期间才有内容在内存中。

### 第四步：时间旅行调试

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# 从历史检查点分叉
target = history[3].config  # 倒退三步
for event in app.stream(None, target, stream_mode="values"):
    pass  # 从那个点向前重放
```

将 `None` 作为输入传入会从给定检查点重放；传入一个值会在恢复前将其作为更新追加到该检查点的状态中。这就是如何在不重跑整个对话的情况下复现一次糟糕的智能体运行。

### 第五步：为生产替换检查点器

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis 和 Postgres 均已内置。`MemorySaver` 仅用于测试。任何需要在重启后持久化的场景都需要真实的存储。

## 技能指南

> 你要把智能体构建为图，而不是 `while True` 循环。

在使用 LangGraph 之前，先做 60 秒的设计：

1. **命名节点。** 每一个离散的决策或有副作用的动作都是一个节点。"智能体思考"、"工具运行"、"审批人确认"、"响应流式输出"。如果你列举不出来，说明这个任务还没有智能体形态。
2. **声明状态。** 最小化的 TypedDict，为每个列表字段设置归约器。不要把所有东西都塞进 `messages`；把任务专属字段（一个工作中的 `plan`、一个 `budget` 计数器、一个 `retrieved_docs` 列表）提升到顶层。
3. **画出边。** 除非下一步依赖模型输出，否则用静态边。每条条件边都需要一个带命名分支的路由函数。
4. **提前选择检查点器。** 测试用 `MemorySaver`，其他场景用 Postgres/Redis/SQLite。没有检查点器就不要上线——没有检查点意味着没有恢复、没有中断、没有时间旅行。
5. **在工具运行前设置中断，而不是之后。** 审批放在进入有副作用节点的边上，这样可以在造成伤害前取消；验证放在模型输出的边上，这样可以低成本地拒绝错误调用。
6. **默认使用流式传输。** UI 用 `mode="updates"`，模型节点内部 token 级流式用 `mode="messages"`，评估时完整快照用 `mode="values"`。

拒绝上线没有检查点器的 LangGraph 智能体。拒绝在副作用*之后*才中断的实现。拒绝 `messages` 字段没有 `add_messages` 归约器的代码。

## 练习

1. **简单。** 使用计算器工具和网络搜索工具实现上面的四节点 ReAct 图。验证 `list(app.get_state_history(config))` 对于两轮对话至少返回四个检查点。
2. **中等。** 添加一个在 `agent` 之前运行的 `planner` 节点，将结构化的 `plan: list[str]` 写入状态。让 `agent` 标记计划步骤为已完成。如果 `plan` 在检查点恢复后丢失（归约器错误），测试应该失败。
3. **困难。** 构建一个主管图，使用 `Send` 在三个子图（`researcher`、`writer`、`reviewer`）之间路由。每个子图有自己的状态和检查点器。在外部图上添加 `interrupt_before=["writer"]`，使人工可以审批研究简报。确认从历史检查点进行时间旅行只会重跑分叉的分支。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| StateGraph | "LangGraph 的图" | 你在编译前向其添加节点和边的构建对象。 |
| 归约器（Reducer） | "字段如何合并" | 当节点为该字段返回更新时应用的函数 `(old, new) -> merged`；默认是覆盖，`add_messages` 是追加。 |
| 线程（Thread） | "对话 ID" | 限定一个会话所有检查点范围的 `thread_id` 字符串。 |
| 检查点（Checkpoint） | "暂停的状态" | 节点转移后完整图状态的持久化快照，以 `(thread_id, checkpoint_id)` 为键。 |
| 中断（Interrupt） | "等待人工" | `interrupt_before` / `interrupt_after` 在节点边界停止执行；用 `Command(resume=...)` 恢复。 |
| 时间旅行（Time-travel） | "从先前步骤分叉" | `graph.invoke(None, config_with_old_checkpoint_id)` 从该检查点向前重放。 |
| Send | "并行子图分派" | 一个节点可以返回的构造器，用于生成目标节点的 N 次并行执行。 |
| 子图（Subgraph） | "作为节点的已编译图" | 在另一张图中用作节点的已编译 StateGraph；保留自己的状态作用域。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — StateGraph、归约器、检查点器和中断的权威参考。
- [LangGraph 概念：状态、归约器、检查点器](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 本章使用的心智模型，直接来自官方。
- [LangGraph 持久化与检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/) — Postgres/SQLite/Redis 存储、检查点命名空间和线程 ID 的详细说明。
- [LangGraph 人机协作](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`、`interrupt_after`、`Command(resume=...)`和编辑状态模式。
- [Yao 等，"ReAct: Synergizing Reasoning and Acting in Language Models"（ICLR 2023）](https://arxiv.org/abs/2210.03629) — 每个 LangGraph 智能体都在实现的模式；阅读它以了解推理链的设计理由。
- [Anthropic — 构建高效智能体（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents) — 哪些图形态（链、路由器、编排器-工作器、评估器-优化器）在什么情况下更优。
- Phase 11 · 09（函数调用） — 每个 LangGraph 智能体节点复用的工具调用原语。
- Phase 11 · 14（模型上下文协议） — 通过 MCP 适配器插入 LangGraph `ToolNode` 的外部工具发现。
- Phase 11 · 17（智能体框架权衡） — 何时选择 LangGraph 而非 CrewAI、AutoGen 或 Agno。

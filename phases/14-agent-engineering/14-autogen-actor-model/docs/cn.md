# AutoGen v0.4：行为者模型与智能体框架（AutoGen v0.4: Actor Model and Agent Framework）

> AutoGen v0.4（微软研究院，2025 年 1 月）围绕行为者模型重新设计了智能体编排。异步消息交换、事件驱动的智能体、每个行为者的故障隔离、天然并发。该框架现处于维护模式，微软智能体框架（2025 年 10 月公开预览）成为其继任者。

**类型：** 学习 + 构建  
**语言：** Python（标准库）  
**前置知识：** Phase 14 · 01（智能体循环）、Phase 14 · 12（工作流模式）  
**预计时间：** 约 75 分钟

## 学习目标

- 描述行为者模型：智能体作为行为者，消息是唯一的 IPC，每个行为者的故障隔离。
- 说出 AutoGen v0.4 的三个 API 层——核心层（Core）、AgentChat、扩展层（Extensions）——及各自的用途。
- 解释为什么将消息传递与处理解耦能实现故障隔离和天然并发。
- 用标准库在 Python 中实现一个行为者运行时，并将双智能体代码审查流程移植到其上。

## 问题所在

大多数智能体框架是同步的：一个智能体生产，一个智能体消费，在调用栈中。失败会崩溃调用栈。并发是后续添加的。分布式需要重写。

AutoGen v0.4 的答案：行为者模型。每个智能体是一个带私有收件箱的行为者。消息是唯一的交互方式。运行时将传递与处理解耦。失败隔离在单个行为者内。并发是天然的。分布式只是不同的传输方式。

## 核心概念

### 行为者

行为者具有：

- 私有状态（从外部永远无法直接触达）。
- 收件箱（消息队列）。
- 处理器：`receive(message) -> effects`，效果可以是"回复"、"发送给其他行为者"、"生成新行为者"、"更新状态"、"停止自己"。

两个行为者不能共享内存。它们只能发送消息。

### AutoGen v0.4 的三个 API 层

1. **核心层（Core）。** 低级行为者框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换，事件驱动。
2. **AgentChat。** 任务驱动的高级 API（替代 v0.2 的 ConversableAgent）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **扩展层（Extensions）。** 集成——OpenAI、Anthropic、Azure、工具、记忆。

### 为什么解耦很重要

在 v0.2 模型中，调用 `agent_a.chat(agent_b)` 会同步阻塞 agent_a，直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 将消息放入 agent_b 的收件箱并返回。运行时稍后传递。三个结果：

- **故障隔离。** 智能体 B 崩溃不会崩溃智能体 A——运行时在 B 的处理器中捕获失败并决定如何处理（记录、重试、死信）。
- **天然并发。** 同时有许多消息在传输中；行为者并发处理各自的收件箱。
- **分布式就绪。** 无论行为者是在进程内还是在另一台主机上，收件箱 + 传输是相同的抽象。

### 拓扑

- **RoundRobinGroupChat。** 智能体按固定轮次轮流发言。
- **SelectorGroupChat。** 选择器智能体根据对话上下文选择下一个发言者。
- **Magentic-One。** 用于网页浏览、代码执行、文件处理的参考多智能体团队。基于 AgentChat 构建。

### 可观测性

内置 OpenTelemetry 支持。每条消息发射一个跨度；工具调用按 2026 年 OTel GenAI 语义约定（第 23 课）携带 `gen_ai.*` 属性。

### 状态：维护模式

2026 年初：AutoGen v0.7.x 对研究和原型开发稳定。微软已将积极开发转移到微软智能体框架（2025 年 10 月 1 日公开预览；1.0 GA 目标在 2026 年 Q1 末）。AutoGen 模式可以顺畅地向前移植——行为者模型是持久的思想。

## 构建它

`code/main.py` 实现了一个标准库行为者运行时：

- `Message` — 带 `sender`、`recipient`、`topic`、`body` 的类型化载荷。
- `Actor` — 带 `receive(message, runtime)` 的抽象类。
- `Runtime` — 带共享队列、传递和故障隔离的事件循环。
- 双行为者演示：`ReviewerAgent` 审查代码，`ChecklistAgent` 运行检查清单；它们交换消息直到达成共识。

运行：

```
python3 code/main.py
```

追踪显示消息传递、一个行为者的模拟失败（不会崩溃另一个行为者），以及收敛到共同的判决。

## 使用它

- **AutoGen v0.4/v0.7**（维护模式）— 对研究、原型开发、多智能体模式稳定。
- **微软智能体框架**（公开预览）— 前进路径；相同的行为者模型思想，刷新的 API。
- **LangGraph 群拓扑**（第 13 课）— 通过共享工具切换的类似模式。
- **自定义行为者运行时** — 当你需要特定传输（NATS、RabbitMQ、gRPC）时。

## 交付它

`outputs/skill-actor-runtime.md` 生成一个最小行为者运行时和一个给定多智能体任务的团队模板（RoundRobin 或 Selector）。

## 练习

1. 添加死信队列：当处理器引发异常时，停放失败的消息供人工检查。在你的玩具中 DLQ 被触发的频率如何？
2. 实现 `SelectorGroupChat`：选择器行为者根据对话状态选择谁处理下一条消息。
3. 添加分布式传输：将进程内队列替换为 JSON-over-HTTP 服务器，使行为者可以在独立进程中运行。
4. 为每条消息连接一个 OTel 跨度（或无操作替代）。按第 23 课发射 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构文章。将你的玩具移植到真实的 `autogen_core` API。你跳过了哪些在生产中很重要的内容？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 行为者（Actor） | "智能体" | 私有状态 + 收件箱 + 处理器；无共享内存。 |
| 消息（Message） | "事件" | 类型化载荷；行为者交互的唯一方式。 |
| 收件箱（Inbox） | "邮箱" | 每个行为者的待处理消息队列。 |
| 运行时（Runtime） | "智能体宿主" | 路由消息和隔离失败的事件循环。 |
| 主题（Topic） | "频道" | 行为者之间命名的发布-订阅路由。 |
| 故障隔离（Fault isolation） | "让它崩溃" | 一个行为者失败不会崩溃其他行为者。 |
| RoundRobinGroupChat | "固定轮次团队" | 智能体按顺序轮流。 |
| SelectorGroupChat | "上下文路由团队" | 选择器选择下一个发言者。 |
| Magentic-One | "参考团队" | 用于网页 + 代码 + 文件的多智能体小组。 |

## 延伸阅读

- [AutoGen v0.4，微软研究院](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 重新设计的文章
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 图形替代方案
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — AutoGen 默认发射的跨度

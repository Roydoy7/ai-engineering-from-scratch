# OpenAI Agents SDK：交接、护栏与追踪（OpenAI Agents SDK: Handoffs, Guardrails, Tracing）

> OpenAI Agents SDK 是构建在 Responses API 之上的轻量级多智能体框架。五个基本元素：Agent、Handoff、Guardrail、Session、Tracing。交接（Handoff）是名为 `transfer_to_<agent>` 的工具。护栏（Guardrail）在输入或输出时触发。追踪（Tracing）默认开启。

**类型：** 学习 + 构建  
**语言：** Python（标准库）  
**前置知识：** Phase 14 · 01（智能体循环）、Phase 14 · 06（工具使用）  
**预计时间：** 约 75 分钟

## 学习目标

- 说出 OpenAI Agents SDK 的五个基本元素。
- 解释交接（Handoff）：为什么它被建模为工具，模型看到的名称是什么形状，以及上下文如何传递。
- 区分输入护栏、输出护栏和工具护栏；解释 `run_in_parallel` 模式与阻塞模式。
- 实现一个带有交接、护栏和跨度式追踪的标准库运行时。

## 问题所在

无法干净委托的智能体最终会把所有东西塞进一个提示词。没有护栏的智能体会泄露 PII、违反策略的输出，或无限循环。OpenAI 的 SDK 将使多智能体工作变得可控的三个基本元素编码化。

## 核心概念

### 五个基本元素

1. **Agent（智能体）。** LLM + instructions + tools + handoffs。
2. **Handoff（交接）。** 委托给另一个智能体。以名为 `transfer_to_<agent_name>` 的工具的形式呈现给模型。
3. **Guardrail（护栏）。** 对输入（仅第一个智能体）、输出（仅最后一个智能体）或工具调用（每个函数工具）的验证。
4. **Session（会话）。** 跨轮次的自动对话历史。
5. **Tracing（追踪）。** 针对 LLM 生成、工具调用、交接、护栏的内置跨度。

### 交接作为工具

模型在其工具列表中看到 `transfer_to_billing_agent`。调用它会指示运行时：

1. 复制对话上下文（或通过 `nest_handoff_history` 测试版折叠）。
2. 用目标智能体的 instructions 初始化它。
3. 用目标智能体继续运行。

这是产品化了的监督者模式（第 13 课 / 第 28 课）。

### 护栏

三种类型：

- **输入护栏（Input guardrails）。** 在第一个智能体的输入上运行。在任何 LLM 调用之前拒绝不安全或超出范围的请求。
- **输出护栏（Output guardrails）。** 在最后一个智能体的输出上运行。捕获 PII 泄露、策略违规、格式错误的响应。
- **工具护栏（Tool guardrails）。** 每个函数工具运行。验证参数、检查权限、审计执行。

模式：

- **并行（Parallel，默认）。** 护栏 LLM 与主 LLM 并行运行。尾延迟更低。如果触发，主 LLM 的工作被丢弃（浪费 token）。
- **阻塞（Blocking，`run_in_parallel=False`）。** 护栏 LLM 先运行。如果触发，主调用不会浪费 token。

触发时抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### 追踪

默认开启。每次 LLM 生成、工具调用、交接和护栏都发射一个跨度。`OPENAI_AGENTS_DISABLE_TRACING=1` 可选择退出。`add_trace_processor(processor)` 将跨度扇出到你自己的后端，与 OpenAI 并行。

### 会话

`Session` 在后端（SQLite、Redis、自定义）存储对话历史。`Runner.run(agent, input, session=session)` 自动加载并追加。

### 这个模式在哪里出错

- **交接漂移。** 智能体 A 交接给智能体 B，智能体 B 再交接回智能体 A。添加跳数计数器。
- **护栏绕过。** 工具护栏只在函数工具上触发；内置工具（文件读取器、网络获取）需要单独的策略。
- **过度追踪。** 跨度中包含敏感内容。与 OTel GenAI 内容捕获规则（第 23 课）配合使用——在外部存储，通过 ID 引用。

## 构建它

`code/main.py` 用标准库实现了 SDK 形态：

- `Agent`、`FunctionTool`、`Handoff`（作为带传输语义的函数工具）。
- 带有输入/输出/工具护栏、交接调度和跳数计数器的 `Runner`。
- 一个简单的跨度发射器，展示追踪形态。
- 一个分诊智能体，根据用户查询交接给账单或支持智能体；护栏在某个输入上触发。

运行：

```
python3 code/main.py
```

追踪显示两次成功的交接、一次输入护栏触发，以及镜像真实 SDK 发射内容的跨度树。

## 使用它

- **OpenAI Agents SDK** 用于 OpenAI 优先的产品。
- **Claude Agent SDK**（第 17 课）用于 Claude 优先的产品。
- **LangGraph**（第 13 课）用于需要明确状态和持久恢复的场景。
- **自定义** 用于需要精确控制的场景（语音、多提供商、联邦部署）。

## 交付它

`outputs/skill-agents-sdk-scaffold.md` 搭建一个带有分诊智能体、交接、输入/输出/工具护栏、会话存储和追踪处理器的 Agents SDK 应用脚手架。

## 练习

1. 添加交接跳数计数器：N 次转移后拒绝。追踪行为。
2. 将 `nest_handoff_history` 实现为一个选项——在转移前将先前消息折叠为一份摘要。
3. 编写一个阻塞式输出护栏。对比会触发它的提示词与通过的提示词的延迟。
4. 将 `add_trace_processor` 连接到 JSON 日志记录器。每个跨度发射什么形状？
5. 阅读 SDK 文档。将你的标准库玩具移植到 `openai-agents-python`。你对什么建模有误？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Agent（智能体） | "LLM + instructions" | SDK 中的 Agent 类型；拥有工具和交接 |
| Handoff（交接） | "转移" | 模型调用以委托给另一个智能体的工具 |
| Guardrail（护栏） | "策略检查" | 对输入/输出/工具调用的验证 |
| Tripwire（触发线） | "护栏触发" | 护栏拒绝时抛出的异常 |
| Session（会话） | "历史存储" | 在运行之间持久化的对话记忆 |
| Tracing（追踪） | "跨度" | 对 LLM + 工具 + 交接 + 护栏的内置可观测性 |
| Blocking guardrail（阻塞护栏） | "顺序检查" | 护栏先运行；触发时不浪费 token |
| Parallel guardrail（并行护栏） | "并发检查" | 护栏并行运行；延迟更低，触发时浪费 token |

## 延伸阅读

- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 基本元素、交接、护栏、追踪
- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude 风格的对应物
- [Anthropic，构建有效智能体](https://www.anthropic.com/research/building-effective-agents) — 何时使用交接
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Agents SDK 跨度映射的标准

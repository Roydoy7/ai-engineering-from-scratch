# OpenTelemetry GenAI——端到端追踪工具调用（OpenTelemetry GenAI — Tracing Tool Calls End-to-End）

> 一个智能体调用了五个工具、三个 MCP 服务器和两个子智能体。你需要一条横跨所有这些的追踪链路。OpenTelemetry GenAI 语义约定（v1.37 及以上版本的稳定属性）是 2026 年的标准，原生支持 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps。本章列举所需属性，讲解跨度层次结构（智能体 → LLM → 工具），并提供一个可插入任意 OTel 导出器的标准库跨度发射器。

**类型：** 构建  
**语言：** Python（标准库，OTel 跨度发射器）  
**前置知识：** Phase 13 · 07（MCP 服务器）、Phase 13 · 08（MCP 客户端）  
**预计时间：** 约 75 分钟

## 学习目标

- 列出 LLM 跨度和工具执行跨度所需的 OTel GenAI 属性。
- 构建涵盖智能体循环、LLM 调用、工具调用和 MCP 客户端分发的追踪层次结构。
- 决定哪些内容需要捕获（选择加入）vs 哪些需要脱敏（默认）。
- 在不重写工具代码的情况下，向本地收集器（Jaeger、Langfuse）发送跨度。

## 问题所在

2026 年 2 月的一次调试：用户反馈"我的智能体有时需要 30 秒才能响应，有时只需 3 秒。"没有追踪链路。日志显示了 LLM 调用，但没有工具分发、没有 MCP 服务器往返、也没有子智能体的信息。只能靠猜。最终发现：某个 MCP 服务器偶尔在冷启动时卡住了。

没有端到端追踪，根本找不到这个问题。OTel GenAI 解决了这个痛点。

这些约定于 2025-2026 年在 OpenTelemetry 语义约定工作组下稳定下来。它们定义了稳定的属性名称，让 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析相同的跨度。埋点一次；推送到任意后端。

## 核心概念

### 跨度层次结构

```
agent.invoke_agent  （顶层，INTERNAL 跨度）
 ├── llm.chat       （CLIENT 跨度）
 ├── tool.execute   （INTERNAL）
 │    └── mcp.call  （CLIENT 跨度）
 ├── llm.chat       （CLIENT 跨度）
 └── subagent.invoke（INTERNAL）
```

整体嵌套在同一个 trace id 下。跨度 ID 编码父子关系。

### 必需属性

根据 2025-2026 年语义约定：

- `gen_ai.operation.name` — `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` — 请求的模型字符串（如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际服务的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 用于关联的提供商响应 ID。

工具跨度属性：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.call.id` — 具体的调用 ID。
- `gen_ai.tool.description` — 工具描述（可选）。

智能体跨度属性：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### 跨度类型

- `SpanKind.CLIENT`：用于跨越进程边界的调用（LLM 提供商、MCP 服务器）。
- `SpanKind.INTERNAL`：用于智能体自身的循环步骤和工具执行。

### 选择加入的内容捕获

默认情况下，跨度携带指标和时序——而非提示词或补全内容。大型载荷和 PII 默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 以及特定的内容捕获环境变量来包含内容。在生产环境启用前请仔细审查。

### 跨度上的事件

可以将令牌级事件添加为跨度事件：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.completion` — 输出消息。
- `gen_ai.content.tool_call` — 已记录的工具调用。

事件在跨度内按时间排序，便于详细回放。

### 导出器

OTel 跨度可导出到：

- **Jaeger / Tempo。** 开源，本地部署。
- **Langfuse。** 专注 LLM 可观测性；可视化 token 使用情况。
- **Arize Phoenix。** 评估 + 追踪一体。
- **Datadog。** 商业产品；原生解析 `gen_ai.*` 属性。
- **Honeycomb。** 列式存储；查询友好。

所有后端都使用 OTLP 线路格式。你的代码不需要关心具体后端。

### 跨 MCP 的上下文传播

当 MCP 客户端调用服务器时，将 W3C traceparent 头注入请求中。Streamable HTTP 支持标准 HTTP 头。Stdio 原生不携带 HTTP 头；规范的 2026 年路线图讨论了在 JSON-RPC 调用的 `_meta.traceparent` 字段上添加支持。

在该特性发布之前：手动将 traceparent 包含在每个请求的 `_meta` 中。服务器记录 trace ID。

### 指标

除跨度之外，GenAI 语义约定还定义了以下指标：

- `gen_ai.client.token.usage` — 直方图。
- `gen_ai.client.operation.duration` — 直方图。
- `gen_ai.tool.execution.duration` — 直方图。

这些指标用于不需要单次调用详情的仪表板。

### AgentOps 层

AgentOps（2024 年成立）专注于 GenAI 可观测性。它封装了主流框架（LangGraph、Pydantic AI、CrewAI）以自动发射 OTel 跨度。如果你的技术栈使用了受支持的框架则很有用；否则使用手动埋点。

## 动手使用

`code/main.py` 将 OTel 格式的跨度输出到 stdout（以类 OTLP-JSON 格式），针对一个调用 LLM、分发两个工具并进行一次 MCP 往返的智能体。没有真实的导出器——本章专注于跨度结构和属性集。将输出粘贴到支持 OTLP 的查看器，或者直接阅读。

要关注的内容：

- Trace ID 在所有跨度中共享。
- 父子关系通过 `parentSpanId` 编码。
- 必需的 `gen_ai.*` 属性已填充。
- 内容捕获默认关闭；一个场景通过环境变量开启。

## 输出产物

本章生成 `outputs/skill-otel-genai-instrumentation.md`。给定一个智能体代码库，该技能生成埋点计划：在哪里添加跨度、填充哪些属性，以及目标导出器。

## 练习

1. 运行 `code/main.py`。统计跨度数量，并识别哪些是 CLIENT 类型，哪些是 INTERNAL 类型。

2. 开启内容捕获（环境变量），确认 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件出现。注意 PII 方面的影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration`，并为每次调用发射一个直方图样本。

4. 将 traceparent 从父智能体跨度传播到 MCP 请求的 `_meta.traceparent` 字段。验证 MCP 服务器能看到相同的 trace ID。

5. 阅读 OTel GenAI 语义约定规范。找出规范中列出但本章代码没有发射的一个属性。添加它。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| OTel | "OpenTelemetry" | 追踪、指标、日志的开放标准。 |
| GenAI 语义约定（GenAI semconv） | "GenAI 语义约定" | LLM/工具/智能体跨度的稳定属性名称。 |
| `gen_ai.*` | "属性命名空间" | 所有 GenAI 属性共享此前缀。 |
| 跨度（Span） | "计时操作" | 有开始时间、结束时间和属性的工作单元。 |
| 追踪（Trace） | "跨跨度的祖先关系" | 共享 trace ID 的跨度树。 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | 跨度方向的提示信息。 |
| OTLP | "OpenTelemetry 线路协议" | 导出器的传输格式。 |
| 选择加入内容（Opt-in content） | "提示词/补全捕获" | 默认关闭；通过环境变量开启。 |
| traceparent | "W3C 头" | 跨服务传播追踪上下文。 |
| 导出器（Exporter） | "特定后端的发送器" | 将跨度发送到 Jaeger/Datadog 等的组件。 |

## 延伸阅读

- [OpenTelemetry — GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI 跨度、指标和事件的规范约定
- [OpenTelemetry — GenAI 跨度](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和工具执行跨度的属性列表
- [OpenTelemetry — GenAI 智能体跨度](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — 智能体级 `invoke_agent` 跨度
- [open-telemetry/semantic-conventions — GenAI 跨度](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 托管的权威来源
- [Datadog — LLM OTel 语义约定](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产集成演练

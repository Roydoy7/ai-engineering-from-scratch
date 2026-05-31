# OpenTelemetry GenAI 语义约定（OpenTelemetry GenAI Semantic Conventions）

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了智能体遥测的标准 schema。跨度名称、属性和内容捕获规则在各供应商之间收敛，使智能体追踪在 Datadog、Grafana、Jaeger 和 Honeycomb 中具有相同的含义。

**类型：** 学习 + 构建  
**语言：** Python（标准库）  
**前置知识：** Phase 14 · 13（LangGraph）、Phase 14 · 24（可观测性平台）  
**预计时间：** 约 60 分钟

## 学习目标

- 说出 GenAI 跨度类别：model/client、agent、tool。
- 区分 `invoke_agent` 的 CLIENT 跨度与 INTERNAL 跨度，以及各自的适用场景。
- 列出顶级 GenAI 属性：提供商名称、请求模型、数据源 ID。
- 解释内容捕获契约：选择加入，`OTEL_SEMCONV_STABILITY_OPT_IN`，外部引用建议。

## 问题所在

每个供应商都发明自己的跨度名称。运维团队最终为每个框架构建单独的仪表板。OpenTelemetry 的 GenAI SIG 通过定义整个生态系统都瞄准的一个标准来解决这个问题。

## 核心概念

### 跨度类别

1. **Model / client 跨度。** 覆盖原始 LLM 调用。由提供商 SDK（Anthropic、OpenAI、Bedrock）和框架模型适配器发射。
2. **Agent 跨度。** `create_agent`（当智能体被构建时）和 `invoke_agent`（当它运行时）。
3. **Tool 跨度。** 每次工具调用一个；通过父子关系连接到智能体跨度。

### 智能体跨度命名

- 跨度名称：如果已命名，则为 `invoke_agent {gen_ai.agent.name}`；否则回退到 `invoke_agent`。
- 跨度类型：
  - **CLIENT** — 用于远程智能体服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 用于进程内智能体框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — 模型 ID。
- `gen_ai.response.model` — 解析后的模型（由于路由，可能与请求不同）。
- `gen_ai.agent.name` — 智能体标识符。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — 对于 RAG：查询了哪个语料库或存储。

针对 Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 存在技术特定的约定。

### 内容捕获

默认规则：Instrumentation 默认情况下不应捕获输入/输出。捕获通过以下方式选择加入：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：在外部存储内容（S3、你的日志存储），在跨度上记录引用（指针 ID，而非散文）。这是连接到可观测性的第 27 课内容投毒防御。

### 稳定性

截至 2026 年 3 月，大多数约定是实验性的。通过以下方式选择加入稳定预览版：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 将 GenAI 属性原生映射到其 LLM Observability schema 中。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 这个模式在哪里出错

- **在跨度中捕获完整提示词。** PII、密钥、客户数据出现在运维人员可以读取的追踪中。在外部存储。
- **没有 `gen_ai.provider.name`。** 当归因缺失时，多提供商仪表板会损坏。
- **没有父链接的跨度。** 孤立的工具跨度。始终传播上下文。
- **不设置稳定性选择加入。** 你的属性可能在后端升级时被重命名。

## 构建它

`code/main.py` 实现了一个符合 GenAI 约定的标准库跨度发射器：

- 带有 GenAI 属性 schema 的 `Span`。
- 带有 `start_span`、嵌套上下文的 `Tracer`。
- 一个发射以下内容的脚本化智能体运行：`create_agent`、`invoke_agent`（INTERNAL）、每个工具的跨度、LLM 调用的 `chat` 跨度。
- 一种内容捕获模式，在外部存储提示词并在跨度上记录 ID。

运行：

```
python3 code/main.py
```

输出：带有所有必需 GenAI 属性的跨度树，以及显示选择加入内容引用的"外部存储"。

## 使用它

- **Datadog LLM Observability**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（第 24 课）——自动 instrument 生态系统。
- **Jaeger / Honeycomb / Grafana Tempo** — 原始 OTel 追踪；从 GenAI 属性构建仪表板。
- **自托管** — 运行带有 GenAI 处理器的 OTel Collector。

## 交付它

`outputs/skill-otel-genai.md` 将 OTel GenAI 跨度连接到现有智能体，带有内容捕获默认值和外部引用存储。

## 练习

1. 用 `invoke_agent`（INTERNAL）+ 每个工具跨度对第 01 课的 ReAct 循环进行 instrument。发送到 Jaeger 实例。
2. 以"仅引用"模式添加内容捕获：提示词存储到 SQLite，跨度属性只携带行 ID。
3. 阅读 `gen_ai.data_source.id` 的规范。将其连接到第 09 课的 Mem0 搜索。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 并验证你的属性不会被 collector 重命名。
5. 构建仪表板："哪些工具错误与哪些模型相关"——仅使用 GenAI 属性。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| GenAI SIG | "OpenTelemetry GenAI 工作组" | 定义 schema 的 OTel 工作组 |
| invoke_agent | "智能体跨度" | 代表智能体运行的跨度名称 |
| CLIENT span（CLIENT 跨度） | "远程调用" | 调用远程智能体服务的跨度 |
| INTERNAL span（INTERNAL 跨度） | "进程内" | 进程内智能体运行的跨度 |
| gen_ai.provider.name | "提供商" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG 来源" | 检索命中了哪个语料库/存储 |
| Content capture（内容捕获） | "提示词日志" | 选择加入的消息捕获；在生产中外部存储 |
| Stability opt-in（稳定性选择加入） | "预览模式" | 固定实验性约定的环境变量 |

## 延伸阅读

- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认发射 GenAI 跨度
- [AutoGen v0.4（微软研究院）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置 OTel 跨度
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C 追踪上下文传播

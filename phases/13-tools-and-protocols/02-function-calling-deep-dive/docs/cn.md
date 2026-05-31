# 函数调用深度解析——OpenAI、Anthropic、Gemini（Function Calling Deep Dive — OpenAI, Anthropic, Gemini）

> 三家前沿提供商在 2024 年收敛于同一工具调用循环，然后在其他所有方面出现分歧。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 和唯一 ID 关联。本章并排对比三者，使在一个提供商上发布的代码在移植时不会出错。

**类型：** 构建  
**语言：** Python（标准库，模式转换器）  
**前置知识：** Phase 13 · 01（工具接口）  
**预计时间：** 约 75 分钟

## 学习目标

- 说出 OpenAI、Anthropic 和 Gemini 函数调用有效载荷的三种形状差异（声明、调用、结果）。
- 将一个工具声明翻译为所有三种提供商格式，并预测严格模式约束在哪里会有所不同。
- 在每个提供商中使用 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解每个提供商的硬性限制（工具数量、模式深度、参数长度）以及超限时各自发出的错误信号。

## 问题所在

函数调用请求的形状因提供商而异。以下是 2026 年生产栈中的三个具体示例：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是你必须解析的 JSON 字符串。严格模式（`strict: true`）通过约束解码强制执行模式合规性。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 的形式返回。`input` 已经解析（是一个对象，而非字符串）。你用包含 `{type: "tool_result", tool_use_id, content}` 块的新 `user` 消息回复。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 的形式到达，其中 `id` 在 Gemini 3 及以上版本中对于并行调用关联是唯一的。你用 `{functionResponse: {name, id, response}}` 回复。

循环相同。字段名不同，嵌套不同，字符串 vs 对象的约定不同，关联机制不同。一个在 OpenAI 上编写天气智能体的团队，移植到 Anthropic 需要两天，再移植到 Gemini 又需要一天，仅仅是为了处理管道差异。

本章构建一个将三种格式统一为一个标准工具声明并在边缘路由的转换器。Phase 13 · 17 将同一模式推广为 LLM 网关。

## 核心概念

### 公共结构

每个提供商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入模式。
2. **工具选择。** 强制指定工具、禁止工具或让模型决定。
3. **调用输出。** 命名工具和参数的结构化输出。
4. **调用 ID。** 将响应关联到正确的调用（对并行调用很重要）。
5. **结果注入。** 将结果与调用绑定的消息或块。

### 逐字段形状差异

| 方面 | OpenAI | Anthropic | Gemini |
|------|--------|-----------|--------|
| 声明包装 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| 模式字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | 助手消息上的 `tool_calls[]` | 类型为 `tool_use` 的 `content[]` | 类型为 `functionCall` 的 `parts[]` |
| 参数类型 | JSON 字符串化 | 已解析对象 | 已解析对象 |
| ID 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | 角色 `tool`，`tool_call_id` | 带 `tool_result`、`tool_use_id` 的 `user` | 带匹配 `id` 的 `functionResponse` |
| 强制指定工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格模式 | `strict: true` | 模式即契约（始终执行） | 请求级 `responseSchema` |

### 你实际会遇到的限制

- **OpenAI。** 每次请求 128 个工具。模式深度 5。参数字符串 <= 8192 字节。严格模式要求无 `$ref`、无重叠的 `oneOf`/`anyOf`/`allOf`，所有属性都列在 `required` 中。
- **Anthropic。** 每次请求 64 个工具。模式深度实际上无界，但实际限制约为 10。没有严格模式标志；模式是契约，模型倾向于遵从。
- **Gemini。** 每次请求 64 个函数。模式类型是 OpenAPI 3.0 子集（与 JSON Schema 2020-12 略有差异）。自 Gemini 3 起并行调用有唯一 ID。

### `tool_choice` 行为

每个人都支持三种模式，命名不同：

- **自动（Auto）。** 模型选择工具或文本。默认值。
- **必须/任意（Required/Any）。** 模型必须至少调用一个工具。
- **无（None）。** 模型不得调用工具。

加上每个提供商独有的一种模式：

- **OpenAI。** 按名称强制指定特定工具。
- **Anthropic。** 按名称强制指定特定工具；`disable_parallel_tool_use` 标志区分单次与多次调用。
- **Gemini。** `mode: "VALIDATED"` 无论模型意图如何都通过模式验证器路由每个响应。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）在一条助手消息中输出多个调用。你全部运行，并用包含每个 `tool_call_id` 一个条目的批量工具角色消息回复。Anthropic 历史上是单次调用；`disable_parallel_tool_use: false`（Claude 3.5 默认）启用多次调用。Gemini 2 允许并行调用但没有稳定的 ID；Gemini 3 添加了 UUID，使乱序响应可以干净地关联。

### 流式传输

三者都支持流式工具调用。wire 格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量块递增到达。你累积直到 `finish_reason: "tool_calls"`。
- **Anthropic。** 块开始/块增量/块停止事件。`input_json_delta` 块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）以带 `functionCallId` 的块输出，使多个并行调用可以交错。

Phase 13 · 03 深入讲解并行 + 流式重组。本章专注于声明和单次调用形状。

### 错误与修复

无效参数错误看起来也不同：

- **OpenAI（非严格）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，你注入错误消息并重新调用。
- **OpenAI（严格）。** 验证在解码期间发生；无效 JSON 是不可能的，但可能出现 `refusal`。
- **Anthropic。** `input` 可能包含意外字段；模式是建议性的。在服务器端验证。
- **Gemini。** OpenAPI 3.0 的怪癖：对象字段上的 `enum` 被静默忽略；自行验证。

### 转换器模式

你代码中的标准工具声明看起来像这样（你选择形状）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数将其翻译为三种提供商形状。`code/main.py` 中的工具库正是这样做的，然后通过每个提供商的响应形状对虚拟工具调用进行往返。无需网络——本章教授形状，而非 HTTP。

生产团队将此转换器封装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。Phase 13 · 17 发布一个在任意三者前面暴露 OpenAI 形状 API 的网关。

## 动手使用

`code/main.py` 定义一个标准 `Tool` 数据类和三个输出 OpenAI、Anthropic 和 Gemini 声明 JSON 的转换器。然后它将每种形状的手工制作提供商响应解析为同一标准调用对象，证明语义在外表之下是相同的。运行它并并排对比三个声明。

要关注的内容：

- 三个声明块只在包装和字段名上有所不同。
- 三个响应块在调用所在位置上有所不同（顶级 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `canonical_call()` 函数从三种响应形状中提取 `{id, name, args}`。

## 输出产物

本章生成 `outputs/skill-provider-portability-audit.md`。给定针对一个提供商的函数调用集成，该技能生成可移植性审计：它依赖哪些提供商限制，哪些字段需要重命名，以及移植到每个其他提供商时什么会出错。

## 练习

1. 运行 `code/main.py` 并验证三个提供商的声明 JSON 都序列化同一底层 `Tool` 对象。修改标准工具以添加枚举参数，并确认只有 Gemini 转换器需要处理 OpenAPI 怪癖。

2. 为每个提供商添加 `ListToolsResponse` 解析器，提取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 原生没有这个功能；注意这个不对称性。

3. 实现 `tool_choice` 转换：将标准 `ToolChoice(mode="force", tool_name="x")` 映射到所有三种提供商形状。然后映射 `mode="any"` 和 `mode="none"`。检查本章的差异表。

4. 选择三个提供商之一，从头到尾阅读其函数调用指南。找到其模式规范中另外两个不支持的一个字段。候选：OpenAI `strict`，Anthropic `disable_parallel_tool_use`，Gemini `function_calling_config.allowed_function_names`。

5. 编写一个测试向量：一个参数违反声明模式的工具调用。通过每个提供商的验证器（第 01 课的标准库版本可以作为代理）运行它，并记录哪些错误被触发。记录生产中你会使用哪个提供商来保证严格性。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 函数调用（Function calling） | "工具调用" | 用于结构化工具调用输出的提供商级 API。 |
| 工具声明（Tool declaration） | "工具规范" | 名称 + 描述 + JSON Schema 输入有效载荷。 |
| `tool_choice` | "强制/禁止" | 自动/必须/无/特定名称模式。 |
| 严格模式（Strict mode） | "模式执行" | 约束解码以匹配模式的 OpenAI 标志。 |
| `tool_use` 块 | "Anthropic 的调用形状" | 带 id、名称、输入的内联内容块。 |
| `functionCall` 部分 | "Gemini 的调用形状" | 包含名称、参数和 ID 的 `parts[]` 条目。 |
| 参数作为字符串（Arguments-as-string） | "JSON 字符串化" | OpenAI 以 JSON 字符串而非对象的形式返回参数。 |
| 并行工具调用（Parallel tool calls） | "一次轮次的扇出" | 一条助手消息中的多个工具调用。 |
| 拒绝（Refusal） | "模型拒绝" | 严格模式专用的拒绝块，而非调用。 |
| OpenAPI 3.0 子集（OpenAPI 3.0 subset） | "Gemini 模式怪癖" | Gemini 使用的与 JSON Schema 有细微差异的类 JSON Schema 方言。 |

## 延伸阅读

- [OpenAI — 函数调用指南](https://platform.openai.com/docs/guides/function-calling) — 包含严格模式和并行调用的权威参考
- [Anthropic — 工具调用概述](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 和 `tool_result` 块语义
- [Google — Gemini 函数调用](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一 ID 和 OpenAPI 子集
- [Vertex AI — 函数调用参考](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业级接口
- [OpenAI — 结构化输出](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式模式执行细节

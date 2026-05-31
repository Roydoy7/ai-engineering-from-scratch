# MCP 基础——原语、生命周期、JSON-RPC 基础（MCP Fundamentals — Primitives, Lifecycle, JSON-RPC Base）

> MCP 之前的每次集成都是一次性的。模型上下文协议（Model Context Protocol）于 2024 年 11 月由 Anthropic 首次发布，现由 Linux 基金会的 Agentic AI Foundation 托管，它标准化了发现和调用机制，使任何客户端都能与任何服务器通信。2025-11-25 规范定义了六个原语（三个服务器端、三个客户端端）、三阶段生命周期和 JSON-RPC 2.0 wire 格式。掌握这些，本阶段 MCP 章节的其余内容就只是阅读了。

**类型：** 学习  
**语言：** Python（标准库，JSON-RPC 解析器）  
**前置知识：** Phase 13 · 01 至 05（工具接口与函数调用）  
**预计时间：** 约 45 分钟

## 学习目标

- 列举全部六个 MCP 原语（服务器端：工具、资源、提示；客户端端：根、采样、询问）并各举一个用例。
- 逐步讲解三阶段生命周期（初始化、操作、关闭），说明每个阶段由谁发送哪条消息。
- 解析和发出 JSON-RPC 2.0 请求、响应和通知信封。
- 解释 `initialize` 时的能力协商是什么，以及没有它会出什么问题。

## 问题所在

MCP 之前，每个使用工具的智能体都有自己的协议。Cursor 有一个形似 MCP 但不兼容的工具系统。Claude Desktop 带有另一个。VS Code 的 Copilot 扩展有第三个。一个构建了"Postgres 查询"工具的团队需要将同一工具写三遍，分别适配不同宿主的 API。复用它需要复制代码。

结果是一次性集成的寒武纪大爆发，以及生态系统速度的天花板。

MCP 通过标准化 wire 格式解决了这个问题。单个 MCP 服务器可在每个 MCP 客户端中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，到 2026 年 4 月已有 300 多个客户端。每月 SDK 下载量 1.1 亿次。超过 10,000 个公开服务器。Linux 基金会于 2025 年 12 月在新的 Agentic AI Foundation 下接管了管理权。

本阶段使用的规范版本是 **2025-11-25**，新增了异步任务（SEP-1686）、URL 模式询问（SEP-1036）、带工具的采样（SEP-1577）、增量范围同意（SEP-835）和 OAuth 2.1 资源指示器语义。Phase 13 · 09 至 16 涵盖这些扩展。本章止步于基础。

## 核心概念

### 三个服务器端原语

1. **工具（Tools）。** 可调用的动作。与 Phase 13 · 01 中相同的四步循环。
2. **资源（Resources）。** 暴露的数据。通过 URI 可寻址的只读内容：`file:///path`、`db://query/...`、自定义方案。
3. **提示（Prompts）。** 可复用的模板。宿主 UI 中的斜杠命令；服务器提供模板，客户端填充参数。

### 三个客户端端原语

4. **根（Roots）。** 服务器被允许访问的 URI 集合。客户端声明它们；服务器遵守它们。
5. **采样（Sampling）。** 服务器请求客户端的模型执行补全。无需服务器端 API 密钥即可实现服务器托管的智能体循环。
6. **询问（Elicitation）。** 服务器在运行中途向客户端用户请求结构化输入。表单或 URL（SEP-1036）。

MCP 中的每个能力都恰好属于这六个之一。Phase 13 · 10 至 14 深入讲解每一个。

### Wire 格式：JSON-RPC 2.0

每条消息都是具有以下字段的 JSON 对象：

- 请求：`{jsonrpc: "2.0", id, method, params}`。
- 响应：`{jsonrpc: "2.0", id, result | error}`。
- 通知：`{jsonrpc: "2.0", method, params}` —— 无 `id`，不期望响应。

基础规范有约 15 个方法，按原语分组。重要的有：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务器到客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**第一阶段：初始化（initialize）。**

客户端发送 `initialize`，携带其 `capabilities` 和 `clientInfo`。服务器响应自己的 `capabilities`、`serverInfo` 以及其所支持的规范版本。客户端消化响应后发送 `notifications/initialized`。从此双方都可以按协商的能力发送请求。

**第二阶段：操作（operation）。**

双向通信。客户端调用 `tools/list` 发现工具，然后调用 `tools/call` 执行。如果服务器声明了该能力，可以发送 `sampling/createMessage`。当工具集变化时，服务器可以发送 `notifications/tools/list_changed`。当用户更改根范围时，客户端可以发送 `notifications/roots/list_changed`。

**第三阶段：关闭（shutdown）。**

任意一方关闭传输。MCP 中没有结构化的关闭方法；传输（stdio 或 Streamable HTTP，Phase 13 · 09）携带连接结束信号。

### 能力协商

`initialize` 握手中的 `capabilities` 是契约。来自服务器的示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它可以发出 `tools/list_changed` 通知并支持 `resources/subscribe`。客户端通过声明自己的能力来表示同意：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端没有声明 `sampling`，服务器不得调用 `sampling/createMessage`。对称地：如果服务器没有声明 `resources.subscribe`，客户端不得尝试订阅。

这就是防止生态系统漂移的机制。不支持采样的客户端仍然是有效的 MCP 客户端；不调用 `sampling` 的服务器仍然是有效的 MCP 服务器。它们只是不一起使用该功能。

### 结构化内容和错误形状

`tools/call` 返回类型化块的 `content` 数组：`text`、`image`、`resource`。Phase 13 · 14 将 MCP 应用（`ui://` 交互式 UI）添加到该列表中。

错误使用 JSON-RPC 错误代码。规范新增：`-32002`"资源未找到"、`-32603`"内部错误"，以及 `error.data` 中的 MCP 特定错误数据。

### 客户端能力 vs 工具调用细节

一个常见混淆：`capabilities.tools` 表示客户端是否支持工具列表变更通知。客户端是否"会"调用特定工具是由其模型驱动的运行时选择，而非能力标志。能力标志是规范层面的契约。模型的选择是正交的。

### 为什么用 JSON-RPC 而非 REST？

JSON-RPC 2.0（2010）是一个轻量级双向协议。REST 是客户端发起的。MCP 需要服务器发起的消息（采样、通知），所以具有对称请求/响应形状的 JSON-RPC 非常合适。JSON-RPC 也能干净地在 stdio 和 WebSocket/Streamable HTTP 上组合，而无需重新发明 HTTP 的请求形状。

## 动手使用

`code/main.py` 提供了一个最小化的 JSON-RPC 2.0 解析器和发送器，然后手动逐步执行 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印每条消息。没有真实传输；只有消息形状。与延伸阅读中链接的规范进行比较，以验证每个信封。

要关注的内容：

- `initialize` 双向声明能力；响应包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回 `tools` 数组；每个条目有 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应 `content` 是 `{type, text}` 块的数组。

## 输出产物

本章生成 `outputs/skill-mcp-handshake-tracer.md`。给定 pcap 风格的 MCP 客户端-服务器交互记录，该技能为每条消息标注其所属原语、生命周期阶段和所依赖的能力。

## 练习

1. 运行 `code/main.py`。找到能力协商发生的那一行，并描述如果服务器没有声明 `tools.listChanged` 会有什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息形状：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在长时间运行的 `tools/call` 进行中发出它，并确认客户端处理器会显示进度条。

3. 从头到尾阅读 MCP 2025-11-25 规范——整个文档约 80 页。找出大多数服务器不需要的那个能力标志。提示：它与资源订阅有关。

4. 在纸上草绘假设的"定时任务"功能所属的原语。（提示：服务器希望客户端在预定时间调用它。如今的六个原语都不完全适用。）MCP 的 2026 年路线图中有一个关于此的草案 SEP。

5. 解析 GitHub 上某个开放 MCP 服务器的一次会话日志。统计请求、响应和通知消息的数量。计算生命周期流量与操作流量的比例。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| MCP | "模型上下文协议" | 用于模型到工具发现和调用的开放协议。 |
| 服务器端原语（Server primitive） | "服务器暴露的内容" | 工具（动作）、资源（数据）、提示（模板）。 |
| 客户端端原语（Client primitive） | "客户端允许服务器使用的内容" | 根（范围）、采样（LLM 回调）、询问（用户输入）。 |
| JSON-RPC 2.0 | "Wire 格式" | 对称的请求/响应/通知信封。 |
| `initialize` 握手 | "能力协商" | 第一对消息；服务器和客户端声明所支持的功能。 |
| `tools/list` | "发现" | 客户端向服务器请求其当前工具集。 |
| `tools/call` | "调用" | 客户端请求服务器使用参数执行工具。 |
| `notifications/*_changed` | "变更事件" | 服务器告知客户端其原语列表已变更。 |
| 内容块（Content block） | "类型化结果" | 工具结果中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}`。 |
| SEP | "规范演进提案" | 命名的草案提案（例如 SEP-1686 用于异步任务）。 |

## 延伸阅读

- [Model Context Protocol — 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 规范文档正文
- [Model Context Protocol — 架构概念](https://modelcontextprotocol.io/docs/concepts/architecture) — 六原语心智模型
- [Anthropic — 介绍模型上下文协议](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月发布文章
- [MCP 博客 — 第一个 MCP 周年纪念](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一周年回顾与 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 规范更新](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835 和 1724 的摘要

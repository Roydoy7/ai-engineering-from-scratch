# MCP 资源与提示——工具之外的上下文暴露（MCP Resources and Prompts — Context Exposure Beyond Tools）

> 工具获得了 MCP 90% 的关注。另外两个服务器端原语解决不同的问题。资源暴露数据供读取；提示将可复用模板作为斜杠命令暴露。许多服务器应该使用资源而非将读取包装在工具中，使用提示而非在客户端提示中硬编码工作流。本章归纳决策规则，并逐步讲解 `resources/*` 和 `prompts/*` 消息。

**类型：** 构建  
**语言：** Python（标准库，资源 + 提示处理器）  
**前置知识：** Phase 13 · 07（MCP 服务器）  
**预计时间：** 约 45 分钟

## 学习目标

- 针对特定领域，决定将能力暴露为工具、资源还是提示。
- 实现 `resources/list`、`resources/read`、`resources/subscribe` 并处理 `notifications/resources/updated`。
- 实现带参数模板的 `prompts/list` 和 `prompts/get`。
- 识别宿主何时将提示作为斜杠命令呈现，何时作为自动注入的上下文。

## 问题所在

一个笔记应用的简单 MCP 服务器将所有内容暴露为工具：`notes_read`、`notes_list`、`notes_search`。这将每次数据访问都包装在模型驱动的工具调用中。后果：

- 对于可能受益于上下文的每个查询，模型都必须决定是否调用 `notes_read`。
- 只读内容无法被订阅或流式传输到宿主的侧面板。
- 客户端 UI（Claude Desktop 的资源附件面板、Cursor 的"包含文件"选择器）无法呈现数据。

正确的拆分：将数据暴露为资源，将变更或计算操作暴露为工具，将可复用的多步工作流暴露为提示。每个原语都有其 UX 功能和访问模式。

## 核心概念

### 工具 vs 资源 vs 提示——决策规则

| 能力 | 原语 |
|------|------|
| 用户想要搜索、过滤或转换数据 | 工具 |
| 用户想让宿主将这些数据作为上下文包含进来 | 资源 |
| 用户想要一个可以重复运行的模板化工作流 | 提示 |

指导原则：如果模型在每个相关查询上调用它都有益，那么它是工具。如果用户将它附加到对话中有益，那么它是资源。如果整个多步工作流是用户想要复用的单元，那么它是提示。

### 资源

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接收 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义方案）
- `memory://session-2026-04-22/recent`（服务器特定）

`contents[]` 同时支持文本和二进制。二进制使用 `blob` 作为 base64 编码的字符串加上 `mimeType`。

### 资源订阅

在能力中声明 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}`。当资源变更时，服务器发送 `notifications/resources/updated {uri}`。客户端重新读取。

用例：其资源是磁盘上文件的笔记服务器；文件监视器触发更新通知；当文件在宿主外部编辑时，Claude Desktop 将文件重新拉入上下文。

### 资源模板（2025-11-25 新增）

`resourceTemplates` 允许你暴露参数化的 URI 模式：`notes://{id}`，其中 `id` 是补全目标。客户端可以在资源选择器中自动补全 ID。

### 提示

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接收 `{name, arguments}` 并返回 `{description, messages: [{role, content}]}`。

提示是一个模板，填充为宿主馈送给其模型的消息列表。例如，`code_review` 提示接受 `file_path` 参数，并返回一个三消息序列：系统消息、包含文件内容的用户消息，以及带推理模板的助手开场白。

### 宿主与提示

Claude Desktop、VS Code 和 Cursor 在聊天 UI 中将提示作为斜杠命令暴露。用户输入 `/code_review` 并从表单中选择参数。服务器的提示是"用户快捷方式"和"发送给模型的完整提示"之间的契约。

并非每个客户端都支持提示——请检查能力协商。声明了提示能力的服务器与不支持提示的客户端配合时，用户只是看不到斜杠命令而已。

### "列表已变更"通知

资源和提示在集合变更时都会发出 `notifications/list_changed`。刚导入 20 条新笔记的笔记服务器发出 `notifications/resources/list_changed`；客户端重新调用 `resources/list` 来获取新增内容。

### 内容类型约定

文本：`mimeType: "text/plain"`、`text/markdown`、`application/json`。  
二进制：`image/png`、`application/pdf`，加上 `blob` 字段。  
MCP 应用（第 14 课）：`ui://` URI 中的 `text/html;profile=mcp-app`。

### 动态资源

资源 URI 不必对应静态文件。`notes://recent` 可以在每次读取时返回最新的五条笔记。`db://query/users/active` 可以执行参数化查询。服务器可以自由地动态计算内容。

规则：如果客户端可以按 URI 缓存，URI 必须稳定。如果计算是一次性的，URI 应包含时间戳或 nonce，以防客户端缓存过期。

### 订阅 vs 轮询

支持订阅的客户端通过 `notifications/resources/updated` 获得服务器推送。不支持订阅的客户端或宿主通过重新读取来轮询。两者都符合规范。服务器的能力声明告诉客户端它支持哪种。

订阅的代价：服务器上的每个会话状态（谁订阅了什么）。保持订阅集合有界；断开连接的客户端应超时。

### 提示 vs 系统提示

MCP 中的提示不是系统提示。宿主的系统提示（其自己的操作指令）和 MCP 提示（用户调用的服务器提供模板）并列存在。行为良好的客户端永远不允许服务器提示覆盖其自己的系统提示；它将它们层叠在一起。

## 动手使用

`code/main.py` 在第 07 课的笔记服务器基础上扩展了：

- 每个笔记的资源（`notes://note-1` 等），支持 `resources/subscribe`。
- 渲染为三消息模板的 `review_note` 提示。
- 在笔记被修改时发出 `notifications/resources/updated` 的文件监视器模拟。
- 始终返回最新五条笔记的 `notes://recent` 动态资源。

运行演示以查看完整流程。

## 输出产物

本章生成 `outputs/skill-primitive-splitter.md`。给定拟议的 MCP 服务器，该技能将每个能力分类为工具/资源/提示，并附上理由。

## 练习

1. 运行 `code/main.py`。观察初始资源列表，然后触发笔记编辑并验证 `notifications/resources/updated` 事件触发。

2. 添加 `resources/list_changed` 发射器：当创建新笔记时，发送通知以便客户端重新发现。

3. 为 GitHub MCP 服务器设计三个提示：`summarize_pr`、`triage_issue`、`release_notes`。每个都带有参数模式。提示体应该无需进一步编辑即可运行。

4. 取第 07 课服务器中的一个现有工具，判断它应该保持为工具还是拆分为资源加工具对。用一句话说明理由。

5. 阅读规范的 `server/resources` 和 `server/prompts` 章节。找出 `resources/read` 中很少使用但规范支持的一个字段。提示：查看资源内容上的 `_meta`。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 资源（Resource） | "暴露的数据" | 宿主可以读取的 URI 可寻址内容。 |
| 资源 URI（Resource URI） | "数据指针" | 方案前缀标识符（`file://`、`notes://` 等）。 |
| `resources/subscribe` | "监视变更" | 客户端选择加入服务器推送特定 URI 的更新。 |
| `notifications/resources/updated` | "资源已变更" | 向客户端发出订阅资源有新内容的信号。 |
| 资源模板（Resource template） | "参数化 URI" | 带有宿主选择器补全提示的 URI 模式。 |
| 提示（Prompt） | "斜杠命令模板" | 带有参数槽位的命名多消息模板。 |
| 提示参数（Prompt arguments） | "模板输入" | 宿主在渲染前收集的类型化参数。 |
| `prompts/get` | "渲染模板" | 服务器返回填充好的消息列表。 |
| 内容块（Content block） | "类型化块" | `{type: text \| image \| resource \| ui_resource}`。 |
| 斜杠命令 UX（Slash-command UX） | "用户快捷方式" | 宿主将提示作为以 `/` 开头的命令呈现。 |

## 延伸阅读

- [MCP — 概念：资源](https://modelcontextprotocol.io/docs/concepts/resources) — 资源 URI、订阅和模板
- [MCP — 概念：提示](https://modelcontextprotocol.io/docs/concepts/prompts) — 提示模板和斜杠命令集成
- [MCP — 服务器资源规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — 完整的 `resources/*` 消息参考
- [MCP — 服务器提示规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — 完整的 `prompts/*` 消息参考
- [MCP — 协议信息站：资源](https://modelcontextprotocol.info/docs/concepts/resources/) — 扩展官方文档的社区指南

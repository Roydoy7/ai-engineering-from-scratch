# 构建 MCP 客户端——发现、调用、会话管理（Building an MCP Client — Discovery, Invocation, Session Management）

> 大多数 MCP 内容提供服务器教程，对客户端部分一带而过。客户端代码才是复杂编排的所在：进程生成、能力协商、跨多个服务器的工具列表合并、采样回调、重连以及命名空间冲突解决。本章构建一个多服务器客户端，将三个不同的 MCP 服务器提升为模型可见的单一扁平工具命名空间。

**类型：** 构建  
**语言：** Python（标准库，多服务器 MCP 客户端）  
**前置知识：** Phase 13 · 07（构建 MCP 服务器）  
**预计时间：** 约 75 分钟

## 学习目标

- 将 MCP 服务器作为子进程生成，完成 `initialize`，并发送 `notifications/initialized`。
- 维护每个服务器的会话状态（能力、工具列表、最近收到的通知 ID）。
- 将多个服务器的工具列表合并为一个命名空间，并处理冲突。
- 将工具调用路由到拥有该工具的服务器，并重组响应。

## 问题所在

真实的智能体宿主（Claude Desktop、Cursor、Goose、Gemini CLI）会同时加载多个 MCP 服务器。用户可能同时运行文件系统服务器、Postgres 服务器和 GitHub 服务器。客户端的工作是：

1. 生成每个服务器。
2. 独立与每个服务器握手。
3. 对每个服务器调用 `tools/list` 并将结果展平。
4. 当模型发出 `notes_search` 时，在合并的命名空间中查找它并路由到正确的服务器。
5. 处理来自任何服务器的通知（`tools/list_changed`），且不阻塞。
6. 在传输失败时重连。

手动实现所有这些，才是区分"玩具"与"可用产品"的关键。官方 SDK 封装了这些，但心智模型必须是你自己的。

## 核心概念

### 子进程生成

使用 `subprocess.Popen`，设置 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。设置 `bufsize=1` 并使用文本模式逐行读取。每个服务器是一个进程；客户端每个服务器持有一个 `Popen` 句柄。

### 每个服务器的会话状态

每个服务器一个 `Session` 对象，包含：

- `process` —— Popen 句柄。
- `capabilities` —— 服务器在 `initialize` 时声明的能力。
- `tools` —— 最近一次 `tools/list` 的结果。
- `pending` —— 请求 ID 到等待响应的 promise/future 的映射。

请求本质上是异步的；向服务器 A 发送 `tools/call` 时，如果服务器 B 正在处理中，不能阻塞。要么使用带队列的线程，要么使用 asyncio。

### 合并命名空间

当客户端看到聚合工具列表时，名称可能冲突。两个服务器都可能暴露 `search`。客户端有三个选项：

1. **按服务器名称添加前缀。** `notes/search`、`files/search`。清晰但稍显丑陋。
2. **静默先到先得。** 后来的服务器的 `search` 覆盖之前的。有风险；隐藏了冲突。
3. **拒绝冲突。** 拒绝加载第二个服务器；通知用户。对安全敏感的宿主最安全。

Claude Desktop 使用按服务器名称添加前缀。Cursor 使用拒绝冲突并显示明确错误。VS Code MCP 也采用了按服务器名称添加前缀。

### 路由

合并后，调度表将 `tool_name -> session` 映射起来。模型按名称发出调用；客户端找到会话并向该服务器的 stdin 写入 `tools/call` 消息，然后等待响应。

### 采样回调

如果服务器在 `initialize` 时声明了 `sampling` 能力，它可能会发送 `sampling/createMessage`，请求客户端运行其 LLM。客户端必须：

1. 在采样解决之前阻止对该服务器的进一步请求，或如果其实现支持并发则进行流水线处理。
2. 调用其 LLM 提供商。
3. 将响应发回给服务器。

第 11 课端到端讲解采样。本章为完整性作了存根处理。

### 通知处理

`notifications/tools/list_changed` 意味着重新调用 `tools/list`。`notifications/resources/updated` 意味着如果正在使用该资源则重新读取。通知不能产生响应——不要尝试确认它们。

一个常见的客户端 bug：在流中有通知等待时，`tools/call` 阻塞了读取循环。使用后台读取线程将每条消息推入队列；主线程从队列中取出并调度。

### 重连

传输可能失败：服务器崩溃、操作系统杀死进程、stdio 管道断裂。客户端检测 stdout 上的 EOF 并将会话标记为已失效。选项：

- 静默重启服务器并重新握手。适用于纯只读服务器。
- 向用户呈现失败。适用于有用户可见会话的有状态服务器。

Phase 13 · 09 涵盖 Streamable HTTP 的重连语义；stdio 更简单。

### 保活和会话 ID

Streamable HTTP 使用 `Mcp-Session-Id` 头部。Stdio 没有会话 ID——进程身份就是会话。保活 ping 是可选的；stdio 管道在空闲时不会断开。

## 动手使用

`code/main.py` 将三个模拟 MCP 服务器作为子进程生成，与每个握手，合并工具列表，并将工具调用路由到正确的服务器。"服务器"实际上是运行玩具响应器的其他 Python 进程（没有真实 LLM）。运行它可以看到：

- 三次初始化，每次都有自己的能力集。
- 三个 `tools/list` 结果合并为一个 7 工具的命名空间。
- 基于工具名称的路由决策。
- 通过命名空间前缀防止的冲突。

要关注的内容：

- `Session` 数据类干净地持有每个服务器的状态。
- 后台读取线程在不阻塞主线程的情况下排空 stdout 上的每一行。
- 调度表是简单的 `dict[str, Session]`。
- 冲突处理是显式的：当两个服务器声明相同名称时，后来的服务器以前缀重命名。

## 输出产物

本章生成 `outputs/skill-mcp-client-harness.md`。给定 MCP 服务器的声明式列表（名称、命令、参数），该技能生成一个框架，生成服务器、合并工具列表，并提供带冲突解决的路由函数。

## 练习

1. 运行 `code/main.py` 并观察服务器生成日志。用 SIGTERM 杀死其中一个模拟服务器进程，观察客户端如何检测 EOF 并将该会话标记为已失效。

2. 实现命名空间前缀。当两个服务器暴露 `search` 时，将第二个重命名为 `<server>/search`。更新调度表并验证工具调用路由正确。

3. 为服务器重启添加连接池风格的退避：连续失败时指数退避，最多 30 秒，三次失败后向用户发出通知。

4. 草拟一个支持 100 个并发 MCP 服务器的客户端。什么数据结构替换简单的调度 dict？（提示：前缀命名空间用 trie，加上每个服务器工具计数的指标。）

5. 将客户端移植到官方 MCP Python SDK。SDK 封装了 `stdio_client` 和 `ClientSession`。代码应从约 200 行缩减到约 40 行，同时保留多服务器路由。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| MCP 客户端（MCP client） | "智能体宿主" | 生成服务器并编排工具调用的进程。 |
| 会话（Session） | "每个服务器的状态" | 能力、工具列表和待处理请求的记账。 |
| 合并命名空间（Merged namespace） | "一个工具列表" | 跨所有活跃服务器的工具名称的扁平集合。 |
| 命名空间冲突（Namespace collision） | "两个服务器同名工具" | 客户端必须添加前缀、拒绝或先到先得处理重复项。 |
| 路由（Routing） | "谁来处理这个调用？" | 从工具名称到拥有该工具的服务器的调度。 |
| 后台读取器（Background reader） | "非阻塞 stdout" | 将服务器 stdout 排入队列的线程或任务。 |
| 采样回调（Sampling callback） | "LLM 即服务" | 客户端对服务器 `sampling/createMessage` 的处理器。 |
| `notifications/*_changed` | "原语已变更" | 客户端必须重新发现或重新读取的信号。 |
| 重连策略（Reconnection policy） | "当服务器挂掉时" | 传输失败时的重启语义。 |
| Stdio 会话（Stdio session） | "进程即会话" | 没有会话 ID；子进程生命周期就是会话。 |

## 延伸阅读

- [Model Context Protocol — 客户端规范](https://modelcontextprotocol.io/specification/2025-11-25/client) — 规范客户端行为
- [MCP — 快速入门客户端指南](https://modelcontextprotocol.io/quickstart/client) — 使用 Python SDK 的 hello-world 客户端教程
- [MCP Python SDK — 客户端模块](https://github.com/modelcontextprotocol/python-sdk) — 参考 `ClientSession` 和 `stdio_client`
- [MCP TypeScript SDK — 客户端](https://github.com/modelcontextprotocol/typescript-sdk) — TypeScript 对应实现
- [VS Code — 扩展中的 MCP](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code 如何在单个编辑器宿主中多路复用多个 MCP 服务器

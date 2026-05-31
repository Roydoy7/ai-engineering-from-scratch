# MCP 传输——stdio vs Streamable HTTP vs SSE 迁移（MCP Transports — stdio vs Streamable HTTP vs SSE Migration）

> stdio 只在本地可用，无法用于远程。Streamable HTTP（2025-03-26）是远程标准。旧的 HTTP+SSE 传输已被弃用，将于 2026 年中期移除。选错传输意味着一次迁移；选对意味着一个可远程托管、具备会话连续性和 DNS 重绑定防护的 MCP 服务器。

**类型：** 学习  
**语言：** Python（标准库，Streamable HTTP 端点骨架）  
**前置知识：** Phase 13 · 07、08（MCP 服务器与客户端）  
**预计时间：** 约 45 分钟

## 学习目标

- 根据部署形态（本地 vs 远程、单进程 vs 集群）在 stdio 和 Streamable HTTP 之间做出选择。
- 实现 Streamable HTTP 单端点模式：POST 处理请求，GET 建立会话流。
- 强制执行 `Origin` 验证和会话 ID 语义以防止 DNS 重绑定攻击。
- 在 2026 年中期移除截止日期前，将旧版 HTTP+SSE 服务器迁移到 Streamable HTTP。

## 问题所在

第一个 MCP 远程传输（2024-11）是 HTTP+SSE：两个端点，一个用于客户端 POST，一个服务器推送事件（SSE）通道用于服务器到客户端的流。它可以工作，但也很笨拙：每个会话两个端点，在某些 CDN 前缓存损坏，以及对某些 WAF 会主动终止的长连接 SSE 的硬依赖。

2025-03-26 规范用 Streamable HTTP 取而代之：单端点，POST 用于客户端请求，GET 用于建立会话流，两者共享 `Mcp-Session-Id` 头。此后构建或迁移的每个服务器都使用 Streamable HTTP。旧的 SSE 模式正在被弃用——Atlassian Rovo 于 2026 年 6 月 30 日移除了它；Keboola 于 2026 年 4 月 1 日移除；大多数剩余的企业服务器将在 2026 年底前移除。

stdio 仍然适用于本地服务器。Claude Desktop、VS Code 和每个 IDE 形状的客户端都通过 stdio 生成服务器。正确的心智模型：stdio 用于"这台机器"，Streamable HTTP 用于"通过网络"。没有交叉使用。

## 核心概念

### stdio

- 子进程传输。客户端生成服务器，通过 stdin/stdout 通信。
- 每行一个 JSON 对象，换行符分隔。
- 没有会话 ID；进程身份就是会话。
- 无需身份验证（子进程继承父进程的信任边界）。
- 永远不要用于远程服务器——你需要 SSH 或 socat 来隧道，那时候不如直接使用 Streamable HTTP。

### Streamable HTTP

单端点 `/mcp`（或任何路径）。支持三种 HTTP 方法：

- **POST /mcp。** 客户端发送 JSON-RPC 消息。服务器响应单个 JSON 响应，或一个或多个响应的 SSE 流（适用于与该请求相关的批量响应和通知）。
- **GET /mcp。** 客户端打开长连接 SSE 通道。服务器用它发送服务器到客户端的请求（采样、通知、询问）。
- **DELETE /mcp。** 客户端显式终止会话。

会话由 `Mcp-Session-Id` 头标识，该头由服务器在第一次响应时设置，客户端在后续每个请求中回传。会话 ID 必须是密码学随机的（128 位以上）；客户端选择的 ID 出于安全原因会被拒绝。

### 单端点 vs 双端点

旧规范的双端点模式在 2026 年仍可调用——规范将其声明为"旧版兼容"。但所有新服务器都应使用单端点。官方 SDK 发出单端点；只有在与未迁移的远程服务器通信时才使用旧版模式。

### `Origin` 验证和 DNS 重绑定

浏览器不是 MCP 客户端（目前如此），但攻击者可以构造一个网页，说服浏览器向 `localhost:1234/mcp` 发送 POST——用户的本地 MCP 服务器就在那里监听。如果服务器不检查 `Origin`，浏览器的同源策略无法保护它，因为 `Origin: http://evil.com` 是有效的跨源请求。

2025-11-25 规范要求服务器拒绝 `Origin` 不在允许列表上的请求。允许列表通常包含 MCP 客户端宿主（`https://claude.ai`、`vscode-webview://*`）和本地 UI 的 localhost 变体。

### 会话 ID 生命周期

1. 客户端发送第一个请求，不带 `Mcp-Session-Id`。
2. 服务器分配一个随机 ID，在响应头中设置 `Mcp-Session-Id`。
3. 客户端在所有后续请求和 `GET /mcp` 流上回传该头。
4. 会话可以被服务器吊销；客户端在后续请求中看到 404 并必须重新初始化。
5. 客户端可以显式 DELETE 会话以干净关闭。

### 保活和重连

SSE 连接会断开。客户端通过使用相同的 `Mcp-Session-Id` 重新 GET 来重建连接。服务器必须排队在断线期间错过的事件（在合理的窗口内），并通过客户端回传的 `last-event-id` 头重放。

Phase 13 · 13 涵盖任务，可以让长时间运行的工作甚至在完整会话重连后也能存活。

### 向后兼容性探测

想要同时支持新旧服务器的客户端：

1. POST 到 `/mcp`。
2. 如果响应是 `200 OK` 带 JSON 或 SSE，这是 Streamable HTTP。
3. 如果响应是 `200 OK` 带 `Content-Type: text/event-stream` 且有指向辅助端点的 `Location` 头，这是旧版 HTTP+SSE；跟随 `Location`。

### Cloudflare、ngrok 和托管

2026 年生产远程 MCP 服务器运行在 Cloudflare Workers（使用其 MCP Agents SDK）、Vercel Functions 或容器化的 Node/Python 上。关键：你的托管必须支持 SSE GET 的长连接。Vercel 免费版最多 10 秒，不适用。Cloudflare Workers 支持无限期流。

### 网关组合

当你用网关（Phase 13 · 17）在多个 MCP 服务器前面时，网关是一个单一的 Streamable HTTP 端点，负责重写会话 ID 并多路复用上游。工具在网关层合并；客户端看到的是单个逻辑服务器。

### 传输失效模式

- **stdio SIGPIPE。** 子进程在写入时死亡会引发 SIGPIPE；服务器应干净地退出。客户端应检测 EOF 并将会话标记为已失效。
- **HTTP 502/504。** Cloudflare、nginx 和其他代理在上游失败时发出这些。Streamable HTTP 客户端应在短暂退避后重试一次。
- **SSE 连接断开。** TCP RST、代理超时或客户端网络变化关闭流。客户端使用 `Mcp-Session-Id` 和可选的 `last-event-id` 重连以恢复。
- **会话吊销。** 服务器使会话 ID 失效；客户端在下一个请求上看到 404。客户端必须重新握手。
- **时钟偏差。** 客户端的资源 TTL 计算与服务器不同步。客户端应将服务器时间戳视为权威。

### 何时绕过 Streamable HTTP

一些企业在其内部网络中将 MCP 服务器部署在 gRPC 或消息队列传输之后。这是非标准的——MCP 规范没有正式定义这些。网关可以向 MCP 客户端暴露 Streamable HTTP 表面，同时在内部使用 gRPC。保持外部表面符合规范；网关负责翻译。

## 动手使用

`code/main.py` 使用 `http.server`（标准库）实现了一个最小化的 Streamable HTTP 端点。它处理 `/mcp` 上的 POST、GET 和 DELETE，在第一次响应时设置 `Mcp-Session-Id`，验证 `Origin`，并拒绝来自非允许列表来源的请求。处理器复用了第 07 课笔记服务器的调度逻辑。

要关注的内容：

- POST 处理器读取 JSON-RPC 请求体，调度，并写入 JSON 响应（单响应变体；SSE 变体结构类似）。
- `Origin` 检查拒绝默认的 `http://evil.example` 探测，但接受 `http://localhost`。
- 会话 ID 是随机的 128 位十六进制字符串；服务器在内存中保存每个会话的状态。

## 输出产物

本章生成 `outputs/skill-mcp-transport-migrator.md`。给定 HTTP+SSE（旧版）MCP 服务器，该技能生成迁移到 Streamable HTTP 的计划，包含会话 ID 连续性、Origin 检查和向后兼容探测支持。

## 练习

1. 运行 `code/main.py`。用 `curl` POST 一个 `initialize` 并观察 `Mcp-Session-Id` 响应头。POST 第二个请求时回传该头，并验证会话连续性。

2. 添加一个打开 SSE 流的 GET 处理器。每五秒发送一个 `notifications/progress` 事件。通过使用相同会话 ID 重新 GET 来重连，并确认服务器接受它。

3. 实现 `last-event-id` 重放逻辑。重连时，重放自该 ID 以来生成的所有事件。

4. 扩展 `Origin` 验证以支持通配符模式（`https://*.example.com`），并确认它接受 `https://app.example.com` 但拒绝 `https://evil.example.com.attacker.net`。

5. 从官方注册表中取一个旧版 HTTP+SSE 服务器（有好几个），草拟迁移方案：端点处理、会话 ID 生成和头语义有哪些变化。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| stdio 传输（stdio transport） | "本地子进程" | 通过 stdin/stdout 的 JSON-RPC，换行符分隔。 |
| Streamable HTTP | "远程传输" | 单端点 POST + GET + 可选 SSE，2025-03-26 规范。 |
| HTTP+SSE | "旧版" | 将于 2026 年中期移除的双端点模型。 |
| `Mcp-Session-Id` | "会话头" | 服务器分配的随机 ID，在每个后续请求中回传。 |
| `Origin` 允许列表（`Origin` allowlist） | "DNS 重绑定防御" | 拒绝来源不在允许列表中的请求。 |
| 单端点（Single endpoint） | "一个 URL" | `/mcp` 处理所有会话操作的 POST/GET/DELETE。 |
| `last-event-id` | "SSE 重放" | 用于恢复断开流而不丢失事件的头。 |
| 向后兼容探测（Backwards-compat probe） | "新旧检测" | 客户端响应形状检查，自动选择传输方式。 |
| 长连接 HTTP（Long-lived HTTP） | "SSE 流" | 服务器在一个 TCP 连接上推送分钟或小时级别的事件。 |
| 会话吊销（Session revocation） | "强制重初始化" | 服务器使会话 ID 失效；客户端必须重新握手。 |

## 延伸阅读

- [MCP — 基础传输规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio 和 Streamable HTTP 的规范参考
- [MCP — 基础传输规范 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — 引入 Streamable HTTP 的修订版
- [Cloudflare — MCP 传输](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Workers 托管的 Streamable HTTP 模式
- [AWS — MCP 传输机制](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — 跨部署形态的比较
- [Atlassian — HTTP+SSE 弃用通知](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 具体迁移截止日期示例

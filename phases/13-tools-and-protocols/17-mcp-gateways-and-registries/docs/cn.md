# MCP 网关与注册表——企业控制平面（MCP Gateways and Registries — Enterprise Control Planes）

> 企业无法让每个开发者随意安装 MCP 服务器。网关集中处理认证、RBAC、审计、速率限制、缓存和工具投毒检测，然后将合并后的工具面以单个 MCP 端点的形式暴露。官方 MCP 注册表（Anthropic + GitHub + PulseMCP + Microsoft，命名空间经过验证）是规范的上游来源。本章说明网关的位置，讲解最小化实现，并概览 2026 年的供应商格局。

**类型：** 学习  
**语言：** Python（标准库，最小化网关）  
**前置知识：** Phase 13 · 15（工具投毒）、Phase 13 · 16（OAuth 2.1）  
**预计时间：** 约 45 分钟

## 学习目标

- 解释 MCP 网关的位置（在 MCP 客户端和多个后端 MCP 服务器之间）。
- 实现五个网关职责：认证、RBAC、审计、速率限制、策略。
- 在网关层执行已锁定工具哈希的清单。
- 区分官方 MCP 注册表和元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题所在

一个财富 500 强企业有 30 个经批准的 MCP 服务器、5000 名开发者、合规和审计要求，以及一个希望集中策略的安全团队。让每个开发者在自己的 IDE 中随意安装服务器根本行不通。

网关模式：

1. 网关作为开发者连接的单个 Streamable HTTP 端点运行。
2. 网关持有每个后端 MCP 服务器的凭据。
3. 每个开发者请求都通过网关自己的 OAuth 进行认证和范围限定。
4. 网关将调用路由到后端服务器，并应用策略。
5. 所有调用都记录用于审计。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway——这些都在 2025-2026 年发布了网关或网关功能。

与此同时，官方 MCP 注册表作为规范的上游发布：精选的、命名空间经过验证的、以反向 DNS 命名的服务器，网关可以从中拉取。元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）从多个来源聚合服务器。

## 核心概念

### 五个网关职责

1. **认证（Auth）。** OAuth 2.1 识别开发者；映射到用户角色。
2. **RBAC。** 每用户策略：哪些服务器、哪些工具、哪些范围。
3. **审计（Audit）。** 记录每个调用的人员、内容、时间、结果。
4. **速率限制（Rate limit）。** 每用户/每工具/每服务器上限，防止滥用。
5. **策略（Policy）。** 拒绝毒化描述，执行二规则，编辑 PII。

### 网关作为单个端点

对开发者来说，网关看起来像一个 MCP 服务器。内部它路由到 N 个后端。会话 ID（Phase 13 · 09）在边界处被重写。

### 凭据保管

开发者永远看不到后端令牌。网关持有它们（或代理到持有它们的身份提供商）。在网关上拥有 `notes:read` 的开发者可能通过网关自己的后端凭据传递性地访问笔记 MCP 服务器——但只有在策略绑定了传递访问的情况下。

### 网关层的工具哈希锁定

网关持有已批准工具描述的清单（SHA256 哈希）。在发现时，它获取每个后端的 `tools/list`，将哈希与清单比较，并移除任何描述已变更的工具。这是 Phase 13 · 15 中集中应用的地毯拉出防御。

### 策略即代码

高级网关用 OPA/Rego、Kyverno 或 Styra 表达策略。像"用户 `alice` 只能在 `acme` 组织的仓库上调用 `github.open_pr`"这样的规则以声明式编码。简单网关使用手工编码的 Python。两种形式都有效。

### 感知会话的路由

当用户的会话包含多个服务器的混合时，网关进行多路复用：开发者的单个 MCP 会话持有 N 个后端会话，每个服务器一个。来自任何后端的通知都通过网关路由到开发者的会话。

### 命名空间合并

网关合并所有后端的工具命名空间，通常在冲突时添加前缀。`github.open_pr`、`notes.search`。这使路由无歧义。

### 注册表

- **官方 MCP 注册表（`registry.modelcontextprotocol.io`）。** 在 Anthropic、GitHub、PulseMCP、Microsoft 管理下发布。命名空间经过验证（反向 DNS：`io.github.user/server`）。经过基本质量预过滤。
- **Glama。** 以搜索为中心的元注册表，聚合多个来源。
- **MCPMarket。** 偏商业性的目录，包含供应商列表。
- **MCP.so。** 社区目录；开放提交。
- **Smithery。** 包管理器风格的安装流程。
- **LobeHub。** 在其 LobeChat 应用中集成 UI 的注册表。

企业网关默认从官方注册表拉取，允许管理员从元注册表精选添加，并拒绝任何未锁定的内容。

### 反向 DNS 命名

官方注册表为公共服务器强制要求反向 DNS 名称：`io.github.alice/notes`。命名空间防止抢注，并使信任委托更清晰。

### 供应商概览，2026 年 4 月

| 供应商 | 优势 |
|--------|------|
| Cloudflare MCP Portals | 边缘托管；集成 OAuth；免费层 |
| Kong AI Gateway | Kubernetes 原生；细粒度策略；日志到 OpenTelemetry |
| IBM ContextForge | 企业 IAM；合规；审计导出 |
| TrueFoundry | 偏 DevOps；以指标为先 |
| MintMCP | 面向开发者平台 |
| Envoy AI Gateway | 开源；可自定义过滤器 |

Phase 17（生产基础设施）更深入地讲解网关操作。

## 动手使用

`code/main.py` 提供了约 150 行的最小化网关：通过假 Bearer 令牌认证用户，持有每用户 RBAC 策略，将请求路由到两个后端 MCP 服务器，将每个调用写入审计日志，执行速率限制，并拒绝任何描述哈希与锁定清单不匹配的后端工具。

要关注的内容：

- `RBAC` 字典以 `user_id` 为键，包含允许的 `server_tool` 条目。
- `AUDIT_LOG` 是一个仅追加的事件列表。
- 速率限制对每个用户使用令牌桶。
- 锁定清单是 `server::tool -> hash` 的字典。

## 输出产物

本章生成 `outputs/skill-gateway-bootstrap.md`。给定企业 MCP 计划（用户、后端、合规性），该技能生成网关配置规范。

## 练习

1. 运行 `code/main.py`。以允许的用户发出调用；然后以不允许的用户；然后进行超出速率限制的爆发。验证所有三个流程。

2. 添加一个在返回客户端之前编辑结果中 PII 的策略。使用简单的正则表达式传递 SSN 形状的字符串；注意缺口（电子邮件、电话号码）。

3. 扩展审计日志以发出 OpenTelemetry GenAI spans。Phase 13 · 20 涵盖了确切的属性。

4. 为有五个后端（笔记、GitHub、Postgres、Jira、Slack）的 50 人开发团队设计 RBAC 策略。谁在每个上有只读权限？谁有写权限？

5. 从头到尾阅读 Cloudflare 企业 MCP 文章。找出 Cloudflare 提供但这个标准库网关没有的一项功能。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 网关（Gateway） | "MCP 代理" | 客户端和后端之间的集中服务器。 |
| 凭据保管（Credential vaulting） | "后端令牌留在服务器端" | 开发者永远看不到上游令牌。 |
| 感知会话的路由（Session-aware routing） | "多后端会话" | 网关为每个开发者会话多路复用 N 个后端会话。 |
| 工具哈希锁定（Tool-hash pinning） | "已批准清单" | 每个已批准工具描述的 SHA256；集中阻止地毯拉出。 |
| RBAC | "每用户策略" | 工具和服务器的基于角色的访问控制。 |
| 策略即代码（Policy-as-code） | "声明式规则" | 在网关执行的 OPA/Rego、Kyverno、Styra 策略。 |
| 审计日志（Audit log） | "人员、内容、时间" | 用于合规的仅追加事件日志。 |
| 速率限制（Rate limit） | "每用户令牌桶" | 每分钟上限，防止滥用。 |
| 官方 MCP 注册表（Official MCP Registry） | "规范上游" | `registry.modelcontextprotocol.io`，命名空间经过验证。 |
| 反向 DNS 命名（Reverse-DNS naming） | "注册表命名空间" | `io.github.user/server` 约定。 |

## 延伸阅读

- [官方 MCP 注册表](https://registry.modelcontextprotocol.io/) — 规范上游，命名空间经过验证
- [Cloudflare — 企业 MCP](https://blog.cloudflare.com/enterprise-mcp/) — 带 OAuth 和策略的网关模式
- [agentic-community — MCP 网关注册表](https://github.com/agentic-community/mcp-gateway-registry) — 开源参考网关
- [TrueFoundry — 什么是 MCP 网关？](https://www.truefoundry.com/blog/what-is-mcp-gateway) — 功能比较文章
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) — IBM 的企业网关

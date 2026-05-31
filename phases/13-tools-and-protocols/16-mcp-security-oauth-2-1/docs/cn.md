# MCP 安全 II——OAuth 2.1、资源指示器、增量范围（MCP Security II — OAuth 2.1, Resource Indicators, Incremental Scopes）

> 远程 MCP 服务器需要的是授权，而不仅仅是认证。2025-11-25 规范与 OAuth 2.1 + PKCE + 资源指示器（RFC 8707）+ 受保护资源元数据（RFC 9728）保持一致。SEP-835 通过在 403 WWW-Authenticate 上的逐步升级授权添加了增量范围同意。本章将逐步升级流程作为状态机实现，让你能看清每一跳。

**类型：** 构建  
**语言：** Python（标准库，OAuth 状态机模拟器）  
**前置知识：** Phase 13 · 09（传输）、Phase 13 · 15（安全 I）  
**预计时间：** 约 75 分钟

## 学习目标

- 区分资源服务器和授权服务器的职责。
- 逐步讲解受 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource`（RFC 8707）和受保护资源元数据（RFC 9728）防止混淆代理攻击。
- 实现逐步升级授权：服务器以 403 + WWW-Authenticate 响应，请求更高范围；客户端重新提示用户同意并重试。

## 问题所在

早期 MCP（2025 年前）以临时 API 密钥甚至没有认证来部署远程服务器。2025-11-25 规范用完整的 OAuth 2.1 配置文件弥补了这一差距。

三个真实世界的需求：

- **普通远程服务器。** 用户安装一个访问其 Notion/GitHub/Gmail 的远程 MCP 服务器。带 PKCE 的 OAuth 2.1 是正确的形状。
- **范围升级。** 被授予 `notes:read` 的笔记服务器以后可能需要 `notes:write` 来执行特定操作。逐步升级（SEP-835）请求额外范围，而不是重新做整个流程。
- **混淆代理预防。** 客户端持有受众范围为服务器 A 的令牌。服务器 A 是恶意的，试图将令牌呈现给服务器 B。资源指示器（RFC 8707）将令牌锁定到其预期受众。

OAuth 2.1 不是新的。新的是 MCP 的配置文件：特定的必需流程（仅限授权码 + PKCE；默认不允许隐式流或客户端凭据），每个令牌请求都必须使用资源指示器，以及发布受保护资源元数据让客户端知道去哪里。

## 核心概念

### 角色

- **客户端。** MCP 客户端（Claude Desktop、Cursor 等）。
- **资源服务器。** MCP 服务器（笔记、GitHub、Postgres 等）。
- **授权服务器。** 颁发令牌。可以与资源服务器是同一服务，或是单独的 IdP（Auth0、Keycloak、Cognito）。

在 MCP 的配置文件中，资源服务器和授权服务器可以是同一主机，但应该通过 URL 加以区分。

### 授权码 + PKCE

流程：

1. 客户端生成 `code_verifier`（随机）和 `code_challenge`（SHA256）。
2. 客户端将用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意。授权服务器重定向到 `redirect_uri?code=...`。
4. 客户端 POST 到 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`。
5. 授权服务器将验证器的哈希与存储的挑战进行验证，并颁发访问令牌。
6. 客户端使用令牌：在每个资源服务器请求中使用 `Authorization: Bearer ...`。

PKCE 防止授权码拦截攻击。资源指示器防止令牌在其他地方有效。

### 受保护资源元数据（RFC 9728）

资源服务器发布 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端从资源服务器发现授权服务器。减少了配置——客户端只需要资源 URL。

### 资源指示器（RFC 8707）

令牌请求中的 `resource` 参数将令牌的预期受众锁定。颁发的令牌包含 `aud: "https://notes.example.com"`。另一个接收此令牌的 MCP 服务器检查 `aud` 并拒绝它。

### 范围模型

范围是空格分隔的字符串。常见 MCP 约定：

- `notes:read`、`notes:write`、`notes:delete`
- `admin:*` 用于管理员能力（谨慎使用）
- `profile:read` 用于身份

范围选择应遵循最小权限原则：请求你现在需要的，在需要更多时逐步升级。

### 逐步升级授权（SEP-835）

用户授予 `notes:read`。他们后来要求智能体删除一条笔记。服务器响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端看到 `insufficient_scope` 错误，向用户显示额外范围的同意对话框，为其执行一个迷你 OAuth 流程，并用新令牌重试请求。

### 令牌受众验证

每个请求：服务器检查 `token.aud == self.resource_url`。不匹配 = 401。这阻止了跨服务器令牌复用。

### 短期令牌和轮换

访问令牌应该是短期的（默认 1 小时）。刷新令牌在每次刷新时轮换。客户端在后台处理静默刷新。

### 禁止令牌传递

采样服务器（Phase 13 · 11）不得将客户端的令牌传递给其他服务。采样请求是边界。

### 混淆代理预防

令牌绑定到 `aud`。客户端绑定到 `client_id`。每个请求都针对两者进行验证。规范明确禁止了在 MCP 之前的远程工具生态系统中常见的旧式"传递令牌"模式。

### 客户端 ID 发现

每个 MCP 客户端在固定 URL 发布其元数据。授权服务器可以获取客户端的元数据文档以发现重定向 URI 和联系信息。这消除了手动客户端注册。

### 网关与 OAuth

Phase 13 · 17 展示了企业网关如何处理 OAuth：网关持有上游服务器的凭据，对客户端的令牌是网关颁发的，上游令牌永远不会离开网关。这翻转了信任模型——用户向网关认证一次；网关处理 N 个服务器授权。

## 动手使用

`code/main.py` 将完整的 OAuth 2.1 逐步升级流程作为状态机模拟。它实现了：

- PKCE 代码验证器/挑战生成。
- 带资源指示器的授权码流程。
- 受保护资源元数据端点。
- 带受众检查的令牌验证。
- 在 `insufficient_scope` 时的逐步升级。

本课中没有 HTTP 服务器；状态机在内存中运行，以便你可以追踪每一跳。Phase 13 · 17 的网关课将其连接到实际传输。

## 输出产物

本章生成 `outputs/skill-oauth-scope-planner.md`。给定带工具的远程 MCP 服务器，该技能设计范围集、锁定规则和逐步升级策略。

## 练习

1. 运行 `code/main.py`。追踪两个范围的逐步升级流程。注意哪些跳在逐步升级时重复。

2. 添加刷新令牌轮换：每次刷新都颁发新的刷新令牌并使旧的失效。模拟被盗的刷新令牌在轮换后被使用，并确认它失败。

3. 使用标准库 `http.server` 将受保护资源元数据端点实现为真实的 HTTP 响应。镜像第 09 课的 `/mcp` 端点。

4. 为 GitHub MCP 服务器设计一个范围层次结构：读取仓库、写 PR、批准 PR、合并 PR、管理员。在每个级别之间使用逐步升级。

5. 阅读 RFC 8707 和 RFC 9728。找出 9728 中 MCP 与 RFC 示例使用方式不同的一个字段。（提示：它与 `scopes_supported` 有关。）

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| OAuth 2.1 | "现代 OAuth" | 强制要求 PKCE 并禁止隐式流的综合 RFC。 |
| PKCE | "所有权证明" | 代码验证器 + 挑战，防御授权码拦截。 |
| 资源指示器（Resource indicator） | "令牌受众" | RFC 8707 `resource` 参数，将令牌锁定到一个服务器。 |
| 受保护资源元数据（Protected-resource metadata） | "发现文档" | RFC 9728 `.well-known/oauth-protected-resource`。 |
| 逐步升级授权（Step-up authorization） | "增量同意" | SEP-835 按需添加范围的流程。 |
| `insufficient_scope` | "带 WWW-Authenticate 的 403" | 服务器要求重新同意更大范围的信号。 |
| 混淆代理（Confused deputy） | "跨服务令牌复用" | 受信任的持有者不当转发令牌的攻击。 |
| 短期令牌（Short-lived token） | "访问令牌 TTL" | 快速过期的 Bearer；刷新令牌负责续期。 |
| 范围层次（Scope hierarchy） | "最小权限栈" | 带有级别间逐步升级的分级范围集。 |
| 客户端 ID 元数据（Client ID metadata） | "客户端发现文档" | 客户端发布其 OAuth 元数据的 URL。 |

## 延伸阅读

- [MCP — 授权规范](https://modelcontextprotocol.io/specification/draft/basic/authorization) — MCP OAuth 配置文件规范
- [den.dev — MCP 11 月授权规范](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 变更演练
- [RFC 8707 — OAuth 2.0 的资源指示器](https://datatracker.ietf.org/doc/html/rfc8707) — 受众锁定 RFC
- [RFC 9728 — OAuth 2.0 受保护资源元数据](https://datatracker.ietf.org/doc/html/rfc9728) — 发现文档 RFC
- [Aembit — MCP OAuth 2.1、PKCE 和 AI 授权的未来](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 实用的逐步升级流程演练

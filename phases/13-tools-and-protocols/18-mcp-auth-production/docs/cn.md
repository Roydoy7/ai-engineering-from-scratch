# 生产环境中的 MCP 认证——DCR、JWKS 轮换、iii 原语上的受众锁定令牌（MCP Auth in Production — DCR, JWKS Rotation, Audience-Pinned Tokens on iii Primitives）

> 第 16 课在内存中搭建了 OAuth 2.1 状态机。到 2026 年，你向真实组织发布的每个 MCP 服务器都需要生产认证：动态客户端注册（RFC 7591）、授权服务器元数据发现（RFC 8414）、不会在凌晨 3 点中断令牌验证的 JWKS 轮换，以及拒绝混淆代理复用的受众锁定令牌。本章将这一切通过 iii 原语串联起来——`iii.registerTrigger` 用于 HTTP 和 cron，`iii.registerFunction` 用于认证逻辑，`state::set/get` 用于缓存密钥——使认证面像引擎中的其他工作负载一样可观测、可重启、可回放。

**类型：** 构建  
**语言：** Python（标准库，为课程环境模拟的 iii 原语）  
**前置知识：** Phase 13 · 16（OAuth 2.1 状态机）、Phase 13 · 17（网关）  
**预计时间：** 约 90 分钟

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器并验证契约。
- 实现 RFC 7591 动态客户端注册，使 MCP 客户端无需管理员干预即可注册。
- 使用 cron 触发器缓存和轮换 JWKS 密钥，使签名验证能够在密钥更换时继续工作。
- 使用 RFC 8707 资源指示器将令牌锁定到单个 MCP 资源，并拒绝混淆代理复用。
- 将每个端点和后台作业串联为 iii 原语——HTTP 触发器、cron 触发器、命名函数和 `state::*` 读取——使单次重启能够重建认证面。
- 读取 IdP 能力矩阵，当 IdP 无法满足 MCP 认证配置文件时拒绝部署。

## 问题所在

第 16 课的模拟器在内存中运行 OAuth 2.1。生产环境有三个仅用内存模拟器看不到的运营差距。

第一个差距是注册。真实组织运行数百个 MCP 服务器和数千个 MCP 客户端。操作员不会手动将每个 Cursor 用户注册为 OAuth 客户端。RFC 7591 动态客户端注册让客户端向授权服务器 `POST /register` 并立即收到 `client_id`（以及可选的 `client_secret`）。服务器在其 RFC 8414 元数据中发布 `registration_endpoint`；客户端无需带外配置就能发现它。

第二个差距是密钥轮换。JWT 验证依赖授权服务器的签名密钥，以 JSON Web Key Set（JWKS）的形式发布。授权服务器按计划轮换这些密钥（通常每小时，有时在事件响应下更快）。在启动时获取一次 JWKS 的 MCP 服务器验证正常，直到轮换窗口——然后每个请求都会失败，直到重启。生产环境将 JWKS 串联为带有刷新作业的缓存值，该作业在前一个密钥过期之前覆盖缓存，加上在缓存丢失时的回退获取，用于处理由比缓存更新的密钥签名的令牌到达的情况。

第三个差距是受众绑定。第 16 课介绍了 RFC 8707 资源指示器。在生产中，该指示器成为每个请求的硬性声明检查。MCP 服务器将 `token.aud` 与其自身的规范资源 URL 进行比较，并以 HTTP 401 拒绝不匹配。这是在同一信任网格中对抗上游 MCP 服务器（或持有针对一个服务器的令牌的恶意客户端）将该令牌重放到另一个服务器的唯一防御。

本章将这些差距中的每一个都视为 iii 原语。元数据文档是一个返回函数输出的 HTTP 触发器。JWKS 轮换是一个调用 `auth::rotate-jwks` 的 cron 触发器，它写入 `state::set("auth/jwks/<issuer>", ...)`。JWT 验证是其他通过 `iii.trigger("auth::validate-jwt", token)` 调用的函数。MCP 服务器本身只是另一个在调度之前调用验证的 HTTP 触发器。重启引擎：触发器注册表重建；状态存活；认证面无需手动对账即可运行。

## 核心概念

### RFC 8414——OAuth 授权服务器元数据

`/.well-known/oauth-authorization-server` 处的文档描述了客户端需要的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

给定 MCP 资源 URL 的客户端链式发现：RFC 9728 中的 `oauth-protected-resource`（资源服务器文档）命名了发行者，然后 `oauth-authorization-server`（本 RFC）命名了每个端点。客户端永远不需要硬编码授权 URL。

在为 MCP 信任 IdP 之前你要验证的契约：

- `code_challenge_methods_supported` 包含 `S256`（RFC 7636 的 PKCE）。
- `grant_types_supported` 包含 `authorization_code` 并拒绝 `password` 和 `implicit`。
- `registration_endpoint` 存在（RFC 7591 支持）。
- 对于 OAuth 2.1，`response_types_supported` 正好是 `["code"]`。

如果任何一项缺失，MCP 服务器拒绝针对此 IdP 部署。是部署清单错了，不是代码。

### RFC 9728（回顾）——受保护资源元数据

第 16 课涵盖了 RFC 9728。在生产中的增量：这个文档是客户端查找被*这个* MCP 服务器信任的授权服务器的唯一地方。单个 MCP 服务器可以接受来自多个 IdP 的令牌（一个用于员工，一个用于合作伙伴）。RFC 9728 声明了该集合；RFC 8414 记录了每个 IdP 支持的内容。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### RFC 7591——动态客户端注册

没有 DCR，每个 MCP 客户端（Cursor、Claude Desktop、自定义智能体）都需要与 IdP 管理员进行带外交换。有了 DCR，客户端发布：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器响应 `client_id` 和用于后续更新的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` 是在用户设备上运行的 MCP 客户端的正确默认值。它们只获得 `client_id`——没有可以泄露的 `client_secret`。PKCE 提供了公共客户端需要的所有权证明。

三个生产陷阱：

- 注册端点必须按源 IP 速率限制。没有这个，恶意行为者可以用脚本注册数百万个假客户端并耗尽 `client_id` 命名空间。iii 使这变得简单：注册 HTTP 触发器在调度到注册器之前调用 `auth::rate-limit` 函数。
- `software_statement`（为客户端背书的签名 JWT）被一些企业 IdP 要求。本课的模拟跳过了它；生产连接了一个验证步骤，拒绝来自非 localhost 重定向 URI 的任何未签名注册。
- `registration_access_token` 必须存储为哈希，而非明文。这个令牌的盗窃意味着攻击者可以重写客户端的重定向 URI。

### RFC 8707（回顾）——资源指示器

第 16 课建立了形状。生产规则：每个令牌请求都包含 `resource=<canonical-mcp-url>`，MCP 服务器在每次调用时验证 `token.aud` 与其自身资源 URL 匹配。如果 MCP 服务器可在 `https://notes.example.com/mcp` 访问，规范 URL 是 `https://notes.example.com`——路径组件被排除，以便单个服务器在一个受众下托管多条路径。

### JWKS 轮换模式与 iii

生产失效模式是过期的 JWKS 缓存。用 cron 触发器和 `state::*` 缓存解决：

```python
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *", "name": "auth::jwks-refresh"},
    "auth::rotate-jwks",
)
```

每六小时，cron 触发器调用 `auth::rotate-jwks`，它获取 `<issuer>/.well-known/jwks.json` 并写入 `state::set("auth/jwks/<issuer>", {keys, fetched_at})`。验证器从 `state::get` 读取。`kid` 在缓存中缺失的令牌触发同步的 `auth::rotate-jwks` 调用作为回退。这同时处理两种情况：计划轮换（cron）和密钥重叠窗口（同步回退）。

### IdP 能力矩阵

并非每个 IdP 都支持完整的 MCP 配置文件。下表记录了截至 2025-11-25 规范的实际能力声明。这是一个*部署门控*，而非建议。

| IdP 类别 | RFC 8414 元数据 | RFC 7591 DCR | RFC 8707 资源 | RFC 7636 S256 PKCE | 备注 |
|---|---|---|---|---|---|
| 自托管（Keycloak） | 是 | 是 | 是（自 24.x 起） | 是 | 本课 MCP 配置文件的参考 IdP；端到端支持所有 RFC。 |
| 企业 SSO（Microsoft Entra ID） | 是 | 是（高级层） | 是 | 是 | DCR 可用性因租户层级而异；在部署前在目标租户中验证。 |
| 企业 SSO（Okta） | 是 | 是（Okta CIC/Auth0） | 是 | 是 | DCR 在 Auth0（现 Okta CIC）上可用；经典 Okta 组织需要管理员预注册。 |
| 社交登录 IdP（通用） | 不同 | 很少 | 很少 | 是 | 大多数社交 IdP 将客户端视为静态合作伙伴；不要依赖 DCR。仅用作身份来源，在其上添加你自己的 MCP 感知授权服务器。 |
| 自定义/自建 | 取决于 | 取决于 | 取决于 | 取决于 | 如果你自己发布，发布完整配置文件。跳过上述任何一个 RFC 都会破坏 MCP 认证契约。 |

部署清单的拒绝规则：如果所选 IdP 不返回 `registration_endpoint` 且不在 `code_challenge_methods_supported` 中列出 `S256`，MCP 服务器拒绝启动。没有降级模式。

### iii 原语串联

五个原语组成认证面：

```python
# 1. RFC 8414 元数据文档
iii.registerTrigger(
    "http",
    {"path": "/.well-known/oauth-authorization-server", "method": "GET"},
    "auth::serve-asm",
)

# 2. RFC 7591 动态客户端注册
iii.registerTrigger(
    "http",
    {"path": "/register", "method": "POST"},
    "auth::register-client",
)

# 3. JWT 验证作为可调用函数（资源服务器触发它）
iii.registerFunction("auth::validate-jwt", validate_jwt_handler)

# 4. 增量范围的逐步升级颁发（来自第 16 课的 SEP-835）
iii.registerFunction("auth::issue-step-up", issue_step_up_handler)

# 5. Cron 驱动的 JWKS 轮换
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *"},
    "auth::rotate-jwks",
)
iii.registerFunction("auth::rotate-jwks", rotate_jwks_handler)
```

MCP 服务器本身永远不直接调用验证。它做：

```python
result = iii.trigger("auth::validate-jwt", {"token": bearer_token, "resource": self.resource})
if not result["valid"]:
    return {"status": 401, "WWW-Authenticate": result["www_authenticate"]}
```

这个间接层是 iii 的赌注。明天你将验证器换成并行咨询两个 IdP 的扇出，或者添加 span 发射器，或者缓存正向验证。MCP 服务器不变。

## 动手使用

`code/main.py` 用标准库 Python 和一个小型 `iii_mock` 注册表走完整个生产流程，该注册表模拟 `iii.registerFunction`、`iii.registerTrigger`、`iii.trigger` 和 `state::set/get`。流程：

1. 授权服务器在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 元数据。
2. MCP 客户端调用元数据端点，发现注册端点。
3. MCP 客户端 POST 到 `/register`（RFC 7591）并收到 `client_id`。
4. MCP 客户端运行带资源指示器（RFC 8707）的 PKCE 保护授权码流程（RFC 7636）。
5. MCP 客户端用 `Authorization: Bearer ...` 调用 MCP 服务器上的工具。
6. MCP 服务器触发 `auth::validate-jwt`，它从 `state::get` 读取 JWKS。
7. cron 触发器触发 `auth::rotate-jwks`，替换 state 中的 JWKS。
8. 下一个调用针对新密钥验证而无需重启。
9. 针对不同 MCP 资源的混淆代理尝试得到受众不匹配的 401。

这里的模拟 JWT 使用带共享密钥的 HS256（使课程仅在标准库上运行）。生产使用上述 JWKS 模式的 RS256 或 EdDSA；验证逻辑其他方面完全相同。

## 输出产物

本章生成 `outputs/skill-mcp-auth-iii.md`。给定 MCP 服务器配置和 IdP 能力集，该技能发出要注册的 iii 原语、JWKS 轮换计划、范围映射，以及当 IdP 不支持完整 RFC 配置文件时要应用的拒绝规则。

## 练习

1. 运行 `code/main.py`。追踪 9 步流程。注意 `state::get` 在 `auth::rotate-jwks` 覆盖之前立即返回过期数据的位置，以及下一个请求如何针对新密钥验证。

2. 在受保护资源元数据的 `authorization_servers` 列表中添加新的 IdP。颁发由新 IdP 签名的令牌并确认验证器接受它。颁发由未列出的 IdP 签名的令牌并确认验证器以 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"` 拒绝。

3. 将 `auth::rate-limit` 实现为 iii 函数，并在注册器运行之前从注册 HTTP 触发器内部调用它。使用 `state::set("auth/ratelimit/<ip>", ...)` 中持有的每源 IP 令牌桶。

4. 阅读 RFC 7591，找出本课 `/register` 处理器未验证的两个字段。添加验证。（提示：`software_statement` 和 `redirect_uris` URI 方案。）

5. 阅读 MCP 规范 2025-11-25 授权章节。找出本课验证器当前未发出的 `WWW-Authenticate` 头的一个规范性要求。添加它。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| ASM | "OAuth 元数据文档" | RFC 8414 `/.well-known/oauth-authorization-server` JSON。 |
| DCR | "自助服务客户端注册" | RFC 7591 `POST /register` 流程。 |
| JWKS | "JWT 验证公钥" | JSON Web Key Set，从 `jwks_uri` 获取，按 `kid` 索引。 |
| 资源指示器（Resource indicator） | "受众参数" | RFC 8707 `resource` 参数，将令牌锁定到一个服务器。 |
| `aud` 声明 | "受众" | 验证器与规范资源 URL 比较的 JWT 声明。 |
| 混淆代理（Confused deputy） | "令牌重放" | 为服务器 A 颁发的令牌被呈现给服务器 B 的攻击。 |
| `iss` 允许列表 | "受信任的授权服务器" | 受保护资源元数据 `authorization_servers` 中命名的集合。 |
| 密钥轮换（Key rotation） | "滚动 JWKS" | 带重叠窗口的签名密钥定期更换。 |
| 公共客户端（Public client） | "原生或浏览器客户端" | 没有 `client_secret` 的 OAuth 客户端；PKCE 作为补偿。 |
| `WWW-Authenticate` | "401/403 响应头" | 携带驱动客户端恢复的 `Bearer error=...` 指令。 |

## 延伸阅读

- [MCP — 授权规范（2025-11-25）](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 本课实现的 MCP 认证配置文件
- [RFC 8414 — OAuth 2.0 授权服务器元数据](https://datatracker.ietf.org/doc/html/rfc8414) — 发现契约
- [RFC 7591 — OAuth 2.0 动态客户端注册协议](https://datatracker.ietf.org/doc/html/rfc7591) — DCR
- [RFC 7636 — 代码交换证明密钥（PKCE）](https://datatracker.ietf.org/doc/html/rfc7636) — 公共客户端所有权证明
- [RFC 8707 — OAuth 2.0 的资源指示器](https://datatracker.ietf.org/doc/html/rfc8707) — 受众锁定
- [RFC 9728 — OAuth 2.0 受保护资源元数据](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现
- [OAuth 2.1 草案](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 综合 OAuth 基础

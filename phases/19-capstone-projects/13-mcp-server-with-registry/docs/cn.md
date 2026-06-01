# 压轴项目 13——带注册表和治理的 MCP 服务器（Capstone 13 — MCP Server with Registry and Governance）

> 模型上下文协议在 2026 年不再只是未来，而成为默认的工具使用规范。Anthropic、OpenAI、Google 和每家主要 IDE 都内置了 MCP 客户端。Pinterest 发布了其内部 MCP 服务器生态系统。AAIF 注册表在 `.well-known` 正式化了能力元数据。AWS ECS 发布了参考无状态部署。Block 的 goose-agent 将同样的协议放入了托管助手中。2026 年的生产形态是：StreamableHTTP 传输、OAuth 2.1 范围、OPA 策略门控，以及让平台团队发现、验证和启用服务器的注册表。端到端构建这一切。

**类型：** 压轴项目  
**语言：** Python（服务器，通过 FastMCP）或 TypeScript（@modelcontextprotocol/sdk），Go（注册表服务）  
**前置知识：** Phase 11（LLM 工程）、Phase 13（工具和 MCP）、Phase 14（智能体）、Phase 17（基础设施）、Phase 18（安全）  
**涉及的阶段：** P11 · P13 · P14 · P17 · P18  
**预计时间：** 25 小时

## 问题所在

MCP 成为工具使用的通用语言。Claude Code、Cursor 3、Amp、OpenCode、Gemini CLI 和每个托管智能体现在都使用 MCP 服务器。生产挑战不在于编写服务器（FastMCP 让这变得容易），而在于以企业要求进行大规模部署：每租户 OAuth 范围、破坏性工具上的 OPA 策略、StreamableHTTP 无状态扩展、发现注册表、每次工具调用的审计日志。Pinterest 的内部 MCP 生态系统和 AAIF 注册表规范设定了 2026 年的标准。

你将构建一个 MCP 服务器，暴露 10 个内部工具（Postgres 只读、S3 列表、Jira、Linear、Datadog 等），一个平台发现的注册表 UI，以及破坏性工具的人工批准门控。负载测试演示 StreamableHTTP 水平扩展。审计轨迹满足企业安全审查。

## 核心概念

MCP 2026 修订版将 StreamableHTTP 指定为默认传输。与早期的 stdio-and-SSE 形态不同，StreamableHTTP 默认是无状态的：单个 HTTP 端点接受 JSON-RPC 请求，流式传输响应，并支持通知的长期连接。无状态意味着负载均衡器后面可以水平扩展。

授权是带每工具范围的 OAuth 2.1。令牌携带如 `jira:read`、`s3:list`、`postgres:query:readonly` 等范围。MCP 服务器在工具调用时（而非仅在会话开始时）检查范围。对于高风险工具，服务器拒绝任何在过去 N 分钟内未将范围提升为 `approved:by:human` 的调用——该提升来自 Slack 审查卡片。

注册表是一个独立的服务。每个 MCP 服务器暴露一个 `.well-known/mcp-capabilities` 文档，包含其工具清单、传输 URL、认证要求。注册表轮询、验证和索引。平台团队使用注册表 UI 查看有哪些工具可用，需要哪些范围，以及哪些团队拥有它们。

## 架构

```
MCP 客户端（Claude Code，Cursor 3，...）
          |
          v
HTTPS 上的 StreamableHTTP（JSON-RPC + 流式传输）
          |
          v
负载均衡器后面的 MCP 服务器（FastMCP）
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 列表      Jira       Linear     Datadog
（只读）   （分页）    （读取）   （读取）    （查询）
          |
   +------+-------------+
   v                    v
 OPA 策略门控   破坏性工具 MCP（单独服务器）
                        |
                        v
                   通过 Slack 的人工批准
                        |
                        v
                   审计日志（仅追加，每租户）

  注册表服务
     |
     v  从每个服务器 GET /.well-known/mcp-capabilities
     v
     UI：搜索 / 验证 / 启用-禁用 / 所有权
```

## 技术栈

- 服务器框架：FastMCP（Python）或 `@modelcontextprotocol/sdk`（TypeScript）
- 传输：HTTPS 上的 StreamableHTTP（无状态）
- 认证：OAuth 2.1，通过 SPIFFE / SPIRE 进行工作负载身份
- 策略：每工具 OPA / Rego 规则；每请求策略决策服务
- 注册表：自托管，消费 `.well-known/mcp-capabilities` 清单
- 人工批准：破坏性工具的 Slack 交互消息
- 部署：AWS ECS Fargate 或 Fly.io，每租户一个服务器或带租户范围的共享
- 审计：每租户桶的结构化 JSONL，带每调用溯源

## 构建它

1. **工具界面。** 暴露 10 个内部工具：Postgres 只读查询、S3 对象列表、Jira 搜索/获取、Linear 搜索/获取、Datadog 指标查询、PagerDuty 值班查询、GitHub 只读、Notion 搜索、Slack 搜索、Salesforce 读取。每个工具都有类型化 schema 和范围标签。

2. **FastMCP 服务器。** 挂载工具。配置 StreamableHTTP 传输。添加 OAuth 令牌内省和范围执行的中间件。

3. **OPA 策略。** 每工具的 Rego 策略：什么范围允许调用，适用什么 PII 删除，适用什么有效负载大小上限。每次工具调用时调用决策服务。

4. **注册表服务。** 单独的 Go 或 TS 服务，轮询来自已注册服务器的 `.well-known/mcp-capabilities`，用 JSON Schema 验证，并暴露列表/搜索/验证/启用-禁用 UI。

5. **能力清单。** 每个服务器暴露 `.well-known/mcp-capabilities`，包含：工具列表、认证要求、传输 URL、所有者团队、SLO。

6. **破坏性工具分离。** 会改变状态的工具（Jira 创建、Linear 创建、Postgres 写入）位于第二个 MCP 服务器上，具有更严格的认证流程：令牌必须具有通过 Slack 卡片在 15 分钟内提升的 `approved:by:human` 范围。

7. **审计日志。** 每租户仅追加 JSONL：`{timestamp, user, tool, args_redacted, response_redacted, outcome}`。写入前通过 Presidio 进行 PII 删除。

8. **负载测试。** StreamableHTTP 上的 100 个并发客户端。通过添加第二个副本演示水平扩展；显示负载均衡器在没有会话粘性的情况下重新分配。

9. **合规性测试。** 对两个服务器运行官方 MCP 合规性套件。通过所有强制性部分。

## 使用它

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[注册表]  能力已验证：postgres.readonly v1.2
[策略]    范围 postgres:query:readonly 存在；已允许
[审计]    已记录：user=u42 tool=postgres.readonly outcome=ok
响应：    { "result": { "rows": [[1]] } }
```

## 交付它

`outputs/skill-mcp-server.md` 描述了可交付成果。一个用于内部工具的生产级 MCP 服务器 + 注册表 + 审计层，带 OAuth 2.1 范围和 OPA 门控。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 规范合规性 | StreamableHTTP + 能力清单通过 MCP 合规性测试 |
| 20 | 安全性 | 范围执行，每工具 OPA 覆盖，密钥卫生 |
| 20 | 可观测性 | 带 PII 删除的每工具调用审计日志 |
| 20 | 规模 | 100 客户端负载测试水平扩展演示 |
| 15 | 注册表用户体验 | 发现/验证/启用-禁用工作流 |
| **100** | | |

## 练习

1. 添加一个新工具（Confluence 搜索）。通过注册表验证流程发布它，而不触及核心服务器。

2. 写一个 OPA 策略，删除包含名为 `email`、`ssn` 或 `phone` 的列的 Postgres 查询结果。用探测查询进行测试。

3. 在本地延迟上对 StreamableHTTP vs stdio 进行基准测试。报告每次调用的 p50/p95。

4. 实现每租户配额：每工具每租户每分钟最多 N 次调用。通过第二个 OPA 规则执行。

5. 从 [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) 运行 MCP 合规性套件，并修复每个失败。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| StreamableHTTP | "2026 年 MCP 传输" | 无状态 HTTP + 流式传输；替换网络服务器的 SSE + stdio |
| Capability manifest（能力清单） | "Well-known 文档" | `.well-known/mcp-capabilities`，带工具列表、认证、传输 URL |
| OPA / Rego | "策略引擎" | 用于根据外部规则授权工具调用的 Open Policy Agent |
| Scope elevation（范围提升） | "人工批准" | 通过 Slack 批准授予的短期范围，破坏性工具需要 |
| Registry（注册表） | "工具发现" | 从能力清单索引 MCP 服务器的服务 |
| Workload identity（工作负载身份） | "SPIFFE / SPIRE" | 用于 OAuth 令牌颁发的加密服务身份 |
| Conformance suite（合规性套件） | "规范测试" | 用于 StreamableHTTP + 工具清单正确性的官方 MCP 测试套件 |

## 延伸阅读

- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP，能力元数据，注册表
- [AAIF MCP 注册表规范](https://github.com/modelcontextprotocol/registry) — 2026 年注册表规范
- [AWS ECS 参考部署](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — 参考生产部署
- [Pinterest 内部 MCP 生态系统](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — 参考内部部署
- [Block `goose` MCP 使用](https://block.github.io/goose/) — 参考智能体消费模式
- [FastMCP](https://github.com/jlowin/fastmcp) — Python 服务器框架
- [Open Policy Agent](https://www.openpolicyagent.org/) — 策略引擎参考
- [SPIFFE / SPIRE](https://spiffe.io) — 工作负载身份参考

# 综合项目——构建完整的工具生态系统（Capstone — Build a Complete Tool Ecosystem）

> Phase 13 讲解了每一个组件。本综合项目将它们连接成一个生产形态的完整系统：一个包含工具 + 资源 + 提示 + 任务 + UI 的 MCP 服务器、边缘处的 OAuth 2.1、一个 RBAC 网关、一个多服务器客户端、一个 A2A 子智能体调用、输出到收集器的 OTel 追踪、CI 中的工具投毒检测，以及一个 AGENTS.md + SKILL.md 包。完成后，你能为每一个架构决策进行辩护。

**类型：** 构建  
**语言：** Python（标准库，端到端生态系统框架）  
**前置知识：** Phase 13 · 01 到 21  
**预计时间：** 约 120 分钟

## 学习目标

- 组合一个暴露工具、资源、提示和带有 `ui://` 应用的任务的 MCP 服务器。
- 用实施 RBAC 和哈希锁定的 OAuth 2.1 网关保护服务器。
- 编写一个使用 OTel GenAI 属性进行端到端追踪的多服务器客户端。
- 将部分工作负载委托给 A2A 子智能体；验证不透明性得到保护。
- 用 AGENTS.md + SKILL.md 打包整个技术栈，使其他智能体能够驱动它。

## 问题所在

交付"研究与报告"系统：

- 用户提问："总结 2026 年 arXiv 上关于智能体协议被引用最多的三篇论文。"
- 系统：通过 MCP 搜索 arXiv；通过 A2A 将论文摘要委托给专门的写作智能体；聚合结果；将交互式报告渲染为 MCP Apps `ui://` 资源；将每个步骤记录到 OTel。

Phase 13 的所有原语都会用到。这不是玩具——Anthropic（Claude Research 产品）、OpenAI（带 Apps SDK 的 GPTs）和第三方在 2026 年发布的生产级研究助手系统就是这个形态。

## 核心概念

### 架构

```
[用户] -> [客户端] -> [网关（OAuth 2.1 + RBAC）] -> [研究 MCP 服务器]
                                                    |
                                                    +- MCP 工具: arxiv_search（纯查询）
                                                    +- MCP 资源: notes://recent
                                                    +- MCP 提示: /research_topic
                                                    +- MCP 任务: generate_report（长时）
                                                    +- MCP Apps UI: ui://report/current
                                                    +- A2A 调用: writer-agent（tasks/send）
                                                    |
                                                    +- OTel GenAI 跨度
```

### 追踪层次结构

```
agent.invoke_agent
 ├── llm.chat（发起）
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── 任务状态转换（不透明内部）
 ├── mcp.call -> tools/call generate_report（任务增强）
 │    └── tasks/status 轮询
 │    └── tasks/result（已完成，返回 ui:// 资源）
 └── llm.chat（最终综合）
```

一个 trace ID。每个跨度都有正确的 `gen_ai.*` 属性。

### 安全态势

- OAuth 2.1 + PKCE，使用资源指示器将令牌受众锁定到网关。
- 网关持有上游凭据；用户永远看不到它们。
- RBAC：`alice` 拥有 `research:read`、`research:write`，可以调用所有工具。`bob` 拥有 `research:read`，不能调用 `generate_report`。
- 锁定的描述清单：丢弃任何工具哈希已变更的服务器。
- 二规则审计：没有工具同时组合不受信任的输入、敏感数据和后果性操作。

### 渲染

最终的 `generate_report` 任务返回内容块加上 `ui://report/current` 资源。客户端宿主（Claude Desktop 等）在沙箱 iframe 中渲染交互式仪表板。仪表板包含按排序的论文列表、引用计数，以及用户点击任意论文时调用 `host.callTool('summarize_paper', {arxiv_id})` 的按钮。

### 打包

整个系统以如下结构发布：

```
research-system/
  AGENTS.md                     # 项目约定
  skills/
    run-research/
      SKILL.md                  # 顶层工作流
  servers/
    research-mcp/               # MCP 服务器
      pyproject.toml
      src/
  agents/
    writer/                     # A2A 智能体
  gateway/
    config.yaml                 # RBAC + 锁定清单
```

用户使用 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 用户可以通过调用 `run-research` 技能来驱动系统。

### 每个 Phase 13 章节的贡献

| 章节 | 综合项目使用了什么 |
|------|------------------|
| 01-05 | 工具接口、提供商可移植性、并行调用、模式设计、静态检查 |
| 06-10 | MCP 原语、服务器、客户端、传输、资源 + 提示 |
| 11-14 | 采样、根目录 + 询问、异步任务、`ui://` 应用 |
| 15-17 | 工具投毒、OAuth 2.1、网关 + 注册表 |
| 18 | A2A 子智能体委托 |
| 19 | OTel GenAI 追踪 |
| 20 | LLM 层的路由网关 |
| 21 | SKILL.md + AGENTS.md 打包 |

## 动手使用

`code/main.py` 将前面各章的模式拼接成一个可运行的演示。全部标准库，全部在进程内，以便你能从头到尾阅读。它运行"研究与报告"场景的完整流程：与网关握手、模拟 OAuth 2.1、合并 tools/list、以任务形式运行 generate_report、A2A 调用写作智能体、返回 ui:// 资源、发射 OTel 跨度。

要关注的内容：

- 一个 trace ID 贯穿每个跳点。
- 网关策略阻止第二个用户进行写操作。
- 任务生命周期经历 working → completed，并返回文本和 ui:// 内容。
- A2A 调用的内部状态对编排者不透明。
- AGENTS.md 和 SKILL.md 是另一个智能体复现工作流所需的唯一文件。

## 输出产物

本章生成 `outputs/skill-ecosystem-blueprint.md`。给定产品需求（研究、摘要、自动化），该技能生成完整架构：使用哪些 MCP 原语、哪些网关控制、哪些 A2A 调用、哪些遥测配置、如何打包。

## 练习

1. 运行 `code/main.py`。观察单一 trace ID 和跨度的嵌套方式。统计演示触及了 Phase 13 的多少个原语。

2. 扩展演示：添加第二个后端 MCP 服务器（如 `bibliography`），确认网关将其工具合并到同一命名空间中。

3. 将虚假的 A2A 写作智能体替换为在子进程中运行的真实智能体。使用第 19 章的框架。

4. 在路由网关的编排者和 LLM 之间添加一个 PII 脱敏步骤。确认用户查询中的电子邮件地址被清除。

5. 为将维护此系统的团队成员编写一个 AGENTS.md。它应该在五分钟内读完，并提供他们在 Cursor 或 Codex 中驱动综合项目所需的一切信息。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 综合项目（Capstone） | "Phase 13 集成演示" | 使用所有原语的端到端系统。 |
| 研究与报告（Research and report） | "场景" | 搜索、摘要、渲染的模式。 |
| 生态系统（Ecosystem） | "所有组件的组合" | 服务器 + 客户端 + 网关 + 子智能体 + 遥测 + 包。 |
| 追踪层次结构（Trace hierarchy） | "单一 trace ID" | 每个跳点的跨度共享追踪；通过跨度 ID 编码父子关系。 |
| 网关颁发的令牌（Gateway-issued token） | "传递式授权" | 客户端只看到网关的令牌；网关持有上游凭据。 |
| 合并命名空间（Merged namespace） | "所有工具在一个平铺列表中" | 在网关处进行多服务器合并，冲突时加前缀。 |
| 不透明边界（Opacity boundary） | "A2A 调用隐藏内部" | 子智能体的推理对编排者不可见。 |
| 三层技术栈（Three-layer stack） | "AGENTS.md + SKILL.md + MCP" | 项目上下文 + 工作流 + 工具。 |
| 纵深防御（Defense-in-depth） | "多重安全层" | 哈希锁定、OAuth、RBAC、二规则、审计日志。 |
| 规范合规矩阵（Spec compliance matrix） | "我们交付的规范要求清单" | 将可交付物映射到 2025-11-25 要求的检查表。 |

## 延伸阅读

- [MCP — 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 综合参考文档
- [MCP 博客 — 2026 年路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 协议的发展方向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 参考文档
- [OpenTelemetry — GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范追踪约定
- [Anthropic — Claude Agent SDK 概述](https://code.claude.com/docs/en/agent-sdk/overview) — 生产智能体运行时模式

# LLM 路由层——LiteLLM、OpenRouter、Portkey（LLM Routing Layer — LiteLLM, OpenRouter, Portkey）

> 绑定单一提供商代价高昂。不同的工具调用工作负载适合不同的模型。路由网关提供统一的 API 接口、重试、故障转移、成本追踪和护栏。2026 年有三种主流架构：LiteLLM（开源自托管）、OpenRouter（托管 SaaS）、Portkey（生产级，2026 年 3 月开源）。本章列举决策标准，并演示一个标准库路由网关。

**类型：** 学习  
**语言：** Python（标准库，路由 + 故障转移 + 成本追踪器）  
**前置知识：** Phase 13 · 02（函数调用）、Phase 13 · 17（网关）  
**预计时间：** 约 45 分钟

## 学习目标

- 区分自托管、托管和生产级路由选项。
- 实现一个按优先级顺序在提供商故障时重试的回退链。
- 跨提供商追踪每请求的成本和 token 用量。
- 针对给定的生产约束，在 LiteLLM、OpenRouter 和 Portkey 之间做出选择。

## 问题所在

提供商路由重要性的场景：

1. **成本。** Claude Sonnet 比 Haiku 贵 3 倍。对于分类任务，Haiku 就够了；对于综合任务，Sonnet 值得。按请求路由。

2. **故障转移。** OpenAI 出现了一个小时的故障。每个请求都失败。你希望自动切换到 Anthropic，无需重新部署。

3. **延迟。** 实时聊天界面需要低首字符延迟。批量摘要生成则不需要。按延迟 SLA 路由。

4. **合规。** 欧盟用户必须留在欧盟区域。按地区路由。

5. **实验。** 在同一工作负载上对两个模型进行 A/B 测试。按测试桶路由。

在每个集成中手动编写所有这些既重复又繁琐。路由网关提供一个兼容 OpenAI 的 API，其余的交给它处理。

## 核心概念

### OpenAI 兼容的代理形式

所有人都使用 OpenAI 格式。路由网关暴露 `/v1/chat/completions`，接受 OpenAI 模式，并在内部代理到 Anthropic / Gemini / Cohere / Ollama / 任意后端。客户端不需要关心。

### 模型别名

你的代码不再写 `claude-3-5-sonnet-20251022`，而是写 `our_smart_model`。网关将别名映射到真实模型。当 Anthropic 发布 Claude 4 时，你只需在服务端修改别名；代码不需要动。

### 回退链

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

网关在配置中定义此规则。重试计入预算，防止回退级联导致成本爆炸。

### 语义缓存

相同或近似相同的提示命中缓存而不是调用提供商。重复智能体循环的节省可达 30%~60%。缓存键基于嵌入；近似相同的提示共享一个缓存槽。

### 护栏

网关层面的防护：

- **PII 脱敏。** 在发送提示前进行正则或 ML 过滤。
- **策略违规。** 拒绝包含禁止内容的提示。
- **输出过滤。** 检查补全内容是否有信息泄露。

Portkey 和 Kong 都内置了详细的护栏。LiteLLM 将其设为可选。

### 每密钥速率限制

一个 API 密钥 = 一个团队。每密钥预算防止某个团队耗尽共享配额。大多数网关支持此功能。

### 自托管 vs 托管的取舍

| 因素 | LiteLLM（自托管） | OpenRouter（托管） | Portkey（生产级） |
|------|------------------|-------------------|------------------|
| 代码 | 开源，Python | 托管 SaaS | 开源（2026 年 3 月）+ 托管 |
| 部署 | 部署代理 | 注册即用 | 两种均可 |
| 提供商 | 100+ | 300+ | 100+ |
| 计费 | 使用自己的密钥 | OpenRouter 积分 | 使用自己的密钥 |
| 可观测性 | OpenTelemetry | 仪表板 | 完整 OTel + PII 脱敏 |
| 最适合 | 需要完全控制的团队 | 快速原型开发 | 需要开箱即用合规的生产环境 |

当你有 SRE 团队并且需要数据主权时，选 LiteLLM。当你想要单一订阅且不想维护基础设施时，选 OpenRouter。当你需要开箱即用的护栏和合规功能时，选 Portkey。

### 成本追踪

每个请求携带 `provider`、`model`、`input_tokens`、`output_tokens`。乘以网关维护的每模型每 token 价格（从定价表拉取）。按用户/团队/项目聚合。

### MCP 加路由

网关可以同时路由 LLM 调用和 MCP 采样请求。当采样请求的 modelPreferences 偏向特定模型时，网关将其转换到正确的后端。这就是 Phase 13 · 17（MCP 网关）和本章路由网关有时合并为同一服务的原因。

### 路由策略

- **静态优先级。** 按列表顺序；出错时回退。
- **负载均衡。** 轮询或加权分配。
- **成本感知。** 选择满足延迟/质量要求的最便宜模型。
- **延迟感知。** 选择过去 N 分钟内最快的模型。
- **任务感知。** 提示分类器将编程任务路由到某模型，将摘要任务路由到另一模型。

## 动手使用

`code/main.py` 用约 150 行实现了一个路由网关：接受 OpenAI 格式的请求，转换为各提供商的存根，运行优先级回退链，追踪每请求成本，并对输入执行 PII 脱敏。运行三个场景：正常请求、主提供商故障触发回退、PII 泄露被脱敏拦截。

要关注的内容：

- `ROUTES` 字典：别名 -> 按优先级排序的具体提供商列表。
- 回退循环在 5xx 时重试。
- 成本追踪器将 token 用量乘以每模型价格。
- PII 脱敏器在转发前清除 SSN 格式的模式。

## 输出产物

本章生成 `outputs/skill-routing-config-designer.md`。给定工作负载配置文件（延迟、成本、合规），该技能选择 LiteLLM / OpenRouter / Portkey 并生成路由配置。

## 练习

1. 运行 `code/main.py`。触发故障场景；确认回退落在第二个提供商上，且成本归因正确。

2. 添加语义缓存：提示的 SHA256 作为查找键；缓存命中时立即返回。测量重复调用的成本节省。

3. 添加一个提示分类器，将"code ..."提示路由到偏向智能的别名，将"summarize ..."提示路由到偏向速度的别名。

4. 设计每团队预算：每个团队有月度消费上限；超出上限后网关拒绝请求。选择一个执行粒度（每请求或时间窗口）。

5. 并排阅读 LiteLLM、OpenRouter 和 Portkey 文档。找出每个产品独有而另外两个没有的功能。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 路由网关（Routing gateway） | "LLM 代理" | 多提供商前面的统一 API 接入层。 |
| OpenAI 兼容（OpenAI-compatible） | "使用 OpenAI 格式" | 接受 `/v1/chat/completions` 格式，转换到任意后端。 |
| 模型别名（Model alias） | "our_smart_model" | 代码中的名称，由网关映射到具体模型。 |
| 回退链（Fallback chain） | "重试列表" | 故障时按序尝试的提供商有序列表。 |
| 语义缓存（Semantic caching） | "提示嵌入缓存" | 以提示嵌入为键；近似重复共享缓存命中。 |
| 护栏（Guardrails） | "输入/输出过滤" | 脱敏 PII，拒绝策略违规。 |
| 每密钥速率限制（Per-key rate limit） | "团队预算" | 作用于 API 密钥的配额。 |
| 成本追踪（Cost tracking） | "每请求花费" | 聚合 token 用量 × 每模型价格。 |
| LiteLLM | "开放代理" | 可自托管的开源路由网关。 |
| OpenRouter | "托管 SaaS" | 基于积分计费的托管网关。 |
| Portkey | "生产级选项" | 内置护栏的开源 + 托管方案。 |

## 延伸阅读

- [LiteLLM — 文档](https://docs.litellm.ai/) — 自托管路由网关
- [OpenRouter — 快速入门](https://openrouter.ai/docs/quickstart) — 托管路由 SaaS
- [Portkey — 文档](https://portkey.ai/docs) — 内置护栏的生产级路由
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — 决策指南
- [Relayplane — 2026 年 LLM 网关对比](https://relayplane.com/blog/llm-gateway-comparison-2026) — 供应商调研

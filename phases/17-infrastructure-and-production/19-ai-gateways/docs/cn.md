# AI 网关——LiteLLM、Portkey、Kong AI Gateway、Bifrost（AI Gateways — LiteLLM, Portkey, Kong AI Gateway, Bifrost）

> 网关位于你的应用和模型供应商之间。核心功能是供应商路由、故障转移、重试、速率限制、密钥引用、可观测性、护栏。2026 年的市场分化：**LiteLLM** 是 MIT 开源，支持 100+ 供应商，兼容 OpenAI，但在约 2000 RPS 时崩溃（8 GB 内存，已发布基准中出现级联故障）；最适合 Python，<500 RPS，开发/原型场景。**Portkey** 定位为控制面（护栏、PII 删除、越狱检测、审计轨迹），2026 年 3 月开源为 Apache 2.0，延迟开销 20-40 毫秒，生产层 49 美元/月。**Kong AI Gateway** 基于 Kong Gateway 构建——Kong 自己在同等 12 CPU 上的基准：比 Portkey 快 228%，比 LiteLLM 快 859%；定价 100 美元/模型/月（Plus 层最多 5 个）；如果你已经在用 Kong，适合企业场景。**Bifrost**（Maxim AI）——带可配置退避的自动重试，在 OpenAI 429 时回退到 Anthropic。**Cloudflare / Vercel AI Gateways** — 托管，零运维，基本重试。数据驻留决定了自托管选择；Portkey 和 Kong 处于中间位置，提供开源 + 可选托管。

**类型：** 学习  
**语言：** Python（标准库，玩具网关路由模拟器）  
**前置知识：** Phase 17 · 01（托管 LLM 平台）、Phase 17 · 16（模型路由）  
**预计时间：** 约 60 分钟

## 学习目标

- 列举六个核心网关功能（路由、故障转移、重试、速率限制、密钥、可观测性、护栏）。
- 将四个 2026 年的网关（LiteLLM、Portkey、Kong AI、Bifrost）映射到规模上限和使用场景。
- 引用 Kong 基准（比 Portkey 快 228%，比 LiteLLM 快 859%）并解释其对 >500 RPS 的重要性。
- 根据数据驻留和运维预算选择自托管 vs 托管。

## 问题所在

你的产品调用 OpenAI、Anthropic 和自托管 Llama。每个供应商都有不同的 SDK、错误模型、速率限制和认证方案。你希望有故障转移（如果 OpenAI 429，则尝试 Anthropic）、单一凭证存储、统一可观测性和每租户速率限制。

在应用层重新发明这些功能会将每个服务与每个供应商耦合。网关层将其整合到一个进程中，使用一个 API（通常兼容 OpenAI）向供应商分发请求。

## 核心概念

### 六个核心功能

1. **供应商路由** — OpenAI、Anthropic、Gemini、自托管等统一在一个 API 背后。
2. **故障转移** — 在 429、5xx 或质量失败时，重试其他供应商。
3. **重试** — 指数退避，有限次数。
4. **速率限制** — 每租户、每密钥、每模型。
5. **密钥引用** — 运行时从保险库拉取凭证（永远不放在应用中）。
6. **可观测性** — OTel + GenAI 属性（Phase 17 · 13）+ 成本归因。
7. **护栏** — PII 删除、越狱检测、允许话题过滤器。

### LiteLLM——MIT 开源，Python

- 100+ 供应商，兼容 OpenAI，路由配置，故障转移，基本可观测性。
- 在 Kong 的基准中约 2000 RPS 时崩溃；8 GB 内存占用，持续负载下级联失败。
- 最适合：Python 应用，<500 RPS，开发/暂存网关，实验性路由。
- 成本：开源为 0 美元；有云端免费层。

### Portkey——控制面定位

- 截至 2026 年 3 月为 Apache 2.0 开源。护栏、PII 删除、越狱检测、审计轨迹。
- 每请求延迟开销 20-40 毫秒。
- 带保留和 SLA 的生产层 49 美元/月。
- 最适合：需要捆绑护栏 + 可观测性的受监管行业。

### Kong AI Gateway——规模化方案

- 基于 Kong Gateway 构建（成熟的 API 网关产品，lua+OpenResty）。
- Kong 自己在 12 CPU 等效上的基准：比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定价：100 美元/模型/月，Plus 层最多 5 个。
- 最适合：已在使用 Kong；>1000 RPS；愿意授权。

### Bifrost（Maxim AI）

- 带可配置退避的自动重试。
- OpenAI 429 时回退到 Anthropic 是经典方案。
- 较新的参与者；商业产品。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管，零运维。基本重试和可观测性。
- 最适合：在 Cloudflare/Vercel 上的边缘服务 JavaScript 应用。
- 在护栏和速率限制方面不如 Kong/Portkey 全面。

### 自托管 vs 托管

数据驻留是决定性因素。医疗和金融默认自托管（LiteLLM 或 Portkey OSS 或 Kong）。消费产品默认托管（Cloudflare AI Gateway）或中间层（Portkey 托管）。混合：受监管租户自托管，其他使用托管。

### 延迟预算

- LiteLLM：通常 5-15 毫秒开销。
- Portkey：20-40 毫秒开销。
- Kong：3-8 毫秒开销。
- Cloudflare/Vercel：1-3 毫秒开销（边缘优势）。

网关延迟直接加到 TTFT 上。TTFT P99 < 100 毫秒 SLA，用 Kong 或 Cloudflare。P99 < 500 毫秒，任何都可以。

### 速率限制语义很重要

简单的令牌桶适用于中等规模。多租户需要滑动窗口 + 突发限额 + 每租户分层。LiteLLM 使用令牌桶；Kong 使用滑动窗口；Portkey 使用分层。

### 网关 + 可观测性 + 路由组合

Phase 17 · 13（可观测性）+ 16（模型路由）+ 19（网关）在生产中是同一层。选择一个覆盖所有三者的工具，或仔细连接它们：大多数 2026 年的部署将 Helicone（可观测性）或 Portkey（护栏）与 Kong（规模）组合用于分工角色。

### 你应该记住的数字

- LiteLLM：约 2000 RPS 时崩溃，8 GB 内存。
- Portkey：20-40 毫秒开销；2026 年 3 月起 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：100 美元/模型/月，Plus 层最多 5 个。
- Cloudflare/Vercel：边缘延迟开销 1-3 毫秒。

## 使用它

`code/main.py` 在注入 429/5xx 的情况下，模拟跨 3 个供应商的带故障转移的网关路由。报告延迟、重试率和故障转移命中率。

## 交付它

本课产出 `outputs/skill-gateway-picker.md`。给定规模、运维立场、合规性、延迟预算，选择网关。

## 练习

1. 运行 `code/main.py`。配置从 OpenAI→Anthropic→自托管的故障转移。在 5% 供应商错误率下预期命中率是多少？
2. 你的 SLA 是 TTFT P99 < 200 毫秒，基线 300 毫秒。哪些网关在预算内？
3. 一个医疗客户需要自托管 + PII 删除 + 审计。选择 Portkey OSS 还是 Kong。
4. 比较 LiteLLM vs Kong：团队应该在什么 RPS 上限迁移？
5. 为多租户 SaaS 设计速率限制策略：免费层、试用层、付费层。令牌桶还是滑动窗口？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Gateway（网关） | "API 中间代理" | 位于应用和供应商之间的进程 |
| LiteLLM | "MIT 的那个" | Python 开源，100+ 供应商，2K RPS 时崩溃 |
| Portkey | "护栏网关" | 控制面 + 可观测性，Apache 2.0 |
| Kong AI Gateway | "规模化的那个" | 基于 Kong Gateway 构建，基准领先 |
| Bifrost | "Maxim 的网关" | 重试 + Anthropic 故障转移方案 |
| Cloudflare AI Gateway | "边缘托管" | 边缘部署的托管网关，零运维 |
| PII redaction（PII 删除） | "数据清洗" | 发送给模型前的正则表达式 + NER 屏蔽 |
| Jailbreak detection（越狱检测） | "提示注入守护" | 对用户输入的分类器 |
| Audit trail（审计轨迹） | "合规日志" | 每次 LLM 调用的不可变记录 |
| Token-bucket（令牌桶） | "简单速率限制" | 基于补充的速率限制器 |
| Sliding-window（滑动窗口） | "精确速率限制" | 基于时间窗口的速率限制器；公平性更好 |

## 延伸阅读

- [Kong AI Gateway 基准](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — 2026 年 AI 网关比较](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — 2026 年顶级 LLM 网关工具](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway 文档](https://docs.konghq.com/gateway/latest/ai-gateway/)

# 托管 LLM 平台——Bedrock、Vertex AI、Azure OpenAI（Managed LLM Platforms — Bedrock, Vertex AI, Azure OpenAI）

> 三家超大规模云厂商，三种截然不同的策略。AWS Bedrock 是一个模型市场——Claude、Llama、Titan、Stability、Cohere 统一在同一 API 背后。Azure OpenAI 是独家 OpenAI 合作关系，加上用于独占容量的预置吞吐量单元（PTU）。Vertex AI 以 Gemini 为核心，在长上下文和多模态方面拥有最佳故事。2026 年，Artificial Analysis 测量到 Azure OpenAI 中位延迟约为 50 毫秒，而 Bedrock 在 Llama 3.1 405B 同等配置上约为 75 毫秒——PTU 解释了这一差距，因为独占容量胜过共享按需容量。决策规则不是"哪个更快"，而是"哪个模型目录和 FinOps 界面与我的产品匹配"。本课教你把权衡写下来做决策，而不是凭感觉。

**类型：** 学习  
**语言：** Python（标准库，玩具成本与延迟比较器）  
**前置知识：** Phase 11（LLM 工程）、Phase 13（工具与协议）  
**预计时间：** 约 60 分钟

## 学习目标

- 说出三种平台策略（市场式 vs 独家合作 vs Gemini 优先），并将每种策略与具体产品使用场景匹配。
- 解释 Azure OpenAI 中预置吞吐量单元（PTU）能买到什么，以及为什么按需 Bedrock 在 405B 规模上通常慢约 25 毫秒。
- 画出每个平台的 FinOps 归因界面（Bedrock 的应用推理配置文件 vs Vertex 的按团队项目 vs Azure 的范围 + PTU 预留）。
- 写下"至少双供应商"策略并解释为什么 2026 年单一供应商锁定是代价高昂的错误。

## 问题所在

你为产品选择了 Claude 3.7 Sonnet。现在需要提供服务。你可以直接调用 Anthropic API，或者通过 AWS Bedrock 调用，或者通过网关调用。直接 API 最简单；Bedrock 增加了 BAA、VPC 端点、IAM 和 CloudWatch 归因。网关增加了故障转移、统一计费以及跨供应商的速率限制。

更深层的问题是目录。如果你的产品同时需要 Claude、Llama 和 Gemini，除非同时使用 Bedrock 加 Vertex 加 Azure OpenAI，否则无法从一个地方获得所有这些。超大规模云厂商不可互换——它们各自对谁拥有模型层下了不同的赌注。

本课绘制三种赌注、延迟差距、FinOps 差距和锁定风险的全貌。

## 核心概念

### 三种策略

**AWS Bedrock** — 市场模式。Claude（Anthropic）、Llama（Meta）、Titan（AWS 自有）、Stability（图像）、Cohere（嵌入）、Mistral，加上图像和嵌入子目录。一个 API，一个 IAM 界面，一个 CloudWatch 导出。Bedrock 的赌注是：客户对可选性的需求高于对单一模型的需求。

**Azure OpenAI** — 独家合作关系。你可以在 Azure 数据中心使用 GPT-4 / 4o / 5 / o 系列、DALL·E、Whisper，以及 OpenAI 模型的微调。"Azure OpenAI Service"目录中没有非 OpenAI 模型——那些模型放在 Azure AI Foundry（独立产品）中。Azure 的赌注是：OpenAI 仍然代表前沿，客户希望在这一特定关系上拥有企业级控制权。

**Vertex AI** — Gemini 优先，其他其次。Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，加上 Model Garden（第三方）。Vertex 的赌注是多模态长上下文——1M token 的 Gemini 上下文是其差异化因素。

### 大规模下的延迟差距

Artificial Analysis 持续运行基准测试。在等效的 Llama 3.1 405B 部署（共享按需）上，Azure OpenAI 中位首 token 延迟约为 50 毫秒；Bedrock 约为 75 毫秒。这个差距不是 AWS 的失败——而是容量模型的差异。Azure 出售 PTU（预置吞吐量单元），为你的租户预留 GPU 容量。Bedrock 的等价方案（预置吞吐量）存在，但起价约为每单元每小时 21 美元，大多数客户仍然使用共享按需模式。

共享按需容量与所有其他客户的流量竞争。独占容量则不会。如果你的产品 SLA 要求 P99 的 TTFT < 100 毫秒，你要么在 Azure 上购买 PTU，要么购买 Bedrock 预置吞吐量，要么接受默认的延迟波动。

### 预置吞吐量经济学

Azure PTU：推理计算的预留块。对于可预测的工作负载，与按需相比可节省高达约 70%。成本按小时固定，无论流量多少——即使空闲也需要支付预留费用。盈亏平衡通常在持续利用率约 40-60% 时出现。

Bedrock 预置吞吐量：根据模型和地区，每小时 21-50 美元。数学逻辑类似——盈亏平衡约在峰值利用率的一半。需要按月承诺。

Vertex 的预置容量按 Gemini SKU 出售；定价因模型和地区而异，公开宣传较少。

### FinOps 界面——真正的差异化因素

**Bedrock 应用推理配置文件**是市场中最清晰的归因。用 `team`、`product`、`feature` 标记配置文件；将所有模型调用通过它路由；CloudWatch 无需后处理即可按配置文件分解成本。2025 年新增，仍是最细粒度的超大规模云厂商原生方案。

**Vertex** 归因采用按团队项目加标签到处都是的模式。将每个团队建模为一个 GCP 项目，给每个资源加标签，使用 BigQuery 账单导出 + DataStudio 进行汇总。工作量更大，但 BigQuery 可以对成本数据进行任意 SQL 查询。

**Azure** 依赖订阅/资源组范围加标签，PTU 预留作为一等成本对象。标签从资源组继承，而非从请求继承，因此按请求归因需要 Application Insights 自定义指标或一个能够标记请求头的网关。

规律：Bedrock 原生最简洁，Vertex 通过 BigQuery 最灵活，Azure 最不透明（除非你进行了监控埋点）。

### 锁定是 2026 年的风险

当一个模型主导市场时，单一超大规模云厂商承诺还说得过去。2026 年，前沿每月都在移动——一个季度是 Claude 3.7，下一个季度是 Gemini 2.5，再下一个季度是 GPT-5。锁定到一个平台就等于将自己锁在三分之二的前沿之外。

工作团队采用的模式：对任何关键 LLM 调用实行至少双供应商原则。Bedrock 加 Azure OpenAI 是常见组合——从一个获取 Claude，从另一个获取 GPT，两者之间进行故障转移，同一网关。成本增幅可忽略不计，因为网关会路由到最优；在中断期间（如 Azure OpenAI 2025 年 1 月事故、AWS us-east-1 中断）的可用性提升是决定性的。

### 数据驻留、BAA 和受监管行业

Bedrock：大多数地区有 BAA；VPC 端点；护栏。常见金融科技默认选择。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；EU 数据驻留；企业受监管行业默认选择。
Vertex：HIPAA、GDPR、按地区的数据驻留；Google Cloud 合规栈。

三者都满足基本要求。差异在于数据保留策略、日志处理方式，以及滥用监控是否读取你的流量（大多数默认开启；企业版可选择退出）。

### 你应该记住的数字

- Azure OpenAI 在 Llama 3.1 405B 同等配置上的中位 TTFT：约 50 毫秒（含 PTU）。
- Bedrock 按需中位 TTFT：约 75 毫秒。
- Bedrock 预置吞吐量：每单元每小时 21-50 美元。
- Azure PTU 盈亏平衡：约 40-60% 持续利用率。
- 高利用率下 PTU vs 按需的节省：高达 70%。

## 使用它

`code/main.py` 在合成工作负载上比较三个平台——对按需 vs PTU 经济学、TTFT 波动和成本归因精度进行建模。运行它可以看到 PTU 在哪些情况下值得，以及市场的模型广度在哪些情况下优于 TTFT 差距。

## 交付它

本课产出 `outputs/skill-managed-platform-picker.md`。给定工作负载配置文件（所需模型、TTFT SLA、日均流量、合规要求），它推荐主要平台、备用平台和 FinOps 监控计划。

## 练习

1. 运行 `code/main.py`。在什么持续利用率下，Azure PTU 对于 70B 级模型优于按需？计算盈亏平衡，并与宣传的 40-60% 区间进行比较。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计一个双供应商部署——哪个去哪家超大规模云厂商，前面放什么网关，故障转移策略是什么？
3. 一个受监管的医疗客户需要 BAA、美国东部数据驻留和 P99 < 100 毫秒的 TTFT。选择一个平台并用三个具体特性来论证。
4. 你发现本月的 Bedrock 账单在没有流量变化的情况下增加了 4 倍。没有应用推理配置文件时，如何找到原因？有配置文件时需要多长时间？
5. 阅读 Azure OpenAI 和 Bedrock 的定价页面。对于每月 1 亿 token 的 Claude 工作负载，哪个更便宜——直接 Anthropic API、Bedrock 按需，还是 Bedrock 预置吞吐量？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Bedrock | "AWS LLM 服务" | 跨 Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | "Azure 的 ChatGPT" | Azure 数据中心中具有企业控制的独家 OpenAI 模型 |
| Vertex AI | "Google 的 LLM" | Gemini 优先平台，带有第三方模型的 Model Garden |
| PTU | "独占容量" | 预置吞吐量单元——按小时计价的预留推理 GPU |
| Application Inference Profile（应用推理配置文件） | "Bedrock 标记" | 带标签的按产品成本/使用配置文件，CloudWatch 原生 |
| Model Garden | "Vertex 目录" | Vertex AI 的第三方模型区，与 Gemini 分离 |
| Two-provider minimum（至少双供应商） | "LLM 冗余" | 在 ≥2 家超大规模云厂商上运行每条关键 LLM 路径的策略 |
| BAA | "HIPAA 文件" | 业务伙伴协议；处理 PHI 所需；三者均提供 |
| Abuse monitoring（滥用监控） | "日志观察者" | 供应商对提示词/输出的安全扫描；企业版可选择退出 |

## 延伸阅读

- [AWS Bedrock 定价](https://aws.amazon.com/bedrock/pricing/) — 权威价格表和预置吞吐量定价
- [Azure OpenAI Service 定价](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU 经济学和价格表
- [Vertex AI 生成式 AI 定价](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini 层级和 Model Garden 附加费
- [Artificial Analysis LLM 排行榜](https://artificialanalysis.ai/) — 跨供应商的持续延迟和吞吐量基准
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO 指南 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 企业决策框架
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — 归因机制并排对比

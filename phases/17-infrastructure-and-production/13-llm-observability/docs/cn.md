# LLM 可观测性栈选型（LLM Observability Stack Selection）

> 2026 年的可观测性市场分为两类。开发平台（LangSmith、Langfuse、Comet Opik）将监控与评估、提示词管理、会话回放捆绑在一起。网关/埋点工具（Helicone、SigNoz、OpenLLMetry、Phoenix）专注于遥测数据。Langfuse 的核心采用 MIT 许可，有良好的开源平衡（云端免费 5 万事件/月）。Phoenix 是 Elastic License 2.0 下的 OpenTelemetry 原生工具——非常适合漂移/RAG 可视化，但不是持久的生产后端。Arize AX 使用零拷贝 Iceberg/Parquet 集成，声称比单体可观测性便宜 100 倍。LangSmith 在 LangChain/LangGraph 中领先，39 美元/用户/月，仅企业版支持自托管。Helicone 基于代理，15-30 分钟即可设置，每月 10 万请求免费，但在智能体轨迹方面深度不足。常见生产模式：网关（Helicone/Portkey）+ 评估平台（Phoenix/TruLens），用 OpenTelemetry 粘合。

**类型：** 学习  
**语言：** Python（标准库，玩具轨迹采样模拟器）  
**前置知识：** Phase 17 · 08（推理指标）、Phase 14（智能体工程）  
**预计时间：** 约 60 分钟

## 学习目标

- 区分开发平台（捆绑：评估 + 提示词 + 会话）与网关/遥测工具（仅轨迹 + 指标）。
- 将六个主要工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）映射到其许可证、定价和最适合的使用场景。
- 解释让你能将网关工具与独立评估平台组合的 OpenTelemetry 粘合模式。
- 说出 2026 年的成本差异化因素（Arize AX 的零拷贝方法 vs 单体摄入），并给出大约 100 倍的乘数。

## 问题所在

你发布了一个 LLM 功能。它可以运行。但你对提示词失败、工具循环、延迟退化、成本峰值或提示词缓存命中率毫无可见性。你搜索"LLM 可观测性"，得到八个工具，都声称在三个不同价位解决同一个问题。

它们解决的不是同一个问题。LangSmith 回答的是"这个 LangGraph 运行为什么失败？"Phoenix 回答的是"我的 RAG 管道是否在漂移？"Helicone 回答的是"哪个应用在烧 token？"Langfuse 回答的是"我可以自托管整个系统吗？"不同工具，不同受众。

选型涉及四个维度：技术栈（LangChain？原始 SDK？多供应商？）、许可证容忍度（仅 MIT？Elastic OK？商业可以？）、预算（免费层？100 美元/月？1000 美元/月？）和自托管（必须？最好有？绝不？）。

## 核心概念

### 两个类别

**开发平台**将可观测性与评估、提示词管理、数据集版本控制、会话回放捆绑在一起。你运行实验，查看哪个提示词有效，对新提示词与旧赢家进行数据集回归测试。LangSmith、Langfuse、Comet Opik。

**网关/遥测工具**对推理调用进行埋点——提示词、响应、token、延迟、模型、成本。Helicone、SigNoz、OpenLLMetry、Phoenix。极简。可通过 OpenTelemetry 与独立评估工具组合。

### Langfuse——开源平衡

- 核心采用 Apache / MIT 许可；通过 Docker 自托管。
- 云端免费层：5 万事件/月。付费：团队版 29 美元/月。
- 评估、提示词管理、轨迹、数据集。合理覆盖所有四个开发平台特性。
- 最适合：你想要 LangSmith 级别的功能，但必须自托管或保持在 OSS 许可证。

### Phoenix（Arize）——遥测优先，OpenTelemetry 原生

- Elastic License 2.0；自托管极为简单。
- 在 RAG 和漂移可视化方面表现出色。嵌入空间散点图作为一等功能提供。
- 不是设计为持久生产后端——主要是开发时可观测性。
- 最适合：RAG 管道开发、漂移调试，与独立网关配合用于生产。

### Arize AX——规模化方案

- 商业产品。通过 Iceberg/Parquet 进行零拷贝数据湖集成。
- 声称比单体可观测性（Datadog 级别）在规模下便宜约 100 倍。数学：你将轨迹存储在 S3 上自己的 Parquet 中；Arize 直接读取。
- 最适合：每天 >1000 万轨迹、有现有数据湖、想要 LLM 专属仪表板而不承受 Datadog 定价。

### LangSmith——LangChain/LangGraph 优先

- 商业产品，39 美元/用户/月。仅企业版支持自托管。
- 对于 LangChain 和 LangGraph 栈是最佳选择。如果你两者都不使用，吸引力就小很多。
- 最适合：团队已经押注 LangChain，愿意付费。

### Helicone——基于代理的最小可行方案

- 通过将你的 `OPENAI_API_BASE` 换成 Helicone 代理，15-30 分钟即可设置。
- MIT 许可；每月 10 万请求免费，付费 20 美元/月起。
- 包含故障转移、缓存、速率限制——同时充当网关。
- 在智能体/多步轨迹方面深度不足。
- 最适合：快速启动、单栈应用、需要网关 + 可观测性于一体。

### Opik（Comet）——OSS 开发平台

- Apache 2.0，完全开源。
- 与 Langfuse 相近的功能集，有 Comet 传承。
- 最适合：已经在使用 Comet 的 ML 团队，希望在同一界面中获得 LLM 可观测性。

### SigNoz——OpenTelemetry 优先的全栈 APM

- Apache 2.0。通过 OpenTelemetry 处理通用 APM 加 LLM 调用。
- 最适合：跨服务和 LLM 调用的统一可观测性。

### 粘合剂：OpenTelemetry + GenAI 语义约定

OpenTelemetry 在 2025 年底发布了 GenAI 语义约定（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。使用 OTel 的工具可以互操作。正在形成的生产模式：

1. 在每次 LLM 调用时使用 GenAI 约定发送 OTel。
2. 路由到网关（Helicone / Portkey）用于日常操作。
3. 双发到评估平台（Phoenix / Langfuse）用于回归测试。
4. 存档到数据湖（Iceberg）用于通过 Arize AX 或 DuckDB 进行长期分析。

### 陷阱：在错误层次埋点

在智能体框架内埋点（例如，添加 LangSmith 轨迹）会将你与该框架耦合。在 HTTP/OpenAI-SDK 层埋点（通过 OpenLLMetry 或你的网关）是可移植的。

### 采样——你无法保留所有内容

在每天 >100 万请求时，全量轨迹保留的成本比 LLM 调用本身还贵。按规则采样：100% 错误，100% 高成本，5% 成功。始终保留聚合数据；只为长尾保留原始数据。

### 你应该记住的数字

- Langfuse 免费云端：5 万事件/月。
- LangSmith：39 美元/用户/月。
- Helicone 免费：10 万请求/月。
- Arize AX 声明：规模下比单体便宜约 100 倍。
- OpenTelemetry GenAI 约定：2025 年发布，2026 年广泛采用。

## 使用它

`code/main.py` 模拟跨保留策略（100% 摄入、采样、采样 + 错误）的每天 100 万轨迹。报告每种策略的存储成本和丢失内容。

## 交付它

本课产出 `outputs/skill-observability-stack.md`。给定栈、规模、预算、许可证立场，选择工具。

## 练习

1. 你的团队使用 LangChain，希望 OSS 自托管可观测性。选择 Langfuse 或 Opik 并论证。
2. 在每天 500 万轨迹的情况下，Datadog 报价 15 万美元/月，计算 Arize AX 的盈亏平衡点。
3. 设计你的组织指导方针应在每次 LLM 调用上强制要求的 OpenTelemetry GenAI 属性集。
4. 论证 Phoenix 单独是否足以用于生产。何时不够用？
5. Helicone 有 20 毫秒的代理开销。在 P99 TTFT 300 毫秒时可以接受吗？如果 SLA 是 100 毫秒呢？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| OpenLLMetry | "LLM 的 OTel" | 面向 LLM 的开源 OpenTelemetry 埋点 |
| GenAI conventions（GenAI 约定） | "OTel 属性" | LLM 调用的标准 OTel 属性名称 |
| LangSmith | "LangChain 可观测性" | 与 LangChain 生态捆绑的商业平台 |
| Langfuse | "OSS LangSmith" | 具有类似功能集的 MIT OSS |
| Phoenix | "Arize 开发工具" | OpenTelemetry 原生开发/评估平台 |
| Arize AX | "规模化可观测性" | 商业零拷贝 Iceberg/Parquet 可观测性 |
| Helicone | "代理可观测性" | 收集 LLM 遥测 + 网关功能的 HTTP 代理 |
| Opik | "Comet LLM" | 来自 Comet 的 Apache 2.0 OSS 开发平台 |
| Session replay（会话回放） | "轨迹重放" | 重放包含工具调用的完整智能体会话 |
| Eval（评估） | "离线测试" | 在标注数据集上运行候选模型/提示词 |

## 延伸阅读

- [SigNoz — 2026 年顶级 LLM 可观测性工具](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX 替代方案分析](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — 设置 Langfuse、LangSmith、Helicone、Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix 文档](https://docs.arize.com/phoenix)
- [Helicone 文档](https://docs.helicone.ai/)

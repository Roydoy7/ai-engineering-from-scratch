# 智能体可观测性：Langfuse、Phoenix、Opik（Agent Observability: Langfuse, Phoenix, Opik）

> 三个开源智能体可观测性平台主导 2026 年。Langfuse（MIT）——每月 600 万+ 安装，追踪 + 提示词管理 + 评估 + 会话回放。Arize Phoenix（Elastic 2.0）——深度智能体特定评估、RAG 相关性、OpenInference 自动 Instrumentation。Comet Opik（Apache 2.0）——自动化提示词优化、护栏、LLM 评判幻觉检测。

**类型：** 学习  
**语言：** Python（标准库）  
**前置知识：** Phase 14 · 23（OTel GenAI）  
**预计时间：** 约 45 分钟

## 学习目标

- 说出三个主要开源智能体可观测性平台及其许可证。
- 区分各自最擅长的领域：Langfuse（提示词管理 + 会话）、Phoenix（RAG + 自动 Instrumentation）、Opik（优化 + 护栏）。
- 解释为什么 89% 的组织在 2026 年报告已部署智能体可观测性。
- 实现一个带 LLM 评判评估的标准库追踪到仪表板流水线。

## 问题所在

OTel GenAI（第 23 课）给你 schema。你仍然需要能够摄入跨度、运行评估、存储提示词版本并呈现回归的平台。三个竞争者各自强调生命周期的不同部分。

## 核心概念

### Langfuse（MIT）

- 每月 600 万+ SDK 安装，19k+ GitHub Stars。
- 功能：追踪、带版本控制 + 操场的提示词管理、评估（LLM 作为评判、用户反馈、自定义）、会话回放。
- 2025 年 6 月：前商业模块（LLM 作为评判、标注队列、提示词实验、操场）以 MIT 开源。
- 最强项：带紧密提示词管理循环的端到端可观测性。

### Arize Phoenix（Elastic License 2.0）

- 更深度的智能体特定评估：追踪聚类、异常检测、RAG 的检索相关性。
- 原生 OpenInference 自动 Instrumentation。
- 与托管的 Arize AX 配对用于生产。
- 无提示词版本控制——定位为与更广泛平台配合的漂移/行为回归工具。
- 最强项：RAG 相关性、行为漂移、异常检测。

### Comet Opik（Apache 2.0）

- 通过 A/B 实验自动化提示词优化。
- 护栏（PII 编辑、主题约束）。
- LLM 评判幻觉检测。
- 来自 Comet 自身测量的基准测试：Opik 日志 + 评估耗时 23.44s，而 Langfuse 为 327.15s（约 14 倍差距）——将供应商基准测试视为方向性参考。
- 最强项：优化循环、自动化实验、护栏执行。

### 行业数据

根据 Maxim（2026 年现场分析）：89% 的组织已部署智能体可观测性；质量问题是首要的生产障碍（32% 的受访者引用）。

### 如何选择

| 需求 | 选择 |
|------|------|
| 带提示词管理的一体化方案 | Langfuse |
| 深度 RAG 评估 + 漂移 | Phoenix |
| 自动化优化 + 护栏 | Opik |
| 开放许可证，无 ELv2 | Langfuse（MIT）或 Opik（Apache 2.0） |
| Datadog / New Relic 集成 | 任意——它们都导出 OTel |

### 这个模式在哪里出错

- **没有评估策略。** 没有评估的追踪只是昂贵的日志记录。
- **没有基础的自制 LLM 评判。** CRITIC 模式（第 05 课）适用——评判者需要外部工具进行事实验证。
- **提示词版本未绑定到追踪。** 当生产回归时，你无法二分到导致问题的提示词。

## 构建它

`code/main.py` 实现了一个标准库追踪收集器 + LLM 评判评估器：

- 摄入 GenAI 形态的跨度。
- 按会话分组，标记失败运行（护栏触发、低置信度评估）。
- 一个根据评分标准对智能体响应评分的脚本化 LLM 评判。
- 类仪表板摘要：失败率、主要失败原因、评估分数分布。

运行：

```
python3 code/main.py
```

输出：每会话评估分数和失败分类，与 Langfuse/Phoenix/Opik 显示的内容相匹配。

## 使用它

- **Langfuse** 自托管或云端；通过 OTel 或其 SDK 连接。
- **Arize Phoenix** 自托管；自动 Instrument OpenInference。
- **Comet Opik** 自托管或云端；自动化优化循环。
- **Datadog LLM Observability** 适合已经运行 Datadog 的混合运维 + ML 团队。

## 交付它

`outputs/skill-obs-platform-wiring.md` 选择一个平台，将追踪 + 评估 + 提示词版本连接到现有智能体。

## 练习

1. 将一周的 OTel 追踪导出到 Langfuse 云（免费版）。哪些会话失败了？为什么？
2. 为你的领域编写 LLM 评判评分标准（事实正确性、语气、范围遵从性）。在 50 条追踪上测试。
3. 对比 Langfuse 提示词版本控制与 Phoenix 的追踪聚类。哪个更快告诉你什么出错了？
4. 阅读 Opik 的护栏文档。将 PII 编辑护栏连接到你的一个智能体运行。
5. 在你的语料库上对三者进行基准测试。忽略供应商发布的数字；测量你自己的。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Tracing（追踪） | "跨度收集器" | 摄入 OTel / SDK 跨度；按会话索引 |
| Prompt management（提示词管理） | "提示词 CMS" | 绑定到追踪的版本化提示词 |
| LLM-as-judge（LLM 作为评判） | "自动化评估" | 单独的 LLM 根据评分标准对智能体输出评分 |
| Session replay（会话回放） | "追踪回放" | 逐步查看过去运行以进行调试 |
| RAG relevancy（RAG 相关性） | "检索质量" | 检索到的上下文是否与查询匹配 |
| Trace clustering（追踪聚类） | "行为分组" | 对相似运行聚类以进行漂移检测 |
| Guardrail enforcement（护栏执行） | "日志时策略" | 对记录内容进行 PII/毒性/范围检查 |

## 延伸阅读

- [Langfuse 文档](https://langfuse.com/) — 追踪、评估、提示词管理
- [Arize Phoenix 文档](https://docs.arize.com/phoenix) — 自动 Instrumentation、漂移
- [Comet Opik](https://www.comet.com/site/products/opik/) — 优化 + 护栏
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 三者消费的 schema

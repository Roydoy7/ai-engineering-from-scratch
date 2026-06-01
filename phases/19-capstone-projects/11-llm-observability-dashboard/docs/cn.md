# 压轴项目 11——LLM 可观测性与评估仪表板（Capstone 11 — LLM Observability & Eval Dashboard）

> Langfuse 采用了开放核心模式。Arize Phoenix 发布了 2026 年 GenAI semconv 映射。Helicone 和 Braintrust 都在每用户成本归因上加倍投入。Traceloop 的 OpenLLMetry 成为事实上的 SDK 埋点标准。生产形态是 ClickHouse 用于追踪，Postgres 用于元数据，Next.js 用于 UI，以及一小批在采样追踪上运行的评估任务（DeepEval、RAGAS、LLM 裁判）。构建一个自托管的系统，从至少四个 SDK 系列摄入数据，并演示在五分钟内捕获注入的回归。

**类型：** 压轴项目  
**语言：** TypeScript（UI），Python / TypeScript（摄入 + 评估），SQL（ClickHouse）  
**前置知识：** Phase 11（LLM 工程）、Phase 13（工具）、Phase 17（基础设施）、Phase 18（安全）  
**涉及的阶段：** P11 · P13 · P17 · P18  
**预计时间：** 25 小时

## 问题所在

2026 年，每个运行生产流量的 AI 团队都在模型旁边维护一个可观测性层。成本归因、幻觉检测、漂移监控、越狱信号、SLO 仪表板、PII 泄露告警。开源参考——Langfuse、Phoenix、OpenLLMetry——以 OpenTelemetry GenAI 语义约定作为摄入 schema 收敛了。你现在可以用一个 SDK 对 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM 进行埋点，并发送兼容的 span。

你将构建一个自托管仪表板，从至少四个 SDK 系列摄入，在采样追踪上运行一小组评估任务，检测漂移并发出告警。测量标准：给定故意注入的回归（一个开始生成 PII 的提示词），仪表板在五分钟内捕获并触发告警。

## 核心概念

摄入是 OTLP HTTP。SDK 生成 GenAI-semconv span：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。span 进入 ClickHouse 进行列式分析；元数据（用户、会话、应用）进入 Postgres。

评估作为批量任务在采样追踪上运行。DeepEval 对忠实度、毒性和答案相关性评分。当追踪携带检索上下文时，RAGAS 对检索指标评分。自定义 LLM 裁判运行特定领域检查（PII 泄露、违反政策的响应）。评估运行将评估 span 写回同一 ClickHouse，链接到父追踪。

漂移检测监视随时间变化的嵌入空间分布（提示词嵌入上的 PSI 或 KL 散度）加上评估分数趋势。告警通过 Prometheus Alertmanager 然后 Slack / PagerDuty 传递。UI 是 Next.js 15 加 Recharts。

## 架构

```
生产应用：
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  带 GenAI semconv 的 OpenTelemetry SDK
       |
       v  OTLP HTTP
  收集器（摄入，采样，扇出）
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 存档
   (span)        (元数据)    (原始事件)
       |
       +---> 评估任务（DeepEval，RAGAS，LLM 裁判）
       |     采样或全追踪
       |     将评估 span 写回
       |
       +---> 漂移检测器（提示词嵌入上的 PSI / KL）
       |
       +---> Prometheus 指标 -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 仪表板（Recharts）
```

## 技术栈

- 摄入：带 GenAI 语义约定的 OpenTelemetry SDK；OTLP HTTP 传输
- 收集器：带尾部采样处理器的 OpenTelemetry Collector（用于成本控制）
- 存储：ClickHouse 用于 span，Postgres 用于元数据，S3 用于原始事件存档
- 评估：DeepEval，RAGAS 0.2，Arize Phoenix 评估器包，自定义 LLM 裁判
- 漂移：池化提示词嵌入上的 PSI / KL（sentence-transformers）每周
- 告警：Prometheus Alertmanager -> Slack / PagerDuty
- UI：Next.js 15 App Router + Recharts + 服务器操作
- 开箱支持的 SDK：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

## 构建它

1. **收集器配置。** OpenTelemetry Collector，带 OTLP HTTP 接收器，保留 100% 错误追踪和 10% 成功追踪的尾部采样器，以及到 ClickHouse 和 S3 的导出器。

2. **ClickHouse schema。** `spans` 表，列镜像 GenAI semconv：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，加上用于长有效负载的 JSON 包。按 user_id 和 app_id 添加二级索引。

3. **SDK 覆盖率测试。** 用每个 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）用 OpenLLMetry 自动埋点编写一个小型客户端应用。验证每个都产生落入 ClickHouse 的规范 GenAI span。

4. **评估任务。** 定期任务读取过去 15 分钟的采样追踪并运行 DeepEval 忠实度、毒性和答案相关性。输出是链接到父追踪的评估 span。

5. **自定义 LLM 裁判。** PII 泄露裁判：给定响应，调用守护 LLM 对 PII 泄露可能性评分。高分响应进入分类队列。

6. **漂移检测。** 每周任务计算本周池化提示词嵌入与过去 4 周基线之间的 PSI。如果 PSI 超过阈值，发出告警。

7. **仪表板。** Next.js 15，带页面：概览（span/秒，每用户成本，p95 延迟），追踪（搜索 + 瀑布），评估（忠实度趋势，毒性），漂移（随时间的 PSI），告警。

8. **告警链。** Prometheus 导出器读取评估分数聚合和延迟百分位数；Alertmanager 对警告路由到 Slack，对严重违规路由到 PagerDuty。

9. **回归探测。** 注入一个 bug：被评估的聊天机器人开始 1% 的时间泄露假 SSN。测量 MTTR：从 bug 部署到 Slack 告警。

## 使用它

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[收集器]  接受 1 个追踪，3 个 span
[clickhouse] 插入 3 个 span（app=chat，user=u_42）
[评估]   DeepEval 忠实度 0.82，毒性 0.03
[漂移]   每周 PSI 0.08（低于 0.2 阈值）
[ui]     实时显示在 https://obs.example.com
```

## 交付它

`outputs/skill-llm-observability.md` 是可交付成果。给定 LLM 应用，仪表板摄入其追踪，运行评估，对漂移发出告警，并在 Next.js 中显示每用户成本分解。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 追踪 schema 覆盖率 | 产生规范 GenAI span 的 SDK 系列数（目标：6+） |
| 20 | 评估正确性 | DeepEval / RAGAS 分数 vs 手工标注集 |
| 20 | 仪表板用户体验 | 注入回归的 MTTR（目标：五分钟以内） |
| 20 | 成本 / 规模 | 在 1k span/秒的持续摄入下没有积压 |
| 15 | 告警 + 漂移检测 | Prometheus/Alertmanager 链端到端演练 |
| **100** | | |

## 练习

1. 为 Haystack 框架添加自定义埋点。验证规范 span 带有忠实的 `gen_ai.*` 属性落入 ClickHouse。

2. 在相同追踪上将 DeepEval 换为 Phoenix 评估器。测量两个评估引擎之间的分数漂移。

3. 细化漂移检测器：按 app-id 而非全局计算 PSI。显示每应用漂移轨迹。

4. 添加"用户影响"页面：每用户成本和每用户失败率，带迷你图。

5. 构建一个尾部采样策略，保留 100% 毒性 > 0.5 的追踪加上其余的 10% 分层样本。测量引入的采样偏差。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| GenAI semconv | "OTel LLM 属性" | 2025 年 LLM span 属性的 OpenTelemetry 规范（系统、模型、token） |
| Tail sampling（尾部采样） | "追踪后采样" | 收集器在追踪完成后决定保留或丢弃（可以查看错误） |
| PSI | "种群稳定性指数" | 比较两个分布的漂移指标；> 0.2 通常表示有意义的漂移 |
| LLM-judge（LLM 裁判） | "以模型为评估" | 用一个 LLM 按规则对另一个 LLM 的输出评分（忠实度，毒性，PII） |
| Tail-sampling policy（尾部采样策略） | "保留规则" | 决定保留哪些追踪 vs 丢弃的规则；错误追踪 + 采样率 |
| Eval span（评估 span） | "链接的评估追踪" | 链接到原始 LLM 调用 span 的子 span，携带评估分数 |
| Cost per user（每用户成本） | "单位经济学" | 在一个窗口内归因于 user_id 的美元成本；关键产品指标 |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) — 参考开放核心可观测性平台
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 带强漂移支持的备用参考
- [OpenLLMetry（Traceloop）](https://github.com/traceloop/openllmetry) — 自动埋点 SDK 系列
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 摄入 schema
- [Helicone](https://www.helicone.ai) — 备用托管可观测性
- [Braintrust](https://www.braintrust.dev) — 备用评估优先平台
- [ClickHouse 文档](https://clickhouse.com/docs) — 列式 span 存储
- [DeepEval](https://github.com/confident-ai/deepeval) — 评估器库

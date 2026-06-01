# LLM API 负载测试——为什么 k6 和 Locust 会说谎（Load Testing LLM APIs — Why k6 and Locust Lie）

> 传统负载测试工具并非为流式响应、可变输出长度、token 级指标或 GPU 饱和而设计。两个陷阱会让大多数团队踩坑。GIL 陷阱：Locust 的 token 级测量在 Python GIL 下运行分词，这在高并发时与请求生成相竞争；分词积压使报告的 token 间延迟虚高——你的客户端才是瓶颈，而非服务器。提示词均一性陷阱：在循环中使用相同提示词测试的是 token 分布上的一个点；真实流量具有可变长度和多样的前缀匹配。LLMPerf 通过 `--mean-input-tokens` + `--stddev-input-tokens` 解决了这个问题。2026 年的工具映射：LLM 专属工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于 token 级精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**——流式感知，通过 TestRun/PrivateLoadZone CRD 实现 Kubernetes 原生分布式，最适合 CI/CD 门控；Vegeta 用于 Go 恒速饱和测试；Locust 2.43.3 仅在使用 LLM-Locust 扩展用于流式时才适用。负载模式：稳态、斜坡、尖峰（自动扩缩容测试）、浸泡（内存泄漏）。

**类型：** 构建  
**语言：** Python（标准库，玩具真实提示词生成器 + 延迟收集器）  
**前置知识：** Phase 17 · 08（推理指标）、Phase 17 · 03（GPU 自动扩缩容）  
**预计时间：** 约 75 分钟

## 学习目标

- 解释使通用负载测试工具对 LLM API 说谎的两个反模式（GIL 陷阱、提示词均一性陷阱）。
- 为特定目的选择工具：LLMPerf（基准测试），k6 + 流式扩展（CI 门控），guidellm（大规模合成），GenAI-Perf（NVIDIA 参考）。
- 设计四种负载模式（稳态、斜坡、尖峰、浸泡），并说出每种捕获的故障模式。
- 使用输入 token 的均值 + 标准差（而非固定长度）构建真实的提示词分布。

## 问题所在

你用 k6 对 LLM 端点进行了 500 并发用户测试。它撑住了。你发布了。在实际 200 用户的生产中，服务崩溃了——P99 TTFT 爆炸，GPU 钉死了。

发生了两件事。首先，k6 发送了 500 个相同提示词——你的请求合并和前缀缓存让它看起来像是你在处理 500 个并发解码，而你实际上只在处理一个。其次，k6 不像眼睛感受那样追踪流式响应上的 token 间延迟；它看到一个 HTTP 连接，而不是以不同间隔到达的 500 个 token。

LLM 的负载测试是一门独立学科。

## 核心概念

### GIL 陷阱（Locust）

Locust 使用 Python，在 GIL 下客户端运行分词。在高并发下，分词器在请求生成后面排队。报告的 token 间延迟包括客户端分词积压。你以为服务器慢；实际上是测试框架慢。

修复：LLM-Locust 扩展将分词移到独立进程，或使用编译语言框架（k6，使用 tokenizers.rs 的 LLMPerf）。

### 提示词均一性陷阱

所有已知的负载测试工具都允许你配置一个提示词。在 10000 次迭代的循环测试中，每次都发送完全相同的提示词。服务器每次都看到相同前缀——前缀缓存命中率接近 100%，吞吐量看起来很好。

修复：从提示词分布中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`——多样的长度，多样的内容。

### 四种负载模式

1. **稳态** — 恒定 RPS 持续 30-60 分钟。捕获：基线性能回归。
2. **斜坡** — 在 15 分钟内从 0 线性增加到目标 RPS。捕获：容量断点，预热异常。
3. **尖峰** — 突然 3-10 倍 RPS 持续 2 分钟然后恢复。捕获：自动扩缩容延迟，队列饱和，冷启动影响。
4. **浸泡** — 稳态持续 4-8 小时。捕获：内存泄漏，连接池漂移，可观测性溢出。

### 2026 年工具映射

**LLMPerf**（Anyscale）— Python 但 Rust 支持的分词。均值/标准差提示词。流式感知。性能运行的最佳默认选择。

**NVIDIA GenAI-Perf** — NVIDIA 的参考工具。使用 Triton 客户端；全面的指标覆盖。注意其 ITL 不包括 TTFT；LLMPerf 的包括。两个工具对同一服务器产生不同的 TPOT。

**LLM-Locust**（TrueFoundry）— 修复 GIL 陷阱的 Locust 扩展。熟悉的 Locust DSL + 流式指标。

**guidellm** — 大规模合成基准测试。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本身（Go，编译，无 GIL）添加了流式感知指标。
- k6 Operator 使用 TestRun / PrivateLoadZone CRD 进行 Kubernetes 原生分布式测试。
- 最适合 CI/CD 门控和 SLA 测试。

**Vegeta** — Go，比 k6 更简单。恒速 HTTP 饱和测试。不是 LLM 感知的，但适合网关/速率限制测试。

**Locust 2.43.3 原版** — 对 LLM 有 GIL 陷阱。仅在使用 LLM-Locust 扩展时才有效。

### CI 中的 SLA 门控

在 PR 上运行 k6：

- 每个基线 RPS 下 30-50 次迭代。
- 门控：P50/P95 TTFT，5xx < 5%，TPOT 低于阈值。
- 违规时中断构建。

### 真实提示词分布

从真实流量样本（如果有的话）或已发布分布（例如，聊天使用 ShareGPT 提示词，代码使用 HumanEval）构建。将均值 + 标准差提供给 LLMPerf。不惜一切代价避免循环使用单一提示词。

### 你应该记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：流式感知指标。
- 典型 LLMPerf 运行：并发数 X 下 100-1000 个请求。
- 典型 CI 门控：每个 PR 30-50 次迭代。
- 四种模式：稳态、斜坡、尖峰、浸泡。

## 使用它

`code/main.py` 模拟具有真实提示词分布的负载测试，测量有效 TPOT，并演示均一提示词陷阱。

## 交付它

本课产出 `outputs/skill-load-test-plan.md`。给定工作负载和 SLA，选择工具并设计四种负载模式。

## 练习

1. 运行 `code/main.py`。比较均一 vs 真实分布——差距在哪里？
2. 编写 CI 门控的 k6 脚本：100 并发时 TTFT P95 < 800 毫秒，运行时间 5 分钟。
3. 你的浸泡测试显示内存每小时增长 50 MB。说出三个原因和区分它们的监控手段。
4. 尖峰测试从 10 RPS 到 100 RPS。如果 Karpenter + vLLM production-stack 到位（Phase 17 · 03 + 18），预期恢复时间是多少？
5. GenAI-Perf 报告同一服务器的 TPOT=6ms；LLMPerf 报告 TPOT=11ms。解释原因。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| LLMPerf | "LLM 测试框架" | Anyscale 基准测试工具，流式感知 |
| GenAI-Perf | "NVIDIA 工具" | NVIDIA 参考测试框架 |
| LLM-Locust | "LLM 的 Locust" | 修复 GIL 陷阱的 Locust 扩展 |
| guidellm | "合成基准" | 大规模合成工具 |
| k6 Operator | "K8s 版 k6" | 基于 CRD 的分布式 k6 |
| GIL trap（GIL 陷阱） | "Python 客户端开销" | 分词积压使报告延迟虚高 |
| Prompt-uniformity trap（提示词均一性陷阱） | "单提示词谎言" | 循环使用同一提示词命中缓存，使吞吐量虚高 |
| Steady-state（稳态） | "恒定负载" | 固定 RPS 持续 N 分钟 |
| Ramp（斜坡） | "线性增加" | 在持续时间内从 0 增加到目标 |
| Spike（尖峰） | "突发测试" | 突然倍数增加然后恢复 |
| Soak（浸泡） | "长时间测试" | 数小时用于泄漏检测 |

## 延伸阅读

- [TianPan — LLM 应用负载测试](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — 2026 年 LLM 负载测试](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — LLM 推理基准测试简介](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)

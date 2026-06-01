# 推理指标——TTFT、TPOT、ITL、吞吐量、P99（Inference Metrics — TTFT, TPOT, ITL, Goodput, P99）

> 四个指标决定了推理部署是否正常工作。TTFT 是预填充加队列加网络。TPOT（等同于 ITL）是内存受限的每 token 解码成本。端到端延迟是 TTFT 加 TPOT 乘以输出长度。吞吐量是整个集群每秒聚合的 token 数。但产品真正关心的是吞吐量（goodput）——满足所有 SLO 的请求比例。高吞吐量但低 goodput 意味着你在处理永远无法及时到达用户的 token。2026 年 Llama-3.1-8B-Instruct 在 TRT-LLM 上的参考数字：均值 TTFT 162 毫秒，均值 TPOT 7.33 毫秒，均值端到端 1093 毫秒。始终报告 P50、P90、P99——永远不要只报告均值。警惕测量陷阱：GenAI-Perf 将 TTFT 排除在 ITL 计算之外，LLMPerf 则包含在内；两个工具对同一次运行的 TPOT 有分歧。

**类型：** 学习  
**语言：** Python（标准库，玩具百分位数计算器和 goodput 报告器）  
**前置知识：** Phase 17 · 04（vLLM 服务内部原理）  
**预计时间：** 约 60 分钟

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、吞吐量和 goodput，并说出每个测量的组件。
- 解释为什么均值对于 LLM 服务是错误的统计量，以及如何读取 P50/P90/P99。
- 构建多约束 SLO（例如 TTFT<500 毫秒 AND TPOT<15 毫秒 AND E2E<2 秒）并据此计算 goodput。
- 说出两个对同一次运行的 TPOT 有分歧的基准工具，并解释原因。

## 问题所在

"我们的吞吐量是每秒 15000 个 token。"那又怎样？如果 40% 的请求超过了 2 秒端到端，用户已经放弃了会话。单纯的吞吐量无法告诉你产品是否正常工作。

推理有多个延迟维度，每个维度的失败方式不同。预填充是计算密集型的，随提示词长度扩展。解码是内存带宽受限的，随批次大小扩展。排队延迟是运营问题。网络是物理距离问题。你需要为每个维度设置不同的指标，需要百分位数，还需要一个单一的复合指标来说明"用户是否得到了他们期望的"——这就是 goodput。

## 核心概念

### TTFT——首 token 时间

`TTFT = 队列时间 + 网络请求 + 预填充时间`

提示词很长时，预填充占主导。在 H100 上运行 Llama-3.3-70B FP8，32k token 提示词需要约 800 毫秒的纯预填充时间。队列时间是负载下的调度器行为。网络请求是包括 TLS 在内的线路时间。TTFT 是用户在任何内容流回来之前看到的延迟。

### TPOT / ITL——token 间隔延迟

同一量的多个名称。`TPOT`（每输出 token 时间）、`ITL`（token 间隔延迟）、`每 token 解码延迟`——都是同一回事。这是第一个 token 之后连续流式 token 之间的时间。

`TPOT = (解码前向时间 + 调度器开销) / 产生的 token 数`

在同一个 Llama-3.3-70B H100 栈上开启分块预填充时，TPOT 均值约 7 毫秒。不开启分块预填充时，相邻序列上有长预填充期间，TPOT 可能飙升到 50 毫秒。关注 P99，而非均值。

### 端到端延迟

`E2E = TTFT + TPOT * 输出 token 数 + 网络响应`

对于长输出（>500 个 token），E2E 由 TPOT 主导。对于提示词长但输出短的情况，E2E 由 TTFT 主导。报告按输出长度条件化的 E2E。

### 吞吐量

`吞吐量 = 总输出 token 数 / 经过时间`

聚合指标。告诉你集群效率。不告诉你单个请求的健康状况。

### Goodput——你真正关心的指标

`goodput = 满足 (TTFT <= a) AND (TPOT <= b) AND (E2E <= c) 的请求比例`

SLO 是多约束的。只有每个约束都满足，请求才是"好的"。Goodput 是其比例。高吞吐量下 60% goodput 是失败。较低吞吐量下 99% goodput 才是目标。

2026 年，goodput 是 MLPerf Inference v6.0 提交中使用的指标，也是 AI 平台提供商内部 SLA 追踪中使用的指标。

### 为什么均值是错误的统计量

LLM 延迟分布是右偏的。一个有长预填充邻居的解码批次可能发出 500 个 token，TPOT 约 7 毫秒，以及 20 个 token，TPOT 约 60 毫秒。均值 TPOT 是 9 毫秒。P99 TPOT 是 65 毫秒。用户会频繁遇到 P99——这就是他们离开的原因。

始终报告三元组（P50、P90、P99）。对于用户体验，P99 是你要优化的。

### 参考数字——2026 年 Llama-3.1-8B-Instruct 在 TRT-LLM 上

- 均值 TTFT：162 毫秒
- 均值 TPOT：7.33 毫秒
- 均值 E2E：1093 毫秒
- P99 TPOT：根据分块预填充配置，在 10-25 毫秒之间变动。

这些是 NVIDIA 发布的参考数字。它们随模型大小（70B 会显示 3-5 倍）、硬件（H100 vs B200 约 3 倍）和负载而变化。

### 测量陷阱

2026 年最常用的两个基准工具对同一次运行的 TPOT 有分歧：

- **NVIDIA GenAI-Perf**：将 TTFT 排除在 ITL 计算之外。ITL 从第 2 个 token 开始。
- **LLMPerf**：包含 TTFT。ITL 从第 1 个 token 开始。

对于 TTFT 500 毫秒、100 个输出 token、总解码 700 毫秒的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 毫秒`，LLMPerf 报告 `ITL = 1200/100 = 12.00 毫秒`。工具选择改变数字。

始终说明使用的工具。始终发布定义。

### 构建 SLO

2026 年面向消费者的 70B 聊天模型的合理 SLO：

- TTFT P99 <= 800 毫秒。
- TPOT P99 <= 25 毫秒。
- E2E P99 <= 3 秒（<300 token 输出）。
- Goodput 目标 >= 99%。

企业 SLO 收紧 TTFT（200-400 毫秒）并放宽 E2E。重点是将它们写下来，测量全部三个，并将 goodput 作为单一复合指标追踪。

### 如何测量

- 运行真实流量或真实的合成流量（LLMPerf 配合 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- 以目标峰值并发的 2 倍运行基准测试。
- 运行 30-50 次迭代，对合并样本取百分位数。
- 发布时附带工具名称、工具版本、模型、硬件、并发度、提示词分布。

## 使用它

`code/main.py` 是一个玩具 goodput 计算器。生成合成延迟分布，应用 SLO，计算 goodput。还在同一轨迹上展示 GenAI-Perf vs LLMPerf 的 TPOT 差异。

## 交付它

本课产出 `outputs/skill-slo-goodput-gate.md`。给定工作负载和 SLO，它产出一个 CI/CD 就绪的基准方案，以 goodput 而非吞吐量作为部署门控。

## 练习

1. 运行 `code/main.py`。生成一个具有 1% 尾部峰值的分布。当你将 P99 TPOT 从 30 毫秒收紧到 15 毫秒时，goodput 如何变化？
2. 供应商报价"H100 上 Llama 3.3 70B 每秒 15000 个 token"。在信任它之前你应该问哪三个问题？
3. 为什么分块预填充保护 P99 TPOT 但不保护均值 TPOT？
4. 为语音助手（第一个 token 是被听到的，而非阅读的）构建面向消费者的 SLO。哪个指标对用户最可见？
5. 阅读 LLMPerf 的 README 和 GenAI-Perf 文档。识别两个工具有分歧的其他三个指标。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| TTFT | "首 token 时间" | 队列 + 网络 + 预填充；长提示词时由预填充主导 |
| TPOT | "每输出 token 时间" | 第一个 token 之后的内存带宽受限解码成本 |
| ITL | "token 间隔延迟" | 在大多数工具中与 TPOT 相同（不是全部——见 GenAI-Perf） |
| E2E | "端到端" | TTFT + TPOT × 输出长度；加上响应端网络 |
| Throughput（吞吐量） | "tok/s" | 集群效率；没有延迟百分位数时毫无意义 |
| Goodput | "SLO 满足率" | 同时满足所有 SLO 约束的请求比例 |
| P99 | "尾部" | 百分之一最差情况延迟；用户体验指标 |
| SLO multi-constraint（SLO 多约束） | "联合约束" | 所有三个延迟限制的 AND；任何一个违反则请求失败 |
| GenAI-Perf vs LLMPerf | "工具陷阱" | 工具对 ITL 是否包含 TTFT 有分歧 |

## 延伸阅读

- [NVIDIA NIM — LLM 基准指标](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的权威定义
- [Anyscale — LLM 服务基准指标](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 替代定义和测量方案
- [BentoML — LLM 推理指标](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 真实部署上的应用测量
- [LLMPerf](https://github.com/ray-project/llmperf) — 基于 Ray 的开源基准
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA 的基准工具
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 行业公认的基于 goodput 的基准

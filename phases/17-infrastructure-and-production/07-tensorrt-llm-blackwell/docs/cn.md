# Blackwell 上的 TensorRT-LLM：FP8 与 NVFP4（TensorRT-LLM on Blackwell with FP8 and NVFP4）

> TensorRT-LLM 只能用于 NVIDIA，但它在 Blackwell 上获胜。在配合 Dynamo 编排的 GB200 NVL72 上，SemiAnalysis InferenceX 在 2026 年第一二季度测量到 120B 模型每百万 token 成本 0.012 美元，而 H100 + vLLM 为 0.09 美元/百万 token——7 倍的经济差距。这个栈由三种浮点制度叠加而成：FP8 对 KV 缓存和注意力内核至关重要，因为它们需要其动态范围；NVFP4（4 位微缩放）处理权重和激活；多 token 预测（MTP）和分离式预填充/解码在此之上再增加 2-3 倍。Day-0 模型支持直接加载 FP4 权重，无需训练后转换。2026 年工程团队的坑：TRT-LLM 是封闭的 NVIDIA 栈，采用它意味着用可移植性换取吞吐量。在承诺之前，先对你的模型和硬件组合进行核算。

**类型：** 学习  
**语言：** Python（标准库，玩具 FP8/NVFP4 内存和成本计算器）  
**前置知识：** Phase 17 · 04（vLLM 服务内部原理）、Phase 10 · 13（量化）  
**预计时间：** 约 75 分钟

## 学习目标

- 解释为什么即使权重在 NVFP4 中，FP8 对于 KV 缓存和注意力仍然至关重要。
- 计算前沿模型在 BF16、FP8 和 NVFP4 下的 HBM 占用，并推理节省来自何处。
- 说出 TRT-LLM 利用的 Blackwell 专属特性（Day-0 FP4、MTP、分离式服务、全对全原语）。
- 判断 TRT-LLM 的 NVIDIA 锁定何时值得以换取相对于 Hopper 上 vLLM 的 7 倍成本差距。

## 问题所在

2026 年推理经济学的前沿是"每美元多少 token"。答案取决于四个叠加的选择：硬件代数（Hopper H100/H200 vs Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、服务引擎（vLLM vs SGLang vs TRT-LLM）和编排方式（普通 vs 分离式 vs Dynamo）。

在 Hopper 加 vLLM 上，120B MoE 的运行成本约为每百万 token 0.09 美元。在 Blackwell 加 TRT-LLM + Dynamo 上，同一模型的成本约为 0.012 美元——便宜 7 倍。一部分差距来自硬件（Blackwell 的每 GPU LLM 吞吐量是 Hopper 的 11-15 倍）。一部分来自栈：FP4 权重、MTP 草稿、分离式预填充/解码，以及用于 MoE 专家通信的 NVLink 5 全对全。

你无法在 NVIDIA 栈之外复制这个结果。这就是权衡——可移植性换取经济性。理解哪些栈选择贡献了多大份额的差距，是本课的重点。

## 核心概念

### 为什么 FP8 仍然是 KV 缓存的底线

2026 年的常见错误：假设 NVFP4 适用于所有地方。它不适用。KV 缓存需要 FP8（8 位浮点），因为它存储跨越宽动态范围的注意力键和值。将 KV 量化到 FP4 会造成灾难性的精度损失——分布尾部丢失，注意力分数崩溃。FP8 的指数位为 KV 缓存提供了所需的范围。

NVFP4（2025-2026）适用于权重和激活。微缩放：每个权重块都有自己的缩放因子，因此小块可以跨越不同的动态范围而不会有每张量缩放损失。对于激活，FP4 能hold住，因为激活在一层内范围较小。

典型的 Blackwell 配置：

- 权重：NVFP4（4 位微缩放）。
- 激活：NVFP4。
- KV 缓存：FP8。
- 注意力累加器：FP32（softmax 稳定性）。

### TRT-LLM 使用的 Blackwell 专属原语

- **Day-0 FP4 权重**：模型提供商直接发布 FP4 权重；TRT-LLM 无需训练后转换即可加载。FP4 不需要 AWQ / GPTQ 步骤。
- **多 token 预测（MTP）**：与 EAGLE 相同的思路（Phase 17 · 05），但集成到 TRT-LLM 构建中。
- **分离式服务**：预填充和解码在独立的 GPU 池上，KV 缓存通过 NVLink 或 InfiniBand 传输。与 Dynamo 思路相同（Phase 17 · 20）。
- **全对全通信原语**：NVLink 5 将 MoE 专家通信延迟相比 Hopper 降低 3 倍。TRT-LLM 的 MoE 内核针对此进行了调优。
- **NVFP4 + MXFP8 微缩放**：Blackwell Tensor Core 上的硬件加速缩放因子处理。

### 你应该记住的数字

- HGX B200 通过 TRT-LLM 在 GPT-OSS-120B 上的成本：0.02 美元/百万 token。
- GB200 NVL72 通过 Dynamo（编排 TRT-LLM）的成本：0.012 美元/百万 token。
- H100 + vLLM 在可比工作负载上约 0.09 美元/百万 token。
- 2026 年三个月内 TRT-LLM 更新带来 2.8 倍吞吐量提升。
- Blackwell vs Hopper 的每 GPU LLM 吞吐量：11-15 倍。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 在每个提交任务上占主导地位。

### FP4 实际上在质量上的代价

NVFP4 很激进。在推理密集型工作负载（思维链、数学、长上下文代码生成）上，FP4 权重的退化是可见的。按块校准能缓解但无法消除。发布推理模型的团队通常使用 FP8 权重 + FP4 激活作为折中，或坚持在 H200 上全程使用 FP8。

规则：在承诺使用 NVFP4 权重之前，务必在你的评估集上验证任务质量。

### 为什么这是一个 NVIDIA 锁定决策

TRT-LLM 是 C++ + CUDA + 封闭源内核。模型需要针对特定 GPU SKU 编译。没有 AMD，没有 Intel，没有 ARM。如果你的基础设施策略是多供应商，TRT-LLM 在 TRT-LLM 服务层是行不通的——你仍然可以在混合硬件上用 vLLM 服务。如果你只用 NVIDIA，7 倍的差距值得接受锁定。

### 2026 年的实用方案

对于年推理账单超过 1 亿美元的情况，在 Hopper + vLLM 上运行意味着放弃了 7-10 倍的效益。将成本主导型工作负载迁移到 Blackwell + TRT-LLM + Dynamo。在 H100 + vLLM 上保留实验层，以获得模型迭代速度。在生产之前验证每个 NVFP4 转换模型的质量。

### 分离式服务的额外收益

TRT-LLM 的分离式服务（独立的预填充和解码池）在 Phase 17 · 20 中深入介绍。在 Blackwell 上，倍数叠加：FP4 权重 × MTP 加速 × 分离式部署 × 缓存感知路由。7 倍的数字假设了这个完整栈。

## 使用它

`code/main.py` 计算跨三个栈的模型 HBM 占用、解码吞吐量（内存带宽受限制度）和每百万 token 成本：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它可以看到叠加效应以及每个变化贡献的差距份额。

## 交付它

本课产出 `outputs/skill-trtllm-blackwell-advisor.md`。给定工作负载、模型大小和年 token 量，决定 Blackwell + TRT-LLM 栈是否值得接受 NVIDIA 锁定。

## 练习

1. 运行 `code/main.py`。对于具有 30% 活跃参数的 120B MoE，计算在 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 上的内存带宽受限解码吞吐量。最大的跳跃来自哪里？
2. 一个客户每年在 H100 + vLLM 上花费 200 万美元。鉴于 7 倍的经济差距，他们需要购买多少 Blackwell GPU 才能在 12 个月内摊销迁移到 TRT-LLM 的成本？
3. 你看到 NVFP4 权重转换后 MATH 基准下降了 3 个百分点。说出两条恢复路径：一条质量优先（保持 FP8 权重），一条成本优先（用领域内数据校准）。
4. 阅读 MLPerf v6.0 推理结果。哪个任务的 Blackwell 优于 Hopper 差距最小，为什么？
5. 计算 405B 模型在 NVFP4 权重 + FP8 KV 缓存 + 128k 上下文下所需的 HBM。能放进一个 GB200 NVL72 节点吗？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| FP8 | "8 位浮点" | 8 位浮点；用于 KV 缓存和注意力，因其动态范围 |
| NVFP4 | "4 位微缩放" | NVIDIA 的 4 位微缩放浮点格式；Blackwell 上的权重和激活 |
| MXFP8 | "MX 8" | 微缩放 FP8 变体；Blackwell Tensor Core 上硬件加速 |
| Day-0 FP4 | "发布 FP4 权重" | 模型提供商直接以 FP4 发布权重；无需训练后转换步骤 |
| MTP | "多 token 预测" | TRT-LLM 集成的推测解码草稿（Phase 17 · 05） |
| Disaggregated serving（分离式服务） | "拆分预填充/解码" | 预填充和解码在独立 GPU 池上；KV 通过 NVLink/IB 传输 |
| All-to-all（全对全） | "MoE 专家通信" | 将 token 路由到专家 GPU 的通信模式；NVLink 5 降低 3 倍延迟 |
| InferenceX | "SemiAnalysis 推理基准" | 2026 年行业公认的每 token 成本基准 |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf 结果
- [NVIDIA — Blackwell 上的 MoE 推理](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 全对全和 MoE 内核
- [TensorRT-LLM 概述](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官方引擎文档
- [NVIDIA — 介绍 Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM 之上的分离式编排
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell 数字的基准套件

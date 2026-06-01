# 分离式预填充/解码——NVIDIA Dynamo 与 llm-d（Disaggregated Prefill/Decode — NVIDIA Dynamo and llm-d）

> 预填充是计算密集型的；解码是内存带宽受限的。在同一 GPU 上同时运行两者会浪费其中一种资源。分离式架构将它们分别部署到独立的 GPU 池，并通过 NIXL（RDMA/InfiniBand 或 TCP 备用）在它们之间传输 KV 缓存。NVIDIA Dynamo（GTC 2025 发布，1.0 GA）位于 vLLM/SGLang/TRT-LLM 之上——其 Planner Profiler + SLA Planner 自动调整预填充:解码比率以满足 SLO。NVIDIA 发布的吞吐量提升数据：developer.nvidia.com（2025-06）显示 DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 的中等延迟情况下约有 6 倍提升，Dynamo 产品页面（developer.nvidia.com，未注明日期）宣传 GB300 NVL72 + Dynamo vs Hopper 的 MoE 吞吐量提升高达 50 倍。"30 倍"的说法是跨完整栈 Blackwell + Dynamo + DeepSeek-R1 报告的社区聚合值；我们未找到精确说明 30 倍的单一主要资料，因此将其视为方向性声明。llm-d（Red Hat + AWS）是 Kubernetes 原生的：预填充/解码/路由器作为独立的 Service，每个角色有独立的 HPA。llm-d 0.5 添加了层级 KV 卸载、缓存感知 LoRA 路由、UCCL 网络和缩容到零。经济数据：多个客户披露的内部汇总表明，在恒定 SLA 下从共置服务切换到使用 Dynamo 的分离式服务，200 万美元级推理支出可节省 30-40%（即每年 60-80 万美元）；这个具体数字是内部综合数据，而非单一已发布案例研究——将其作为数量级参考，而非引用来源。短提示词（<512 个 token，短输出）不值得承担传输成本。

**类型：** 学习  
**语言：** Python（标准库，玩具分离式 vs 共置服务模拟器）  
**前置知识：** Phase 17 · 04（vLLM 服务内部原理）、Phase 17 · 08（推理指标）  
**预计时间：** 约 75 分钟

## 学习目标

- 解释为什么预填充和解码有不同的最优 GPU 分配，并量化共置情况下的浪费。
- 画出分离式架构图：预填充池、解码池、通过 NIXL 进行 KV 传输、路由器。
- 说出分离式架构不值得的条件（短提示词，短输出）。
- 区分 NVIDIA Dynamo（栈上层）和 llm-d（Kubernetes 原生），并将每个与运营场景匹配。

## 问题所在

你在 8 块 H100 上运行 Llama 3.3 70B。在混合工作负载（长提示词 + 短输出）下，GPU 在解码期间空闲，因为大部分计算都用于预填充了。在不同工作负载（短提示词 + 长输出）下，情况相反。共置预填充 + 解码意味着两者都过度配置。

预算影响：20-40% 的 GPU 时间浪费在错误的资源上。你在购买 H100 计算能力来运行内存带宽受限的解码，或者购买 H100 HBM 带宽来运行计算密集型的预填充。两者都是昂贵的浪费。

分离式架构将预填充和解码分到按各自瓶颈配置的独立池。KV 缓存通过高带宽互连从预填充池传输到解码池。

## 核心概念

### 为什么瓶颈不同

**预填充** — 在单次前向传递中对完整输入提示词运行变换器。矩阵乘法占主导；计算密集型。H100 FP8 提供约 2000 TFLOPS 的有效吞吐量。批次效率良好——单次前向处理多个 token。

**解码** — 每次迭代逐 token 生成，读取完整权重。内存带宽受限。HBM3 提供约 3 TB/s。批次效率仅在高并发时良好——权重读取在批次中摊销。

共置两者：你购买为两者优化的 GPU。H100 在两方面都很好，但成本相同。在大规模下，你希望预填充池使用 H100/计算密集型；解码池使用 H200/内存密集型，或使用激进量化。

### 架构图

```
            ┌──────────────┐
  请求 ──► │    路由器    │ ────────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ （仅提示词）                   │
            ┌──────────────┐    KV 缓存    ┌────────▼─────┐
            │  预填充池    │ ── NIXL ────► │   解码池     │
            │  （计算）    │               │  （内存）    │
            └──────────────┘               └──────┬───────┘
                                                  │ token
                                                  ▼
                                                客户端
```

NIXL 是 NVIDIA 的节点间传输协议。可用时使用 RDMA/InfiniBand，否则回退到 TCP。传输延迟是真实存在的——70B FP8 上 4K token 提示词的 KV 缓存传输通常为 20-80 毫秒。这就是为什么短提示词不值得分离式架构：传输税超过了节省。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 发布，1.0 GA）：
- 作为编排器位于 vLLM、SGLang、TRT-LLM 之上。
- Planner Profiler 测量工作负载，SLA Planner 自动配置预填充:解码比率。
- Rust 核心，Python 可扩展性。
- 吞吐量提升：NVIDIA 报告 DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 中等延迟情况下 6 倍提升（developer.nvidia.com，2025-06）；关于完整 Blackwell + Dynamo + DeepSeek-R1 栈"高达 30 倍"的社区报告缺乏单一主要来源，应视为方向性说法。
- GB300 NVL72 + Dynamo：相比 Hopper，MoE 吞吐量提升高达 50 倍（Dynamo 产品页面，未注明日期）。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- 预填充/解码/路由器作为独立的 Kubernetes Service。
- 使用队列深度（预填充）/KV 利用率（解码）信号的每角色 HPA。
- `topologyConstraint packDomain: rack` 将预填充 + 解码集群打包在同一机架上，以实现高带宽 KV 传输。
- llm-d 0.5（2026）：层级 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、缩容到零。

如果你想要一个托管的栈上层编排器，使用 Dynamo。如果你想要 Kubernetes 原生原语并致力于 CNCF 生态系统，使用 llm-d。

### 经济数据

内部综合数据（非单一已发布案例研究——数量级参考锚点）：

- 共置服务的年推理支出 200 万美元。
- 切换到使用 Dynamo 的分离式服务。
- 相同请求量，相同 P99 延迟 SLA。
- 报告节省：每年 60-80 万美元（减少 30-40%）。
- 没有新硬件。

我们从多个客户披露中综合这一数字，而非单一可引用案例研究；最接近的已发布数据点是 Baseten 使用 Dynamo KV 路由实现 TTFT 快 2 倍/吞吐量提高 61%（baseten.co，2025-10），以及 VAST + CoreWeave 在 40-60% KV 命中率下每美元 token 数增加 60-130% 的预测（vastdata.com，2025-12）。节省来自对每个池的合理配置；预填充密集型工作负载（8K+ 前缀的 RAG）比均衡工作负载受益更多。

### 不适合分离式架构的场景

- 提示词 < 512 个 token 且输出 < 200 个 token：传输税超过收益。
- 小集群（< 4 个 GPU）：没有足够的池多样性。
- 团队无法操作两个 GPU 池并进行每角色扩缩容：Dynamo 有帮助但并非轻而易举。
- 没有 RDMA 网络：TCP 传输税更重。

### 路由器与 Phase 17 · 11 集成

分离式路由器是 KV 缓存感知的（Phase 17 · 11）。请求落在持有其前缀的解码池上——如果没有匹配，则流经预填充→解码。缓存命中率和分离式架构形成复合效应——缓存感知路由器决定是否需要新的预填充。

### MoE 在 Blackwell 上是真正数字所在的地方

GB300 NVL72 + Dynamo 相比 Hopper 基线显示 50 倍 MoE 吞吐量。MoE 专家路由在预填充时计算密集，在解码时内存密集（专家缓存），因此分离式架构是双赢。2026 年前沿模型服务以 MoE 为主（DeepSeek-V3，未来的 GPT-5 变体）。

### 你应该记住的数字

基准数字会漂移——NVIDIA 和推理栈每季度都会发布更新结果。引用之前重新核实。

- GB200 NVL72 + Dynamo 上的 DeepSeek-R1：中等延迟情况下吞吐量约 6 倍（developer.nvidia.com，2025-06）；完整 Blackwell + Dynamo 栈上"高达 30 倍"的社区声明是方向性聚合，没有单一主要来源。
- GB300 NVL72 + Dynamo：相比 Hopper，MoE 吞吐量提升高达 50 倍（developer.nvidia.com，未注明日期）。
- 节省锚点（内部综合，非单一案例研究）：在恒定 SLA 下，200 万美元年度支出节省 60-80 万美元/年。
- 分离式架构阈值：提示词 >512 个 token + 输出 >200 个 token。
- 通过 NIXL 的 KV 传输：70B FP8 上 4K 提示词 KV 约 20-80 毫秒。

## 使用它

`code/main.py` 模拟共置 vs 分离式服务。报告吞吐量、每请求成本和提示词长度交叉点。

## 交付它

本课产出 `outputs/skill-disaggregation-decider.md`。给定工作负载和集群，决定是否进行分离式部署。

## 练习

1. 运行 `code/main.py`。在什么提示词长度时，分离式架构优于共置？
2. 为 P99 前缀长度 8K、输出 300 token 的 RAG 服务设计预填充池和解码池。
3. Dynamo vs llm-d：为一个没有 Python 运行时偏好的纯 Kubernetes 团队选择一个。
4. 计算 KV 传输成本：70B FP8 的 4K 预填充 ≈ 500 MB KV。在 RDMA 100 GB/s 时，传输 = 5 毫秒。在 TCP 10 GB/s 时 = 50 毫秒。哪个对你的 SLA 更重要？
5. MoE 专家路由改变了 KV 访问模式。分离式架构如何处理每个 token 激活不同专家的 MoE？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Disaggregated serving（分离式服务） | "拆分预填充/解码" | 每个阶段使用独立 GPU 池 |
| NIXL | "NVIDIA 传输" | Dynamo 的节点间 KV 传输（RDMA/TCP） |
| NVIDIA Dynamo | "编排器" | vLLM/SGLang/TRT-LLM 的栈上层协调器 |
| llm-d | "Kubernetes 原生" | Red Hat + AWS K8s 分离式栈 |
| Planner Profiler | "Dynamo 自动配置" | 测量工作负载，配置池比率 |
| SLA Planner | "Dynamo 策略" | 自动调整预填充:解码比率以满足 SLO |
| `packDomain: rack` | "llm-d 拓扑" | 将预填充 + 解码打包在同一机架以实现快速 KV |
| UCCL | "统一集合" | llm-d 0.5 用于缩容到零的网络层 |
| MoE expert routing（MoE 专家路由） | "每 token 的专家" | DeepSeek-V3 模式；分离式架构有帮助 |

## 延伸阅读

- [NVIDIA — 介绍 Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Kubernetes 上的分离式 LLM 推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM 分离式服务博客](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 发行说明](https://github.com/llm-d/llm-d/releases)

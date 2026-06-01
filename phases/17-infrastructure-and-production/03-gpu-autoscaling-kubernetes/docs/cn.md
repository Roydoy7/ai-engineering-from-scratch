# Kubernetes 上的 GPU 自动扩缩容——Karpenter、KAI Scheduler、Gang 调度（GPU Autoscaling on Kubernetes — Karpenter, KAI Scheduler, Gang Scheduling）

> 三个层次，而非一个。Karpenter 动态预置节点（一分钟以内，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理 gang 调度、拓扑感知和分层队列——它能防止"8 个 GPU 只等到 7 个"的部分分配陷阱，即七个节点等待并消耗资源却差一个 GPU 无法启动。应用级自动扩缩容器（NVIDIA Dynamo Planner、llm-d 工作负载变体自动扩缩容器）基于推理专用信号扩缩——队列深度、KV 缓存利用率——而非 CPU/DCGM 占空比。经典的 HPA 陷阱是：`DCGM_FI_DEV_GPU_UTIL` 是占空比测量值：100% 可能意味着 10 个请求或 100 个。vLLM 预分配 KV 缓存内存，因此内存永远不会触发缩容。本课教你组合这三个层次，并避免默认 Karpenter `WhenEmptyOrUnderutilized` 策略在推理过程中终止正在运行的 GPU 任务。

**类型：** 学习  
**语言：** Python（标准库，玩具队列深度自动扩缩容模拟器）  
**前置知识：** Phase 17 · 02（推理平台经济学）、Phase 17 · 04（vLLM 服务内部原理）  
**预计时间：** 约 75 分钟

## 学习目标

- 画出三个自动扩缩容层次（节点预置、gang 调度、应用级），并说出每层使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 对于 vLLM 是错误的 HPA 信号，并说出两个替代信号（队列深度、KV 缓存利用率）。
- 描述 gang 调度以及 KAI Scheduler 所能防止的部分分配故障模式（8 个 GPU 中的 7 个闲置）。
- 说出终止运行中 GPU 任务的 Karpenter 合并策略（`WhenEmptyOrUnderutilized`），并给出 2026 年的安全替代方案。

## 问题所在

你的团队在 Kubernetes 上部署了 LLM 服务。你用 `DCGM_FI_DEV_GPU_UTIL` 作为信号设置了 HPA。在业务高峰期，服务的利用率钉在 100%。HPA 从不扩容——它已经认为你满了。你手动添加一个副本；TTFT 下降。HPA 仍然不扩容。信号在欺骗你。

另一个问题是，你用 Cluster Autoscaler 管理节点。凌晨 2 点来了一个 1M token 的提示词；集群花了 3 分钟预置节点，请求超时了。

还有另一个问题，你部署了一个需要跨 2 个节点使用 8 个 GPU 的 70B 模型。集群有 7 个空闲 GPU，还有 1 个分散在 3 个节点上。Cluster Autoscaler 为那个缺失的 1 个 GPU 预置了一个节点。七个节点等待 4 分钟，白白烧钱，等着 Kubernetes 把最后一个 GPU 启动起来。

三个层次，三种不同的故障模式。2026 年的 GPU 感知自动扩缩容不是"开启 HPA"。而是组合节点预置、gang 调度和应用信号自动扩缩容。

## 核心概念

### 第一层——节点预置（Karpenter）

Karpenter 监视待处理的 Pod，并在约 45-60 秒内预置节点（Cluster Autoscaler 在 GPU 节点上通常需要 90-120 秒）。它根据 `NodePool` 约束动态选择实例类型——如果你的 Pod 需要 8 个 H100 而集群没有匹配节点，Karpenter 会直接预置一个，而不是扩展现有组。

**合并陷阱**：Karpenter 的默认 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池很危险。它会终止一个运行中的 GPU 节点，将 Pod 迁移到更便宜的合适规格实例。对于推理工作负载，这意味着驱逐正在运行的请求，并在新节点上重新加载 70B 模型。损失是数分钟的容量加上请求失败。

GPU 池的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

让 Karpenter 在一小时后合并真正空闲的节点，但绝不驱逐正在运行的任务。

### 第二层——gang 调度（KAI Scheduler）

KAI Scheduler（项目曾名为"Karp"，后更名）处理默认 kube-scheduler 无法做到的事情：

**Gang 调度** — 全有或全无式调度。一个需要 8 个 GPU 的分布式推理 Pod，要么所有 8 个一起启动，要么一个都不启动。没有这个机制，就会出现部分分配陷阱：8 个 Pod 中的 7 个启动了，无限等待，白白烧钱。

**拓扑感知** — 知道哪些 GPU 共享 NVLink，哪些在同一机架，哪些之间有 InfiniBand。相应地放置 Pod。DeepSeek-V3 67B 张量并行工作负载必须保持在同一 NVLink 域内；KAI Scheduler 尊重这一点。

**分层队列** — 多个团队用优先级和配额竞争同一 GPU 池。只有优先级规则允许，团队 B 的训练任务才能抢占团队 A 的生产高峰期任务。

KAI 作为辅助调度器与 kube-scheduler 并排部署；你通过注解工作负载来使用它。Ray 和 vLLM 生产栈都集成了 KAI。

### 第三层——应用级信号

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是占空比指标——它测量 GPU 在每个采样间隔内是否在工作。100% 利用率可能意味着 10 个并发请求或 100 个；无论哪种情况，GPU 都是忙碌的。基于占空比扩缩等于盲目扩缩。

更糟糕的是，vLLM 和类似引擎会预分配 KV 缓存内存（最多到 `--gpu-memory-utilization`）。即使只有一个请求，内存使用也接近 90%。基于内存的 HPA 永远不会缩容。

**2026 年的替代信号**：

- 队列深度（等待预填充的请求数量）。
- KV 缓存利用率（分配给活跃序列的块的比例）。
- 每副本 P99 TTFT（你的 SLA 信号）。
- 吞吐量（每秒满足所有 SLO 的请求数）。

NVIDIA Dynamo Planner 和 llm-d 工作负载变体自动扩缩容器消费这些信号并扩缩副本。它们完全取代 LLM 服务中的 HPA。

### 各工具的使用场景

| 扩缩决策 | 工具 |
|----------|------|
| 添加/移除节点 | Karpenter |
| 调度多 GPU 任务 | KAI Scheduler |
| 添加/移除副本 | Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA） |
| 选择 GPU 类型 | Karpenter NodePool |
| 抢占低优先级任务 | KAI Scheduler 队列 |

### 分离式预填充/解码让一切更复杂

如果你运行分离式预填充/解码（Phase 17 · 17），你有两类 Pod，触发扩缩的信号不同：预填充 Pod 基于队列深度扩缩，解码 Pod 基于 KV 缓存压力扩缩。llm-d 将这些作为带有每角色 HPA 的独立 `Service` 暴露出来。不要试图在两者前面放一个单一的 HPA。

### 冷启动在这里也很重要

冷启动缓解（Phase 17 · 10）是节点预置时间变成用户可见延迟的地方。Karpenter 约 45-60 秒的预热时间，加上 20GB 模型加载，再加上引擎初始化，意味着从零开始的请求需要 2-5 分钟。对于有 SLO 要求的关键路径，保持一个热池（`min_workers=1`），或在应用层使用 Modal 风格的检查点。

### 你应该记住的数字

- Karpenter 节点预置：约 45-60 秒，vs Cluster Autoscaler 约 90-120 秒（GPU 节点）。
- KAI Scheduler 防止部分分配浪费——8 个 GPU 只等到 7 个的陷阱。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA 信号：有问题；使用队列深度或 KV 利用率。
- Karpenter `WhenEmptyOrUnderutilized`：终止正在运行的 GPU 任务。推理时使用 `WhenEmpty + consolidateAfter: 1h`。

## 使用它

`code/main.py` 在突发性 GPU 工作负载上模拟三层自动扩缩容器。比较朴素 HPA（占空比）、队列深度 HPA 和 KAI gang 调度扩缩。报告未满足的请求数、闲置 GPU 分钟数和综合评分。

## 交付它

本课产出 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载形状和 SLO，设计一个三层自动扩缩容方案。

## 练习

1. 运行 `code/main.py`。在突发性工作负载下，朴素占空比 HPA 会丢弃多少个请求是队列深度 HPA 能捕获的？差异从哪里来？
2. 为一个在 H100 SXM5 上服务 Llama 3.3 70B FP8 的集群设计一个 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个阻止非 GPU 工作负载使用这些节点的污点。
3. 你的团队报告部署卡在 Pending，原因是"GPU 可用但 Pod 无法调度"。诊断——是 Karpenter、kube-scheduler 还是 KAI Scheduler 的问题？哪些指标能确认？
4. 为分离式预填充 Pod 选择一个自动扩缩容信号，为解码 Pod 选择一个不同的信号。分别论证。
5. 计算 `WhenEmptyOrUnderutilized` 合并陷阱对一个 24x7 生产服务的成本，该服务平均每天有 60 次请求丢弃事件，P99 TTFT > 10 秒。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Karpenter | "节点预置器" | Kubernetes 节点自动扩缩容器；分钟内预置 |
| Cluster Autoscaler | "旧扩缩容器" | Kubernetes 节点自动扩缩容器的前身；更慢，基于组 |
| KAI Scheduler | "GPU 调度器" | 用于 gang + 拓扑 + 队列的辅助调度器 |
| Gang scheduling（gang 调度） | "全有或全无" | 原子性地调度 N 个 Pod 或全部延迟 |
| Topology awareness（拓扑感知） | "机架感知" | 基于 NVLink/IB/机架位置放置 Pod |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU 利用率" | 占空比指标；不是 LLM 的扩缩信号 |
| Queue depth（队列深度） | "等待中的请求" | 预填充绑定扩缩的正确 HPA 信号 |
| KV cache utilization（KV 缓存利用率） | "内存压力" | 解码绑定扩缩的正确 HPA 信号 |
| Consolidation（合并） | "Karpenter 合并" | 将节点终止以使用更便宜的实例类型 |
| `WhenEmpty + 1h` | "安全合并" | 不驱逐正在运行的 GPU 任务的策略 |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 设计文档和配置示例
- [Karpenter 中断控制](https://karpenter.sh/docs/concepts/disruption/) — 合并策略语义和 GPU 安全默认值
- [NVIDIA — Kubernetes 上的分离式 LLM 推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 扩缩信号
- [Ray 文档 — RayClusters 的 KAI Scheduler](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 集成模式
- [AWS EKS 计算和自动扩缩容最佳实践](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 托管 Kubernetes 的专项指导
- [llm-d GitHub](https://github.com/llm-d/llm-d) — 工作负载变体自动扩缩容器设计

# 无服务器 LLM 的冷启动缓解（Cold Start Mitigation for Serverless LLMs）

> 一个 20 GB 的模型镜像从冷启动到服务就绪，7B 模型需要 5-10 分钟，70B 以上需要 20 分钟以上。在真正的无服务器世界中，这不是预热——而是中断。缓解措施在五个层次上运作：预植节点镜像（AWS 上的 Bottlerocket，双卷架构）、模型流式加载（NVIDIA Run:ai Model Streamer，vLLM 原生内置）、GPU 内存快照（Modal 检查点，重启速度最高快 10 倍）、热池（`min_workers=1`）、分层加载（ServerlessLLM 的 NVMe→DRAM→HBM 管道，延迟降低 10-200 倍），以及传输输入 token（KB 级别）而非 KV 缓存（GB 级别）的实时迁移。Modal 发布的冷启动下限为 2-4 秒；Baseten 默认 5-10 秒，预热后低于 1 秒。本课教你测量、预算并叠加这五个层次。

**类型：** 学习  
**语言：** Python（标准库，玩具冷启动路径模拟器）  
**前置知识：** Phase 17 · 02（推理平台经济学）、Phase 17 · 03（GPU 自动扩缩容）  
**预计时间：** 约 60 分钟

## 学习目标

- 列举冷启动缓解的五个层次，并在每个层次上说出一种工具或模式。
- 将 70B 模型的总冷启动时间计算为（节点预置）+（权重下载）+（权重加载到 HBM）+（引擎初始化）的总和。
- 解释为什么实时迁移传输输入 token（KB）而非 KV 缓存（GB），以及代价是什么（重新计算）。
- 说出热池的权衡（为空闲 GPU 付费或接受冷启动尾部），以及 `min_workers > 0` 成为必需的 SLA 阈值。

## 问题所在

你的无服务器 LLM 端点在夜间缩容到零。早上 8 点流量激增。第一个请求在等待：

1. Karpenter 预置一个 GPU 节点：45-60 秒。
2. 容器拉取带权重的 30 GB 镜像：120-300 秒。
3. 引擎将权重加载到 HBM：45-120 秒，取决于模型大小和存储速度。
4. vLLM 或 TRT-LLM 初始化 CUDA 图、KV 缓存池、分词器：10-30 秒。

总计：220-510 秒（约 3-8 分钟）后才会返回一个 token。你的 SLA 是 2 秒。你设置了热池（`min_workers=1`），问题似乎消失了——但现在你为一个空闲 GPU 付费 24x7。如果你的服务有 5 个产品，每个都有一个热副本，那就是每月 5 × 24 × 30 = 3600 GPU 小时，无论是否有用户调用。

冷启动缓解是在保持无服务器经济性的同时近似常驻服务延迟的方法。

## 核心概念

### 第一层——预植节点镜像（Bottlerocket）

在 AWS 上，Bottlerocket 的双卷架构将操作系统与数据分离。用预拉取的容器镜像快照数据卷；在 `EC2NodeClass` 中引用快照 ID。新节点启动时本地 NVMe 上已有权重——步骤 2 和部分步骤 3 消失。与 Karpenter 原生配合工作。典型节省：大型模型每次冷启动节省 2-4 分钟。

GCP 上的等价方案：带预烘焙容器层的自定义 VM 镜像。Azure 上：使用相同模式的托管磁盘快照。

### 第二层——模型流式加载（Run:ai Model Streamer）

不等待完整文件下载再回答第一个请求，而是逐层将权重流式传输到 GPU 内存，一旦第一个 Transformer 块就绪就开始处理。NVIDIA Run:ai Model Streamer 在 vLLM 2026 中原生内置。支持 S3、GCS 和本地 NVMe。通过将 I/O 与计算设置重叠，将大型模型的权重加载时间大致减半。

### 第三层——GPU 内存快照（Modal）

Modal 在首次加载后对 GPU 状态（权重、CUDA 图、KV 缓存区域）进行检查点。后续重启直接反序列化到 HBM——比重新初始化快 10 倍。这是最接近"2 秒内启动热 GPU"的方案。权衡：快照是针对每种 GPU 拓扑的，因此如果 Karpenter 将你迁移到不同 SKU，需要重新建立检查点。

### 第四层——热池（min_workers=1）

最简单的缓解措施：始终保持一个副本就绪。成本是一个 GPU 24x7 的小时费率。对于小模型，这笔账很难看（花 0.85-1.50 美元/小时避免 30 秒冷启动），对于大模型则合理（花 4 美元/小时避免 5 分钟冷启动）。热池成为必需的 SLA 阈值：通常是 70B 以上模型 TTFT P99 < 60 秒。

### 第五层——分层加载（ServerlessLLM）

ServerlessLLM 将存储视为层次结构：NVMe（快但大）、DRAM（中等但分层）、HBM（小但即时）。权重预加载到 DRAM；按需加载到 HBM。论文报告相比从磁盘到 HBM 的朴素方法，冷加载延迟降低 10-200 倍。生产采用还在早期，但与 vLLM 的集成已存在。

### 第六层——实时迁移（额外模式）

当节点不可用（Spot 实例被回收、节点排空）时，传统模式是冷启动另一个副本并排空请求队列。实时迁移将输入 token（千字节）移动到已加载模型的目标节点，并在目标节点上重新计算 KV 缓存。重新计算比通过网络传输 GB 级 KV 缓存更便宜。适用于分离式部署。

### 热池数学

对于 P99 TTFT SLA 为 2 秒的服务，问题不是"热池要不要"，而是"多少热副本，哪些路径需要"。

- 高价值交互式路径（实时聊天、语音智能体）：`min_workers=1-2`。
- 后台批处理路径（夜间分类）：可以接受缩容到零，5-10 分钟冷启动可以容忍。
- 高级层级：每个租户设置 `min_workers`，使用独占容量。

### 优化前先测量

70B 模型在新节点上的冷启动解剖（示例）：

| 阶段 | 时间 | 缓解措施 |
|------|------|---------|
| 节点预置 | 50 秒 | Bottlerocket + 预植镜像，热池 |
| 镜像拉取 | 180 秒 | 预植数据卷（消除） |
| 权重加载到 HBM | 75 秒 | 模型流式加载（减半）；GPU 快照（消除） |
| 引擎初始化 | 20 秒 | 持久 CUDA 图缓存 |
| 第一次前向传递 | 3 秒 | 最低固有延迟 |
| **总冷启动** | **328 秒** | |
| **应用缓解后总计** | **约 15 秒** | 降低 22 倍 |

### 你应该记住的数字

- Modal 冷启动：2-4 秒（使用 GPU 快照）。
- Baseten 默认冷启动：5-10 秒；预热后低于 1 秒。
- 原始 70B 冷启动：3-8 分钟。
- Run:ai Model Streamer：约 2 倍权重加载速度提升。
- ServerlessLLM 分层加载：延迟降低 10-200 倍（论文数字）。

## 使用它

`code/main.py` 模拟有无各种缓解措施的冷启动路径。报告总冷启动时间、热池成本，以及热池在多少请求率以上开始值回本的盈亏平衡点。

## 交付它

本课产出 `outputs/skill-cold-start-planner.md`。给定 SLA、模型大小和流量形状，选择需要叠加的缓解措施。

## 练习

1. 运行 `code/main.py`。计算热副本在多少请求率以上比承担冷启动税（通过额外请求丢失）更便宜。
2. 你部署了一个 13B 模型，P99 TTFT SLA 为 3 秒。选择能实现该目标的最少缓解层叠加（最少层数）。
3. Bottlerocket 预植消除了镜像拉取，但权重仍然需要从快照加载到 HBM。如果快照支持的 NVMe 读取速度为 7 GB/s，计算 70B 模型的挂钟时间。
4. 你的无服务器供应商提供 GPU 快照（Modal），但团队以"快照泄露 PII"为由拒绝使用。论证两边——实际风险是什么，缓解措施是什么（临时快照、加密、命名空间隔离）？
5. 设计一个分层热池策略：付费用户、试用用户和批处理工作负载分别需要多少热副本？展示数学计算过程。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Cold start（冷启动） | "大暂停" | 从请求到新副本首个 token 的时间 |
| Warm pool（热池） | "始终在线最低值" | `min_workers >= 1` 以保持至少一个副本就绪 |
| Pre-seeded image（预植镜像） | "预烘焙 AMI" | 带容器权重预驻留的节点镜像 |
| Bottlerocket | "AWS 节点操作系统" | AWS 容器优化操作系统，支持双卷快照 |
| Model streamer（模型流式加载器） | "流式加载" | 将权重 I/O 与计算设置重叠 |
| GPU snapshot（GPU 快照） | "HBM 检查点" | 序列化加载后的 GPU 状态；重启时反序列化 |
| Tiered loading（分层加载） | "NVMe + DRAM + HBM" | 存储层次结构；按需加载 |
| Live migration（实时迁移） | "移动 token" | 传输输入（KB），在目标节点重新计算 KV |
| `min_workers` | "热副本" | 无服务器最小保活数量 |
| Scale-to-zero（缩容到零） | "完全无服务器" | 空闲时无成本；接受完整冷启动税 |

## 延伸阅读

- [Modal — 冷启动性能](https://modal.com/docs/guide/cold-start) — Modal 发布的基准和检查点架构
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — 预植数据卷快照模式
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — 将权重加载与计算设置重叠
- [Baseten — 冷启动缓解](https://www.baseten.co/blog/cold-start-mitigation/) — 预热操作手册
- [ServerlessLLM 论文（USENIX OSDI'24）](https://www.usenix.org/conference/osdi24/presentation/fu) — 分层加载设计
- [NVIDIA — Kubernetes 上的分离式 LLM 推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — 分离式部署的实时迁移

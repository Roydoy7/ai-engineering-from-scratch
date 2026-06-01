# 多区域 LLM 服务与 KV 缓存局部性（Multi-Region LLM Serving and KV Cache Locality）

> 轮询负载均衡对缓存 LLM 推理是有害的。落不到持有其前缀节点的请求要付出全额预填充成本——在长提示词上约为 800 毫秒 P50，而命中缓存只需约 80 毫秒。2026 年的生产模式是缓存感知路由器（Rust 版 vLLM Router，llm-d router），它消费 KV 缓存事件并基于前缀哈希匹配进行路由。近期研究（GORGO）将跨区域网络延迟作为路由目标中的显式项。商业"跨区域推理"产品（Bedrock 跨区域推理、GKE 多集群网关）将推理视为不透明——它们处理可用性，而非 TTFT。JPMorgan 和 Mayo Clinic 在 2024 年 11 月进行了 us-east-1 故障转移演练，耗时约 22 分钟。DR 现实：32% 的 LLM DR 故障是因为团队备份了权重，却忘记了分词器文件或量化配置。

**类型：** 学习  
**语言：** Python（标准库，玩具前缀缓存感知路由器模拟器）  
**前置知识：** Phase 17 · 04（vLLM 服务）、Phase 17 · 06（SGLang RadixAttention）  
**预计时间：** 约 60 分钟

## 学习目标

- 解释为什么轮询负载均衡会破坏缓存推理，并量化 TTFT 惩罚。
- 画出缓存感知路由器图：输入（KV 缓存事件）、算法（前缀哈希匹配）、平局决胜（GPU 利用率）。
- 说出 LLM 中 32% DR 故障的驱动因素（缺少分词器文件/量化配置），并给出三文件 DR 检查清单。
- 区分商业跨区域产品（Bedrock CRI、GKE 多集群网关）与 KV 感知路由。

## 问题所在

你的服务运行在 us-east-1、us-west-2 和 eu-west-1。你在前面放了一个轮询 ALB。生产中的前缀缓存命中率降至 8%。TTFT P50 变为三倍。你的 vLLM 日志显示每个请求都在付全额预填充成本。

轮询对无状态服务是最优的。LLM 推理在设计上是有状态的——KV 缓存编码了模型见过的所有内容。盲目路由就是路由到错误的缓存。

另一方面，你的团队有 DR 计划。你将模型权重备份到 S3 跨区域。区域中断发生；你尝试故障转移；副本拒绝启动。你忘记了 tokenizer.json、量化配置和 RoPE 缩放配置在另一个没有同步的桶里。

多区域 LLM 服务是一个缓存问题、路由问题和 DR 卫生问题——而非负载均衡问题。

## 核心概念

### 缓存感知路由

请求携带提示词到达。路由器对前缀进行哈希（比如，前 512 个 token）；它询问每个副本"你缓存了这个前缀吗？"副本在分配和驱逐块时在发布/订阅频道上发布 KV 缓存事件。路由器选择匹配的副本，如果没有匹配则回退到基于 GPU 利用率的平局决胜。

**vLLM Router**（Rust，2026 生产栈）：订阅 `kv.cache.block_added` 事件，维护前缀哈希 → 副本索引，以 O(1) 查找进行路由。无匹配时回退到最小队列深度。

**llm-d router**：相同模式，Kubernetes 原生。通过 ControlPlane API 发布事件。

**SGLang RadixAttention**（Phase 17 · 06）是副本内部的等价物。跨副本路由严格在上游处理。

### 数字

Llama 3.3 70B FP8，H100 上，2K token 提示词的 TTFT P50：
- 缓存命中（同一副本，前缀常驻）：约 80 毫秒。
- 缓存未命中（冷预填充）：约 800 毫秒。

10 倍差距。如果你的路由器在各副本间命中 60-80% 的前缀缓存，你就以 N 副本容量近似了单副本性能。如果命中 10%，你近似的就是朴素扩展。

### 跨区域有新的约束——网络延迟

区域间 RTT：
- us-east-1 ↔ us-west-2：约 65 毫秒。
- us-east-1 ↔ eu-west-1：约 75 毫秒。
- us-east-1 ↔ ap-southeast-1：约 220 毫秒。

如果路由将请求从 us-east-1 发到 ap-southeast-1 的热前缀，节省的预填充时间（800 → 80 毫秒）会被 440 毫秒往返时间所淹没。GORGO（2026 年研究）将此明确化——联合最小化 `预填充时间 + 网络延迟`，而非单独最小化预填充时间。通常答案是保持区域内路由，除非是预填充占主导的超大 MB 级前缀。

### 商业"跨区域推理"在这里帮不上忙

AWS Bedrock 跨区域推理在容量压力下自动将请求路由到其他区域。它优化可用性，而非 TTFT，并将推理视为不透明。GKE 多集群网关也是如此——服务级故障转移，不感知 KV 缓存。

即使使用这些产品，你仍然需要应用层的缓存感知路由器。它们处理"us-east-1 着火了"的情况。缓存感知路由处理 TTFT 的情况。

### DR 卫生——32% 缺失文件问题

2026 年广泛引用的统计：32% 的 LLM DR 故障发生，因为团队备份了权重，却忘记了：

- `tokenizer.json` 或 `tokenizer.model`
- 量化配置（`quantize_config.json`，AWQ 缩放，GPTQ 零点）
- 模型专属配置（RoPE 缩放、注意力掩码、聊天模板）
- 引擎配置（`vllm_config.yaml`，采样默认值，LoRA 适配器清单）

修复方案是三文件最小 DR 清单：

1. HF 模型仓库下的所有文件（权重 + 配置 + 分词器）。
2. 引擎专属服务配置。
3. 部署清单（K8s YAML、Dockerfile、依赖锁定）。

加上：每季度进行 DR 演练。JPMorgan us-east-1 演练在 2024 年 11 月仅用 22 分钟恢复，正是因为操作手册已经经过演练。

### 数据驻留是正交问题

欧盟客户的 PHI 不能离开欧盟。如果你的缓存感知路由器将来自巴黎的请求发到 us-east-1 进行前缀匹配，无论 TTFT 收益如何，你都违反了 GDPR。在为缓存优化之前，先按驻留边界对路由器进行分区。

### 你应该记住的数字

- 缓存命中 vs 未命中 TTFT 差距：约 10 倍（2K 提示词上 80 毫秒 vs 800 毫秒）。
- 美国-欧盟区域间 RTT：约 75 毫秒。
- DR 故障：32% 缺少分词器/量化配置。
- JPMorgan us-east-1 故障转移 2024 年 11 月：22 分钟（30 分钟 SLA）。

## 使用它

`code/main.py` 在多区域工作负载上模拟三种路由策略（轮询、区域缓存感知、全局缓存感知）。报告缓存命中率、TTFT P50/P99 和跨区域费用。

## 交付它

本课产出 `outputs/skill-multi-region-router.md`。给定区域、驻留约束和 SLA，设计一个路由方案。

## 练习

1. 运行 `code/main.py`。给定 75 毫秒 RTT，在什么提示词长度时跨区域路由优于仅区域内路由？
2. 你的缓存命中率从 70% 降至 12%。诊断三个可能的原因及能确认每个原因的可观测指标。
3. 为一个在 vLLM 中服务、带 5 个 LoRA 适配器的 70B AWQ 量化模型设计 DR 清单。列出每个文件和配置。
4. 就"Bedrock 跨区域推理是否足够"对一个有严格 TTFT SLO 的金融科技公司进行论证。引用具体行为。
5. 一个来自巴黎的请求在 us-east-1 匹配到前缀。你是否路由过去？写出策略。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Cache-aware routing（缓存感知路由） | "智能负载均衡" | 基于前缀哈希匹配路由到持有 KV 缓存的副本 |
| KV-cache events（KV 缓存事件） | "缓存发布订阅" | 副本发布块添加/驱逐事件；路由器建立索引 |
| Prefix hash（前缀哈希） | "缓存键" | 前 N 个 token 的哈希，用作路由器查找 |
| GORGO | "跨区域路由研究" | arXiv 2602.11688；网络延迟作为显式项 |
| Cross-region inference（跨区域推理） | "Bedrock CRI" | AWS 产品；可用性故障转移，非 TTFT 感知 |
| DR manifest（DR 清单） | "备份列表" | 恢复所需的每个文件——不只是权重 |
| Data residency（数据驻留） | "GDPR 边界" | 哪个区域可以看到用户数据的法律约束 |
| RTT | "往返时间" | 网络延迟；美国-欧盟 75 毫秒，美国-亚太 220 毫秒 |
| LLM-aware LB | "缓存命中负载均衡" | 缓存感知路由器作为产品类别 |

## 延伸阅读

- [BentoML — 多云和跨区域推理](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO（2602.11688）](https://arxiv.org/html/2602.11688v1) — 带网络延迟项的跨区域 KV 缓存复用
- [TianPan — 多区域 LLM 服务缓存局部性](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock 跨区域推理](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — 可用性故障转移文档
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — 缓存感知路由器源码

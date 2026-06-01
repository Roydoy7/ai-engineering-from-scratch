# vLLM 生产栈与 LMCache KV 卸载（vLLM Production Stack with LMCache KV Offloading）

> vLLM 的 production-stack 是参考 Kubernetes 部署——路由器、引擎和可观测性连接在一起。LMCache 是 KV 卸载层，将 KV 缓存从 GPU 内存中提取出来，并跨查询和引擎复用（CPU DRAM，然后是磁盘/Ceph）。vLLM 0.11.0 KV 卸载连接器（2026 年 1 月）通过 Connector API（v0.9.0+）使其异步且可插拔。卸载延迟对用户不可见。LMCache 即使在没有共享前缀的情况下也很有价值——当 GPU 的 KV 槽用完时，被抢占的请求可以从 CPU 恢复，而不是重新计算预填充。在 4 个 a3-highgpu-4g 上的 16x H100（80GB HBM）已发布基准：当 KV 缓存超过 HBM 时，原生 CPU 卸载和 LMCache 都大幅提升吞吐量；在 KV 占用小的情况下，所有配置都与基线相当，只有少量开销。

**类型：** 学习  
**语言：** Python（标准库，玩具 KV 溢出模拟器）  
**前置知识：** Phase 17 · 04（vLLM 服务内部原理）、Phase 17 · 06（SGLang/RadixAttention）  
**预计时间：** 约 60 分钟

## 学习目标

- 画出 vLLM production-stack 各层：路由器、引擎、KV 卸载、可观测性。
- 解释 KV 卸载 Connector API（v0.9.0+）以及 0.11.0 异步路径如何隐藏卸载延迟。
- 量化 LMCache CPU-DRAM 何时有帮助（KV > HBM）vs 何时增加开销（KV 小到可以放入 HBM）。
- 在给定部署约束的情况下，在原生 vLLM CPU 卸载和 LMCache 连接器之间做出选择。

## 问题所在

你的 vLLM 服务显示 GPU HBM 处于 100%，每当并发攀升时就会发生抢占事件。请求被驱逐、重新排队，你在一分钟内四次重新预填充同一个 2K token 提示词。GPU 计算被浪费在冗余预填充上；吞吐量远低于原始吞吐量。

增加更多 GPU 线性增加成本。增加更多 HBM 是不可能的。但 CPU DRAM 很便宜——一个插槽有 512 GB+，延迟比 HBM 差几个数量级，但对于"临时热"KV 缓存来说完全可以接受。

LMCache 将 KV 缓存提取到 CPU DRAM，使被抢占的请求快速恢复，并让各引擎间重复的前缀无需各自重新预填充即可共享缓存。

## 核心概念

### vLLM production-stack

`github.com/vllm-project/production-stack` 是参考 Kubernetes 部署：

- **路由器** — 缓存感知（Phase 17 · 11）。消费 KV 事件。
- **引擎** — vLLM 工作进程。每个 GPU 或每个 TP/PP 组一个。
- **KV 缓存卸载** — LMCache 部署或原生连接器。
- **可观测性** — Prometheus 抓取、Grafana 仪表板、OTel 轨迹。
- **控制面** — 服务发现、配置、滚动更新。

作为 Helm chart + operator 提供。

### KV 卸载 Connector API（v0.9.0+）

vLLM 0.9.0 引入了用于可插拔 KV 缓存后端的 Connector API。你的引擎将块卸载到连接器；连接器存储它们（RAM、磁盘、对象存储、LMCache）。请求需要块时，连接器将其加载回来。

vLLM 0.11.0（2026 年 1 月）添加了异步卸载路径——卸载可以在后台进行，因此引擎在通常情况下不会因此阻塞。端到端延迟和吞吐量仍然取决于工作负载形状、KV 缓存命中率和系统压力；vLLM 自己的说明指出，自定义内核卸载在低命中率时可能降低吞吐量，异步调度与推测解码存在已知的交互问题。

### 原生 CPU 卸载 vs LMCache

**原生 vLLM CPU 卸载**：引擎本地。将 KV 块存储在主机 RAM 中。实现简单，零网络跳转。不跨引擎。

**LMCache 连接器**：集群规模。将块存储在共享的 LMCache 服务器（CPU DRAM + Ceph/S3 层）中。块对任何引擎都可访问。已发布 16x H100 基准。

当单个引擎有 HBM 压力时选择原生。当多个引擎共享前缀时选择 LMCache（具有通用系统提示词的 RAG、具有共享模板的多租户）。

### 基准行为

16x H100（80 GB HBM）跨 4 个 a3-highgpu-4g 的测试：

- 低 KV 占用（短提示词，低并发）：所有配置与基线相当，LMCache 增加约 3-5% 开销。
- 中等占用：LMCache 开始在跨引擎前缀复用方面发挥作用。
- KV 超过 HBM：原生 CPU 卸载和 LMCache 都大幅提升吞吐量；LMCache 收益更大，因为跨引擎共享。

### LMCache 起决定性作用的场景

- 系统提示词跨租户共享的多租户服务。
- 文档块跨查询重复的 RAG。
- 同一基础模型上的微调变体（LoRA），基础模型 KV 复用减少冗余工作。
- 抢占密集型工作负载：从 CPU 恢复比重新预填充更便宜。

### 不适合启用的场景

- HBM 压力小——你付出开销却没有收益。
- 短上下文（<1K token）——传输时间 > 重新预填充时间。
- 单租户单提示词工作负载——没有可捕获的复用。

### 与分离式服务的集成

Phase 17 · 17 分离式服务 + LMCache 形成复合效应：从预填充池到解码池的 KV 传输如果未使用则落入 LMCache；后续查询从 LMCache 中拉取。Phase 17 · 11 缓存感知路由器可以路由到其本地缓存或 LMCache 共享缓存与之匹配的引擎。

### 你应该记住的数字

- vLLM 0.9.0：Connector API 发布。
- vLLM 0.11.0（2026 年 1 月）：异步卸载路径；端到端延迟影响取决于工作负载、KV 命中率和系统压力（不是绝对保证）。
- 16x H100 基准：KV 占用超过 HBM 时 LMCache 有帮助。
- HBM 压力小时：3-5% 开销没有收益。

## 使用它

`code/main.py` 模拟有无 LMCache 的抢占密集型工作负载。报告避免的重新预填充次数、吞吐量提升和收支平衡 HBM 利用率。

## 交付它

本课产出 `outputs/skill-vllm-stack-decider.md`。给定工作负载形状和 vLLM 部署，决定使用原生 vs LMCache vs 都不用。

## 练习

1. 运行 `code/main.py`。在什么 HBM 利用率时 LMCache 开始有收益？
2. 一个租户每小时 200 个查询共享一个 6K token 系统提示词。计算每个租户的预期 LMCache 节省。
3. LMCache 服务器是单点故障。设计高可用策略（副本、回退到原生）。
4. LMCache 存储到旋转磁盘上的 Ceph。对于 70B FP8 的 4K token KV（500 MB），读取时间 vs 重新预填充时间是多少？
5. 论证 vLLM 0.11.0 异步路径是否"免费"——开销隐藏在哪里？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Production-stack | "参考部署" | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | "KV 后端接口" | vLLM 0.9.0+ 可插拔 KV 存储接口 |
| Native CPU offload（原生 CPU 卸载） | "引擎本地溢出" | 在同一引擎的主机 RAM 中存储 KV |
| LMCache | "集群 KV 缓存" | CPU DRAM + 磁盘上的跨引擎 KV 缓存服务器 |
| 0.11.0 async | "非阻塞卸载" | 卸载隐藏在引擎流之后 |
| Preemption（抢占） | "为腾空间而驱逐" | HBM 满时的 KV 缓存调整 |
| Prefix reuse（前缀复用） | "相同系统提示词" | 多个查询共享开头；缓存命中 |
| Ceph tier（Ceph 层） | "磁盘层" | 缓存层次中 DRAM 之下的持久存储 |

## 延伸阅读

- [vLLM 博客 — KV 卸载连接器（2026 年 1 月）](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator
- [LMCache 用于企业级 LLM 推理（arXiv:2510.09665）](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — 连接器实现
- [vLLM 0.11.0 发行说明](https://github.com/vllm-project/vllm/releases) — 异步路径详情

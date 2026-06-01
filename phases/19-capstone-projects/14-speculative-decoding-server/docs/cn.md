# 压轴项目 14——推测解码推理服务器（Capstone 14 — Speculative-Decoding Inference Server）

> vLLM 0.7 中的 EAGLE-3 在真实流量上提供了 2.5-3 倍的吞吐量。P-EAGLE（AWS 2026）将并行推测进一步推进。SGLang 的 SpecForge 大规模训练了草稿头。Red Hat 的 Speculators hub 为常见开源模型发布了对齐的草稿。TensorRT-LLM 在 NVIDIA 上将推测解码提升为一等公民。2026 年的生产服务栈是带 EAGLE 系列草稿的 vLLM 或 SGLang，FP8 或 INT4 量化，以及队列等待上的 HPA。本压轴项目是以超过 2.5 倍的基线吞吐量服务两个开源模型，并提供完整的尾部延迟报告。

**类型：** 压轴项目  
**语言：** Python（服务），C++ / CUDA（内核检查），YAML（配置）  
**前置知识：** Phase 3（深度学习）、Phase 7（Transformer）、Phase 10（从头开始的 LLM）、Phase 17（基础设施）  
**涉及的阶段：** P3 · P7 · P10 · P17  
**预计时间：** 30 小时

## 问题所在

推测解码在 2026 年成为了商品。EAGLE-3 草稿头在目标模型的隐藏状态上训练并预测前 N 个 token；目标模型在单次传递中验证所有 token。60-80% 的接受率转化为 2-3 倍的端到端吞吐量。vLLM 0.7 原生集成了这一功能。SGLang + SpecForge 提供了训练管道。Red Hat 的 Speculators 为 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 发布了对齐的草稿。

技艺在于服务运营，而非模型。接受率随流量分布（ShareGPT vs 代码 vs 领域数据）漂移。拒绝时的尾部延迟比没有推测时更差——你必须报告多个批次大小下的 p99，而不仅仅是稳态 tokens/秒。每百万 token 成本 vs Anthropic / OpenAI API 是可信度杠杆。

## 核心概念

推测解码有两层。**草稿**模型（EAGLE-3 头、n-gram 或更小的目标对齐模型）每步提出 k 个候选 token。**目标**模型在一次传递中验证所有 k 个；接受的任何前缀替换贪心路径。接受率取决于草稿-目标对齐和输入分布。

EAGLE-3 在大多数流量上优于 n-gram 草稿。P-EAGLE 为更深的草稿树运行并行推测。权衡：拒绝时的 P99 延迟更高，因为验证传递更大。服务配置必须报告按批次大小分组的延迟以暴露这一点。

部署是 Kubernetes。vLLM 0.7 每个 GPU 或张量并行分片运行一个副本。HPA 根据队列等待而非 CPU 进行自动扩缩容。FP8（Marlin）和 INT4（AWQ）量化将 GPU 内存保持在 H100 / H200 范围内。端到端报告是吞吐量、接受率、批次 1/8/32 的 p50/p99，以及每百万 token 美元成本。

## 架构

```
请求入口
    |
    v
vLLM 服务器（0.7）或 SGLang（0.4）
    |
    +-- 草稿：EAGLE-3 头 | P-EAGLE 并行 | n-gram 备用
    +-- 目标：Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     量化为 FP8-Marlin 或 INT4-AWQ
    |
    v
验证传递：通过目标批量处理 k 个草稿 token
    |
    v（接受前缀；对被拒绝的后缀重新采样）
    v
token 流回客户端
    |
    v
Prometheus 指标：吞吐量、接受率、队列等待、延迟 p50/p99
    |
    v
队列等待指标上的 HPA
```

## 技术栈

- 服务：vLLM 0.7 或 SGLang 0.4
- 推测方法：EAGLE-3 草稿头、P-EAGLE 并行推测、n-gram 备用
- 草稿训练：SpecForge（SGLang）或 Red Hat Speculators
- 目标模型：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- 量化：FP8（Marlin）、INT4 AWQ
- 部署：Kubernetes + NVIDIA 设备插件；队列等待指标上的 HPA
- 评估：ShareGPT、MT-Bench-v2、GSM8K、HumanEval 用于领域扩散接受率测量
- 参考：TensorRT-LLM 推测解码作为供应商基线

## 构建它

1. **目标模型准备。** 选择 Llama 3.3 70B。通过 Marlin 量化为 FP8。在 1xH100（或 2x 张量并行）上的 vLLM 0.7 下部署。

2. **草稿来源。** 从 Red Hat Speculators 拉取对齐的 EAGLE-3 草稿头（或通过 SpecForge 训练一个）。加载到 vLLM 的推测解码配置中。

3. **基线数字。** 推测前：批次 1/8/32 时的 tokens/s，p50/p99 延迟，GPU 利用率。发布。

4. **启用 EAGLE-3。** 翻转配置；重新运行相同基准。报告加速比、接受率、p99 尾部延迟差值。

5. **P-EAGLE。** 启用并行推测；测量更深的草稿树 vs 串行 EAGLE-3。报告 P-EAGLE 有帮助 vs 有害的拐点。

6. **领域流量。** 通过同一服务器运行 ShareGPT vs HumanEval vs 领域特定流量。测量每分布的接受率。识别草稿何时漂移。

7. **第二目标模型。** 在 Qwen3-Coder-30B MoE 上运行相同管道。草稿更棘手（MoE 路由噪声）。报告。

8. **K8s HPA。** 在带追踪 `queue_wait_ms` 的 HPA 的 K8s 下部署。演示当负载增加三倍时的扩展。

9. **成本比较。** 在同一评估上计算 $/1M tokens vs Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4。发布。

## 使用它

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[服务]   vLLM 0.7，Llama 3.3 70B FP8，EAGLE-3 激活
[解码]   bs=8，accepted_tokens_per_step=3.2，acceptance_rate=0.76
[延迟]   首 token 42ms，完整响应 980ms（620 个 token）
[成本]   在持续吞吐量下每百万输出 token $0.34
```

## 交付它

`outputs/skill-inference-server.md` 描述了可交付成果。一个带推测解码的经测量服务栈，完整基准报告，以及 K8s 部署。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 相对基线的测量加速比 | 在两个模型上的相同质量下 2.5 倍以上吞吐量 |
| 20 | 真实流量上的接受率 | 按分布的接受率报告 |
| 20 | P99 尾部延迟规范 | 有和没有推测时批次 1/8/32 的 p99 |
| 20 | 运维 | K8s 部署，队列等待上的 HPA，平滑推出 |
| 15 | 报告和方法论 | 对什么变了以及为什么的清晰解释 |
| **100** | | |

## 练习

1. 测量当草稿比目标落后一个版本时的接受率退化（例如，Llama 3.3 -> 3.4 漂移）。构建监控告警。

2. 实现 n-gram 备用：如果 EAGLE-3 接受率下降到阈值以下，切换到 n-gram 草稿。报告可靠性改进。

3. 运行受控 MoE 实验：相同的 Qwen3-Coder-30B 注入路由噪声 vs 没有注入。测量草稿接受率敏感性。

4. 扩展到 H200（141 GB）。报告获得的每副本模型大小余量，以及是否可以服务未量化的 Llama 3.3 70B。

5. 在相同 H100 硬件上对 TensorRT-LLM 推测解码进行基准测试。报告它在哪里胜过 vLLM。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Draft model（草稿模型） | "推测器" | 提出 N 个 token 供目标验证的小型模型 |
| EAGLE-3 | "2026 年草稿架构" | 在目标隐藏状态上训练的草稿头；约 75% 接受率 |
| P-EAGLE | "并行推测" | 在一次目标传递中验证的草稿分支树 |
| Acceptance rate（接受率） | "命中率" | 无需重新采样就被接受的草稿 token 比例 |
| Quantization（量化） | "FP8 / INT4" | 更低精度权重以将更多模型装入 GPU 内存 |
| Queue wait（队列等待） | "HPA 指标" | 请求在推断开始前在待处理队列中等待的时间 |
| Speculators hub | "对齐草稿" | Red Hat Neural Magic 的常见开源模型 EAGLE 草稿 hub |

## 延伸阅读

- [vLLM EAGLE 和 P-EAGLE 文档](https://docs.vllm.ai) — 参考服务栈
- [P-EAGLE（AWS 2026）](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 并行推测解码论文 + 集成
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 草稿头训练管道
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — 对齐草稿 hub
- [TensorRT-LLM 推测解码](https://nvidia.github.io/TensorRT-LLM/) — 供应商替代方案
- [Fireworks.ai 服务架构](https://fireworks.ai/blog) — 商业参考
- [EAGLE-3 论文（arXiv:2503.01840）](https://arxiv.org/abs/2503.01840) — 方法论文
- [vLLM 仓库](https://github.com/vllm-project/vllm) — 代码和基准

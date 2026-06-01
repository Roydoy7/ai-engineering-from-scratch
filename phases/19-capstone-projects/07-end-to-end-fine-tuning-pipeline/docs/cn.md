# 压轴项目 07——端到端微调管道（数据到 SFT 到 DPO 到服务）（Capstone 07 — End-to-End Fine-Tuning Pipeline: Data to SFT to DPO to Serve）

> 在你自己的数据上训练的 8B 模型，针对你自己的偏好进行 DPO 对齐，量化，推测解码，以可测量的每百万 token 美元成本提供服务。2026 年的开源栈是 Axolotl v0.8、TRL 0.15、Unsloth 用于迭代、GPTQ/AWQ/GGUF 用于量化、带 EAGLE-3 的 vLLM 0.7 用于服务。本压轴项目是可重现地运行整个管道——YAML 输入，服务端点输出——并在 2026 年模型开放框架下发布模型卡片。

**类型：** 压轴项目  
**语言：** Python（管道），YAML（配置），Bash（脚本）  
**前置知识：** Phase 2（ML）、Phase 3（DL）、Phase 7（Transformer）、Phase 10（从头开始的 LLM）、Phase 11（LLM 工程）、Phase 17（基础设施）、Phase 18（安全）  
**涉及的阶段：** P2 · P3 · P7 · P10 · P11 · P17 · P18  
**预计时间：** 35 小时

## 问题所在

2026 年，每个认真的 AI 团队都随时准备好微调管道。不是因为他们发布了前沿基础模型，而是因为下游适配——领域 SFT、针对标注偏好的 DPO、用于推测解码的蒸馏草稿、使用 EAGLE-3 提供服务——才是可测量收益所在的地方。Axolotl v0.8 处理多 GPU SFT 配置。TRL 0.15 处理 DPO 和 GRPO。Unsloth 让你快速进行单 GPU 迭代。带 EAGLE-3 的 vLLM 0.7 在没有质量损失的情况下将解码吞吐量提高 2-3 倍。工具链有效；技艺在于 YAML、数据卫生和评估规范。

你将通过 SFT 然后 DPO 在任务特定数据上运行一个 8B 基础模型（Llama 3.3、Qwen3 或 Gemma 3），量化以供服务，并根据 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 测量收益。你将在 2026 年模型开放框架下生成模型卡片。重点是可重现性——一个命令端到端重新运行整个管道。

## 核心概念

管道有五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC 风格分类器）、PII 清洗、针对公共基准污染的分割卫生检查。**SFT**：Axolotl YAML，8xH100 上的 ZeRO-3，余弦调度，打包序列，2-3 个周期。**DPO 或 GRPO**：TRL 配置，1 个周期，偏好对要么是人工标注的要么是模型判断的，beta 调优。**量化**：GPTQ + AWQ + GGUF 用于部署灵活性。**服务**：带 EAGLE-3 推测头的 vLLM 0.7（或带 SpecForge 的 SGLang），K8s 部署，在队列等待上使用 HPA。

消融是可交付成果：SFT-only vs SFT+DPO vs SFT+GRPO 在三个任务特定基准上。服务指标：批次 1/8/32 时的 tokens/s，EAGLE-3 接受率，每百万 token 美元成本。安全评估：Llama Guard 4 通过率。模型卡片：偏见评估、可重现性种子、数据许可证。

## 架构

```
原始数据（HF 数据集 + 内部）
    |
    v
Datatrove 去重 + Nemotron-CC 质量过滤 + PII 清洗
    |
    v
分割卫生检查（MMLU-Pro 污染检查）
    |
    v
Axolotl SFT 配置（YAML）  ---> 8xH100，ZeRO-3
    |
    v
TRL DPO / GRPO 配置       ---> 4xH100，1 个周期
    |
    v
GPTQ + AWQ + GGUF 量化
    |
    v
vLLM 0.7 + EAGLE-3 推测解码
    |
    v
K8s 部署，队列等待上的 HPA
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
模型卡片（2026 MOF）+ 安全评估（Llama Guard 4）
```

## 技术栈

- 数据：Datatrove 用于去重，Nemotron-CC 分类器用于质量，Presidio 用于 PII
- 基础：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8，带 ZeRO-3、Flash Attention 3、打包序列
- 偏好微调：TRL 0.15 用于 DPO 或 GRPO；Unsloth 用于单 GPU 迭代
- 量化：GPTQ（Marlin）、AWQ、通过 llama.cpp 的 GGUF
- 服务：带 EAGLE-3 推测解码的 vLLM 0.7（或 SGLang 0.4 + SpecForge）
- 评估：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评估：Llama Guard 4、ShieldGemma-2
- 基础设施：Kubernetes + NVIDIA 设备插件，队列等待指标上的 HPA
- 可观测性：W&B 用于训练，Langfuse 用于推断

## 构建它

1. **数据管道。** 在原始语料库上运行 Datatrove 去重。应用 Nemotron-CC 风格质量分类器。Presidio 清洗 PII。用显式种子写入训练/验证分割。

2. **污染检查。** 对每个验证分割，针对 MMLU-Pro、MT-Bench-v2、RewardBench-2 测试集计算 MinHash。拒绝任何重叠。

3. **Axolotl SFT。** YAML 带 ZeRO-3、FA3、序列打包。在 8xH100 上运行 2-3 个周期。记录到 W&B。

4. **TRL DPO / GRPO。** 使用 SFT 检查点，在偏好对上运行一个 DPO 周期（或在数学/代码上带可验证奖励的 GRPO）。调整 beta。

5. **量化。** 生成三种量化：GPTQ-INT4-Marlin、AWQ-INT4、GGUF-Q4_K_M 用于 llama.cpp。记录大小和名义吞吐量。

6. **带推测解码的服务。** vLLM 0.7 配置，带通过 Red Hat Speculators 训练的 EAGLE-3 草稿头。测量批次 1/8/32 时的接受率和尾部延迟。报告每百万 token 美元成本 vs Anthropic/OpenAI 在同一评估上。

7. **评估矩阵。** 在基础、SFT-only、SFT+DPO、SFT+GRPO 上运行 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。生成一张表。

8. **安全评估。** 在开发集上的 Llama Guard 4 通过率。ShieldGemma-2 输出过滤器。

9. **模型卡片。** MOF 2026 模板：数据、训练、评估、安全、许可证、带 YAML 和提交 SHA 的可重现性部分。

## 使用它

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[数据]    30 万去重，1.2 万过滤，28 万接受（seed=7）
[SFT]     3 个周期，8xH100，6 小时 12 分钟，val loss 1.42 -> 1.03
[DPO]     1 个周期，beta=0.08，4xH100，1 小时 40 分钟
[量化]   GPTQ-INT4 4.6 GB，AWQ-INT4 4.8 GB，GGUF-Q4_K_M 5.1 GB
[服务]   vLLM 0.7，EAGLE-3 接受率 0.74，p99 126ms @ bs=8
[评估]   MMLU-Pro +3.2，MT-Bench-v2 +0.41，RewardBench-2 +0.08
[卡片]   在 2026 MOF 下生成 model-card.md
```

## 交付它

`outputs/skill-finetuning-pipeline.md` 描述了可交付成果。单个命令运行数据到 SFT 到 DPO 到量化到服务到评估，并发出模型卡片 + 服务端点。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 评估相对基础的提升 | 目标任务（MMLU-Pro、MT-Bench-v2、任务特定）上的测量收益 |
| 20 | 管道可重现性 | 单个命令用相同种子端到端重新运行 |
| 20 | 数据卫生 | 去重率、PII 清洗覆盖率、污染检查为绿色 |
| 20 | 服务效率 | bs=1/8/32 时的 tokens/s，EAGLE-3 接受率，每百万 token 美元成本 |
| 15 | 模型卡片 + 安全评估 | 2026 MOF 完整性 + Llama Guard 4 通过率 |
| **100** | | |

## 练习

1. 在同一任务特定基准上运行 SFT-only vs SFT+DPO vs SFT+GRPO。报告哪种偏好方法胜出以及差距有多大。

2. 将 Llama 3.3 8B 换为 Qwen3 14B。测量在相同质量下每百万 token 美元成本。

3. 测量 EAGLE-3 在领域数据 vs 通用 ShareGPT 上的接受率。报告差异及其对延迟预算的含义。

4. 注入 1% 的污染（将 MMLU-Pro 答案泄漏到训练数据中）并重新运行评估。观察 MMLU-Pro 准确率不现实地跳升。构建一个捕获此情况的污染检查 CI 门控。

5. 添加 LoRA SFT 作为全量微调的替代方案。以 10 倍更低内存测量质量差距。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Axolotl | "SFT 训练器" | 用于 SFT、DPO 和蒸馏的统一 YAML 驱动训练器 |
| TRL | "偏好微调器" | Hugging Face 用于 LLM 上 DPO、GRPO、PPO 的库 |
| GRPO | "群组相对策略优化" | DeepSeek R1 的带可验证奖励的 RL 方案 |
| EAGLE-3 | "推测解码草稿" | 预测 N 个前瞻 token 的草稿头；vLLM 用目标模型验证 |
| MOF | "模型开放框架" | 2026 年在数据、代码、许可证上给模型发布评级的标准 |
| Contamination check（污染检查） | "分割卫生" | 基于 MinHash 的测试集泄漏到训练集的检测 |
| Acceptance rate（接受率） | "EAGLE / MTP 指标" | 目标模型接受的草稿 token 比例 |

## 延伸阅读

- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/) — 参考 SFT / DPO 训练器
- [TRL 文档](https://huggingface.co/docs/trl) — DPO 和 GRPO 参考实现
- [Unsloth](https://github.com/unslothai/unsloth) — 单 GPU 迭代参考
- [DeepSeek R1 论文（arXiv:2501.12948）](https://arxiv.org/abs/2501.12948) — GRPO 方法论
- [vLLM + EAGLE-3 文档](https://docs.vllm.ai) — 参考服务栈
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 备用推测解码训练器
- [模型开放框架 2026](https://isocpp.org/) — 开放发布评级标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 标准评估运行器

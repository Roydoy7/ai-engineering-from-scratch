# Janus-Pro：解耦编码器的统一多模态模型（Janus-Pro: Decoupled Encoders for Unified Multimodal Models）

> 统一多模态模型存在一个不可避免的张力。理解任务需要语义特征——富含概念级信息的 SigLIP 或 DINOv2 输出向量。生成任务需要重建友好的编码——能还原为清晰像素的 VQ token。这两个目标在单一编码器中是不兼容的。Janus（DeepSeek，2024 年 10 月）和 Janus-Pro（DeepSeek，2025 年 1 月）的论点是停止尝试兼顾：解耦两个编码器。在任务之间共享 Transformer 主干，但理解路径走 SigLIP，生成路径走 VQ 分词器。在 70 亿参数下，Janus-Pro 在 GenEval 上击败 DALL-E 3，同时在 MMMU 上匹配 LLaVA。本章解读为什么两个编码器在单编码器失败的地方成功。

**类型：** 构建  
**语言：** Python（标准库，双编码器路由 + 共享主干信号）  
**前置知识：** Phase 12 · 13（Transfusion）、Phase 12 · 14（Show-o）  
**预计时间：** 约 120 分钟

## 学习目标

- 解释为什么单一共享编码器会在理解质量或生成质量上做出妥协。
- 描述 Janus-Pro 的路由：理解侧输入使用 SigLIP 特征，生成侧输入和输出使用 VQ token。
- 追踪让 Janus-Pro 在 Janus 失败之处成功的数据规模扩展。
- 对比解耦（Janus-Pro）、耦合连续（Transfusion）和耦合离散（Show-o）三种架构。

## 问题所在

统一模型在理解和生成之间共享 Transformer 主干。之前的尝试（Chameleon、Show-o、Transfusion）对两个方向都使用同一个视觉分词器。这个分词器是一种妥协：

- 为重建优化（生成）：VQ-VAE 捕捉细粒度像素细节，但产生语义一致性弱的 token。
- 为语义优化（理解）：SigLIP 嵌入将"猫"的图像聚集在"猫"的 token 附近，但无法实现良好的重建。

Show-o 和 Transfusion 在其中一个方向上支付了明显的质量代价。Janus-Pro 问：当任务有不同需求时，为什么要强求使用一个分词器？

## 核心概念

### 解耦视觉编码

Janus-Pro 的架构分离了两个编码器：

- **理解路径。** 输入图像 → SigLIP-SO400m → 2 层 MLP → Transformer 主干。
- **生成路径（条件输入）。** 输入图像（如果以现有图像为条件）→ VQ 分词器 → token ID → Transformer 主干。
- **输出生成。** Transformer 预测的图像 token → VQ 解码器 → 像素。

Transformer 主干是共享的。主干上下游的一切都是任务专属的。

输入通过提示词格式区分：`<understand>` 标签走 SigLIP 路径；`<generate>` 标签走 VQ 路径。或者路由从任务中隐式推断。

### 为何有效

理解损失获得 SigLIP 特征，这些特征由 CLIP 风格预训练针对语义相似性调整过。由于输入特征更适合任务，模型的感知基准优于 Show-o/Transfusion。

生成损失获得 VQ token，这些 token 由分词器针对重建调整过。由于 VQ 编码能干净地还原回像素，图像质量优于 Show-o。

共享 Transformer 主干看到两种输入分布（SigLIP 和 VQ），并学会与两者协作。主张是：足够的数据 + 足够的参数，主干可以吸收这种切换。

### 数据规模——Janus vs Janus-Pro

Janus（原版，arXiv 2410.13848）引入了解耦概念，但规模较小（13 亿参数，有限数据）。Janus-Pro（arXiv 2501.17811）进行了规模化：

- 70 亿参数（vs 13 亿）。
- 第一阶段（对齐）9000 万图文对（vs 7200 万）。
- 第二阶段（统一）7200 万（vs 2600 万）。
- 第三阶段增加了 20 万个图像生成指令样本。

结果：Janus-Pro-7B 在 MMMU 上匹配 LLaVA（60.3 vs ~58），在 GenEval 上击败 DALL-E 3（0.80 vs 0.67）。一个开放模型，在统一谱系的两端都有竞争力。

### JanusFlow——整流流变体

JanusFlow（arXiv 2411.07975）将 VQ 生成路径替换为整流流生成路径（连续）。分裂变成了理解用 SigLIP + 生成用整流流。质量上限进一步提升。架构保持解耦编码器 + 共享主干的模式。

### 共享主干的职责

Transformer 主干处理统一序列，但有两种输入分布。它的职责是：

- **理解时：** 消耗 SigLIP 特征 + 文本 token → 自回归输出文本。
- **生成时：** 消耗文本 token + （可选的图像 VQ token）→ 自回归输出图像 VQ token。

主干每个块中没有模态专属的权重。它就是你在 Qwen 或 Llama 内部期望找到的文本风格 Transformer，加上两个输入适配器。

有趣的是，这意味着 Janus-Pro 的主干可以从预训练 LLM 初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。这个选择很重要：LLM 贡献了纯从头训练的统一模型难以达到的推理能力。

### 与 InternVL-U 的对比

InternVL-U（Lesson 12.10）是 2026 年的后续产品。它结合了：

- 原生多模态预训练（InternVL3 骨干）。
- 解耦编码器路由（SigLIP 输入，VQ + 扩散头输出）。
- 统一的理解 + 生成 + 编辑能力。

InternVL-U 将 Janus-Pro 的架构选择纳入了更大的框架。解耦编码器的想法现在已成为大规模统一模型的默认选择。

### 局限性

解耦编码器增加了架构复杂度。需要训练两个分词器，维护两条输入路径，处理两套故障模式。对于不需要生成的产品，Janus-Pro 是过度设计——选 LLaVA 系列理解模型即可。

对于不需要理解的产品，Janus-Pro 是大材小用——选 Stable Diffusion 3/Flux 模型即可。

对于需要两者兼备的产品，Janus-Pro 现在是参考开放架构。

## 动手使用

`code/main.py` 模拟 Janus-Pro 路由：

- 两个模拟编码器：类 SigLIP（产生 256 维语义向量）和类 VQ（产生整数编码）。
- 一个根据任务标签选择编码器的提示词路由器。
- 一个（替代）共享主干，处理无论哪个编码器产生的 token 序列。
- 从第一阶段（对齐）到第三阶段（指令调整）的加权采样调度切换。

打印三个示例的路由路径：图像问答、文本生成图像、图像编辑。

## 输出产物

本章生成 `outputs/skill-decoupled-encoder-picker.md`。给定一个希望在接近前沿质量上实现统一生成 + 理解的产品，在 Janus-Pro、JanusFlow 或 InternVL-U 之间做出选择，并给出具体的数据规模建议。

## 练习

1. Janus-Pro-7B 在 GenEval 上击败 DALL-E 3。解释为什么一个 70 亿参数的开放模型在生成上能匹敌前沿专有模型，但在理解上却不能。

2. 实现一个路由器函数：给定提示词文本，分类为 `understand` 或 `generate`。如何处理"描述然后素描"这样的模糊提示词？

3. JanusFlow 将 VQ 路径替换为整流流。Transformer 主干现在输出什么，损失有什么变化？

4. 提出 Janus-Pro 架构可以用再一个解耦编码器处理的第四个任务。示例：图像分割（DINO 风格）、深度估计（MiDaS 风格）。

5. 阅读 Janus-Pro 第 4.2 节关于数据规模的内容。哪个数据阶段对文本生成图像质量提升（相比 Janus）贡献最大？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 解耦编码（Decoupled encoding） | "两个视觉编码器" | 每个方向独立的分词器或编码器：理解用语义编码，生成用重建编码。 |
| 共享主干（Shared body） | "一个 Transformer" | 单个 Transformer 处理任一编码器的输出；没有模态专属的权重。 |
| 理解用 SigLIP（SigLIP for understanding） | "语义特征" | CLIP 家族视觉塔，提供丰富的概念特征，但重建效果差。 |
| 生成用 VQ（VQ for generation） | "重建编码" | 能干净解码回像素的向量量化 token。 |
| JanusFlow | "整流流变体" | 用连续流匹配生成头替换 VQ 的 Janus-Pro。 |
| 路由标签（Routing tag） | "任务标签" | 提示词标记（`<understand>` / `<generate>`），用于选择输入编码器。 |

## 延伸阅读

- [Wu 等 — Janus（arXiv:2410.13848）](https://arxiv.org/abs/2410.13848)
- [Chen 等 — Janus-Pro（arXiv:2501.17811）](https://arxiv.org/abs/2501.17811)
- [Ma 等 — JanusFlow（arXiv:2411.07975）](https://arxiv.org/abs/2411.07975)
- [InternVL-U（arXiv:2603.09877）](https://arxiv.org/abs/2603.09877)
- [Dong 等 — DreamLLM（arXiv:2309.11499）](https://arxiv.org/abs/2309.11499)

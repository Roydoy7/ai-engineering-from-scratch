# Flamingo 与门控交叉注意力——少样本 VLM（Flamingo and Gated Cross-Attention for Few-Shot VLMs）

> DeepMind 的 Flamingo（2022）做了两件领先于所有人的事。它证明了单个模型能处理任意交错排列的图像、视频和文本序列；也证明了 VLM 可以进行上下文学习——给出三个（图像、描述）示例的少样本提示，模型无需任何梯度步骤就能描述新图像。机制：在冻结 LLM 的现有层之间插入门控交叉注意力层，带有一个初始化为零的可学习 tanh 门，使 LLM 的文本能力在初始化时得以保留。本章解读 Flamingo 的 Perceiver 重采样器与门控交叉注意力架构——Gemini 交错输入和 Idefics2 视觉 token 的祖先。

**类型：** 学习  
**语言：** Python（标准库，门控交叉注意力 + Perceiver 重采样器演示）  
**前置知识：** Phase 12 · 03（BLIP-2 Q-Former）  
**预计时间：** 约 120 分钟

## 学习目标

- 解释门控交叉注意力如何通过 `tanh(gate) = 0` 在初始化时保留冻结 LLM 的文本能力。
- 逐步介绍 Perceiver 重采样器：通过交叉注意力将 N 个图像图块压缩为 K 个固定"潜在"查询。
- 描述 Flamingo 如何通过因果掩码处理交错的图像-文本序列，并尊重图像的位置。
- 复现少样本多模态提示结构（3 个图文示例，然后查询图像）。

## 问题所在

BLIP-2 将 32 个视觉 token 喂给冻结 LLM 的输入层，每个提示词处理一张图像没有问题。但如果你想喂入*多张*与文本交错的图像，例如"这是图像 A，描述它；这是图像 B，描述它；现在这是图像 C，描述它"呢？LLM 的自注意力需要在单个流中处理图像 token 和文本 token，哪些位置能关注哪些图像的问题就变得棘手了。

Flamingo 的答案：完全不改变 LLM 的输入流。在现有 LLM 块之间插入额外的交叉注意力层。文本 token 仍然像往常一样流经 LLM 的因果自注意力。在每几个 LLM 块之间，文本 token 还会通过新的门控层交叉关注图像特征。门控（初始化为零）意味着在第 0 步时新层是无操作的——模型表现得完全像预训练的 LLM。随着训练推进，门控打开，视觉信息开始流入。

Flamingo 回答的第二个问题：如何处理每个提示词中数量不定（0、1 或多张）的图像？Perceiver 重采样器——一个小型交叉注意力模块，接受任意数量的图块并产出固定数量的视觉潜在 token。无论提示词中有多少张图像，LLM 的交叉注意力层看到的形状都相同。

## 核心概念

### 冻结 LLM

Flamingo 以冻结的 Chinchilla 70B LLM 为起点，所有 700 亿权重保持不变，现有的文本自注意力和 FFN 正常运行。

### Perceiver 重采样器

对于提示词中的每张图像，ViT 产生 N 个图块 token。Perceiver 重采样器有 K 个固定的可学习潜在向量（Flamingo 使用 K=64）。每个重采样器块有两个子步骤：

1. 交叉注意力：K 个潜在向量关注 N 个图块 token（Q 来自潜在向量，K/V 来自图块）。
2. 潜在向量内部的自注意力 + FFN。

经过 6 个重采样器块后，无论 ViT 产生了多少图块，输出都是 K=64 个维度为 1024 的视觉 token。224×224 的图像（196 个图块）和 480×480 的图像（900 个图块）都输出为 64 个重采样器 token。

对于视频，重采样器按时间维度应用：每帧的图块产出 64 个潜在向量，时间位置编码使模型能区分 t=0 和 t=N。整个视频变成 T×64 个视觉 token。

### 门控交叉注意力

在冻结 LLM 的每 M 层之间（Flamingo 使用 M=4），插入一个新的门控交叉注意力块：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是初始化为零的可学习标量。
- `tanh(0) = 0`，因此在初始化时门控分支贡献为零。
- 随着 `alpha` 偏离零，交叉注意力的贡献平滑增长。
- 残差连接意味着即使门控完全打开，也不会覆盖 LLM 的文本表示；只是在其上叠加视觉信息。

这是 Flamingo 中最重要的设计决策：视觉条件是加性的、有门控的，且在初始化时为零。第 0 步的 Flamingo 在纯文本输入上就是完美的 Chinchilla 70B。

### 交错输入的掩码交叉注意力

在"`<图像A> 描述A <图像B> 描述B <图像C> ?`"这样的提示词中，每个文本 token 应该只看到序列中它之前的图像。交叉注意力掩码强制执行：位置 `t` 的文本 token 只关注图像索引 `i < i_t` 的重采样器 token，其中 `i_t` 是位置 `t` 之前最近的图像索引。"只看最近的前置图像"或"看所有前置图像"都是有效的选择；Flamingo 选择了前者。

### 上下文少样本学习

Flamingo 的提示词形如：

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型看到补全模式后输出"bird"（或 image3 所展示的任何内容）。无需梯度步骤。冻结 LLM 的上下文学习能力通过门控交叉注意力传递——这是论文的核心贡献，也是它的重要性所在。

### 训练数据

Flamingo 在三个数据集上训练：

1. MultiModal MassiveWeb（M3W）：4300 万个网页，包含按阅读顺序交错的图像和文本。
2. 图文对（ALIGN + LTIP）：44 亿对。
3. 视频-文本对（VTP）：2700 万个短视频片段。

OBELICS（2023）是交错网络语料库的开放复现，Idefics、Idefics2 和大多数开放"Flamingo 类"模型在其上训练。

### OpenFlamingo 与 Otter

OpenFlamingo（2023）是开放复现，架构完全相同（Perceiver 重采样器 + 基于冻结 LLaMA 或 MPT 的门控交叉注意力），检查点有 3B、4B、9B 三种规模。由于基础 LLM 较小、数据较少，质量落后于 Flamingo。

Otter（2023）在 OpenFlamingo 基础上用 MIMIC-IT（一个多模态指令数据集）进行指令微调，证明门控交叉注意力同样适用于指令跟随。

### 后代

- Idefics / Idefics2 / Idefics3：Hugging Face 的门控交叉注意力传承，逐步简化（Idefics2 放弃了重采样器，转而使用带自适应池化的直接图块 token）。
- Flamingo 到 Chameleon 的过渡：到 2024 年许多团队转向早期融合（Lesson 12.11）；Flamingo 风格的门控交叉注意力在需要骨干网络冻结的生产场景中仍然存在。
- Gemini 的交错输入：概念上继承了 Flamingo 的交错格式灵活性，但具体机制是专有的。

### 与 BLIP-2 的对比

| | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥梁 | Q-Former 一次性在输入层 | 在每 M 层插入门控交叉注意力 |
| 每图像视觉 token | 32 个 | 每个交叉注意力层 64 个 |
| 冻结 LLM | 是 | 是 |
| 少样本上下文学习 | 弱 | 强——论文的核心 |
| 交错输入 | 不原生支持 | 是，设计目标 |
| 训练数据 | 1.3 亿对 | 13 亿对 + 4300 万交错页面 |
| 参数量（训练部分） | 1.88 亿 | 约 100 亿（交叉注意力层） |
| 计算成本 | 数天（8 块 A100） | 数周（数千块 TPUv4） |

预算有限的单图像 VQA 选 BLIP-2；交错、少样本或多图像推理选 Flamingo/Idefics2。

## 动手使用

`code/main.py` 演示：

1. 在 36 个假图块 token 上运行带 8 个可学习潜在向量的 Perceiver 重采样器（纯 Python 交叉注意力）。
2. 一个门控交叉注意力步骤：`alpha = 0` 时输出等于输入（LLM 不变），`alpha = 2.0` 时视觉贡献混入。
3. 一个交错掩码构建器，为"（图像1）（文本1）（图像2）（文本2）"序列生成二维注意力掩码。

## 输出产物

本章生成 `outputs/skill-gated-bridge-diagnostic.md`。给定开放 VLM 的配置（是否有重采样器、交叉注意力频率、门控方案），它识别 Flamingo 谱系元素并解释冻结策略。适用于调试微调为何降低了文本性能（答案：门控打开太快太宽）。

## 练习

1. 计算 Flamingo-9B 的视觉参数量：90 亿 LLM + 14 亿门控交叉注意力层 + 6400 万重采样器。训练部分占总参数的多少比例？

2. 用 PyTorch 实现门控残差 `y = tanh(alpha) * cross + x`。通过实验证明 `alpha=0` 时初始化时 `y==x` 精确成立。

3. 阅读 OpenFlamingo 第 3.2 节（arXiv:2308.01390），了解当批次中每个提示词有不同图像数量时如何处理。描述其填充策略。

4. 为什么 Flamingo 的交叉注意力掩码让文本 token 只关注*最近的*前置图像，而不是所有前置图像？阅读 Flamingo 论文第 2.4 节并解释这一权衡。

5. 少样本上下文：为新的 Flamingo 变体构建一个包含 4 个"图像 → 主要对象颜色"示例的提示词。描述当示例数量从 0 变化到 8 时预期的准确率模式。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Perceiver 重采样器 | "固定潜在向量交叉注意力" | 从可变数量的输入图块产出 K 个固定 token 的模块。 |
| 门控交叉注意力（Gated cross-attention） | "Tanh 门控桥梁" | 残差层 `y = tanh(alpha)*cross + x`，可学习 alpha，初始化为 0。 |
| 交错输入（Interleaved input） | "混合序列" | 图像和文本按阅读顺序自由混合的提示词格式。 |
| 冻结 LLM（Frozen LLM） | "不计 LLM 梯度" | 文本 LLM 的权重不更新；只有重采样器 + 交叉注意力层训练。 |
| 少样本（Few-shot） | "上下文示例" | 在提示词中给出少量（图像、答案）对；模型无需微调即可泛化。 |
| OBELICS | "交错网络语料库" | 包含按阅读顺序排列图像和文本的 1.41 亿个网页的开放数据集。 |
| Chinchilla | "700 亿冻结基础模型" | Flamingo 的冻结文本 LLM，来自 DeepMind 的 Chinchilla 论文。 |
| 门控计划（Gate schedule） | "alpha 如何变化" | 训练期间交叉注意力门控打开的速率。 |
| 交叉注意力频率（Cross-attn frequency） | "每 M 层" | 插入门控交叉注意力块的频率；Flamingo 使用 M=4。 |
| OpenFlamingo | "开放复现" | MosaicML/LAION 的开放检查点（3-9B）；架构与 Flamingo 完全相同。 |

## 延伸阅读

- [Alayrac 等 — Flamingo（arXiv:2204.14198）](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla 等 — OpenFlamingo（arXiv:2308.01390）](https://arxiv.org/abs/2308.01390) — 开放复现。
- [Laurençon 等 — OBELICS（arXiv:2306.16527）](https://arxiv.org/abs/2306.16527) — 交错网络语料库。
- [Jaegle 等 — Perceiver IO（arXiv:2107.14795）](https://arxiv.org/abs/2107.14795) — 通用 Perceiver 架构。
- [Li 等 — Otter（arXiv:2305.03726）](https://arxiv.org/abs/2305.03726) — 指令微调的 Flamingo 后代。
- [Laurençon 等 — Idefics2（arXiv:2405.02246）](https://arxiv.org/abs/2405.02246) — Flamingo 方法的现代简化版。

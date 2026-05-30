# 视觉自回归建模（VAR）：下一尺度预测

> 扩散模型在时间上迭代采样（去噪步骤）。VAR 在尺度上迭代采样——它预测一个 1×1 的 token，然后 2×2，然后 4×4，直到最终分辨率，每个尺度以前一个为条件。2024 年的论文表明，VAR 匹配 GPT 风格的图像生成缩放定律，并在相同计算预算下优于 DiT。本课构建核心机制。

**类型：** 构建
**语言：** Python（使用 PyTorch）
**前置知识：** 第7阶段第03课（多头注意力）、第8阶段第06课（DDPM）
**预计时间：** 约90分钟

## 问题背景

自回归生成主导了语言建模，因为它的扩展是可预测的：更多计算量，更多参数，更低困惑度，更好的输出。图像生成在 2024 年之前有两次主要的 AR 尝试：PixelRNN/PixelCNN（逐像素）和 DALL-E 1 / Parti / MuseGAN（VQ-VAE 编码上的逐 token）。

两者都有生成顺序问题。像素和 token 排列在二维网格中，但 AR 模型必须以一维光栅顺序访问它们。早期的角落像素对图像最终会变成什么毫不知情。生成质量的缩放比 GPT-on-text 差，在匹配计算量下从未达到扩散模型的质量。

VAR 通过改变被生成的内容来解决生成顺序问题。VAR 不是在空间中逐一预测图像 token，而是以越来越高的分辨率预测整张图像。步骤 1：预测 1×1 的 token（整体图像"摘要"）。步骤 2：预测 2×2 的 token 网格（较粗的特征）。步骤 3：预测 4×4 网格。步骤 K：预测最终的 (H/8)×(W/8) 网格。

每个尺度关注所有之前的尺度（在"尺度顺序"上是因果的），并在其自身尺度内并行。顺序问题消失：尺度 k 上的整张图像在一次 Transformer 前向传播中产生。

## 核心概念

### VQ-VAE 多尺度分词器

VAR 需要一个**多尺度离散分词器**。对于图像 x，它产生一系列分辨率逐渐提高的 token 网格：

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个 z_k 使用相同的码本（典型大小 4096-16384）。每个尺度的分词化不是独立的——它经过训练使得各尺度的残差之和能重建 f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是一种**残差 VQ** 变体。尺度 k 捕获尺度 1..k-1 遗漏的内容。解码器取所有尺度嵌入的总和并产生图像。

多尺度 VQ 分词器训练一次（类似 VQGAN）然后冻结。所有生成工作由其上的自回归模型完成。

### 下一尺度预测

生成模型是一个 Transformer，它看到所有之前尺度的 token 并预测下一尺度的 token。

输入序列结构：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

位置嵌入同时编码尺度索引和尺度内的空间位置。注意力在尺度顺序上是因果的：尺度 k、位置 (i, j) 处的 token 可以关注尺度 1..k 的所有 token 以及尺度 k 本身中按任何尺度内顺序排列的较早 token（VAR 使用固定位置注意力，没有尺度内因果性——尺度内的所有位置并行预测）。

训练损失：在每个尺度 k，给定所有之前尺度的 token，预测 token z_k。对离散 VQ 编码使用交叉熵损失。结构与 GPT 相同，只是"序列"现在是按尺度结构化的。

### 生成

推理时：
```
generate z_1 = sample from p(z_1)                    # 1 个 token
generate z_2 = sample from p(z_2 | z_1)              # 4 个 token 并行
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 个 token 并行
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

对于 K = 10 个尺度，生成是 10 次 Transformer 前向传播。每次传播并行产生其整个尺度——尺度内没有逐 token 的自回归。对于 256×256 的图像，这大约是 10 次传播，而 DiT 需要 28-50 次。

### 为何下一尺度优于下一 token

三个结构性优势：
1. **由粗到细与自然图像统计一致。** 人类视觉感知和图像数据集都表现出与尺度相关的规律性：低频结构稳定且可预测；高频细节以低频内容为条件。下一尺度预测利用了这一点。
2. **尺度内并行生成。** 与 GPT 风格的 token AR 不同，VAR 在一步中产生一个尺度的所有 token。有效生成长度是对数级而非线性级。
3. **无生成顺序偏差。** 尺度 k 的 token 看到所有尺度 k-1 的内容；不存在强制早期 token 在晚期上下文可用之前就做出承诺的"左侧"或"上方"偏差。

### 缩放定律

Tian et al. 证明 VAR 对 ImageNet 上的 FID 遵循幂律缩放曲线——就像 GPT 对困惑度的表现一样。参数或计算量翻倍可靠地将误差减半。这是第一个像语言模型一样干净地展示这种缩放行为的图像生成模型。结果是 VAR 尺度的预测从计算量变得可预测，而不是每个架构的经验猜测。

### 与扩散的关系

VAR 和扩散共享相同的数据压缩故事：两者都将生成问题分解为一系列更简单的子问题。

- 扩散：逐渐添加噪声，学习撤销一步。
- VAR：逐渐增加分辨率，学习预测下一尺度。

它们是通过问题的不同轴。两者都产生可处理的条件分布。从经验上看，VAR 在推理时更快（更少的传播，尺度内全部并行），并在类别条件 ImageNet 上匹配或优于 DiT。文本条件 VAR（VARclip、HART）是一个活跃的研究方向。

## 动手实现

在 `code/main.py` 中你将：
1. 在合成"图像"数据（二维高斯环）上构建一个微型**多尺度 VQ 分词器**。
2. 训练一个 **VAR 风格的 Transformer** 进行下一尺度预测。
3. 通过调用 Transformer 4 次（4 个尺度）并解码来采样。
4. 验证尺度顺序训练使尺度内生成真正并行。

这是一个玩具实现。重点是看到尺度结构化的注意力掩码和尺度内并行生成实际工作。

## 交付物

本课产生 `outputs/skill-var-tokenizer-designer.md`——一个设计多尺度分词器的技能：尺度数量、尺度比例、码本大小、残差共享、解码器架构。

## 练习

1. **尺度数量消融。** 用 4、6、8、10 个尺度训练 VAR。测量重建质量 vs 自回归传播次数。更多尺度 = 更细的残差 = 更好的质量，但需要更多传播。

2. **码本大小。** 用码本大小 512、4096、16384 训练分词器。更大的码本给出更好的重建，但预测更难。找到拐点。

3. **尺度内并行性检查。** 对于训练好的 VAR，显式测量注意力模式。在尺度 k 内，模型是否关注跨尺度位置但不关注尺度内位置？验证掩码实现。

4. **VAR vs DiT 缩放。** 对于相同的 ImageNet 类别条件任务，在匹配参数预算下（例如 3300 万、1.3 亿、4.58 亿）训练 VAR 和 DiT。绘制 FID vs 计算量。VAR 应该在每个规模上超越 DiT——在小规模上复现论文的结果。

5. **文本条件化。** 将 VAR 扩展为通过 adaLN 接受文本嵌入（CLIP 池化）作为额外的条件输入。这是 HART 方案。文本对齐采样的 FID 改善了多少？

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| VAR | "视觉自回归" | 通过 VQ token 金字塔的下一尺度预测进行图像生成。 |
| 下一尺度预测 (Next-scale prediction) | "预测较粗的，再预测较细的" | 模型以越来越高的分辨率尺度预测 token，以所有之前尺度为条件。 |
| 多尺度 VQ 分词器 (Multi-scale VQ tokenizer) | "残差 VQ" | 产生 K 个分辨率递增的 token 网格的 VQ-VAE，解码器对所有尺度求和。 |
| 尺度 k (Scale k) | "金字塔第 k 层" | K 个分辨率级别之一，从 k=1 的 1×1 到 k=K 的 (H/p)×(W/p)。 |
| 尺度内并行 (Parallel-within-scale) | "每尺度一次前向传播" | 尺度 k 的所有 token 在一次 Transformer 传播中预测，而不是自回归地。 |
| 跨尺度因果 (Causal-across-scales) | "尺度顺序注意力" | 尺度 k 的 token 可以关注尺度 1..k 的所有内容，但不能关注尺度 k+1..K。 |
| 残差 VQ (Residual VQ) | "加性分词化" | 每个尺度的 token 编码较低尺度遗留的残差；解码器对所有尺度嵌入求和。 |
| VAR 缩放定律 (VAR scaling law) | "图像 GPT 缩放" | FID 在计算量上遵循可预测的幂律，类似语言模型的困惑度。 |
| HART | "混合 VAR + 文本" | 将 MaskGIT 风格的迭代解码与 VAR 的尺度结构相结合的文本条件 VAR 变体。 |
| 尺度位置嵌入 (Scale position embedding) | "(尺度, 行, 列) 三元组" | 位置编码同时携带尺度索引和尺度内的空间坐标。 |

## 延伸阅读

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — VAR 论文，权威参考
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT，扩散比较基线
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN，VAR 多尺度分词器扩展的分词器家族
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE，离散图像分词的基础
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — 文本条件 VAR

# Chameleon 与早期融合纯 Token 多模态模型（Chameleon and Early-Fusion Token-Only Multimodal Models）

> 我们迄今看到的每个 VLM 都保持图像和文本的分离。视觉 token 来自视觉编码器，流经投影器，然后在 LLM 内部与文本相遇。视觉词汇表和文本词汇表从不重叠。Chameleon（Meta，2024 年 5 月）提出：如果它们重叠会怎样？训练一个 VQ-VAE，将图像转换为共享词汇表中的离散 token 序列。每个多模态文档现在都是一个序列——文本 token 和图像 token 交错，单一的自回归损失。副作用：模型可以生成混合模态输出——在单次推理调用中交替产生文本和图像 token。本章解读早期融合的论点，并端到端构建一个玩具版本。

**类型：** 构建  
**语言：** Python（标准库，VQ-VAE 分词器 + 交错解码器）  
**前置知识：** Phase 12 · 05、Phase 8（生成式 AI）  
**预计时间：** 约 180 分钟

## 学习目标

- 解释为什么共享词汇表 + 单一损失改变了模型能做的事情。
- 描述 VQ-VAE 如何将图像分词为与 Transformer 下一个 token 目标兼容的离散序列。
- 说出 Chameleon 的训练稳定性技巧：QK-Norm、Dropout 放置、LayerNorm 顺序。
- 比较 Chameleon 与 BLIP-2 的 Q-Former 方法，描述各自何时是正确选择。

## 问题所在

基于适配器的 VLM（LLaVA、BLIP-2、Qwen-VL）将文本和图像视为两种不同的东西。文本 token 通过 `embed(text_token)`；图像通过 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两条在中途合并的输入路径。

三个后果：

1. LLM 只能消费图像，不能生成图像。输出只有文本。
2. 混合模态文档（段落和图像交替，如文章）处理起来很笨拙——你要么在模型外解析多模态输入，要么链接多次生成。
3. 分布不匹配。视觉 token 和文本 token 生活在隐藏空间的不同区域，造成微妙的对齐问题。

Chameleon 拒绝这一前提：图像不过是来自共享词汇表的离散 token 序列。在交错文档上训练模型，一个损失，一个自回归解码器，你就免费解锁了混合模态生成。

## 核心概念

### VQ-VAE 作为图像分词器

分词器是向量量化变分自编码器。架构：

- **编码器：** CNN + ViT，将图像映射到空间特征图，比如 32×32 个维度为 256 的特征。
- **码本：** K 个可学习向量（Chameleon 使用 8192 个），同样维度 256。
- **量化：** 对每个空间特征，通过 L2 距离找到最近的码本条目，用整数索引替换连续特征。
- **解码器：** CNN，将量化特征还原为像素。

训练：VAE 重建损失 + 承诺损失 + 码本损失。码本索引形成图像的离散字母表。

对于 Chameleon：一张图像变成 32×32 = 1024 个来自 8192 词汇表的 token。与文本 token（来自 LLM 的 BPE 词汇表，约 32000 个）拼接。最终词汇表：40192。Transformer 看到一个序列，一个损失。

### 共享词汇表

Chameleon 的词汇表结合了文本 token、图像 token 和模态分隔符。每个 token 有唯一的 ID。输入嵌入层将每个 ID 映射到 D 维隐藏向量。输出投影将隐藏映射回词汇 logit。Softmax 选择下一个 token，无论是哪种模态。

分隔符很重要：`<image>` 和 `</image>` 标签括住图像 token 序列。在生成时，如果模型发出 `<image>`，下游软件知道接下来的 1024 个 token 是要发送给解码器渲染像素的 VQ 索引。

### 混合模态生成

推理就是在共享词汇表中的下一个 token 预测。示例提示："画一只猫并描述它。" Chameleon 输出：

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

模型自主选择顺序——它可能先产生图像再产生文本、先文本再图像，或交错产生。相同的解码器，相同的损失。

与只能输出文本的适配器 VLM 相比，Chameleon 重新开放了模型输出模态的问题。

### 训练稳定性——QK-Norm、Dropout、LayerNorm 顺序

早期融合训练在规模上是不稳定的。Chameleon 论文记录了三个技巧：

- **QK-Norm。** 在注意力中，在点积之前对查询和键投影应用 LayerNorm。防止深度处的 logit 量级爆炸。2024 年后的多个大型模型都使用了这一技巧。
- **Dropout 放置。** 在每个残差加法之后应用 Dropout，而不仅仅是在注意力和 MLP 之后。当来自图像 token 的梯度可能占主导时，需要更多正则化。
- **LayerNorm 顺序。** 残差分支上的前置 LN（标准做法），加上最后一个块的跳跃连接上的额外 LN。稳定最终层的梯度流。

没有这些技巧，340 亿参数的 Chameleon 训练在多个检查点发散。有了它们，训练收敛。训练方案与架构一样是核心贡献。

### 分词器的重建上限

VQ-VAE 是有损的。在 8192 个码本条目和每张 512×512 图像 1024 个 token 的情况下，重建 PSNR 上限约为 26-28 dB。这对可识别的图像生成足够，但明显劣于连续空间扩散（Stable Diffusion 3 达到 32+ dB）。

分词器是瓶颈。更好的分词器（MAGVIT-v2、IBQ、SBER-MoVQGAN）提升上限。Emu3（Lesson 12.12）仅通过更好的分词器就实现了 SDXL 级别的生成质量。

### Chameleon vs BLIP-2 / LLaVA

**Chameleon（早期融合，共享词汇表）：**
- 一个损失，一个解码器。
- 生成混合模态输出。
- 分词器是质量上限。
- 推理路径上每张生成图像都需要 VQ-VAE 解码器，成本较高。

**BLIP-2 / LLaVA（晚期融合，独立塔）：**
- 视觉输入，仅文本输出。
- 复用预训练 LLM。
- 理解任务无分词器瓶颈。
- 单次前向传播，成本低。

按任务选择。需要图像生成选 Chameleon 系列；只需要理解，适配器 VLM 更简单，复用更多预训练计算。

### Fuyu 与 AnyGPT

Fuyu（Adept，2023）是相关方法：完全跳过独立的视觉编码器，将原始图像图块通过 LLM 的输入投影送入，就像它们是 token 一样，没有分词器。比 Chameleon 更简单，但失去了共享词汇表输出生成能力。

AnyGPT（Zhan 等，2024）将 Chameleon 扩展到四种模态：文本、图像、语音、音乐。每种模态使用相同的 VQ-VAE 技巧，共享 Transformer。任意到任意生成。Lesson 12.16 有更多介绍。

## 动手使用

`code/main.py` 构建一个玩具端到端早期融合模型：

- 一个小型 VQ-VAE 风格量化器，将 8×8 图块映射到码本索引（K=16）。
- 一个共享词汇表：（文本 id 0..31）+（图像 id 32..47）+（分隔符 48, 49）。
- 一个玩具自回归解码器（二元组表），在合成描述 + 图像 token 序列上训练。
- 给定提示词，输出交错文本 + 图像 token 的采样循环。

代码有意保持 Transformer 极小（二元组），以便你能端到端追踪信号流。

## 输出产物

本章生成 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定产品规格（仅理解 vs 理解 + 生成、所需图像质量、成本预算），它在 Chameleon 系列（早期融合）和 LLaVA 系列（晚期融合）之间做出选择，并用定量经验法则辩护。

## 练习

1. Chameleon 使用 K=8192 个码本条目和每张 512×512 图像 1024 个 token。估计与 24 位 RGB 图像相比的压缩比。它是有损的吗？损失有多大？

2. 在相同 VQ-VAE 密度下，4K 图像（3840×2160）产生多少图像 token？Chameleon 风格的模型能在一次推理调用中生成 4K 图像吗？最先崩溃的是什么——上下文、分词器质量还是 KV 缓存？

3. 用纯 Python 实现 QK-Norm。给定 64 维的查询和键，显示 LayerNorm 前后的点积。为什么在深度上幅度控制很重要？

4. 阅读 Chameleon 第 2.3 节关于训练稳定性的内容。描述论文在没有 QK-Norm 的 340 亿参数下观察到的确切失败模式。"范数爆炸"的特征是什么？

5. 扩展玩具解码器，给定纯文本提示词输出混合模态响应。在训练数据分布为 60% 文本优先 / 40% 图像优先的情况下，测量模型选择图像优先与文本优先的频率。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 早期融合（Early fusion） | "统一 token" | 图像从第一步起就转换为与 Transformer 词汇表共享的离散 token。 |
| VQ-VAE | "图像分词器" | CNN + ViT + 码本，将图像映射到 Transformer 能预测的整数索引。 |
| 共享词汇表（Shared vocabulary） | "一个字典" | 覆盖文本 + 图像 + 模态分隔符的单一 token ID 空间。 |
| QK-Norm | "注意力稳定器" | 在查询和键点积之前应用 LayerNorm，防止范数爆炸。 |
| 混合模态生成（Mixed-modality generation） | "文本 + 图像输出" | 在一次推理中自主产生交错文本和图像 token。 |
| 码本大小（Codebook size） | "K 个条目" | VQ-VAE 可量化的离散向量数量；权衡压缩率与保真度。 |
| 分词器上限（Tokenizer ceiling） | "重建极限" | 解码 VQ token 可达到的最高 PSNR；限制模型的图像质量。 |

## 延伸阅读

- [Chameleon 团队 — Chameleon: Mixed-Modal Early-Fusion Foundation Models（arXiv:2405.09818）](https://arxiv.org/abs/2405.09818)
- [Aghajanyan 等 — CM3（arXiv:2201.07520）](https://arxiv.org/abs/2201.07520)
- [Yu 等 — CM3Leon（arXiv:2309.02591）](https://arxiv.org/abs/2309.02591)
- [Zhan 等 — AnyGPT（arXiv:2402.12226）](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B 博客（adept.ai）](https://www.adept.ai/blog/fuyu-8b)

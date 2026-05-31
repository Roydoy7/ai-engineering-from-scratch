# Show-o 与离散扩散统一模型（Show-o and Discrete-Diffusion Unified Models）

> Transfusion 混合连续与离散表示。Show-o（Xie 等，2024 年 8 月）走的是另一条路：文本 token 使用因果下一个 token 预测，图像 token 以 MaskGIT 的精神使用掩码离散扩散。两者都在一个带混合注意力掩码的 Transformer 中运行。结果是在同一骨干、每种模态一个分词器、一个损失公式（扩展到掩码预测的下一个 token 预测）上统一了视觉问答、文本生成图像、修复和混合模态生成。本章解读 Show-o 的设计——为什么掩码离散扩散是一个并行的少步图像生成器——并与 Transfusion 和 Emu3 对比。

**类型：** 学习  
**语言：** Python（标准库，掩码离散扩散采样器）  
**前置知识：** Phase 12 · 13（Transfusion）  
**预计时间：** 约 120 分钟

## 学习目标

- 解释掩码离散扩散：均匀掩码 token 然后要求 Transformer 恢复它们的调度。
- 在速度和质量上对比并行图像解码（Show-o、MaskGIT）与自回归图像解码（Chameleon、Emu3）。
- 说出 Show-o 在一个检查点中处理的三个任务：文本生成图像、视觉问答、图像修复。
- 选择一个掩码调度（余弦、线性、截断）并推理其对样本质量的影响。

## 问题所在

Transfusion 的双损失训练有效，但动态更棘手——连续扩散损失与离散 NTP 损失在数值量级上不同。平衡损失权重是一次超参数搜索。架构有效但复杂。

Show-o 的答案：保持两种模态都是离散的（像 Chameleon 一样），但通过掩码离散扩散并行生成图像，而非顺序生成。训练目标变成单一的掩码 token 预测，这是下一个 token 预测的自然推广。

## 核心概念

### 掩码离散扩散（MaskGIT）

原始的 Chang 等（2022）MaskGIT 技巧很优雅。从完全掩码的图像开始（每个 token 都是特殊的 `<MASK>` ID）。在每一步，并行预测所有被掩码的 token，然后保留置信度最高的 K 个预测并重新掩码其余的。经过约 8-16 次迭代，所有 token 都被填入。每步解掩码多少 token 的调度需要调整——余弦调度效果好。

训练很简单：从 [0, 1] 均匀采样一个掩码比例，将其应用于图像的 VQ token，训练 Transformer 恢复被掩码的 token。正是 BERT 对文本做的事，规模化到图像生成。

### Show-o：一个 Transformer，混合掩码

Show-o 将 MaskGIT 嵌入一个因果语言模型 Transformer 中。注意力掩码为：

- 文本 token：因果（标准 LLM）。
- 图像 token：图像块内完全双向（这样被掩码的 token 在预测时可以看到每个其他图像 token）。
- 文本生成图像：文本关注之前的图像，图像关注之前的文本。

训练在以下三种样本之间交替：
1. 文本序列上的标准 NTP。
2. 文本生成图像样本：文本 → 图像（带掩码图像 token），使用掩码 token 预测损失。
3. 视觉问答样本：图像 → 文本（带掩码文本 token，本质上就是 NTP）。

统一损失是在 `<MASK>` token 上的交叉熵，涵盖了文本 NTP（只有最后一个 token 被"掩码"）和图像掩码扩散（随机子集被掩码）两种情况。

### 并行采样

Show-o 在约 16 步内生成一张图像，而不是约 1000 步（每 token 自回归）或约 20 步（扩散）。在每一步，并行预测所有被掩码的 token；提交置信度最高的 K 个；重复。

对比：
- Chameleon / Emu3（token 自回归）：N_tokens 次前向传播，每张图像通常 1024-4096 次。
- Transfusion（连续扩散）：约 20 步，每步一次完整 Transformer 前向传播。
- Show-o（掩码离散扩散）：约 16 步，每步一次完整 Transformer 前向传播。

在相同规模的模型下，Show-o 比 Chameleon 更快，步骤数与 Transfusion 大致相当，但每步成本更低（离散词汇 logit vs 连续 MSE 损失）。

### 一个检查点中的多种任务

Show-o 在推理时支持四种任务，由提示词格式选择：

- 文本生成：标准自回归文本输出。
- 视觉问答：图像输入，文本输出。
- 文本生成图像：文本输入，通过掩码离散扩散输出图像。
- 图像修复：带部分 token 被掩码的图像，填入缺失部分。

图像修复能力来自掩码预测训练，无需额外工作。掩码 VQ token 网格的一个区域，输入其余部分加文本提示词，预测被掩码的 token 即可。

### 掩码调度

每步解掩码多少 token 的调度影响质量。Show-o 推荐余弦调度：

```
mask_ratio(t) = cos(π × t / (2 × T))   # t = 0..T
```

第 0 步时，所有 token 被掩码（比例 1.0）。第 T 步时，无 token 被掩码。余弦调度将质量集中在预测信息最丰富的中等比例范围。线性调度也有效，但平台期来得更快。

### Show-o2

Show-o2（2025 年后续，arXiv 2506.15564）对 Show-o 进行了扩展：更大的 LLM 基座、更好的分词器、改进的掩码调度。架构模式不变。

### Show-o 的定位

在 2026 年的分类体系中：

- 离散 token + NTP：Chameleon、Emu3。简单但推理慢。
- 离散 token + 掩码扩散：Show-o、MaskGIT、LlamaGen、Muse。并行采样，仍受分词器损耗限制。
- 连续 + 扩散：Transfusion、MMDiT、DiT。质量最高，训练更复杂。
- 连续 + 在 VLM 中的流匹配：JanusFlow、InternVL-U。最新。

按任务选择：当你想在一个开放模型中以合理速度同时拥有文本生成图像 + 修复 + 视觉问答时选 Show-o；当质量优先且能承受双损失管道时选 Transfusion。

## 动手使用

`code/main.py` 模拟 Show-o 采样：

- 16 个 VQ token 的玩具网格。
- 一个模拟"Transformer"，根据提示词和当前未掩码的 token 预测 logit。
- 带余弦调度的 8 步并行掩码采样。
- 打印中间状态（掩码模式的演变过程）和最终 token。

运行它，观察掩码如何一步步消散。

## 输出产物

本章生成 `outputs/skill-unified-gen-model-picker.md`。给定一个需要同时具备理解（视觉问答、描述）和生成（文本生成图像、修复）能力且有开放权重约束的产品，在 Show-o 家族、Transfusion/MMDiT 家族和 Emu3/Chameleon 家族之间做出选择，并给出具体的权衡说明。

## 练习

1. 掩码离散扩散在约 16 步内完成采样。为什么不用 1 步？如果在第 0 步就解掩码所有内容会发生什么？

2. 修复是掩码扩散的免费能力。提出一个产品用例（真实或假设的），其中 Show-o 的修复能力优于专业模型。

3. 余弦调度 vs 线性调度：追踪 T=8 时每步解掩码的 token 数量。哪个更均衡？

4. 一张 512×512 的 Show-o 图像是 1024 个 token。在词汇量 K=16384 下，模型输出 1024 × log2(16384) = 14336 比特（约 1.75 KiB）的数据。Stable Diffusion 输出 512×512×24 比特 = 6,291,456 比特（约 768 KiB）的原始像素。压缩比是多少？这换来了什么质量？

5. 阅读 LlamaGen（arXiv:2406.06525）。LlamaGen 的类别条件自回归图像模型与 Show-o 的掩码方法有何不同？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 掩码离散扩散（Masked discrete diffusion） | "MaskGIT 风格" | 训练预测被掩码的 token；推理时迭代地解掩码置信度最高的预测。 |
| 余弦调度（Cosine schedule） | "解掩码调度" | 推理步骤中掩码比例的衰减方式；将置信度增长集中在中等比例范围。 |
| 并行解码（Parallel decoding） | "一次预测所有 token" | 每一步在一次前向传播中预测全部被掩码 token 序列，然后提交置信度最高的 K 个。 |
| 混合注意力（Hybrid attention） | "因果 + 双向" | 对文本 token 因果、对图像块内双向的注意力掩码。 |
| 图像修复（Inpainting） | "填充生成" | 以部分 token 被掩码的图像为条件，预测缺失部分；由训练目标免费提供。 |
| 提交率（Commitment rate） | "每步 Top-K" | 每次迭代声明"完成"的 token 数量；控制推理速度与质量之间的权衡。 |

## 延伸阅读

- [Xie 等 — Show-o（arXiv:2408.12528）](https://arxiv.org/abs/2408.12528)
- [Show-o2（arXiv:2506.15564）](https://arxiv.org/abs/2506.15564)
- [Chang 等 — MaskGIT（arXiv:2202.04200）](https://arxiv.org/abs/2202.04200)
- [Sun 等 — LlamaGen（arXiv:2406.06525）](https://arxiv.org/abs/2406.06525)
- [Chang 等 — Muse（arXiv:2301.00704）](https://arxiv.org/abs/2301.00704)

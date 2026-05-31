# Transfusion：自回归文本 + 扩散图像，共享一个 Transformer（Transfusion: Autoregressive Text + Diffusion Image in One Transformer）

> Chameleon 和 Emu3 将一切押注在离散 token 上。它们有效，但量化瓶颈清晰可见——图像质量停留在连续空间扩散模型之下。Transfusion（Meta，Zhou 等，2024 年 8 月）押了相反的注：保持图像连续，完全舍弃 VQ-VAE，用两个损失训练一个 Transformer。文本 token 获得下一个 token 预测；图像图块获得流匹配/扩散损失。两个目标优化同一套权重。Stable Diffusion 3 底层的架构（MMDiT）是它的近亲。本章解读 Transfusion 的论点，构建一个玩具双损失训练器，并追踪让一个 Transformer 同时做两件事的注意力掩码。

**类型：** 构建  
**语言：** Python（标准库，MNIST 规模玩具上的双损失训练器）  
**前置知识：** Phase 12 · 11（Chameleon）、Phase 8（生成式 AI）  
**预计时间：** 约 180 分钟

## 学习目标

- 搭建一个在同一骨干上同时运行两个损失（文本 token 的 NTP，图像图块的扩散 MSE）的 Transformer。
- 解释为什么图像图块上的双向注意力加文本 token 上的因果注意力是正确的掩码选择。
- 在算力、质量和代码复杂度三个维度对比 Transfusion 风格（连续图像、扩散损失）与 Chameleon 风格（离散图像、NTP）。
- 说出 MMDiT 的贡献：每个块中模态专属的权重，残差流上的联合注意力。

## 问题所在

离散 vs. 连续图像 token 的争论早于大语言模型。连续表示（原始像素、VAE 潜变量）保留细节。离散 token（VQ 索引）适配 Transformer 的原生词汇表，但在量化步骤中会损失细节。

Chameleon / Emu3 走离散路线：单一损失，单一架构，但图像保真度受限于分词器质量。

扩散模型走连续路线：卓越的图像质量，但与 LLM 是独立模型，噪声调度工程复杂，且与文本生成没有干净的集成。

Transfusion 的问题是：我们能两者兼得吗？保持图像连续，仍然训练一个模型，用两个损失拼在一个梯度步骤中。

## 核心概念

### 双损失架构

一个仅解码器的 Transformer 处理包含以下内容的序列：

- 文本 token（离散，来自 BPE 词汇表）。
- 图像图块（连续，16×16 像素块通过线性嵌入投影到隐藏维度——与 ViT 编码器的输入相同）。
- `<image>` 和 `</image>` 标签，标记连续图块所在的位置。

前向传播只运行一次。损失按 token 类型选择两个头之一：

- 对于文本 token：在词汇表 logit 头上的标准交叉熵。
- 对于图像图块：连续图块上的扩散损失——预测添加到每个图块上的噪声。

梯度流过共享的 Transformer 主干。两个损失同时改进共享权重。

### 注意力掩码：因果文本 + 双向图像

文本 token 必须是因果的——不能让文本 token 关注未来的文本，否则教师强制会失效。而图像图块代表一个快照，它们应该在同一图像块内相互双向关注。

掩码规则：

```
M[i, j] = 1，当：
  (i 是文本 且 j 是文本 且 j <= i)           # 文本因果
  或 (i 是图像 且 j 是图像 且 same_image_block(i, j))  # 图像内双向
  或 (i 是文本 且 j 是图像 且 j < i_image_end)    # 文本关注之前的图像
  或 (i 是图像 且 j 是文本 且 j < i_image_start)   # 图像关注之前的文本
```

在训练和推理时实现为块三角掩码。

### Transformer 内部的扩散损失

扩散损失是标准做法：向图像图块添加噪声，要求模型预测噪声（等价地，也可预测干净图块）。Transfusion 的版本使用流匹配——预测从噪声到干净图像的速度场。

训练过程：
1. 对每个图像图块 x0，采样一个随机时间步 t。
2. 采样噪声 ε，计算 xt = (1-t) × x0 + t × ε（流匹配的线性插值）。
3. Transformer 预测 v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。
4. 与同一序列中的文本 NTP 损失一起反向传播。

推理时，生成过程为：
- 文本 token：标准自回归采样。
- 图像图块：以前序文本 token 为条件的扩散采样循环（通常 10-30 步）。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser 等，2024 年 3 月）在 Transfusion 同期发布了 MMDiT（多模态扩散 Transformer）。两种架构是同胞。

MMDiT 的主要区别：

- **每个块中模态专属的权重。** 每个 Transformer 块对文本 token 和图像图块分别有独立的 Q、K、V 和 MLP 权重。注意力是联合的（跨模态）；其余是模态专属的。
- **整流流训练。** 一种特定的流匹配变体，采样已知，数学比 DDPM 更简单。
- **规模。** MMDiT 是 SD3（20 亿和 80 亿参数变体）的骨干。Transfusion 论文规模到 70 亿。

两者殊途同归：一个 Transformer 对文本运行 NTP，对连续图像表示运行扩散。

### 为何优于 Chameleon 风格

连续扩散与离散 NTP 在图像生成质量上的差距是可测量的。Transfusion 论文报告：

- 在 70 亿参数下，FID 比同等大小的 Chameleon 风格模型好 3-5 分。
- 无需训练分词器——图像编码器更简单（线性投影到隐藏维度，与 ViT 输入层相同）。
- 推理可以并行化图像图块去噪，而自回归图像 token 不行。

缺点：Transfusion 是双损失模型，训练动态更棘手。损失权重需要调整。NTP 和扩散之间的调度不匹配可能导致某一个头主导另一个。

### 下游影响

Janus-Pro（Lesson 12.15）通过为理解和生成解耦视觉编码器来完善 Transfusion 的思路——理解用 SigLIP，生成用 VQ——同时共享 Transformer 主干。Show-o（Lesson 12.14）将扩散换成离散扩散（掩码预测）。在 Transfusion 之后，统一生成家族迅速分支。

2026 年生产中能输出图像的 VLM——Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径——几乎可以肯定使用了这一家族的某个后代。细节是专有的。

## 动手使用

`code/main.py` 在一个类 MNIST 的小玩具问题上构建 Transfusion：

- 文本描述是描述数字（0-9）的短整数序列。
- 图像是 4×4 字节网格。
- 一对共享权重的线性投影充当 Transformer 的替代；文本上的 NTP 损失，带噪图块上的 MSE 损失。
- 训练循环交替进行两个损失，注意力掩码是显式构建的。
- 生成在一次前向传播中产生文本描述和 4×4 图像。

Transformer 是玩具。双损失管道、注意力掩码构建和推理循环才是真正的成果。

## 输出产物

本章生成 `outputs/skill-two-loss-trainer-designer.md`。给定一个新的多模态训练任务（文本 + 图像、文本 + 音频、文本 + 视频），它设计双损失方案（损失权重、掩码形状、共享 vs 模态专属块）并标记实施风险。

## 练习

1. 一个 Transfusion 风格的模型训练 70% 文本 token 和 30% 图像图块。图像扩散损失在量级上约为文本 NTP 损失的 10 倍。用什么损失权重能平衡它们？

2. 为以下序列实现块三角掩码：`[T, T, <image>, P, P, P, P, </image>, T]`。标出每个条目 0 或 1。

3. MMDiT 拥有模态专属的 QKV 权重。与 Transfusion 的完全共享 Transformer 相比，这增加了多少参数量开销？在 70 亿参数下，这值得吗？

4. 生成过程：给定文本提示词，模型运行 NTP 生成 50 个 token，然后遇到 `<image>`，随后在 256 个图块上进行 20 步去噪。总共需要多少次前向传播？

5. 阅读 SD3 论文第 3 节。描述整流流以及它为什么比 DDPM 在更少推理步骤内收敛。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 双损失训练（Two-loss training） | "NTP + 扩散" | 单个 Transformer 在同一梯度步骤中同时优化文本 token 的交叉熵和图像图块的 MSE。 |
| 流匹配（Flow matching） | "整流流" | 扩散变体，预测从噪声到干净数据的速度场；数学比 DDPM 更简单。 |
| MMDiT | "多模态 DiT" | Stable Diffusion 3 的架构：联合注意力，模态专属的 MLP 和归一化层。 |
| 块三角掩码（Block-triangular mask） | "因果文本 + 双向图像" | 文本间因果、图像区域内双向的注意力掩码。 |
| 连续图像表示（Continuous image representation） | "无 VQ" | 图像图块作为实值向量，而非整数码本索引。 |
| 速度预测（Velocity prediction） | "v 参数化" | 网络输出是噪声与数据之间的速度场，而非噪声本身。 |

## 延伸阅读

- [Zhou 等 — Transfusion（arXiv:2408.11039）](https://arxiv.org/abs/2408.11039)
- [Esser 等 — Stable Diffusion 3 / MMDiT（arXiv:2403.03206）](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT（arXiv:2212.09748）](https://arxiv.org/abs/2212.09748)
- [Zhao 等 — MonoFormer（arXiv:2409.16280）](https://arxiv.org/abs/2409.16280)
- [Xie 等 — Show-o（arXiv:2408.12528）](https://arxiv.org/abs/2408.12528)

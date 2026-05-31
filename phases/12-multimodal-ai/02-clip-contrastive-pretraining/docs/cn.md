# CLIP 与对比视觉-语言预训练（CLIP and Contrastive Vision-Language Pretraining）

> OpenAI 的 CLIP（2021）用一个足以支撑接下来五年的想法证明了自己：仅使用嘈杂的网络图文对和对比损失，将图像编码器与文本编码器对齐到同一个向量空间。零监督标签，4 亿图文对。所得到的嵌入空间可以做零样本分类、图文检索，并被插入每个 2026 年的 VLM 作为其视觉塔。SigLIP 2（2025）用 sigmoid 替换了 softmax，以更低成本超越了 CLIP。本章从 InfoNCE 到 sigmoid 逐对损失推导数学原理，并用标准库 Python 构建训练步骤。

**类型：** 构建  
**语言：** Python（标准库，InfoNCE + sigmoid 损失实现）  
**前置知识：** Phase 12 · 01（ViT 图块）、Phase 7（Transformer）  
**预计时间：** 约 180 分钟

## 学习目标

- 从互信息推导 InfoNCE 损失，并实现数值稳定的向量化版本。
- 解释为什么 sigmoid 逐对损失（SigLIP）可以扩展到 batch 32768+ 而无需 softmax 所需的 all-gather 开销。
- 通过构建文本模板（`a photo of a {class}`）并对余弦相似度取 argmax 来运行零样本 ImageNet 分类。
- 说出 CLIP / SigLIP 预训练提供的四个调节旋钮：批量大小、温度、提示词模板、数据质量。

## 问题所在

CLIP 之前的视觉是监督式的。收集标注数据集（ImageNet：120 万张图像，1000 个类别），训练 CNN，上线。标注成本高昂，标注者的共识偏见限制了覆盖范围，且标注无法在不微调的情况下迁移到新任务。

互联网上的图文对超过十亿，都带有松散的"标签"且免费。一张拉布拉多照片配上"我的狗 Max 在公园里"的 alt 文本，就携带了监督信号——文本描述了图像。问题是：你能把这个变成有用的训练吗？

CLIP 的答案：把图文对视为一个匹配任务。给定 N 张图像和 N 条文本，学会在 N-1 个干扰项中将每张图像与它自己的文本匹配。监督信号是"这两个东西属于同一组；这 N-1 个不属于"。没有类别标签，没有人工标注，只有一个对比损失。

所得到的嵌入空间能做的事超过了 CLIP 的训练目标。ImageNet 零样本之所以有效，是因为"a photo of a cat"的嵌入接近那些从未被显式标注为"猫"的猫的图片。这是催生了每个 2026 年 VLM 的那个赌注。

## 核心概念

### 双塔编码器

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，每张图像输出一个 D 维向量。
- 文本编码器 `g`：小型 Transformer，每条文本输出一个 D 维向量。

两个塔都对输出做单位长度归一化。由于都是单位范数，相似度为 `cos(f(x), g(y)) = f(x)^T g(y)`。

对于一批 N 个（图像、文本）对，构建形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是一个可学习的温度参数（CLIP 初始化为 0.07；在对数空间中学习）。

### InfoNCE 损失

CLIP 对行和列分别使用对称的交叉熵：

```
loss_i2t = CE(S, labels=identity)     # 每张图像的正样本是其对应文本
loss_t2i = CE(S^T, labels=identity)   # 每条文本的正样本是其对应图像
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 强制每张图像比批次中所有其他文本更匹配自己的文本。"负样本"就是批次中所有其他项。批量越大 = 负样本越多 = 信号越强。CLIP 以 batch 32k 训练；规模很重要。

### 温度

`tau` 控制 softmax 的锐度。低 tau → 分布尖锐，有硬负样本挖掘效果。高 tau → 平滑，所有样本都有贡献。CLIP 学习 `log(1/tau)`，并剪裁以防止崩溃。SigLIP 2 固定初始 tau，改用可学习偏置。

### 为什么 sigmoid 扩展性更好（SigLIP）

Softmax 需要整个相似度矩阵同步。在分布式训练中，必须在所有副本间做 all-gather，然后再做 softmax。通信开销随节点数平方增长。

SigLIP 用逐元素 sigmoid 替换 softmax：对每对 `(i, j)`，损失是一个二分类问题——"这对是否匹配？"正样本标签是对角线，其余都是负样本。损失为：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 当 `i == j`，否则为 0。每对的损失相互独立，无需 all-gather。每个 GPU 计算自己的本地块后求和。SigLIP 2 能以较低成本扩展到 batch 32k-512k，而 CLIP 则需要相应更多的通信。

### 零样本分类

给定 N 个类别名称，为每个类别构建文本模板：

```
"a photo of a {class}"
```

用文本编码器嵌入每个模板，用图像编码器嵌入图像，对余弦相似度取 argmax 即为预测类别。无需在目标类别上训练。

提示词模板很重要。CLIP 原论文每个类别使用 80 个模板（普通、艺术、照片、绘画等），并对嵌入取平均。这样在 ImageNet 上提升了 3 个百分点。现代用法通常选一两个模板即可。

### 线性探针与微调

零样本是基准。线性探针（在冻结的 CLIP 特征上训练一个线性层用于目标类别）在领域内任务上优于零样本。全量微调在领域内优于线性探针，但可能损害零样本迁移。三种规范，三种权衡。

### SigLIP 2：NaFlex 与密集特征

SigLIP 2（2025）新增：
- NaFlex：单个模型处理可变宽高比和分辨率，无需调整大小。
- 更好的密集特征，用于分割和深度估计，目标是作为 VLM 中的冻结骨干。
- 多语言：在 100+ 种语言上训练，而 CLIP 仅限英语。
- 10 亿参数规模，超越了 CLIP 的 4 亿参数上限。

在 2026 年的开放 VLM 中，SigLIP 2 SO400m/14 是默认视觉塔。CLIP 在纯图文检索场景中仍是默认选择，前提是 LAION-2B 的训练分布与你的查询模式相符。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 相同的思路，18 亿对数据规模，90% 是噪声数据，证明了噪声数据的可扩展性。OpenCLIP（LAION）：CLIP 在 LAION-400M / 2B 上的开放复现，提供多种规模，是开放检查点的首选。EVA-CLIP：从掩码图像建模初始化，是 VLM 的强力骨干。BASIC：Google 的 CLIP+ALIGN 混合方案。都属于同一家族，只是数据和调参不同。

### 零样本上限

CLIP 类模型的 ImageNet 零样本准确率上限约为 76%（CLIP-G、OpenCLIP-G）。突破需要更大数据（SigLIP 2 达到 80%+）或架构变化（监督头、更多参数）。这个基准正在饱和；真正的价值在于下游 VLM 所消费的嵌入空间。

## 动手使用

`code/main.py` 实现了：

1. 一个玩具双塔编码器（基于哈希的图像特征、字符文本特征），让你在不使用 numpy 的情况下看到 InfoNCE 的形状。
2. 纯 Python 的 InfoNCE 损失（通过 log-sum-exp 实现数值稳定性）。
3. sigmoid 逐对损失（作对比）。
4. 零样本分类流程：计算与一组文本提示的余弦相似度，取 argmax 作为预测。

运行它，观察损失曲线。绝对数值是玩具级的；形状与真实 CLIP 训练器输出一致。

## 输出产物

本章生成 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和目标类别列表，它用 CLIP 模板构建文本提示，用指定检查点（如 `openai/clip-vit-large-patch14`）嵌入两侧，并返回带有相似度分数的 top-1 / top-5 预测。该技能拒绝对不在提示词列表中的类别做任何声明。

## 练习

1. 手动为 4 对样本实现 InfoNCE。构建 4×4 相似度矩阵，运行 softmax，取出对角线，计算交叉熵。用你的 Python 实现验证这个手动计算。

2. SigLIP 除温度外还使用偏置参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当批次中类别严重不平衡（每行的负样本远多于正样本）时，`b` 起什么作用？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 构建一个猫狗零样本分类器。尝试两个提示词模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测量准确率。模板集成是否优于单个模板？

4. 计算在 512 GPU、batch 32k 的运行中，softmax InfoNCE 与 sigmoid 逐对的通信成本。哪个是 O(N)，哪个是 O(N²)？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 缩放定律论文（arXiv:2212.07143，Cherti 等）。从图中复现他们关于数据缩放的结论：在固定模型大小下，ImageNet 零样本准确率与训练数据量之间的对数线性关系是什么？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| InfoNCE | "对比损失" | 对批次相似度矩阵的交叉熵；每个样本的正样本是其配对样本，负样本是其余所有。 |
| Sigmoid 损失 | "SigLIP 损失" | 逐对二分类交叉熵；无 softmax，无 all-gather，分布式训练扩展成本低。 |
| 温度（Temperature） | "tau" | 在 softmax/sigmoid 之前缩放 logit 的标量；控制分布的锐度。 |
| 零样本（Zero-shot） | "无微调分类" | 用文本提示构建类别嵌入，通过余弦相似度分类；无需在目标类别上训练。 |
| 提示词模板（Prompt template） | "a photo of a ..." | 类别名称的文本脚手架；影响零样本准确率 1-5 个百分点。 |
| 双塔编码器（Dual encoder） | "双塔" | 一个图像编码器 + 一个文本编码器，输出在共享的 D 维空间中。 |
| 硬负样本（Hard negative） | "棘手的干扰项" | 与正样本足够相似，使模型必须努力区分的负样本。 |
| 线性探针（Linear probe） | "冻结 + 一层" | 在冻结特征上只训练一个线性分类器；衡量特征质量。 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 在任意宽高比和分辨率下处理图像的能力，无需调整大小。 |
| 温度缩放（Temperature scaling） | "对数参数化 tau" | CLIP 参数化 `log(1/tau)` 以使梯度行为良好；通过剪裁防止 tau 接近零时的崩溃。 |

## 延伸阅读

- [Radford 等 — Learning Transferable Visual Models From Natural Language Supervision（arXiv:2103.00020）](https://arxiv.org/abs/2103.00020) — CLIP 原论文。
- [Zhai 等 — Sigmoid Loss for Language Image Pre-Training（arXiv:2303.15343）](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen 等 — SigLIP 2（arXiv:2502.14786）](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。
- [Jia 等 — ALIGN（arXiv:2102.05918）](https://arxiv.org/abs/2102.05918) — 用嘈杂网络数据扩展规模。
- [Cherti 等 — Reproducible scaling laws for contrastive language-image learning（arXiv:2212.07143）](https://arxiv.org/abs/2212.07143) — OpenCLIP 缩放定律。

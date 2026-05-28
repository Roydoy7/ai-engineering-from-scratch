# 视觉 Transformer（ViT）

> 把图像切成 patch，把每个 patch 当作一个词，跑一个标准 Transformer。不再回头。

**类型：** 构建
**语言：** Python
**前置知识：** 第7阶段第2课（自注意力）、第4阶段第4课（图像分类）
**预计时间：** 约45分钟

## 学习目标

- 从零实现 patch 嵌入、可学习位置嵌入、class token 和 Transformer 编码器块，构建一个最小化 ViT
- 解释为何 ViT 曾被认为需要海量预训练数据，直到 DeiT 和 MAE 证明并非如此
- 在架构先验层面对比 ViT、Swin 和 ConvNeXt（分别是：无先验、局部窗口注意力、卷积骨干）
- 使用 `timm` 和标准线性探测 / 微调方案，在小型数据集上微调预训练 ViT

## 问题背景

十年间，卷积与计算机视觉几乎是同义词。CNN 拥有强大的归纳偏置——局部性、平移等变性——没人认为能被替代。然后 Dosovitskiy et al.（2020）证明，一个完全没有卷积机制、直接作用于图像 patch 展平序列的普通 Transformer，在大规模数据下可以媲美甚至超越最好的 CNN。

但前提是"大规模"。ViT 在 ImageNet-1k 上输给了 ResNet。先在 ImageNet-21k 或 JFT-300M 上预训练，再在 ImageNet-1k 上微调的 ViT 才能超越它。结论是：Transformer 缺乏有用的先验知识，但可以从足够多的数据中学到。后续工作（DeiT、MAE、DINO）证明，有了正确的训练方案——强数据增强、自监督预训练、蒸馏——ViT 在小数据上同样能训练好。

到 2026 年，纯 CNN 在边缘设备上仍有竞争力（ConvNeXt 最强），但 Transformer 主导了其他一切：分割（Mask2Former、SegFormer）、检测（DETR、RT-DETR）、多模态（CLIP、SigLIP）、视频（VideoMAE、VJEPA）。ViT 的块结构是必须掌握的。

## 核心概念

### 流水线全景

```mermaid
flowchart LR
    IMG["图像<br/>(3, 224, 224)"] --> PATCH["Patch 嵌入<br/>conv 16x16 步长=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["展平为<br/>(196, 768) token"]
    FLAT --> CAT["在开头加<br/>[CLS] token"]
    CAT --> POS["加上可学习<br/>位置嵌入"]
    POS --> ENC["N 个 Transformer<br/>编码器块"]
    ENC --> CLS["取 [CLS]<br/>token 输出"]
    CLS --> HEAD["MLP 分类器"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

七个步骤：patch → token → 注意力 → 分类器。每个变体（DeiT、Swin、ConvNeXt、MAE 预训练）都只改变其中一两步，其余保持不变。

### Patch 嵌入

第一个卷积是关键所在。卷积核大小为 16、步长为 16，因此一张 224×224 的图像变成了 14×14 的 16×16 patch 网格，每个 patch 投影为一个 768 维嵌入。这一个卷积同时完成了分块和线性投影两个操作。

```
输入:  (3, 224, 224)
Conv (3 -> 768, k=16, s=16, 无 padding):
输出: (768, 14, 14)
展平空间维度: (196, 768)
```

196 个 patch = 196 个 token。每个 token 的特征维度是 768（ViT-B）、1024（ViT-L）或 1280（ViT-H）。

### Class Token

在序列开头拼接一个单独的可学习向量：

```
tokens = [CLS; patch_1; patch_2; ...; patch_196]   shape (197, 768)
```

经过 N 个 Transformer 块后，`[CLS]` 的输出就是全局图像表示。分类头只读取这一个向量。

### 位置嵌入

Transformer 天生没有空间位置的概念。为每个 token 加上一个可学习向量：

```
tokens = tokens + learned_pos_embedding   (形状同为 (197, 768))
```

该嵌入是模型的参数；基于梯度的训练会将其调整以适应图像的 2D 结构。二维正弦波替代方案存在，但实践中很少使用。

### Transformer 编码器块

标准结构：多头自注意力、MLP、残差连接、前置 LayerNorm（Pre-LN）。

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP 为两层，使用 GELU：Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

ViT-B/16 堆叠了 12 个这样的块，每个块有 12 个注意力头，共 8600 万参数。

### 为什么用 Pre-LN

早期 Transformer 使用 Post-LN（`x = LN(x + sublayer(x))`），超过 6-8 层后训练困难，需要精心调整预热策略。Pre-LN（`x = x + sublayer(LN(x))`）无需预热即可稳定训练更深的网络。所有 ViT 和现代 LLM 都使用 Pre-LN。

### Patch 大小的权衡

- 16×16 patch → 196 个 token，标准配置。
- 32×32 patch → 49 个 token，更快但分辨率更低。
- 8×8 patch → 784 个 token，更精细但 O(n²) 注意力代价急剧增加。

Patch 越大 = token 越少 = 越快，但空间细节越少。SwinV2 在层次化窗口中使用 4×4 patch。

### DeiT 在 ImageNet-1k 上训练 ViT 的方案

原始 ViT 需要 JFT-300M 才能超越 CNN。DeiT（Touvron et al., 2020）仅用 ImageNet-1k 就将 ViT-B 训练到 81.8% top-1，做了四处改动：

1. 强数据增强：RandAugment、Mixup、CutMix、Random Erasing。
2. 随机深度（Stochastic depth）：训练时随机丢弃整个 block。
3. 重复增强（Repeated augmentation）：同一图像在每个 batch 中被采样 3 次。
4. 从 CNN 教师蒸馏（可选，能进一步提升精度）。

所有现代 ViT 训练方案都源于 DeiT。

### Swin vs ConvNeXt

- **Swin**（Liu et al., 2021）— 基于窗口的注意力。每个 block 在局部窗口内做注意力；交替的 block 会平移窗口以跨窗口混合信息。引入了类似 CNN 的局部性先验，同时保留注意力算子。
- **ConvNeXt**（Liu et al., 2022）— 重新设计的 CNN，采纳了 Swin 的架构选择（深度可分离卷积、LayerNorm、GELU、倒置瓶颈）。证明差距不在于"注意力 vs 卷积"，而在于"现代训练方案 + 架构设计"。

2026 年，ConvNeXt-V2 和 Swin-V2 都是生产级选项；正确选择取决于推理栈（ConvNeXt 在边缘设备上编译效果更好）和预训练语料库。

### MAE 预训练

掩码自编码器（He et al., 2022）：随机遮住 75% 的 patch，训练编码器只处理可见的 25%，训练一个小型解码器从编码器输出重建被遮住的 patch。预训练结束后，丢弃解码器，对编码器进行微调。

MAE 使 ViT 可以只用 ImageNet-1k 进行训练，达到 SOTA 效果，是目前默认的自监督预训练方案。

## 动手实现

### 第一步：Patch 嵌入

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

一个卷积，一次展平，一次转置。这就是图像到 token 的全部步骤。

### 第二步：Transformer 块

Pre-LN、多头自注意力、带 GELU 的 MLP、残差连接。

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention` 负责切分注意力头、缩放点积运算和输出投影。`batch_first=True` 使形状为 `(N, seq, dim)`。

### 第三步：ViT 整体

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"output: {vit(x).shape}")
print(f"params: {sum(p.numel() for p in vit.parameters()):,}")
```

约 280 万参数——一个在 CPU 上也可运行的微型 ViT。真实的 ViT-B 有 8600 万参数；使用相同的类定义，只需将 `dim=768, depth=12, num_heads=12`。

### 第四步：健全性检查——单张图像推理

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"probs:  {logits.softmax(-1)}")
```

应无报错地运行，且概率加和为 1。

## 工程应用

`timm` 提供所有 ViT 变体及 ImageNet 预训练权重，一行即可加载：

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

`timm` 是 2026 年视觉 Transformer 的生产默认选择。同一 API 下支持 ViT、DeiT、Swin、Swin-V2、ConvNeXt、ConvNeXt-V2、MaxViT、MViT、EfficientFormer 等数十种模型。

多模态工作（图像 + 文本）使用 `transformers`，其中的 CLIP、SigLIP、BLIP-2、LLaVA 的图像编码器都是 ViT 的变体。

## 交付物

本课产出：

- `outputs/prompt-vit-vs-cnn-picker.md` — 一个提示词，根据数据集大小、计算资源和推理栈，在 ViT、ConvNeXt 和 Swin 之间做出选择。
- `outputs/skill-vit-patch-and-pos-embed-inspector.md` — 一个技能文件，验证 ViT 的 patch 嵌入和位置嵌入形状是否与模型预期的序列长度匹配，捕获最常见的移植错误。

## 练习

1. **(简单)** 打印上面微型 ViT 前向传播的每个中间 tensor 的形状。确认：输入 `(N, 3, 64, 64)` → patch `(N, 16, 192)` → 加入 CLS 后 `(N, 17, 192)` → 分类器输入 `(N, 192)` → 输出 `(N, num_classes)`。
2. **(中等)** 在第4课的合成-CIFAR 数据集上微调一个预训练的 `timm` ViT-S/16。与在相同数据上微调的 ResNet-18 对比，报告训练时间和最终精度。
3. **(困难)** 为微型 ViT 实现 MAE 预训练：遮住 75% 的 patch，训练编码器 + 一个小型解码器来重建被遮住的 patch。比较预训练前后在合成数据上的线性探测精度。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| Patch 嵌入 (Patch embedding) | "第一个卷积" | 卷积核大小 = 步长 = patch 大小的卷积；将图像转化为 token 嵌入网格 |
| Class token | "[CLS]" | 在 token 序列开头拼接的可学习向量；其最终输出是全局图像表示 |
| 位置嵌入 (Positional embedding) | "可学习位置" | 加到每个 token 上的可学习向量，让 Transformer 知道每个 patch 来自何处 |
| Pre-LN | "先做 LayerNorm" | 稳定的 Transformer 变体：`x + sublayer(LN(x))`，而非 `LN(x + sublayer(x))` |
| 多头注意力 (Multi-head attention) | "并行注意力" | 将标准 Transformer 注意力拆分为 num_heads 个独立子空间，之后拼接 |
| ViT-B/16 | "Base，patch 16" | 标准尺寸：dim=768，depth=12，heads=12，patch_size=16，image=224；约 8600 万参数 |
| DeiT | "数据高效 ViT" | 仅用 ImageNet-1k 训练的 ViT，配合强数据增强；证明不需要大规模预训练数据集 |
| MAE | "掩码自编码器" | 自监督预训练：遮住 75% 的 patch 再重建；ViT 主流预训练方案 |

## 延伸阅读

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) — ViT 原论文
- [DeiT: Data-efficient Image Transformers (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877) — 如何仅用 ImageNet-1k 训练 ViT
- [Masked Autoencoders are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377) — MAE 预训练
- [timm 文档](https://huggingface.co/docs/timm) — 生产中所有视觉 Transformer 的参考手册

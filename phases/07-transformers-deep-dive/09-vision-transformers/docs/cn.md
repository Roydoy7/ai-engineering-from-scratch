# 视觉 Transformer（ViT）

> 图像是一个图块网格。句子是一个 token 网格。同一个 Transformer 都能处理。

**类型：** 构建
**语言：** Python
**前置知识：** 第7阶段第05课（完整 Transformer）、第4阶段第03课（CNN）、第4阶段第14课（视觉 Transformer 入门）
**预计时间：** 约45分钟

## 问题背景

2020 年之前，计算机视觉意味着卷积。ImageNet、COCO 和检测基准上的所有最先进模型都使用 CNN 骨架。Transformer 是用于语言的。

Dosovitskiy et al.（2020）——"一张图像值 16×16 个词"——证明了你可以完全抛弃卷积。将图像切成固定大小的图块，将每个图块线性投影为嵌入，把序列送入普通的 Transformer 编码器。在足够大的规模下（ImageNet-21k 预训练或更大），ViT 与基于 ResNet 的模型持平甚至更优。

ViT 是 2026 年一种更广泛模式的开端：一种架构，多种模态。Whisper 对音频做 token 化。ViT 对图像做 token 化。机器人的动作 token。视频的像素 token。Transformer 不在乎——给它一个序列，它就能学习。

到 2026 年，ViT 及其后代（DeiT、Swin、DINOv2、ViT-22B、SAM 3）主导了大部分视觉任务。CNN 在边缘设备和延迟敏感任务上仍然胜出。其他所有场景的堆栈中都有 ViT 的身影。

## 核心概念

### 第一步——图块化

将 `H × W × C` 的图像分割成由 `N` 个扁平图块组成的 `N × (P·P·C)` 序列。典型配置：`224 × 224` 图像，`16 × 16` 图块 → 196 个图块，每个有 768 个值。

```
图像 (224, 224, 3) → 14 × 14 个 16x16x3 图块网格 → 196 个长度为 768 的向量
```

图块大小是调节旋钮。图块越小 = token 越多、分辨率越高、注意力计算量越大。图块越大 = 越粗糙、越便宜。

### 第二步——线性嵌入

一个学习到的矩阵将每个扁平图块投影到 `d_model`。等价于核大小为 `P`、步长为 `P` 的卷积。在 PyTorch 中就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`——两行实现。

### 第三步——前置 `[CLS]` token，添加位置嵌入

- 前置一个可学习的 `[CLS]` token。其最终隐状态是用于分类的图像表示。
- 添加可学习位置嵌入（原始 ViT）或正弦 2D 嵌入（后来的变体）。
- 2024 年后 RoPE 扩展到 2D 位置，有时不需要显式嵌入。

### 第四步——标准 Transformer 编码器

堆叠 L 个 `LayerNorm → 自注意力 → + → LayerNorm → MLP → +` 块。与 BERT 完全相同。没有视觉专用层。这是论文的核心观点。

### 第五步——头部

分类：取 `[CLS]` 隐状态 → 线性层 → softmax。对于 DINOv2 或 SAM，舍弃 `[CLS]`，直接使用图块嵌入。

### 重要的变体

| 模型 | 年份 | 变化 |
|------|------|------|
| ViT | 2020 | 原始版本。固定图块大小，完整全局注意力 |
| DeiT | 2021 | 蒸馏；仅在 ImageNet-1k 上可训练 |
| Swin | 2021 | 带移动窗口的分层结构。固定了亚二次方代价 |
| DINOv2 | 2023 | 自监督（无标签）。最佳通用视觉特征 |
| ViT-22B | 2023 | 220亿参数；缩放定律适用 |
| SigLIP | 2023 | ViT + 语言配对，sigmoid 对比损失 |
| SAM 3 | 2025 | 分割一切；ViT-Large + 可提示的掩码解码器 |

### 为什么花了一段时间才成功

ViT 需要*大量*数据才能与 CNN 持平，因为它没有 CNN 的归纳偏置（平移不变性、局部性）。没有 1 亿以上的标注图像或强自监督预训练，CNN 在相同计算量下仍然胜出。DeiT 在 2021 年用蒸馏技巧解决了这个问题；DINOv2 在 2023 年用自监督永久性地解决了这个问题。

## 动手实现

见 `code/main.py`。纯标准库的图块化 + 线性嵌入 + 正确性检查。不需要训练——任何实际规模的 ViT 都需要 PyTorch 和数小时 GPU 时间。

### 第一步：假图像

将一个 24 × 24 的 RGB 图像表示为 `(R, G, B)` 元组的行列表。使用 6×6 图块 → 16 个图块，每个嵌入向量维度 108。

### 第二步：图块化

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

光栅顺序：在网格上按行优先排列。每个 ViT 都使用这种顺序。

### 第三步：线性嵌入

将每个扁平图块乘以随机的 `(图块扁平尺寸, d_model)` 矩阵。前置 `[CLS]` 后验证输出形状为 `(图块数 + 1, d_model)`。

### 第四步：统计实际 ViT 的参数量

打印 ViT-Base 的参数量：12 层，12 头，d=768，图块大小=16。与 ResNet-50（约 2500 万）对比。ViT-Base 约 8600 万，ViT-Large 约 3.07 亿，ViT-Huge 约 6.32 亿。

## 工程应用

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768)：[CLS] + 196 个图块
cls_emb = out[:, 0]                       # 图像表示
```

**DINOv2 嵌入是 2026 年图像特征的默认方案。** 冻结骨架，训练一个小型头部。适用于分类、检索、检测、字幕生成。Meta 的 DINOv2 检查点在所有非文字视觉任务上优于 CLIP。

**图块大小选择。** 小型模型用 16×16（ViT-B/16）。密集预测（分割）用 8×8 或 14×14（SAM、DINOv2）。超大型模型用 14×14。

## 交付物

见 `outputs/skill-vit-configurator.md`。该技能根据数据集大小、分辨率和计算预算，为新视觉任务选择 ViT 变体和图块大小。

## 练习

1. **（简单）** 运行 `code/main.py`。验证图块数等于 `(H/P) * (W/P)`，扁平图块维度等于 `P*P*C`。
2. **（中等）** 实现 2D 正弦位置嵌入——对每个图块的 `行` 和 `列` 分别用独立的正弦编码，然后拼接。将其输入一个微型 PyTorch ViT，在 CIFAR-10 上与可学习位置嵌入对比准确率。
3. **（困难）** 构建一个 3 层 ViT（PyTorch），用 4×4 图块在 1000 张 MNIST 图像上训练。测量测试准确率。再在同样的 1000 张图像上加入 DINOv2 预训练（简化版：只训练编码器预测被掩码图块的嵌入）。准确率是否提升？

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 图块 (Patch) | "视觉 Transformer 的 token" | 图像中 `P × P × C` 区域的扁平像素值向量 |
| 图块化 (Patchify) | "切割 + 展平" | 将图像切成不重叠的图块，每个展平为向量 |
| `[CLS]` token | "图像摘要" | 前置的可学习 token；其最终嵌入是图像表示 |
| 归纳偏置 (Inductive bias) | "模型的假设" | ViT 比 CNN 的先验更少；需要更多数据来弥补 |
| DINOv2 | "自监督 ViT" | 用图像增广 + 动量教师在无标签情况下训练；2026 年最佳通用图像特征 |
| SigLIP | "CLIP 的继任者" | 用 sigmoid 对比损失训练的 ViT + 文字编码器；在相同计算量下优于 CLIP |
| Swin | "窗口化 ViT" | 带局部注意力 + 移动窗口的分层 ViT；亚二次方计算 |
| 寄存器 token (Register tokens) | "2023 年技巧" | 几个额外的可学习 token，吸收注意力汇聚现象；改善 DINOv2 特征 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 论文
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2 的寄存器 token 修复

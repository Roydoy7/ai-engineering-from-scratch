# 条件 GAN 与 Pix2Pix

> 2014—2017 年最大的突破是控制 GAN 的生成内容。附上一个标签、一张图像或一句话。Pix2Pix 做了图像版本，在窄范围的图像到图像任务上，它至今仍优于所有通用文生图模型。

**类型：** 构建
**语言：** Python
**前置知识：** 第8阶段第03课（GAN）、第4阶段第06课（U-Net）、第3阶段第07课（CNN）
**预计时间：** 约75分钟

## 问题背景

无条件 GAN 生成任意人脸。演示有用，生产无用。你想要的是：*把草图映射成照片*，*把地图映射成航拍照片*，*把白天场景映射成夜晚*，*给灰度图上色*。在所有这些任务中，你得到一张输入图像 `x`，必须输出具有某种语义对应关系的 `y`。每个 `x` 对应多个合理的 `y`。均方误差会把它们抹平成一团模糊。对抗损失不会，因为"看起来真实"是清晰的。

条件 GAN（Mirza & Osindero，2014）向 `G` 和 `D` 都添加了条件 `c` 作为输入。Pix2Pix（Isola et al.，2017）对此进行了专门化：条件是完整的输入图像，生成器是 U-Net，判别器是*基于 patch 的*分类器（PatchGAN），损失是对抗损失 + L1。这个方案在窄范围图像到图像领域，哪怕在 2026 年都优于从零开始训练的文生图模型，因为它训练在**配对数据**上——你拥有恰好需要的信号。

## 核心概念

![Pix2Pix：U-Net 生成器，PatchGAN 判别器](../assets/pix2pix.svg)

**条件生成器 G。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 dropout（没有输入噪声——Isola 发现显式噪声会被忽略）。

**条件判别器 D。** `D(x, y) → [0, 1]`。输入是**配对**（条件，输出）。这是关键区别：D 必须判断 `y` 是否与 `x` 一致，而不仅仅是 `y` 是否看起来真实。

**U-Net 生成器。** 跨瓶颈有跳跃连接的编码器-解码器。对于输入和输出共享低级结构（边缘、轮廓）的任务至关重要。没有跳跃连接，高频细节就会消失。

**PatchGAN 判别器。** D 不输出单一的真/假分数，而是输出一个 `N×N` 网格，每个格子判断约 70×70 像素的感受野。取平均。这是马尔可夫随机场假设：真实感是局部的。训练更快，参数更少，输出更清晰。

**损失。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项稳定训练，并将 G 推向已知目标。L1 比 L2 产生更锐利的边缘（中位数，而不是均值）。Pix2Pix 的默认 `λ = 100`。

## CycleGAN——当你没有配对数据时

Pix2Pix 需要配对的 `(x, y)` 数据。CycleGAN（Zhu et al.，2017）以额外的损失为代价放弃了这一要求：*循环一致性损失*。两个生成器 `G: X → Y` 和 `F: Y → X`。训练它们使得 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y`。这让你可以在没有配对样本的情况下将马变成斑马、夏天变成冬天。

2026 年，无配对的图像到图像翻译大多通过扩散（ControlNet、IP-Adapter）完成，而不是 CycleGAN，但循环一致性思想几乎在每篇无配对域适应论文中都留存了下来。

## 动手实现

`code/main.py` 在一维数据上实现了一个小型条件 GAN。条件 `c` 是类别标签（0 或 1）。任务：为给定类别从条件分布中产生样本。

### 第一步：将条件附加到 G 和 D 的输入

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

one-hot 编码是最简单的方式。更大的模型使用学习型嵌入、FiLM 调制或交叉注意力。

### 第二步：条件训练

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配**给定条件下**的真实分布，而不是边缘分布。

### 第三步：验证每类输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 常见陷阱

- **条件被忽略。** G 学会了边缘化，D 不惩罚因为条件信号太弱。修复：更积极地给 D 加条件（在早期层，而不仅仅是晚期层），使用投影判别器（Miyato & Koyama 2018）。
- **L1 权重太低。** G 会漂移到任意看起来真实的输出，而不是忠实的输出。对于 Pix2Pix 风格的任务，从 λ≈100 开始。
- **L1 权重太高。** G 产生模糊输出，因为 L1 仍然是一种 Lp 范数。训练稳定后退火降低。
- **D 中真实标签泄露。** 将 `(x, y)` 连接作为 D 输入，而不仅仅是 `y`。没有这个，D 无法检查一致性。
- **每类模式坍塌。** 每个类别可以独立坍塌。运行类别条件的多样性检查。

## 工程应用

2026 年图像到图像任务的现状：

| 任务 | 最佳方案 |
|------|---------|
| 草图 → 照片，相同领域，有配对数据 | Pix2Pix / Pix2PixHD（仍然快速、仍然清晰） |
| 草图 → 照片，无配对数据 | 使用涂鸦条件模型的 ControlNet |
| 语义分割 → 照片 | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| 风格迁移 | 带 IP-Adapter 或 LoRA 的扩散；GAN 方法已是遗产 |
| 深度 → 照片 | 在 Stable Diffusion 上的 ControlNet-Depth |
| 超分辨率 | Real-ESRGAN（GAN）、ESRGAN-Plus 或 SD-Upscale（扩散） |
| 图像上色 | ColTran、基于扩散的上色器或 Pix2Pix-color |
| 白天 → 夜晚、季节、天气 | CycleGAN 或基于 ControlNet 的方案 |

当（a）有数千个配对样本，（b）任务窄且可重复，（c）需要快速推理时，Pix2Pix 仍然是正确的工具。在通用开放域任务上，扩散胜出。

## 交付物

见 `outputs/skill-img2img-chooser.md`。该技能接受任务描述、数据可用性（配对 vs 无配对，N 个样本）和延迟/质量预算，输出：方案（Pix2Pix、CycleGAN、ControlNet 变体、SDXL + IP-Adapter）、训练数据要求、推理成本和评估协议（LPIPS、FID、任务特定指标）。

## 练习

1. **（简单）** 修改 `code/main.py`，添加第三个类别。确认 G 仍将每个类别的噪声映射到正确的模式。
2. **（中等）** 在一维设置中用感知风格损失替换 L1（例如用一个小的冻结 D 作为特征提取器）。这会改变条件分布的清晰度吗？
3. **（困难）** 在一维设置中勾勒出一个 CycleGAN：两个分布，两个生成器，循环损失。证明它能在没有配对数据的情况下学会在两者之间映射。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 条件 GAN (Conditional GAN) | "带标签的 GAN" | G(z, c)，D(x, c)。两个网络都能看到条件。 |
| Pix2Pix | "图像到图像 GAN" | 带 U-Net G 和 PatchGAN D + L1 损失的配对 cGAN。 |
| U-Net | "带跳跃连接的编码器-解码器" | 对称卷积网络；跳跃连接保留高频信息。 |
| PatchGAN | "局部真实性分类器" | D 输出逐 patch 分数而不是全局分数。 |
| CycleGAN | "无配对图像翻译" | 两个 G + 循环一致性损失；无需配对数据。 |
| SPADE | "GauGAN" | 用语义图归一化中间激活；语义分割到图像。 |
| FiLM | "特征级线性调制" | 从条件进行逐特征仿射变换；廉价的条件化方式。 |

## 生产说明：Pix2Pix 作为延迟约束基线

当你有配对数据和窄任务（草图 → 渲染、语义图 → 照片、白天 → 夜晚）时，Pix2Pix 的单次推理在延迟上比扩散快一个数量级。生产比较通常是：

| 方案 | 步数 | 单个 L4 上 512² 的典型延迟 |
|------|------|--------------------------|
| Pix2Pix（U-Net 前向传播） | 1 | ~30 ms |
| SD-Inpaint 或 SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1—4 | ~0.15—0.35 s |
| ControlNet + SDXL base | 20—30 | ~3—5 s |

Pix2Pix 在静态批次吞吐量上胜出（每个请求消耗相同的 FLOP）。扩散在质量和泛化上胜出。现代做法通常是为窄任务部署一个 Pix2Pix 风格的蒸馏模型，并为长尾输入保留一个扩散备选。

## 延伸阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGAN 论文
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — 投影判别器

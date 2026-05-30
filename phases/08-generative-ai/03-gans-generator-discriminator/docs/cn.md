# GAN——生成器与判别器

> Goodfellow 在 2014 年的技巧是完全绕过密度估计。两个网络。一个制造假货。一个抓假货。它们对抗，直到假货与真品无从分辨。这按理说不该有效。它经常确实无效。但当它奏效时，在窄领域上产出的样本至今仍是文献中最清晰的。

**类型：** 构建
**语言：** Python
**前置知识：** 第3阶段第02课（反向传播）、第3阶段第08课（优化器）、第8阶段第02课（VAE）
**预计时间：** 约75分钟

## 问题背景

VAE 产生模糊样本，因为其 MSE 解码器损失对*均值*图像是贝叶斯最优的——而许多合理数字的均值是一个模糊数字。你想要的损失要奖励*合理性*，而不是对任何一个目标的像素级接近程度。合理性没有封闭形式，必须学习。

Goodfellow 的想法：训练一个分类器 `D(x)` 来区分真实图像和假图像。训练一个生成器 `G(z)` 来欺骗 `D`。`G` 的损失信号是 `D` 当前认为什么让某样东西看起来真实的信号。随着 `G` 改进，这个信号也在更新，追着一个移动的目标跑。如果两个网络都收敛，`G` 就学到了数据分布，而从未写下 `log p(x)`。

这就是对抗训练。数学上是一个极小极大博弈：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

2026 年 GAN 不再是 SOTA 生成器（扩散和流匹配夺走了那顶王冠）。但 StyleGAN 2/3 仍是有史以来最清晰的人脸模型，GAN 判别器被用作扩散训练中的*感知损失*，对抗训练也驱动着快速单步蒸馏方案（SDXL-Turbo、SD3-Turbo、LCM），让实时扩散成为可能。

## 核心概念

![GAN 训练：生成器与判别器的极小极大博弈](../assets/gan.svg)

**生成器 `G(z)`。** 将噪声向量 `z ~ N(0, I)` 映射到样本 `x̂`。形状类似解码器的网络（全连接层或转置卷积）。

**判别器 `D(x)`。** 将样本映射到一个标量概率（或分数）。真实 → 1，假 → 0。

**损失。** 两个交替更新：

- **训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。对真实=1、假=0 进行二元交叉熵。
- **训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的*非饱和*形式（原始的 `log(1 - D(G(z)))` 在 `D` 有把握时会饱和，杀死梯度）。

**训练循环。** 一步 `D`，一步 `G`。重复。

**为何有效。** 如果 `G` 完美匹配 `p_data`，那么 `D` 不能做得比随机猜测更好，处处输出 0.5；`G` 不再获得梯度。均衡达到。

**为何失效。** 模式坍塌（`G` 找到一个 `D` 无法分类的模式，然后永远产出它）、梯度消失（`D` 学得太快，`log D` 饱和）、训练不稳定（学习率、批大小，凡此种种）。

## 让 GAN 奏效的变体

| 年份 | 创新 | 解决的问题 |
|------|------|---------|
| 2015 | DCGAN | 卷积/转置卷积、批归一化、LeakyReLU——第一个稳定架构。 |
| 2017 | WGAN、WGAN-GP | 用 Wasserstein 距离 + 梯度惩罚替换 BCE。修复梯度消失。 |
| 2017 | 谱归一化 (Spectral normalization) | Lipschitz 约束判别器。2026 年判别器仍在使用。 |
| 2018 | Progressive GAN | 先训练低分辨率，再添加层。首次实现百万像素效果。 |
| 2019 | StyleGAN / StyleGAN2 | 映射网络 + 自适应实例归一化。固定领域照片级真实感的最优水平。 |
| 2021 | StyleGAN3 | 无混叠、平移等变——2026 年人脸生成的黄金标准。 |
| 2022 | StyleGAN-XL | 条件化、类别感知、更大规模。 |
| 2024 | R3GAN | 以更强正则化重构；无需技巧即可处理 1024²。 |

## 动手实现

`code/main.py` 在一维数据上训练一个小型 GAN：两个高斯的混合分布。生成器和判别器都是单隐层 MLP。我们手动实现前向传播、反向传播和极小极大循环。目标是在发生时亲眼观察两种关键失效模式（模式坍塌 + 梯度消失）。

### 第一步：非饱和损失

朴素的 Goodfellow 损失 `log(1 - D(G(z)))` 在 D 高置信度地将 G 的假货分类为假时趋向 0。此时 G 的梯度基本为零——G 无法改进。非饱和形式 `-log D(G(z))` 具有相反的渐近行为：当 D 有把握时它会爆炸，给 G 一个强信号。

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 第二步：每步生成器更新对应一步判别器更新

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

G 的更新用新生成的假货，否则梯度是过时的。

### 第三步：监测模式坍塌

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

典型症状：两个真实模式中的一个停止被生成。判别器不再纠正它，因为它从未被当作假货看到。

## 常见陷阱

- **判别器太强。** 将 D 的学习率降低 2—5 倍，或添加实例/层噪声。如果 D 准确率超过 95%，G 就死了。
- **生成器记住了一个模式。** 向 D 的输入添加噪声、使用 minibatch 判别层，或切换到 WGAN-GP。
- **批归一化泄露统计量。** 真实批次和假批次流经同一 BN 层会混合它们的统计量。改用实例归一化或谱归一化。
- **Inception Score 刷分。** FID 和 IS 在样本量少时噪声很大。评估时至少使用 1 万个样本。
- **条件任务的单次采样是谎言。** 你仍然需要 CFG 尺度、截断技巧和重采样才能得到可用的输出。

## 工程应用

2026 年 GAN 的选型：

| 场景 | 选择 |
|------|------|
| 照片级真实感人脸，固定姿态 | StyleGAN3（最清晰、最小） |
| 动漫 / 风格化人脸 | StyleGAN-XL 或 Stable Diffusion LoRA |
| 图像到图像转换 | Pix2Pix / CycleGAN（第8阶段第04课）或 ControlNet（第8阶段第08课） |
| 快速单步文生图 | 扩散模型的对抗蒸馏（SDXL-Turbo、SD3-Turbo） |
| 扩散训练器内的感知损失 | 在图像裁剪上使用小型 GAN 判别器 |
| 任何多模态、开放式任务 | 不要用 GAN——使用扩散或流匹配 |

GAN 清晰但范围窄。一旦你的领域开放——照片、任意文本提示、视频——就切换到扩散。对抗技巧以组件形式延续（感知损失、蒸馏），而不是作为独立生成器。

## 交付物

见 `outputs/skill-gan-debugger.md`。该技能接受一个失败的 GAN 训练运行（损失曲线、样本网格、数据集大小），输出可能原因的排名列表、一行修复方案和重新训练协议。

## 练习

1. **（简单）** 用默认设置运行 `code/main.py`。然后设置 `D_LR = 5 * G_LR` 并重新运行。G 的损失多快坍塌到一个常数？
2. **（中等）** 用 WGAN 损失替换 Goodfellow BCE 损失：`loss_D = E[D(fake)] - E[D(real)]`，`loss_G = -E[D(fake)]`，并将 D 的权重裁剪到 `[-0.01, 0.01]`。训练是否更稳定？比较收敛的挂钟时间。
3. **（困难）** 将一维示例扩展到二维数据（环上的 8 个高斯混合）。在第 1k、5k、10k 步记录生成器捕获了多少个模式。实现 minibatch 判别并重新测量。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 生成器 (Generator) | "G" | 噪声到样本的网络，`G: z → x̂`。 |
| 判别器 (Discriminator) | "D" | 分类器 `D: x → [0, 1]`，区分真假。 |
| 极小极大 (Minimax) | "博弈" | 联合目标的 `min_G max_D`。 |
| 非饱和损失 (Non-saturating loss) | "修复方案" | G 使用 `-log D(G(z))` 而不是 `log(1 - D(G(z)))`。 |
| 模式坍塌 (Mode collapse) | "G 只记住了一件事" | 尽管数据多样，生成器只产出少数几种不同输出。 |
| WGAN | "Wasserstein" | 用地球移动距离 + 梯度惩罚替换 BCE；梯度更平滑。 |
| 谱归一化 (Spectral norm) | "Lipschitz 技巧" | 约束 D 的权重范数以限制其斜率；稳定训练。 |
| StyleGAN | "那个有效的" | 映射网络 + AdaIN；人脸生成的最优水平，2026 年仍然如此。 |

## 生产说明：单次推理是 GAN 持久的优势

GAN 在开放域生成的样本质量上不再胜出，但在推理成本上仍然占优。用生产推理文献的术语来说，GAN 具有：

- **没有预填充，没有解码阶段。** 一次 `G(z)` 前向传播。首 token 时间 ≈ 总延迟。
- **没有 KV 缓存压力。** 唯一的状态是权重。批大小受激活内存限制，而不是缓存。
- **极简的连续批处理。** 由于每个请求消耗固定的 FLOP，在服务器目标占用率下使用静态批通常是最优的。不需要动态调度器。

这就是为什么 GAN 蒸馏（SDXL-Turbo、SD3-Turbo、ADD、LCM）是 2026 年快速文生图的主流技术：它将 20—50 步扩散流水线压缩成 1—4 次 GAN 风格的前向传播，同时保留扩散基础模型的分布。对抗损失作为训练时将慢速生成器变成快速生成器的旋钮，延续了下来。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — GAN 原始论文
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — 第一个稳定架构
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — 谱归一化
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo

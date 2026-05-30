# 流匹配与整流流

> 扩散模型需要 20-50 个采样步骤，因为它们走的是从噪声到数据的弯曲路径。流匹配（Lipman et al.，2023）和整流流（Liu et al.，2022）训练了直线路径。路径更直意味着步骤更少，意味着推理更快。Stable Diffusion 3、Flux.1 和 AudioCraft 2 都在 2024 年切换到了流匹配。

**类型：** 构建
**语言：** Python
**前置知识：** 第8阶段第06课（DDPM）、第1阶段微积分
**预计时间：** 约45分钟

## 问题背景

DDPM 的反向过程是从 `N(0, I)` 回到数据分布的 1000 步随机游走。DDIM 将其压缩到 20-50 步确定性步骤。你想要更少的步骤——理想情况下是一步。障碍在于求解反向过程的 ODE 是刚性的；路径是弯曲的。

如果你能训练模型使得从噪声到数据的路径是**直线**，从 `t=1` 到 `t=0` 的单步 Euler 步就能工作。流匹配直接构建了这一点：定义从 `x_1 ∼ N(0, I)` 到 `x_0 ∼ 数据` 的直线插值，训练一个向量场 `v_θ(x, t)` 来匹配其时间导数，推理时积分。

整流流（Liu 2022）更进一步：用一个 reflow 过程迭代地拉直路径，产生逐渐接近线性的 ODE。经过两次 reflow 迭代，2 步采样器就能匹配 50 步 DDPM 的质量。

## 核心概念

![流匹配：噪声与数据之间的直线插值](../assets/flow-matching.svg)

### 直线流

定义：

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 `x_0 ~ 数据`，`x_1 ~ N(0, I)`。沿这条直线的时间导数是常数：

```
dx_t / dt = x_1 - x_0
```

定义神经向量场 `v_θ(x_t, t)` 并训练它匹配这个导数：

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这就是**条件流匹配**损失（Lipman 2023）。训练是免模拟的：你从不在训练时展开 ODE。只需采样 `(x_0, x_1, t)` 并进行回归。

### 采样

推理时，**反向**积分学习到的向量场：

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 `x_1 ~ N(0, I)` 开始，用 Euler 步向下积分到 `t=0`。

### 整流流（Liu 2022）

直线流有效，但学习到的路径*实际上并不是直线*——它们会弯曲，因为许多 `x_0` 可以映射到同一个 `x_1`。整流流的 reflow 步骤：

1. 用随机配对训练流模型 v_1。
2. 通过从 `x_1` 积分到其落点 `x_0`，采样 N 个配对 `(x_1, x_0)`。
3. 在这些配对样本上训练 v_2。因为这些配对现在是"ODE 匹配"的，它们之间的直线插值实际上更平坦。
4. 重复。

实践中，2 次 reflow 迭代可以达到近线性，支持 2-4 步推理。SDXL-Turbo、SD3-Turbo、LCM 都是从流匹配模型蒸馏而来。

### 为何 2024 年它在图像领域胜出

三个原因：

1. **免模拟训练**——训练时无需展开 ODE，实现简单。
2. **更好的损失几何**——直线路径具有一致的信噪比，而 DDPM ε 损失在调度边缘处信噪比差。
3. **更快的推理**——SDXL-Turbo 质量只需 4-8 步；一致性蒸馏只需 1 步。

## 流匹配 vs DDPM——精确联系

带有高斯条件路径的流匹配就是具有**特定噪声调度**的扩散模型。选择 `x_t = α(t) x_0 + σ(t) x_1` 调度，流匹配恢复为 Stratonovich 重新表述的扩散，其中 `v = α'·x_0 - σ'·x_1`。对于高斯路径，二者在代数上等价。

流匹配增加了什么：目标的*清晰度*（一个简单的速度）、更干净的损失，以及实验非高斯插值的许可。

## 动手实现

`code/main.py` 在两峰高斯混合上实现了一维流匹配。向量场 `v_θ(x, t)` 是一个用直线目标训练的小型 MLP。推理时，对 1、2、4、20 个 Euler 步进行积分并比较样本质量。

### 第一步：训练损失

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### 第二步：多步推理

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 第三步：比较步数

期望 4 步采样器已经能匹配 20 步的质量——对延迟来说意义重大。

## 常见陷阱

- **时间参数化。** 流匹配使用 `t ∈ [0, 1]`，`t=0` 对应数据，`t=1` 对应噪声。DDPM 使用 `t ∈ [0, T]`，`t=0` 对应数据，`t=T` 对应噪声。方向相同，尺度不同。论文经常在这一点上出错。
- **调度选择。** 整流流的直线是"那个"流匹配调度，但你可以使用余弦或 logit-normal t 采样（SD3 这样做）以获得更好的尺度覆盖。
- **Reflow 代价。** 为 reflow 生成配对数据集需要每个样本一次完整的推理过程。只在真正需要 1-2 步推理时才做 reflow。
- **无分类器引导仍然适用。** 只需在线性组合中将 ε 替换为 v：`v_cfg = (1+w) v_cond - w v_uncond`。

## 工程应用

| 用例 | 2026 年的技术栈 |
|------|----------------|
| 文生图，最佳质量 | 流匹配：SD3、Flux.1-dev |
| 文生图，1-4 步 | 蒸馏流匹配：Flux.1-schnell、SD3-Turbo、SDXL-Turbo |
| 实时推理 | 从流匹配基础蒸馏的一致性模型（LCM、PCM） |
| 音频生成 | 流匹配：Stable Audio 2.5、AudioCraft 2 |
| 视频生成 | 流匹配与扩散混合（Sora、Veo、Stable Video） |
| 科学 / 物理（粒子轨迹、分子） | 流匹配 + 等变向量场 |

当 2025-2026 年的论文说"比扩散更快"时，几乎总是指流匹配 + 蒸馏。

## 交付物

见 `outputs/skill-fm-tuner.md`。该技能接受扩散风格的模型规格，并将其转换为流匹配训练配置：调度选择、时间采样分布（均匀 / logit-normal）、优化器、reflow 计划、目标步数、评估协议。

## 练习

1. **（简单）** 运行 `code/main.py`，比较 1 步 vs 20 步与真实数据分布的 MSE。
2. **（中等）** 从均匀 `t` 采样切换到 logit-normal（集中在中间 t 值的采样）。模型质量是否提升？
3. **（困难）** 实现一次 reflow 迭代：通过积分第一个模型生成配对 (x_0, x_1)，在这些配对上训练第二个模型，比较 1 步样本质量。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 流匹配 (Flow matching) | "直线扩散" | 训练 `v_θ(x, t)` 沿插值匹配 `x_1 - x_0`。 |
| 整流流 (Rectified flow) | "Reflow" | 拉直学习流的迭代过程。 |
| 速度场 (Velocity field) | "v_θ" | 模型的输出——移动 `x_t` 的方向。 |
| 直线插值 (Straight-line interpolant) | "那条路径" | `x_t = (1-t)·x_0 + t·x_1`；目标导数简单。 |
| Euler 采样器 (Euler sampler) | "一阶 ODE 求解器" | 最简单的积分器；路径笔直时效果很好。 |
| Logit-normal t | "SD3 采样" | 将 `t` 采样集中在梯度最强的中间值。 |
| 一致性蒸馏 (Consistency distillation) | "单步采样器" | 训练学生直接将任意 `x_t` 映射到 `x_0`。 |
| 带速度的 CFG (CFG with velocity) | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`；相同技巧，新变量。 |

## 生产说明：Flux.1-schnell 是流匹配最快的实现

流匹配的生产胜利是 Flux.1-schnell——一个蒸馏到 1-4 步推理同时保持 Flux-dev 级质量的流匹配 DiT。参考部署方案是：T5 + CLIP 编码，量化 MMDiT 去噪（schnell 用 4 步，dev 用 50 步），VAE 解码。成本核算：

| 变体 | 步数 | L4 上 1024² 的延迟 | 总 FLOP（相对值） |
|------|------|-------------------|----------------|
| Flux.1-dev（原始） | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08×（快 12 倍） |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2 步 | 2 | ~0.3 s | 0.03× |

生产规则：**流匹配基础 + 蒸馏 = 2026 年快速文生图的默认选择。** 每个主要供应商都发布了这个组合：SD3-Turbo（SD3 + 流 + 蒸馏）、Flux-schnell（Flux-dev + 整流流拉直）、CogView-4-Flash。纯扩散基础只存在于遗产检查点中。

## 延伸阅读

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — 整流流
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — 流匹配
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，大规模整流流
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — 涵盖 FM + 扩散的通用框架
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — 扩散/流的单步蒸馏
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — turbo 变体
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — 生产中的流匹配

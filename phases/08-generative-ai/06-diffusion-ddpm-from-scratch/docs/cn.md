# 扩散模型——从零开始实现 DDPM

> Ho、Jain、Abbeel（2020）给了这个领域一个无法放弃的方案。用一千个小步骤向数据中加入噪声。训练一个神经网络来预测噪声。推理时反转这个过程。如今所有主流的图像、视频、3D 和音乐模型都在这个循环上运行，可能还叠加了流匹配或一致性技巧。

**类型：** 构建
**语言：** Python
**前置知识：** 第3阶段第02课（反向传播）、第8阶段第02课（VAE）
**预计时间：** 约75分钟

## 问题背景

你需要一个 `p_data(x)` 的采样器。GAN 玩一个经常发散的极小极大博弈。VAE 从高斯解码器产生模糊样本。你真正想要的训练目标要满足：（a）单一稳定损失（没有鞍点，没有极小极大），（b）`log p(x)` 的下界（这样你就有了似然值），（c）样本质量达到最优水平。

Sohl-Dickstein et al.（2015）有一个理论答案：定义一个马尔可夫链 `q(x_t | x_{t-1})` 逐渐添加高斯噪声，然后训练一个反向链 `p_θ(x_{t-1} | x_t)` 进行去噪。Ho、Jain、Abbeel（2020）证明损失可以简化为一行——预测噪声——并清理了数学推导。2020 年这还是个冷门。2021 年它产出了最优水平的样本。2022 年它成了 Stable Diffusion。2026 年它是一切的基石。

## 核心概念

![DDPM：前向加噪，反向去噪](../assets/ddpm.svg)

**前向过程 `q`。** 在 `T` 个小步骤中添加高斯噪声。封闭形式——数学可处理的原因——是累积步骤也是高斯的：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)`，对应一个 `β_t` 的调度。在 T=1000 步内将 `β_t` 从 1e-4 线性增加到 0.02，`x_T` 近似为 `N(0, I)`。

**反向过程 `p_θ`。** 学习一个神经网络 `ε_θ(x_t, t)` 来预测被添加的噪声。给定 `x_t`，通过以下方式去噪：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 是 `sqrt(β_t)` 或一个学习的方差。这个表达式看起来很丑，但只是代数——根据后验 `q(x_{t-1} | x_t, x_0)` 解出 `x_{t-1}`，并用其噪声预测估计代入 `x_0`。

**训练损失。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据中采样 `x_0`，随机选一个 `t`，采样 `ε ~ N(0, I)`，通过封闭形式一步计算带噪 `x_t`，然后回归噪声。一个损失，没有极小极大，没有 KL，没有重参数化技巧。

**采样。** 从 `x_T ~ N(0, I)` 开始，从 `t = T` 到 `1` 迭代反向步骤。完成。

## 为何有效

三个直觉：

1. **去噪容易；生成困难。** 在 `t=T` 时，数据是纯噪声——网络只需解决一个简单问题。在 `t=0` 时，网络只需清理几个像素。在中间的 `t`，问题很难，但网络从每个噪声级别获得许多梯度流过相同的权重。

2. **伪装的分数匹配。** Vincent（2011）证明预测噪声等价于估计 `∇_x log q(x_t | x_0)`，即"分数"。反向 SDE 利用这个分数沿密度梯度行走——一个引导向高概率区域的随机游走。

3. **ELBO 化简为简单 MSE。** 完整的变分下界每个时间步有一个 KL 项。在 DDPM 的参数化下，这些 KL 项化简为带特定系数的噪声预测 MSE；Ho 去掉了这些系数（称之为"简单"损失），质量反而**提升**了。

## 动手实现

`code/main.py` 实现了一个一维 DDPM。数据是两峰混合分布。"网络"是一个小型 MLP，接受 `(x_t, t)` 并输出预测的噪声。训练就是那一行损失。采样迭代反向链。

### 第一步：前向调度（封闭形式）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 第二步：一步采样 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 第三步：单步训练

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 第四步：反向采样

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于一维问题，40 个时间步和一个 24 单元 MLP，约 200 个 epoch 就能学会两峰混合分布。

## 时间步条件化

网络需要知道它在对哪个时间步去噪。两种标准方案：

- **正弦嵌入。** 类似 Transformer 的位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。通过 MLP 后广播进网络。
- **FiLM / 组归一化条件化。** 在每个块将嵌入投影为逐通道 scale/bias（FiLM）。

我们的玩具代码使用正弦 → 拼接。生产级 U-Net 使用 FiLM。

## 常见陷阱

- **调度非常重要。** 线性 `β` 是 DDPM 默认值，但余弦调度（Nichol & Dhariwal，2021）在相同计算量下给出更好的 FID。如果质量停滞，切换调度。
- **时间步嵌入很脆弱。** 将原始 `t` 作为浮点数传入对玩具一维数据有效，但对图像会失败；始终使用适当的嵌入。
- **V 预测 vs ε 预测。** 对于极端范围（非常小或非常大的 t），`ε` 的信噪比很差。V 预测（`v = α·ε - σ·x`）更稳定；SDXL、SD3 和 Flux 都使用它。
- **无分类器引导。** 推理时，同时计算条件和无条件 `ε`，然后 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，`w ≈ 3-7`。见第08课。
- **1000 步太多了。** 生产使用 DDIM（20-50 步）、DPM-Solver（10-20 步）或蒸馏（1-4 步）。见第12课。

## 工程应用

| 角色 | 2026 年的典型技术栈 |
|------|-------------------|
| 图像像素空间扩散（小型、玩具） | DDPM + U-Net |
| 图像潜在扩散 | VAE 编码器 + U-Net 或 DiT（第07课） |
| 视频潜在扩散 | 时空 DiT（Sora、Veo、WAN） |
| 音频潜在扩散 | Encodec + 扩散 Transformer |
| 科学（分子、蛋白质、物理） | 等变扩散（EDM、RFdiffusion、AlphaFold3） |

扩散是通用的生成骨干。流匹配（第13课）是 2024—2026 年的竞争者，通常在相同质量下赢得推理速度。

## 交付物

见 `outputs/skill-diffusion-trainer.md`。该技能接受数据集 + 计算预算，输出：调度（线性/余弦/sigmoid）、预测目标（ε/v/x）、步数、引导强度、采样器家族和评估协议。

## 练习

1. **（简单）** 在 `code/main.py` 中将 T 从 40 改为 10。样本质量（输出的视觉直方图）如何下降？T 降到多少时两峰结构会坍塌？
2. **（中等）** 从 ε 预测切换到 v 预测。重新推导反向步骤。比较最终样本质量。
3. **（困难）** 添加无分类器引导。对类别标签 `c ∈ {0, 1}` 进行条件化，训练时 10% 的概率丢弃它，采样时使用 `ε = (1+w)·ε_cond - w·ε_uncond`。测量 `w = 0, 1, 3, 7` 时的条件模式命中率。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 前向过程 (Forward process) | "加噪" | 固定马尔可夫链 `q(x_t \| x_{t-1})` 逐步破坏数据。 |
| 反向过程 (Reverse process) | "去噪" | 学习的链 `p_θ(x_{t-1} \| x_t)` 重建数据。 |
| β 调度 (β schedule) | "噪声阶梯" | 逐步方差；线性、余弦或 sigmoid。 |
| α̅（alpha bar） | "Alpha bar" | 累积乘积 `∏(1 - β)`；给出从 `x_0` 到 `x_t` 的封闭形式。 |
| 简单损失 (Simple loss) | "噪声上的 MSE" | `\|\|ε - ε_θ(x_t, t)\|\|²`；所有变分推导都化简为此。 |
| ε 预测 (ε-prediction) | "预测噪声" | 输出是被添加的噪声；标准 DDPM。 |
| V 预测 (V-prediction) | "预测速度" | 输出是 `α·ε - σ·x`；跨 t 的条件化更好。 |
| DDPM | "那篇论文" | Ho et al. 2020；线性 β，1000 步，U-Net。 |
| DDIM | "确定性采样器" | 非马尔可夫采样器，20-50 步，相同训练目标。 |
| 无分类器引导 (Classifier-free guidance) | "CFG" | 混合条件和无条件噪声预测以放大条件信号。 |

## 生产说明：扩散推理是步数问题

DDPM 论文运行 T=1000 个反向步骤。没有人在生产中这样做。每个真实推理栈选择三种策略之一——每种都清晰映射到"延迟从哪里来"的生产框架：

1. **更快的采样器，相同的模型。** DDIM（20-50 步）、DPM-Solver++（10-20 步）、UniPC（8-16 步）。直接替换反向循环；已训练的 `ε_θ` 权重不变。延迟减少 20-50 倍。
2. **蒸馏。** 训练一个学生以更少的步骤匹配教师：渐进式蒸馏（2→1 步）、一致性模型（任意→1-4 步）、LCM、SDXL-Turbo、SD3-Turbo。延迟再减少 5-10 倍，需要重新训练。
3. **缓存与编译。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM 的扩散后端、`xformers`/SDPA 注意力、bf16 权重。每步延迟减少约 2 倍。与（1）和（2）叠加使用。

对于生产扩散服务器，预算讨论与生产文献描述的 LLM 相同：延迟是 `步数 × 单步成本 + VAE 解码`，吞吐量是 `批大小 × (步数 × 单步成本)^-1`。首 token 时间很小（一步）；等效的 TPOT 是整个响应时间，因为从用户角度来看图像生成是"一次全部完成"的。

## 延伸阅读

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 扩散论文，超前于时代
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM，步数更少
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — 余弦调度，学习方差
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — 分类器引导
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — 统一符号，最简洁的方案

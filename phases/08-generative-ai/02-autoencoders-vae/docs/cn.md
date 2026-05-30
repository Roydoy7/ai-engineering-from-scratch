# 自编码器与变分自编码器（VAE）

> 普通自编码器先压缩后重建。它在记忆，不在生成。加一个技巧——强制编码向量看起来像高斯分布——你就得到了一个采样器。这个技巧，`z = μ + σ·ε` 的重参数化，就是为什么你在 2026 年使用的每一个潜在扩散和流匹配图像模型，在输入端都有一个 VAE。

**类型：** 构建
**语言：** Python
**前置知识：** 第3阶段第02课（反向传播）、第3阶段第07课（CNN）、第8阶段第01课（分类）
**预计时间：** 约75分钟

## 问题背景

将一张 784 像素的 MNIST 数字压缩成 16 个数字的编码，然后重建。普通自编码器在重建 MSE 上表现出色，但编码空间杂乱无章。在编码空间中随机取一个点，解码出来只是噪声。它没有采样能力，只是一个包装成生成模型的压缩模型。

你真正想要的是：（a）编码空间是一个干净、平滑的分布，可以从中采样——比如各向同性高斯 `N(0, I)`；（b）解码任何采样都能产生一个合理的数字；（c）编码器和解码器仍然压缩效果良好。三个目标，一个架构，一个损失函数。

Kingma 2013 年的 VAE 通过以下方式解决这个问题：训练编码器输出一个**分布** `q(z|x) = N(μ(x), σ(x)²)`，通过 KL 惩罚将该分布拉向先验 `N(0, I)`，然后在解码之前从 `q(z|x)` 采样 `z`。推理时，丢掉编码器，采样 `z ~ N(0, I)`，解码即可。KL 惩罚正是迫使编码空间有结构的关键。

2026 年 VAE 很少单独部署——在原始图像质量上它已被扩散模型超越——但它是每个潜在扩散模型（SD 1/2/XL/3、Flux、AudioCraft）的首选编码器。学会 VAE，你就学会了你使用的每个图像流水线中那个看不见的第一层。

## 核心概念

![自编码器与 VAE：重参数化技巧](../assets/vae.svg)

**自编码器。** `z = encoder(x)`，`x̂ = decoder(z)`，损失 = `||x - x̂||²`。编码空间无结构。

**VAE 编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。它们定义了 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧。** 从 `q(z|x)` 采样不可微。将采样改写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 加上一个与参数无关的噪声的确定性函数——梯度可以流过 `μ` 和 `σ`。

**损失。** 证据下界（ELBO），两个项：

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重建项将 `x̂` 推向 `x`。KL 项将 `q(z|x)` 推向先验。二者相互权衡。小 β（<1）= 样本更清晰，编码空间不够高斯。大 β（>1）= 编码空间更干净，样本更模糊。β-VAE（Higgins 2017）让这个旋钮声名大噪，开启了解耦表示学习研究。

**采样。** 推理时：从 `N(0, I)` 采样 `z`，通过解码器前向传播。一次前向传播——不像扩散模型那样需要迭代采样。

## 动手实现

`code/main.py` 实现了一个不使用 numpy 或 torch 的小型 VAE。输入是从 8 维空间中二成分高斯混合分布采样的 8 维合成数据。编码器和解码器是单隐层 MLP。我们实现 tanh 激活函数、前向传播、损失和手写反向传播。不追求生产级，只为教学。

### 第一步：编码器前向传播

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

输出 `log σ²` 而不是 `σ`，这样网络输出是无约束的（用 softplus 约束 σ 是个陷阱——在 σ ≈ 0 处梯度会消失）。

### 第二步：重参数化与解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 第三步：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

因为两个分布都是高斯，KL 有精确的封闭形式。不要用数值积分。2026 年仍有代码用蒙特卡洛 KL 估计——毫无理由地慢了 3 倍。

### 第四步：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行代码。

## 常见陷阱

- **后验坍塌（Posterior collapse）。** KL 项将 `q(z|x)` 向 `N(0, I)` 推得太猛，使得 `z` 不再携带任何关于 `x` 的信息。修复：β 退火（从 β=0 开始，逐渐增加到 1）、free bits、或在不活跃的维度上跳过 KL。
- **样本模糊。** 高斯解码器似然意味着 MSE 重建，这对 L2 是贝叶斯最优的（即均值）——一组合理数字的均值是一个模糊数字。修复：离散解码器（VQ-VAE、NVAE），或只将 VAE 用作编码器，在潜变量上叠加扩散（这正是 Stable Diffusion 的做法）。
- **β 过大且升得过早。** 见后验坍塌。从 β≈0.01 开始，缓慢提升。
- **潜变量维度过小。** MNIST 用 16 维可以，ImageNet 256² 用 256 维，ImageNet 1024² 用 2048 维。Stable Diffusion 的 VAE 将 512×512×3 压缩到 64×64×4（空间面积 32 倍下采样，通道 32 倍）。

## 工程应用

2026 年 VAE 的选型：

| 场景 | 选择 |
|------|------|
| 扩散模型的图像潜变量编码器 | Stable Diffusion VAE（`sd-vae-ft-ema`）或 Flux VAE |
| 音频潜变量编码器 | Encodec（Meta）、SoundStream 或 DAC（Descript） |
| 视频潜变量 | Sora 的时空 patch、Latte VAE、WAN VAE |
| 解耦表示学习 | β-VAE、FactorVAE、TCVAE |
| 离散潜变量（用于 Transformer 建模） | VQ-VAE、RVQ（ResidualVQ） |
| 用于生成的连续潜变量 | 普通 VAE，然后在该潜变量空间上条件流/扩散模型 |

潜在扩散模型 = VAE + 生活在编码器和解码器之间的扩散模型。VAE 做粗粒度压缩，扩散模型承担重活。视频（VAE + 视频扩散 DiT）和音频（Encodec + MusicGen Transformer）同样遵循这一模式。

## 交付物

见 `outputs/skill-vae-trainer.md`。该技能接受：数据集特征 + 潜变量维度目标 + 下游用途（重建、采样或潜在扩散输入），输出：架构选择（普通/β/VQ/RVQ）、β 调度、潜变量维度、解码器似然（高斯 vs 分类）和评估方案（重建 MSE、每维 KL、`q(z|x)` 与 `N(0, I)` 之间的 Fréchet 距离）。

## 练习

1. **（简单）** 在 `code/main.py` 中将 `β` 分别改为 `0.01`、`0.1`、`1.0`、`5.0`。记录最终重建 MSE 和 KL 值。哪个 β 在你的合成数据上是帕累托最优的？
2. **（中等）** 将高斯解码器似然替换为伯努利似然（交叉熵损失）。在同一合成数据的二值化版本上比较样本质量。
3. **（困难）** 将 `code/main.py` 扩展为一个迷你 VQ-VAE：用 K=32 条目的码本中的最近邻查找替换连续的 `z`。比较重建 MSE，并报告有多少码本条目被实际使用（码本坍塌是真实存在的问题）。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 自编码器 (Autoencoder) | 编码-解码网络 | `x → z → x̂`，学习 MSE。不具有生成能力。 |
| VAE | 带采样器的 AE | 编码器输出一个分布，KL 惩罚塑造编码空间。 |
| ELBO | 证据下界 | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`；当 `q = p(z\|x)` 时取等。 |
| 重参数化 (Reparameterization) | `z = μ + σ·ε` | 将随机节点改写为确定性 + 纯噪声。使反向传播能流过采样操作。 |
| 先验 (Prior) | `p(z)` | 潜变量的目标分布，通常为 `N(0, I)`。 |
| 后验坍塌 (Posterior collapse) | "KL 项赢了" | 编码器忽略 `x`，输出先验；解码器必须凭空编造。 |
| β-VAE | 可调 KL 权重 | `loss = recon + β·KL`。β 越大 = 越解耦但越模糊。 |
| VQ-VAE | 离散潜变量 | 用最近的码本向量替换连续的 `z`；支持 Transformer 建模。 |

## 生产说明：VAE 是扩散服务中最热的路径

在 Stable Diffusion / Flux / SD3 流水线中，VAE 每次请求被调用两次——一次编码（img2img / 修复时）和一次解码。在 1024² 分辨率下，解码过程通常是整个流水线中单次激活内存峰值最大的环节，因为它要将 `128×128×16` 的潜变量上采样回 `1024×1024×3`。两个实际后果：

- **切片或分块解码。** `diffusers` 提供 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。分块以小的接缝伪影为代价，将内存从 `O(H·W)` 降到 `O(tile²)`。在消费级 GPU 上处理 1024²+ 分辨率时必不可少。
- **解码器用 bf16，最终 resize 用 fp32 数值精度。** SD 1.x VAE 以 fp32 发布，在 1024²+ 时转换为 fp16 会**静默产生 NaN**。SDXL 附带了 `madebyollin/sdxl-vae-fp16-fix`——始终优先使用 fp16-fix 变体，或使用 bf16。

## 延伸阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 论文
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — 解耦 β-VAE
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — 图像 VAE 的顶尖水平
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion；VAE 作为编码器
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec，音频 VAE 标准

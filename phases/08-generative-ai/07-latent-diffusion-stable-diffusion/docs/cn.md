# 潜在扩散与 Stable Diffusion

> 在 512×512 图像的像素空间进行扩散是一场算力上的暴行。Rombach et al.（2022）发现生成一张图像并不需要所有 78.6 万个维度——你只需要足够捕捉语义结构的维度，其余交给一个单独的解码器。在 VAE 的潜变量空间内运行扩散。就这一个想法，就是 Stable Diffusion。

**类型：** 构建
**语言：** Python
**前置知识：** 第8阶段第02课（VAE）、第8阶段第06课（DDPM）、第7阶段第09课（ViT）
**预计时间：** 约75分钟

## 问题背景

在 512² 分辨率上进行像素空间扩散意味着 U-Net 在形状为 `[B, 3, 512, 512]` 的张量上运行。每个采样步骤对于一个 5 亿参数的 U-Net 约需 100 GFLOPS。五十步就是每张图像 5 TFLOPS。在十亿张图像上训练，算力账单荒谬至极。

这些 FLOP 大部分用于将感知上不重要的细节推过网络——高频纹理，一个有损 VAE 本可以压缩掉的东西。Rombach 的想法：训练一次 VAE（*第一阶段*），冻结它，然后完全在 4 通道 64×64 的潜变量空间中运行扩散（*第二阶段*）。相同的 U-Net，1/16 的像素，相当质量下约减少 64 倍的 FLOP。

这就是 Stable Diffusion 的方案。SD 1.x / 2.x 在 `64×64×4` 的潜变量上使用 8.6 亿参数的 U-Net，SDXL 在 `128×128×4` 上使用 26 亿参数的 U-Net，SD3 将 U-Net 换成了带流匹配的扩散 Transformer（DiT）。Flux.1-dev（Black Forest Labs，2024）发布了一个 120 亿参数的 DiT-MMDiT。所有这些都运行在相同的两阶段基础上。

## 核心概念

![潜在扩散：VAE 压缩 + 在潜变量空间中扩散](../assets/latent-diffusion.svg)

**两个阶段，分开训练。**

1. **第一阶段——VAE。** 编码器 `E(x) → z`，解码器 `D(z) → x`。目标压缩：每个空间轴 8 倍下采样 + 调整通道数，使总潜变量大小约为像素数的 1/16。损失 = 重建（L1 + LPIPS 感知）+ KL（小权重，因为我们不需要从 `z` 精确采样）。通常带对抗损失训练，使解码图像清晰。

2. **第二阶段——对 `z` 进行扩散。** 将 `z = E(x_real)` 视为数据。训练一个 U-Net（或 DiT）对 `z_t` 去噪。推理时：通过扩散采样 `z_0`，然后 `x = D(z_0)`。

**文本条件化。** 两个额外组件。一个冻结的文本编码器（SD 1.x 用 CLIP-L，SD 2/XL 用 CLIP-L+OpenCLIP-G，SD3 和 Flux 用 T5-XXL）。一个交叉注意力注入：每个 U-Net 块接受 `[Q = 图像特征, K = V = 文本 token]` 并混合它们。token 是文本影响图像的唯一途径。

**损失函数与第06课完全相同。** 噪声上的相同 DDPM / 流匹配 MSE。你只是切换了数据域。

## 架构变体

| 模型 | 年份 | 骨干 | 潜变量形状 | 文本编码器 | 参数量 |
|------|------|------|----------|-----------|-------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L（77 个 token） | 8.6 亿 |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 8.65 亿 |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 26 亿 + 66 亿 |
| SDXL-Turbo | 2023 | 蒸馏 | 128×128×4 | 同上 | 1-4 步采样 |
| SD3 | 2024 | MMDiT（多模态 DiT） | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 20 亿 / 80 亿 |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 120 亿 |
| Flux.1-schnell | 2024 | 蒸馏 MMDiT | 128×128×16 | T5-XXL + CLIP-L | 120 亿，1-4 步 |

趋势：用 DiT（在潜变量 patch 上的 Transformer）替换 U-Net，扩展文本编码器（T5 在提示词遵循上优于 CLIP），增加潜变量通道（4 → 16 提供更多细节空间）。

## 动手实现

`code/main.py` 将一个玩具一维"VAE"（用于演示的恒等编码器+解码器；真实 VAE 是卷积网络）叠加在第06课的 DDPM 之上，并添加了带无分类器引导的类别条件化。它表明相同的扩散损失无论在原始一维值上还是在编码值上运行都有效——这是核心洞见。

### 第一步：编码器/解码器

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

真实 VAE 有训练好的权重。出于教学目的，这个线性映射足以表明扩散在 `z` 上运行而无需关心原始数据空间。

### 第二步：在 `z` 空间中扩散

与第06课的 DDPM 相同。网络看到的数据是 `z = E(x)`。采样到 `z_0` 后，用 `D(z_0)` 解码。

### 第三步：无分类器引导

训练时，10% 的概率丢弃类别标签（替换为空 token）。推理时，同时计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无引导（完全多样性），`w = 3` = 默认，`w = 7+` = 饱和/过度清晰。

### 第四步：文本条件化（概念，非代码）

将类别标签替换为冻结的文本编码器输出。通过交叉注意力将文本嵌入输入 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这是类别条件扩散模型与 Stable Diffusion 之间唯一实质性的区别。

## 常见陷阱

- **VAE 缩放不匹配。** SD 1.x VAE 在编码后应用一个缩放常数（`scaling_factor ≈ 0.18215`）。忘记这个会让 U-Net 在方差完全错误的潜变量上训练。每个检查点都带有一个。
- **文本编码器静默出错。** SD3 需要带 >=128 个 token 的 T5-XXL，回退到仅 CLIP 是有损的。始终检查 `use_t5=True`，否则提示词遵循度会崩溃。
- **混用潜变量空间。** SDXL、SD3、Flux 使用不同的 VAE。在 SDXL 潜变量上训练的 LoRA 不适用于 SD3。Hugging Face diffusers 0.30+ 拒绝加载不匹配的检查点。
- **CFG 太高。** `w > 10` 产生饱和、油腻的图像，以多样性为代价过度拟合提示词。甜点是 `w = 3-7`。
- **负提示词泄露。** 空负提示词变成空 token；填写了内容的负提示词变成 `ε_uncond`。这两者不同；一些流水线静默地默认为空 token。

## 工程应用

2026 年的生产技术栈：

| 目标 | 推荐骨干 |
|------|---------|
| 窄领域，配对数据，从头训练 | SDXL 微调（LoRA / 全量）——最快投产 |
| 开放域文生图，开放权重 | Flux.1-dev（120 亿，Apache / 非商业）或 SD3.5-Large |
| 最快推理，开放权重 | Flux.1-schnell（1-4 步，Apache）或 SDXL-Lightning |
| 最佳提示词遵循，托管服务 | GPT-Image / DALL-E 3、Midjourney v7、Imagen 4 |
| 编辑工作流 | Flux.1-Kontext（2024 年 12 月）——原生接受图像 + 文本 |
| 研究，基线 | SD 1.5——古老但研究充分 |

## 交付物

见 `outputs/skill-sd-prompter.md`。该技能接受文本提示词 + 目标风格，输出：模型 + 检查点、CFG 尺度、采样器、负提示词、分辨率、可选的 ControlNet/IP-Adapter 组合，以及逐步质检清单。

## 练习

1. **（简单）** 用引导强度 `w ∈ {0, 1, 3, 7, 15}` 运行 `code/main.py`。记录每个类别的样本均值。在哪个 `w` 时类别均值超出了真实数据均值？
2. **（中等）** 将玩具线性编码器替换为带重建损失的 tanh-MLP 编码器/解码器对。在新的潜变量上重新训练扩散模型。样本质量是否有变化？
3. **（困难）** 用 diffusers 搭建一个真实的 Stable Diffusion 推理：加载 `sdxl-base`，用 CFG=7 运行 30 步 Euler 采样，计时。现在切换到 `sdxl-turbo`，4 步，CFG=0。相同主题，不同质量——描述发生了什么变化及原因。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 第一阶段 (First stage) | "那个 VAE" | 已训练的编码器/解码器对；将 512² 压缩到 64²。 |
| 第二阶段 (Second stage) | "那个 U-Net" | 在潜变量空间上的扩散模型。 |
| CFG | "引导尺度" | `(1+w)·ε_cond - w·ε_uncond`；调节条件化强度。 |
| 空 token (Null token) | "空提示词嵌入" | 用于 `ε_uncond` 的无条件嵌入。 |
| 交叉注意力 (Cross-attention) | "文本如何进入" | 每个 U-Net 块将文本 token 作为 K 和 V 进行注意力计算。 |
| DiT | "扩散 Transformer" | 用在潜变量 patch 上的 Transformer 替换 U-Net；扩展性更好。 |
| MMDiT | "多模态 DiT" | SD3 的架构：文本和图像流使用联合注意力。 |
| VAE 缩放因子 (VAE scaling factor) | "那个魔法数字" | 将潜变量除以约 5.4，使扩散在单位方差空间中操作。 |

## 生产说明：在 8GB 消费级 GPU 上运行 Flux-12B

参考 Flux 集成是典型的"我有一块消费级 GPU，能否上线？"方案。技巧与生产推理文献对扩散 DiT 所列出的相同三旋钮方案：

1. **交错加载。** Flux 有三个网络，无需同时存在于显存中：T5-XXL 文本编码器（fp32 约 10 GB）、CLIP-L（小）、120 亿参数的 MMDiT 和 VAE。先编码提示词，*删除*编码器，加载 DiT，去噪，*删除* DiT，加载 VAE，解码。消费级 8GB GPU 每次只能容纳一个阶段。
2. **通过 bitsandbytes 进行 4 位量化。** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` 同时用于 T5 编码器和 DiT。内存减少 8 倍，文生图的质量下降几乎无感。
3. **CPU 卸载。** `pipe.enable_model_cpu_offload()` 随着每个前向传播的推进，自动在 CPU 和 GPU 之间交换模块。增加 10-20% 的延迟，但让流水线能够运行。

内存计算：`10 GB T5 / 8 = 1.25 GB` 量化后，`120 亿参数 × 0.5 字节 = 约 6 GB` 量化 DiT，加上激活值。这是 TP=1 推理的极端情况——没有模型并行，最大量化。生产环境下你会在 H100 上跑 TP=2 或 TP=4；对于单人开发笔记本，这就是方案。

## 延伸阅读

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，MMDiT
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 家族
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — 以上所有检查点的参考实现

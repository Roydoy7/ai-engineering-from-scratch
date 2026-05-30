# ControlNet、LoRA 与条件化

> 单靠文本是一个笨拙的控制信号。ControlNet 让你克隆一个预训练扩散模型，并用深度图、姿态骨架、涂鸦或边缘图来引导它。LoRA 让你通过训练 1000 万个参数来微调一个 20 亿参数的模型。二者合力，将 Stable Diffusion 从一个玩具变成了 2026 年每家机构都在使用的图像流水线。

**类型：** 构建
**语言：** Python
**前置知识：** 第8阶段第07课（潜在扩散）、第10阶段（从零开始的 LLM——LoRA 基础）
**预计时间：** 约75分钟

## 问题背景

"一个女人穿着红裙子在繁忙街道上遛狗"这样的提示词，没有给模型任何关于*狗在哪里*、*女人是什么姿势*或*街道的透视*的信息。文字固定的信息约占你需要指定一张图像所需信息的 10%。其余的是视觉信息，无法用文字高效描述。

为每种信号（姿态、深度、Canny 边缘、分割）从头训练一个新的条件模型代价高昂。你希望保持 26 亿参数的 SDXL 骨干冻结，附加一个读取条件信号的小型侧网络，让它微调骨干的中间特征。这就是 ControlNet。

你还希望在不重新训练完整模型的情况下，教会模型新概念（你的脸、你的产品、你的风格）。你需要一个小 100 倍的增量。这就是 LoRA——插入现有注意力权重的低秩适配器。

ControlNet + LoRA + 文本 = 2026 年实践者的工具箱。大多数生产图像流水线在 SDXL / SD3 / Flux 基础模型之上叠加 2-5 个 LoRA、1-3 个 ControlNet 和一个 IP-Adapter。

## 核心概念

![ControlNet 克隆编码器；LoRA 添加低秩增量](../assets/controlnet-lora.svg)

### ControlNet（Zhang et al.，2023）

取一个预训练的 SD。*克隆* U-Net 的编码器部分。冻结原始模型。训练克隆体接受额外的条件输入（边缘、深度、姿态）。用*零卷积*跳跃连接（初始化为零的 1×1 卷积——从无操作开始，学习一个增量）将克隆体连接回原始模型的解码器部分。

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 一开始是恒等变换——即使在训练前也不会造成损害。在 100 万个（提示词，条件，图像）三元组上使用标准扩散损失训练。

每种模态的 ControlNet 作为小型侧模型发布（SDXL 约 3.6 亿参数，SD 1.5 约 7000 万参数）。你可以在推理时组合它们：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu et al.，2021）

对模型中的任意线性层 `W ∈ R^{d×d}`，冻结 `W` 并添加一个低秩增量：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。注意力层通常使用秩 4-16，重度微调使用秩 64-128。新参数数量：`2 · d · r` 而不是 `d²`。对于 `d=640` 的 SDXL 注意力，`r=16`：每个适配器 2 万个参数而不是 41 万——减少 20 倍。整个模型来看：LoRA 通常是 20-200MB，而基础模型是 5GB。

推理时可以缩放 LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 是正常范围。多个 LoRA 可以加性叠加（但通常有非线性交互的注意事项）。

### IP-Adapter（Ye et al.，2023）

一个小型适配器，接受*图像*作为条件（与文本并列）。使用 CLIP 图像编码器生成图像 token，将其注入交叉注意力，与文本 token 并列。每个基础模型约 20MB。让你可以做"生成这张参考图像风格的图像"，而不需要 LoRA。

## 可组合性矩阵

| 工具 | 控制什么 | 大小 | 何时使用 |
|------|---------|------|---------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 精确布局、构图 |
| LoRA | 风格、主体、概念 | 20-200MB | 个性化、风格 |
| IP-Adapter | 参考图像的风格或主体 | 20MB | 文字无法描述的外观 |
| Textual Inversion | 作为新 token 的单个概念 | 10KB | 遗产技术，已大多被 LoRA 取代 |
| DreamBooth | 对主体的全量微调 | 2-5GB | 强身份认同，高计算量 |
| T2I-Adapter | 更轻量的 ControlNet 替代方案 | 70MB | 边缘设备、推理预算受限 |

ControlNet ≈ 空间控制。LoRA ≈ 语义控制。两者都用。

## 动手实现

`code/main.py` 在一维空间模拟这两种机制：

1. **LoRA。** 一个预训练的线性层 `W`。冻结它。训练低秩 `B @ A` 使 `W + BA` 匹配目标线性层。证明 `r = 1` 足以完美学习一个秩-1 修正。

2. **ControlNet-lite。** 一个"冻结的基础"预测器和一个读取额外信号的"侧网络"。侧网络的输出由初始化为零的可学习标量门控（我们版本的零卷积）。训练并观察门控值如何逐渐增大。

### 第一步：LoRA 数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 第二步：零初始化侧网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

在步骤 0 时，输出与基础模型完全相同。早期训练缓慢更新 `gate`——没有灾难性漂移。

## 常见陷阱

- **LoRA 过度缩放。** `α = 2` 或 `α = 3` 是常见的"让它更强"的做法，会产生过度风格化/损坏的输出。保持 `α ≤ 1.5`。
- **ControlNet 权重冲突。** 在权重 1.0 下使用姿态 ControlNet 并在权重 1.0 下使用深度 ControlNet 通常会过头。权重之和 ≈ 1.0 是安全的默认值。
- **LoRA 用在错误的基础模型上。** SDXL LoRA 在 SD 1.5 上静默失效，因为注意力维度不匹配。diffusers 0.30+ 会发出警告。
- **Textual Inversion 漂移。** 在一个检查点上训练的 token 在另一个检查点上漂移严重。LoRA 的可移植性更好。
- **LoRA 权重合并与存储。** 你可以将 LoRA 烘焙到基础模型权重中以加快推理（无需运行时加法），但会失去在运行时缩放 `α` 的能力。保留两个版本。

## 工程应用

| 目标 | 2026 年的流水线 |
|------|--------------|
| 复制品牌的艺术风格 | 在约 30 张精选图像上训练秩 32 的 LoRA |
| 将我的脸放入生成图像 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 特定姿态 + 提示词 | ControlNet-Openpose + SDXL + 文本 |
| 深度感知构图 | ControlNet-Depth + SD3 |
| 参考图像 + 提示词 | IP-Adapter + 文本 |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 背景替换 | ControlNet-Seg + 修复（第09课） |
| 快速单步风格 | LCM-LoRA on SDXL-Turbo |

## 交付物

见 `outputs/skill-sd-toolkit-composer.md`。该技能接受任务（输入资产：提示词、可选的参考图像、可选的姿态、可选的深度、可选的涂鸦），输出工具栈、权重和可复现的种子协议。

## 练习

1. **（简单）** 在 `code/main.py` 中将 LoRA 秩 `r` 从 1 改到 4。在哪个秩时 LoRA 能精确匹配秩-2 的目标增量？
2. **（中等）** 对两个目标变换分别训练两个 LoRA。同时加载它们并展示加性交互。何时交互会破坏线性？
3. **（困难）** 使用 diffusers 叠加：SDXL-base + Canny-ControlNet（权重 0.8）+ 风格 LoRA（α 0.8）+ IP-Adapter（权重 0.6）。测量在栈权重变化时 FID 与提示词遵循度的权衡。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| ControlNet | "空间控制" | 克隆的编码器 + 零卷积跳跃连接；读取条件图像。 |
| 零卷积 (Zero convolution) | "从恒等开始" | 初始化为零的 1×1 卷积；ControlNet 从无操作开始。 |
| LoRA | "低秩适配器" | `W + B @ A`，`r << d`；比全量微调少 100 倍的参数。 |
| 秩 r (rank r) | "那个旋钮" | LoRA 压缩比；通常 4-16，重度个性化用 64+。 |
| α | "LoRA 强度" | LoRA 增量的运行时缩放因子。 |
| IP-Adapter | "参考图像" | 通过 CLIP 图像 token 实现的小型图像条件适配器。 |
| DreamBooth | "全量主体微调" | 在约 30 张主体图像上训练完整模型。 |
| Textual Inversion | "新 token" | 只学习一个新词嵌入；遗产技术，已大多被替代。 |

## 生产说明：LoRA 热换、ControlNet 通道、多租户服务

真实的文生图 SaaS 在同一个基础检查点上服务数百个 LoRA 和十几个 ControlNet。服务问题与 LLM 多租户非常相似（生产文献在连续批处理和 LoRAX / S-LoRA 下涵盖了 LLM 情况）：

- **热换 LoRA，不要合并。** 将 `W' = W + α·B·A` 合并到基础模型中可以使每步推理快约 3-5%，但会冻结 `α` 和基础模型。将 LoRA 作为秩-r 增量保持在显存中热备；diffusers 提供 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 用于按请求激活。换换代价是 `2 · d · r · 层数` 个权重——MB 级别，亚秒完成。
- **ControlNet 作为第二个注意力通道。** 克隆的编码器与基础模型并行运行。两个 ControlNet 各权重 1.0 = 每步多两次前向传播，而不是一次合并的传播。批大小空间成二次方下降。每个活跃 ControlNet 预计约 1.5 倍的单步成本。
- **LoRA 也量化。** 如果你量化了基础模型（见第07课，8GB 上的 Flux），LoRA 增量也可以干净地量化到 8 位或 4 位。QLoRA 风格的加载让你在 4 位 Flux 基础上叠加 5-10 个 LoRA 而不爆显存。

Flux 专用：将基础模型量化为 4 位，然后叠加风格 LoRA（`pipe.load_lora_weights("user/style-lora")`）在该量化基础上仍然有效。这是 2026 年大多数 SaaS 机构发布的方案。

## 延伸阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初用于 LLM；移植到扩散模型）
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet 的更轻量替代方案
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — 参考流水线

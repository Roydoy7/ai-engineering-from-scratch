# 修复、扩展与图像编辑

> 文生图创造新事物。修复（Inpainting）修正旧事物。在生产中，70% 有偿的图像工作是编辑——换背景、去除 Logo、延伸画布、重新生成手部。修复才是扩散模型证明自身价值的地方。

**类型：** 构建
**语言：** Python
**前置知识：** 第8阶段第07课（潜在扩散）、第8阶段第08课（ControlNet 与 LoRA）
**预计时间：** 约75分钟

## 问题背景

客户发来一张完美的产品照片，背景里有一个碍眼的标牌。你想抹掉标牌，其余一切像素完全保留。你不能从头运行文生图——结果会有不同的颜色、不同的光照、不同的产品角度。你想只重新生成**遮罩区域**，并且希望重新生成的内容尊重周围的上下文。

这就是修复（Inpainting）。变体有：

- **修复（Inpainting）。** 在遮罩内重新生成，遮罩外的像素保持不变。
- **扩展（Outpainting）。** 在遮罩外重新生成（或在画布之外），遮罩内的保持不变。
- **图像编辑（Image editing）。** 重新生成整张图像，但保持与原始图像的语义或结构一致性（SDEdit、InstructPix2Pix）。

2026 年的每个扩散流水线都提供修复模式。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们工作在相同的原理上。

## 核心概念

![修复：感知遮罩的去噪与保留上下文的重注入](../assets/inpainting.svg)

### 朴素方法（以及为何错误）

用遮罩运行标准文生图。在每个采样步骤，将带噪潜变量的未遮罩区域替换为干净图像的前向扩散版本。这能工作……但效果很差。边界伪影会渗出，因为模型没有遮罩区域内容的信息。

### 正确的修复模型

训练一个修改后的 U-Net，接受 9 个输入通道而不是 4 个：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外的通道是 VAE 编码的源图像副本加上一个单通道遮罩。训练时，随机遮罩图像区域，训练模型只对遮罩区域去噪，同时将未遮罩区域作为干净的条件信号给出。推理时，模型可以"看到"遮罩区域周围的内容，并产生连贯的补全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用这种 9 通道（或类似）输入。Diffusers 的 `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit（Meng et al.，2022）——免训练编辑

向源图像添加噪声直到某个中间时间步 `t`，然后用新提示词从 `t` 向下运行反向链到 0。无需重新训练。起始 `t` 的选择在保真度和创意自由度之间权衡：

- `t/T = 0.3` → 与源图像几乎相同，风格上有小变化
- `t/T = 0.6` → 中等程度的编辑，保留粗粒度结构
- `t/T = 0.9` → 从接近纯噪声生成，几乎不保留源图像

### InstructPix2Pix（Brooks et al.，2023）

在 `（输入图像，指令，输出图像）` 三元组上微调扩散模型。推理时，同时以输入图像和文本指令为条件（"让它变成日落"、"加一条龙"）。两个 CFG 尺度：图像尺度和文本尺度。

### RePaint（Lugmayr et al.，2022）

保留标准的无条件扩散模型。在每个反向步骤，重采样——偶尔跳回更嘈杂的状态并重新生成。避免边界伪影。在没有训练过的修复模型时使用。

## 动手实现

`code/main.py` 在 5 维数据上实现了一个玩具一维修复方案。我们在 5 维混合数据上训练 DDPM，每个样本是来自两个簇之一的 5 个浮点数。推理时，我们"遮罩"5 个维度中的 2 个，在每步注入未遮罩的 3 个维度的带噪前向版本，只重新生成遮罩的维度。

### 第一步：5 维 DDPM 数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 第二步：在所有 5 个维度上训练去噪器

标准 DDPM。网络对 5 维带噪输入输出 5 维噪声预测。

### 第三步：推理时感知遮罩的反向过程

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

这是朴素方法，在玩具一维数据上有效。真实图像修复使用 9 通道输入，因为纹理一致性更重要。

### 第四步：扩展（Outpainting）

扩展就是遮罩翻转的修复：遮罩新的（之前不存在的）画布，保留其余的原始内容。相同的训练目标。

## 常见陷阱

- **接缝。** 朴素方法留下可见的边界，因为梯度信息不能跨越遮罩流动。修复：将遮罩扩大 8-16 像素，或使用正确的修复模型。
- **遮罩泄露。** 如果条件图像的未遮罩区域质量差或有噪声，会污染遮罩内的生成。稍微去噪或模糊处理。
- **CFG 与遮罩大小交互。** 在小遮罩上使用高 CFG = 饱和的补块。对小的编辑降低 CFG。
- **SDEdit 保真度悬崖。** 从 `t/T = 0.5` 到 `t/T = 0.6` 可能失去主体的身份。扫描并记录检查点。
- **提示词不匹配。** 提示词应描述*整张*图像，而不仅仅是新内容。"一只猫坐在椅子上"而不是"一只猫"。

## 工程应用

| 任务 | 流水线 |
|------|-------|
| 移除物体，小遮罩 | SD-Inpaint 或 Flux-Fill，标准提示词 |
| 替换天空 | SD-Inpaint + "blue sky at sunset" |
| 扩展画布 | SDXL 扩展模式（8px 羽化）或带扩展遮罩的 Flux-Fill |
| 重新生成手/脸 | 带重新描述主体的提示词的 SD-Inpaint + ControlNet-Openpose |
| 更改一个区域的风格 | 在遮罩区域以 `t/T=0.5` 的 SDEdit |
| "让它变成日落" | InstructPix2Pix 或 Flux-Kontext |
| 背景替换 | SAM 遮罩 → SD-Inpaint |
| 超高保真度 | Flux-Fill 或 GPT-Image（托管）用于最难的情况 |

SAM（Meta 的 Segment Anything，2023）+ 扩散修复是 2026 年的背景移除流水线。SAM 2（2024）支持视频。

## 交付物

见 `outputs/skill-editing-pipeline.md`。该技能接受原始图像 + 编辑描述 + 可选的遮罩（或 SAM 提示词），输出：遮罩生成方法、基础模型、CFG 尺度（图像 + 文本）、SDEdit-t 或修复模式，以及质检清单。

## 练习

1. **（简单）** 在 `code/main.py` 中，将遮罩的维度比例从 0.2 变到 0.8。在哪个比例时，修复质量（遮罩维度的残差）等于无条件生成的质量？
2. **（中等）** 实现 RePaint：每 10 步反向步骤，跳回 5 步（添加噪声）并重新去噪。测量它是否减少遮罩边缘的边界残差。
3. **（困难）** 使用 Hugging Face diffusers 比较：SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill 在 20 个人脸重生成任务上。分别对姿态遵循度和身份保留度打分。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 修复 (Inpainting) | "填充孔洞" | 在遮罩内重新生成；遮罩外像素保持不变。 |
| 扩展 (Outpainting) | "延伸画布" | 在画布外重新生成；内部保持不变。 |
| 9 通道 U-Net | "正确的修复模型" | 以 `带噪潜变量 \| 编码源图像 \| 遮罩` 为输入的 U-Net。 |
| SDEdit | "带噪声级别的 Img2img" | 加噪到时间步 `t`，用新提示词去噪。 |
| InstructPix2Pix | "纯文本编辑" | 在（图像，指令，输出）三元组上微调的扩散模型。 |
| RePaint | "无需重新训练" | 在反向过程中周期性重新加噪以减少接缝。 |
| SAM | "Segment Anything" | 通过点击或框生成遮罩；与修复配对使用。 |
| Flux-Kontext | "带上下文的编辑" | 接受参考图像 + 指令进行编辑的 Flux 变体。 |

## 生产说明：编辑流水线对延迟敏感

用户编辑图像期望 5 秒内得到响应。在 L4 上 30 步 SDXL-Inpaint 在 1024² 分辨率下需要 3-4 秒，加上 SAM 遮罩生成（约 200ms）和 VAE 编码/解码（合计约 500ms）。用生产框架来说，这是首 token 时间（TTFT）约束而非吞吐量约束——批大小 1，低并发，最小化每个阶段：

- **SAM-H 是瓶颈。** SAM-H 在 1024² 上约 200ms；SAM-ViT-B 约 40ms，质量损失轻微。SAM 2（视频）增加时间维度开销；不要用于单张图像编辑。
- **尽可能跳过编码。** `pipe.image_processor.preprocess(img)` 将图像编码为潜变量。如果你已经有了上一次生成的潜变量（典型的迭代编辑 UI 场景），通过 `latents=...` 直接传入以跳过一次 VAE 编码。
- **遮罩膨胀对吞吐量也有影响。** 小遮罩意味着大部分 U-Net 前向传播被浪费（未遮罩的像素无论如何都会被夹住）。diffusers 的 `StableDiffusionInpaintPipeline` 无论如何都运行完整的 U-Net；只有 9 通道正确修复变体才能利用遮罩计算。
- **Flux-Kontext 是 2025 年的答案。** 在 `（源图像，指令）` 上单次前向传播——没有单独的遮罩，没有 SDEdit 噪声扫描。在 H100 上约 1.5 秒完成一次编辑。架构教训：合并阶段。

## 延伸阅读

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 免训练修复
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 文本指令编辑
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM，遮罩来源
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 视频 SAM
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — 注意力级别的编辑
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 工具

# 视频生成

> 图像是一个二维张量。视频是一个三维张量。理论相同，计算量难了 10-100 倍。OpenAI 的 Sora（2024 年 2 月）证明了这是可行的。到 2026 年，Veo 2、Kling 1.5、Runway Gen-3、Pika 2.0 和 WAN 2.2 在 1080p 分辨率上提供生产级文生视频——而开放权重的技术栈（CogVideoX、HunyuanVideo、Mochi-1、WAN 2.2）落后 12 个月。

**类型：** 构建
**语言：** Python
**前置知识：** 第8阶段第07课（潜在扩散）、第7阶段第09课（ViT）、第8阶段第06课（DDPM）
**预计时间：** 约45分钟

## 问题背景

一个 24fps 的 10 秒 1080p 视频是 240 帧的 1920×1080×3 像素。每个片段原始数据约 1.5 GB。像素空间扩散完全不可行。你需要：

1. **时空压缩。** 一个 VAE，能编码视频（而不是帧）为时空 patch 序列。
2. **时间连贯性。** 帧需要在数秒内共享内容、光照和物体身份。网络必须对运动建模。
3. **计算预算。** 视频训练比相同模型大小的图像训练贵 10-100 倍。
4. **条件化。** 文本、图像（第一帧）、音频或另一个视频。大多数生产模型接受全部四种。

解决这个问题的架构是将**扩散 Transformer（DiT）**应用于时空 patch，在大规模（提示词，字幕，视频）数据集上训练。与第06课的扩散损失相同。

## 核心概念

![视频扩散：patch 化、DiT、解码](../assets/video-generation.svg)

### Patch 化

用 3D VAE（学习的时空压缩）对视频编码。潜变量形状为 `[T_latent, H_latent, W_latent, C_latent]`。分成大小为 `[t_p, h_p, w_p]` 的 patch。对于 Sora 风格的模型，`t_p = 1`（逐帧 patch）或 `t_p = 2`（每两帧）。一个 10 秒 1080p 视频压缩为约 2 万—10 万个 patch。

### 时空 DiT

Transformer 处理 patch 的平铺序列。每个 patch 有一个 3D 位置嵌入（时间 + y + x）。注意力通常被分解：

- **空间注意力**：在每帧的 patch 内。
- **时间注意力**：跨相同空间位置的帧。
- **完整 3D 注意力**：代价高 16-100 倍；仅在低分辨率或研究中使用。

### 文本条件化

与大型文本编码器进行交叉注意力（Sora 用 T5-XXL，CogVideoX-5B 用 T5-XXL）。长提示词很重要——Sora 的训练集有 GPT 重新生成的密集字幕，每个片段平均 200 个 token。

### 训练

在时空潜变量上的标准扩散损失（ε 或 v 预测）。数据：网络视频 + 约 1 亿个精选片段 + 合成文本字幕。计算量：即使是小规模研究运行也需要 1 万+ GPU 小时；Sora 规模需要 10 万+。

## 2026 年的生产全景

| 模型 | 日期 | 最大时长 | 最大分辨率 | 开放权重？ | 特点 |
|------|------|---------|----------|-----------|------|
| Sora（OpenAI） | 2024-02 | 60s | 1080p | 否 | 首个大规模展示世界模拟器特性的模型 |
| Sora Turbo | 2024-12 | 20s | 1080p | 否 | 生产级 Sora，推理速度快 5 倍 |
| Veo 2（Google） | 2024-12 | 8s | 4K | 否 | 2025 年最高质量 + 物理模拟 |
| Veo 3 | 2025 Q3 | 15s | 4K | 否 | 原生音频和更强的镜头控制 |
| Kling 1.5 / 2.1（快手） | 2024-2025 | 10s | 1080p | 否 | 2025 年 Q1 最佳人体运动 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | 否 | 上层专业视频工具 |
| Pika 2.0 | 2024-10 | 5s | 1080p | 否 | 最强角色一致性 |
| CogVideoX（清华） | 2024 | 10s | 720p | 是（20 亿、50 亿） | 首个开放 50 亿规模视频模型 |
| HunyuanVideo（腾讯） | 2024-12 | 5s | 720p | 是（130 亿） | 2024 年底开放 SOTA |
| Mochi-1（Genmo） | 2024-10 | 5.4s | 480p | 是（100 亿） | 授权最宽松 |
| WAN 2.2（阿里巴巴） | 2025-07 | 5s | 720p | 是 | 2025 年中最强开放模型 |

开放权重正在比图像领域更快地缩小差距：HunyuanVideo + WAN 2.2 LoRA 到 2026 年中已驱动大多数开源工作流。

## 动手实现

`code/main.py` 模拟了核心时空 DiT 想法：对一个小型合成视频进行 patch 化，为每个 patch 添加位置嵌入，并用 Transformer 风格的 patch 注意力对整个序列去噪。不使用 numpy；纯 Python。我们展示当相邻帧的 patch 共享去噪器和位置嵌入时，即使在一维情况下也能涌现出时间连贯性。

### 第一步：对合成一维"视频"进行 patch 化

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### 第二步：每帧的位置嵌入

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### 第三步：去噪器看到整个序列

我们的小型网络不是独立对每帧去噪，而是将所有帧的值 + 位置嵌入拼接起来，联合预测所有帧的噪声。

### 第四步：时间连贯性测试

训练后，采样一个视频。测量帧间差异。如果模型已经学习了时间结构，差异应小于独立采样每帧。

## 常见陷阱

- **独立逐帧采样 = 闪烁。** 如果对每帧单独运行图像扩散，输出会闪烁，因为每帧的噪声是独立的。视频扩散通过注意力或共享噪声将帧耦合在一起来解决这个问题。
- **朴素 3D 注意力 = 内存溢出。** 在 10 秒 1080p 潜变量上的完整 3D 注意力需要数千亿次操作。分解为空间 + 时间注意力。
- **数据字幕比规模更重要。** Sora 相对于前期工作的主要升级是训练时使用了约 10 倍更详细的字幕（GPT-4 重新标注的片段）。OpenAI 技术报告对此明确说明。
- **第一帧条件化。** 大多数生产模型也接受图像作为第一帧。这是"图像到视频"模式；训练包含这个变体。
- **物理漂移。** 长片段（>10 秒）会积累细微不一致。滑动窗口生成 + 关键帧锚定有帮助。

## 工程应用

| 用例 | 2026 年的选择 |
|------|-------------|
| 最高质量文生视频，托管服务 | Veo 3 或 Sora |
| 镜头控制的电影级效果 | 带运动笔刷的 Runway Gen-3 |
| 跨片段的角色一致性 | Pika 2.0 或 Kling 2.1 |
| 开放权重，快速微调 | WAN 2.2 + LoRA |
| 图像到视频 | WAN 2.2-I2V、Kling 2.1 I2V 或 Runway |
| 音频到视频唇同步 | Veo 3（原生音频）或专用唇同步模型 |
| 视频编辑 | Runway Act-Two、Kling Motion Brush、Flux-Kontext（静帧） |

2024 年到 2026 年，同等质量下每秒视频的成本下降了 20 倍。

## 交付物

见 `outputs/skill-video-brief.md`。该技能接受视频简报（时长、宽高比、风格、镜头计划、主体一致性、音频），输出：模型 + 托管服务、提示词脚手架（镜头语言、主体描述、运动描述符）、种子 + 可复现协议，以及逐帧质检清单。

## 练习

1. **（简单）** 在 `code/main.py` 中，比较（a）独立逐帧采样和（b）联合序列采样的帧间差异。报告差异的均值和方差。
2. **（中等）** 添加第一帧条件：将第 0 帧固定为给定值，采样其余帧。测量固定值如何传播。
3. **（困难）** 使用 HuggingFace diffusers 在本地 GPU 上运行 CogVideoX-2B。对 6 秒片段在 720p 下 20 步推理进行计时。分析时空注意力的瓶颈所在。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 视频 VAE (Video VAE) | "3D VAE" | 将 `(T, H, W, C)` 压缩为时空潜变量的编码器。 |
| Patch | "那些 token" | 潜变量的固定大小 3D 块；DiT 的输入。 |
| 分解注意力 (Factorized attention) | "空间 + 时间" | 先在空间上运行注意力，再在时间上运行；跳过完整 3D 注意力。 |
| 图像到视频 (Image-to-video, I2V) | "让这张照片动起来" | 模型接受图像 + 文本，输出从它开始的视频。 |
| 关键帧条件化 (Keyframe conditioning) | "锚帧" | 固定特定帧以控制视频的走向。 |
| 运动笔刷 (Motion brush) | "方向提示" | 用户在图像上绘制运动向量的 UI 输入。 |
| 重新字幕 (Re-captioning) | "密集字幕" | 用 LLM 以详细提示词重新标注训练片段。 |
| 闪烁 (Flicker) | "时间伪影" | 帧间不一致性；通过耦合去噪解决。 |

## 生产说明：视频潜变量是内存带宽问题

一个 24fps 的 10 秒 1080p 片段是 240 帧 × 1920 × 1080 × 3 ≈ 1.5 GB 的原始像素。经过 4× 视频 VAE 压缩（`2 × 空间 × 2 × 时间`），每个请求的潜变量约 100 MB。批大小为 1，通过时空 DiT 运行 30 步，每步需要移动约 3 GB 的 HBM——内存带宽而非 FLOP 是瓶颈。

三个生产旋钮，均来自生产推理文献的推理章节：

- **DiT 的张量并行（TP）。** 文生视频模型通常 ≥100 亿参数。4 张 H100 上 TP=4 是标准配置；4050 亿级模型使用 PP=2 × TP=2。每步延迟大致随 TP 线性下降，直到达到 all-reduce 上限。
- **帧批处理 = 连续批处理。** 生成时，视频在概念上是通过注意力链接的帧的批次。连续批处理（在途调度）适用：如果模型架构支持滑动窗口生成，则在返回第 t-1 帧时开始渲染第 t+1 帧。
- **片段级预填充缓存。** 对于图像到视频，第一帧条件化类似于 LLM 的提示词预填充：计算一次，在时间解码过程中复用。这实际上是视频的 KV 缓存。

## 延伸阅读

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora 技术报告
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — 2025 年中最强开放模型
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — 开创性的视频扩散论文
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — Stable Video Diffusion 的前身

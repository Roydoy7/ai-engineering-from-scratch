# 神经音频编解码器——EnCodec、SNAC、Mimi、DAC 与语义-声学分离

> 2026 年的音频生成几乎全部基于 token。EnCodec、SNAC、Mimi 和 DAC 将连续波形转换为 Transformer 可以预测的离散序列。语义 token 与声学 token 的分离——首码本为语义，其余为声学——是自 Transformer 以来音频领域最重要的架构转变。

**类型：** 学习
**语言：** Python
**前置知识：** 第6阶段第2课（频谱图）、第10阶段第11课（量化）、第5阶段第19课（子词分词）
**预计时间：** 约60分钟

## 问题背景

语言模型处理离散 token，音频是连续的。如果你想要 LLM 风格的语音/音乐模型——MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus——首先需要一个**神经音频编解码器**：一个将音频离散化为小词表 token 的学习型编码器，以及一个重建波形的匹配解码器。

已出现两大家族：

1. **重建优先的编解码器**——EnCodec、DAC。优化感知音频质量。Token 是"声学"的——捕获一切包括说话人身份、音色、背景噪声。
2. **语义优先的编解码器**——Mimi（Kyutai）、SpeechTokenizer。强制首码本编码语言/音素内容（通常通过从 WavLM 蒸馏）。后续码本是声学细节。

2024–2026 年的洞见：**纯重建编解码器在从文字生成时会产生模糊的语音**。在编解码器 token 上训练的 LLM 必须在同一码本中同时学习语言结构和声学结构，这无法规模化。将两者分离——语义码本 0，声学码本 1-N——才是 Moshi 和 Sesame CSM 能工作的原因。

## 核心概念

### 核心技巧：残差向量量化（RVQ）

所有现代音频编解码器都不用一个大码本（需要数百万个码才能保证质量），而是使用 **RVQ**：小码本的级联。第一个码本量化编码器输出；第二个量化残差；依此类推。每个码本有 1024 个码。8 个码本 = 有效词表大小 1024^8 = 10^24。

推理时，解码器对每帧所有选定的码求和来重建。

### 2026 年重要的四个编解码器

**EnCodec（Meta，2022）**。基线方案。在波形上做编码-解码，RVQ 瓶颈。24 kHz，最多 32 个码本，默认 4 个码本 @ 1.5 kbps。使用 `1D 卷积 + Transformer + 1D 卷积` 架构。MusicGen 使用此编解码器。

**DAC（Descript，2023）**。L2 归一化码本的 RVQ，周期激活函数，改进的损失函数。所有开源编解码器中重建保真度最高——12 个码本下有时与原始语音难以区分。支持 44.1 kHz 全频段。

**SNAC（Hubert Siuzdak，2024）**。多尺度 RVQ——粗码本以比细码本更低的帧率运行。有效地对音频进行层次化建模：约 12 Hz 的粗略"草图"加上 50 Hz 的细节。被 Orpheus-3B 采用，因为层次结构非常适合基于 LM 的生成。

**Mimi（Kyutai，2024）**。2026 年的游戏规则改变者。12.5 Hz 帧率（极低），8 个码本 @ 4.4 kbps。码本 0 **从 WavLM 蒸馏**——训练为预测 WavLM 的语音内容特征。码本 1-7 是声学残差。这种分离驱动了 Moshi（第15课）和 Sesame CSM。

### 帧率对语言建模至关重要

帧率越低 = 序列越短 = LM 越快。

| 编解码器 | 帧率 | 1 秒 = N 帧 | 适用场景 |
|---------|------|------------|---------|
| EnCodec-24k | 75 Hz | 75 | 音乐、通用音频 |
| DAC-44.1k | 86 Hz | 86 | 高保真音乐 |
| SNAC-24k（粗层） | ~12 Hz | 12 | AR-LM 高效 |
| Mimi | 12.5 Hz | 12.5 | 流式语音 |

在 12.5 Hz 下，一段 10 秒的话语只有 125 个编解码器帧——Transformer 可以轻松预测。

### 语义 token 与声学 token

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **语义 token（Mimi 中的码本 0）**。编码说了什么——音素、单词、内容。通过辅助预测损失从 WavLM 蒸馏。
- **声学 token（码本 1-7）**。编码音色、说话人身份、韵律、背景噪声、精细细节。

AR LM 先预测语义 token（以文字为条件），再预测声学 token（以语义 + 说话人参考为条件）。这种因式分解是现代 TTS 能零样本克隆声音的原因：语义模型处理内容，声学模型处理音色。

### 2026 年重建质量（每秒比特，码率越低越好）

| 编解码器 | 码率 | PESQ | ViSQOL |
|---------|------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

Opus 等传统编解码器在每比特感知质量上仍然领先。神经编解码器在**离散 token**（Opus 不产生）和**生成模型质量**（LM 用这些 token 能做什么）上胜出。

## 动手实现

### 第一步：用 EnCodec 编码

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

6 kbps 下 `n_codebooks=8`。每个码是 0–1023（10 位）。

### 第二步：解码并测量重建质量

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### 第三步：语义-声学分离（Mimi 风格）

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

语义码本 0 与 WavLM 对齐。你可以训练一个文字到语义的 Transformer——词表比直接面向音频小得多。然后一个独立的声学到波形解码器以说话人参考为条件。

### 第四步：为什么 AR LM 在编解码器 token 上有效

对于 Mimi 12.5 Hz × 8 码本的 10 秒语音片段：

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 个 token 对 Transformer 来说是微不足道的上下文。一个 2.56 亿参数的 Transformer 在现代 GPU 上可以在几毫秒内生成 10 秒的语音。

## 工程应用

问题到编解码器的映射：

| 任务 | 编解码器 |
|------|---------|
| 通用音乐生成 | EnCodec-24k |
| 最高保真度重建 | DAC-44.1k |
| 语音上的 AR LM（TTS） | SNAC 或 Mimi |
| 流式全双工语音 | Mimi（12.5 Hz） |
| 带文字的音效库 | EnCodec + T5 条件 |
| 精细音频编辑 | DAC + 局部修复 |

经验规则：**如果在构建生成模型，从 Mimi 或 SNAC 开始。如果在构建压缩流水线，使用 Opus。**

## 常见陷阱

- **码本太多**。增加码本线性提升保真度，但 LM 序列长度也线性增加。控制在 8-12 个。
- **帧率不匹配**。在 12.5 Hz Mimi 上训练 LM，然后在 50 Hz EnCodec 上微调，会静默失败。
- **假设所有码本同等重要**。在 Mimi 中，码本 0 携带内容；丢失它会破坏可懂度。丢失码本 7 几乎无感知差异。
- **只用重建质量作为指标**。编解码器可以有出色的重建质量，但如果语义结构很差，对 LM 生成毫无用处。

## 交付物

保存为 `outputs/skill-codec-picker.md`。为给定的生成或压缩任务选择编解码器。

## 练习

1. **（简单）** 运行 `code/main.py`，实现一个玩具标量+残差量化器，随码本数量增加测量重建误差。
2. **（中等）** 安装 `encodec`，在留出语音片段上比较 1、4、8、32 个码本，绘制 PESQ 或 MSE vs 码率曲线。
3. **（困难）** 加载 Mimi，编码一段片段，将码本 0 替换为随机整数后解码；再将码本 7 同样替换后解码，比较两种损坏——码本 0 损坏应该破坏可懂度，码本 7 损坏几乎没有变化。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| RVQ | "残差量化" | 小码本的级联；每个码本量化前一个的残差 |
| 帧率 | "编解码器速度" | 每秒产生多少 token 帧，越低 LM 越快 |
| 语义码本 | "Mimi 的码本 0" | 从 SSL 特征蒸馏的码本，编码内容 |
| 声学码本 | "其余所有码本" | 音色、韵律、噪声、精细细节 |
| PESQ / ViSQOL | "感知质量" | 与 MOS 相关的客观指标 |
| EnCodec | "Meta 编解码器" | RVQ 基线，MusicGen 使用 |
| Mimi | "Kyutai 编解码器" | 12.5 Hz 帧率，语义-声学分离，驱动 Moshi |

## 延伸阅读

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — RVQ 基线
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — 最高保真度开源编解码器
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — 多尺度 RVQ
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — 语义-声学分离，WavLM 蒸馏
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — 两阶段语义/声学范式
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 原始可流式 RVQ 编解码器

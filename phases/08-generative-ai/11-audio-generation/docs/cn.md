# 音频生成

> 音频是 16-48 kHz 的一维信号。一个五秒钟的片段有 8-24 万个采样点。没有 Transformer 能直接关注那么长的序列。2026 年每个生产音频模型的解决方案都是一样的：神经编解码器（Encodec、SoundStream、DAC）以 50-75 Hz 的速率将音频压缩为离散 token，然后由 Transformer 或扩散模型生成 token。

**类型：** 构建
**语言：** Python
**前置知识：** 第6阶段第02课（音频特征）、第6阶段第04课（语音识别）、第8阶段第06课（DDPM）
**预计时间：** 约45分钟

## 问题背景

三类音频生成任务：

1. **文字转语音（TTS）。** 给定文本，生成语音。清晰的语音是窄带的，具有强烈的音素结构——Transformer-over-tokens 能很好地解决。VALL-E（微软）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成。** 给定提示词（文本、旋律、和弦进行、流派），生成音乐。分布宽泛得多。MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音效 / 声音设计。** 给定提示词，生成环境音或拟音。AudioGen、AudioLDM 2、Stable Audio Open。

所有三类都运行在同一基础上：神经音频编解码器 + token 自回归或扩散生成器。

## 核心概念

![音频生成：编解码器 token + Transformer 或扩散](../assets/audio-generation.svg)

### 神经音频编解码器

Encodec（Meta，2022）、SoundStream（Google，2021）、Descript Audio Codec（DAC，2023）。卷积编码器将波形压缩为每时间步的向量；残差向量量化（RVQ）将每个向量转换为 K 个码本索引的级联。解码器反转这个过程。24 kHz 音频以 2 kbps 使用 8 个 RVQ 码本，75 Hz = 600 个 token/秒。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 两种生成范式

**Token 自回归。** 将 RVQ token 展平为序列，运行仅解码器 Transformer。MusicGen 使用"延迟并行"策略，以每流偏移量并行发出 K 个码本流。VALL-E 从文本提示词 + 3 秒语音样本生成语音 token。

**潜在扩散。** 将编解码器 token 打包为连续潜变量，或用分类扩散建模。Stable Audio 2.5 在连续音频潜变量上使用流匹配。AudioLDM 2 使用文本→mel→音频的扩散链。

2024—2026 年的趋势：流匹配在音乐生成中胜出（推理更快，样本更干净），而 token 自回归在语音中仍占主导，因为它天然是因果的，流式传输效果好。

## 生产全景

| 系统 | 任务 | 骨干 | 延迟 |
|------|------|------|------|
| ElevenLabs V3 | TTS | Token-AR + 神经声码器 | 首 token 约 300ms |
| OpenAI GPT-4o audio | 全双工语音 | 端到端多模态 AR | 约 200ms |
| NaturalSpeech 3 | TTS | 潜在流匹配 | 非流式 |
| Stable Audio 2.5 | 音乐 / 音效 | DiT + 音频潜变量流匹配 | 1 分钟片段约 10s |
| Suno v4 | 完整歌曲 | 未公开；疑似 Token-AR | 每首歌约 30s |
| Udio v1.5 | 完整歌曲 | 未公开 | 每首歌约 30s |
| MusicGen 3.3B | 音乐 | Encodec 32kHz 上的 Token-AR | 实时 |
| AudioCraft 2 | 音乐 + 音效 | 流匹配 | 5s 片段约 5s |
| Riffusion v2 | 音乐 | 频谱图扩散 | 约 10s |

## 动手实现

`code/main.py` 模拟了核心思想：在从两种截然不同"风格"生成的合成"音频 token"序列上训练一个小型 next-token Transformer（风格 A 是交替的低高 token，风格 B 是单调递增）。以风格为条件并进行采样。

### 第一步：合成音频 token

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### 第二步：训练小型 token 预测器

以风格为条件的二元组风格预测器。重点是模式：编解码器 token → 交叉熵训练 → 自回归采样。

### 第三步：条件采样

给定风格 token 和起始 token，从预测分布中采样下一个 token。持续 20-40 个 token。

## 常见陷阱

- **编解码器质量限制输出质量。** 如果编解码器无法忠实表示某种声音，再好的生成器质量也无济于事。DAC 是当前开放的最佳选择。
- **RVQ 误差累积。** 每个 RVQ 层建模上一层的残差。第 1 层的误差会传播。在高层上使用温度 0 采样有帮助。
- **音乐结构。** 30 秒的 token 在 75 Hz 下超过 2 万个 token。对 Transformer 来说很难。MusicGen 使用滑动窗口 + 提示词延续；Stable Audio 使用较短的片段 + 交叉淡化。
- **边界处的伪影。** 生成片段之间的交叉淡化需要仔细的重叠相加。
- **对干净数据的渴求。** 音乐生成器需要数万小时的授权音乐。Suno / Udio 的 RIAA 诉讼（2024）将这一问题带到了表面。
- **声音克隆伦理。** 3 秒的样本加上文本提示词就足以让 VALL-E / XTTS / ElevenLabs 克隆一个声音。每个生产模型都需要滥用检测 + 退出名单。

## 工程应用

| 任务 | 2026 年的技术栈 |
|------|----------------|
| 商业 TTS | ElevenLabs、OpenAI TTS 或 Azure Neural |
| 声音克隆（经过同意验证） | XTTS v2（开放）或 ElevenLabs Pro |
| 背景音乐，快速 | Stable Audio 2.5 API、Suno 或 Udio |
| 带歌词的音乐 | Suno v4 或 Udio v1.5 |
| 音效 / 拟音 | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| 实时语音助手 | GPT-4o realtime 或 Gemini Live |
| 开放权重音乐研究 | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音 / 翻译 | HeyGen、ElevenLabs Dubbing |

## 交付物

见 `outputs/skill-audio-brief.md`。该技能接受音频简报（任务、时长、风格、声音、授权），输出：模型 + 托管服务、提示词格式（流派标签、风格描述符、结构标记）、编解码器 + 生成器 + 声码器链、种子协议，以及评估方案（MOS / CLAP 分数 / TTS 的 CER / 用户 A/B 测试）。

## 练习

1. **（简单）** 运行 `code/main.py` 并显式设置风格。验证生成的序列是否符合该风格的模式。
2. **（中等）** 添加延迟并行解码：模拟 2 条 token 流，必须保持偏移 1 步。训练一个联合预测器。
3. **（困难）** 使用 HuggingFace transformers 在本地 GPU 上运行 MusicGen-small。用三个不同的提示词生成 10 秒片段；对风格遵循度进行 A/B 测试。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 编解码器 (Codec) | "神经压缩" | 音频的编码器/解码器；典型输出是 50-75 Hz 的 token。 |
| RVQ | "残差向量量化" | K 个量化器的级联；每个建模前一个的残差。 |
| Token | "一个编解码符号" | 码本中的离散索引；通常 1024 或 2048 个。 |
| 延迟并行 (Delayed parallel) | "偏移码本" | 以交错偏移量发出 K 条 token 流以减少序列长度。 |
| 流匹配 (Flow matching) | "2024 年音频的胜者" | 路径更直的扩散替代方案；采样更快。 |
| 声音提示词 (Voice prompt) | "3 秒样本" | 引导克隆声音的说话人嵌入或 token 前缀。 |
| Mel 频谱图 (Mel spectrogram) | "那个视觉表示" | 对数幅度感知频谱图；许多 TTS 系统使用。 |
| 声码器 (Vocoder) | "Mel 到波形" | 将 mel 频谱图转换回音频的神经组件。 |

## 生产说明：音频是流式传输问题

音频是用户期望**随着生成过程**到达的唯一输出模态，而不是一次全部到达。用生产术语来说，这意味着 TPOT（每输出 token 时间）很重要，因为用户的听音速度是目标吞吐量——而不是阅读速度。对于以约 75 token/秒（Encodec）在 16kHz 编码的音频，服务器必须每用户每秒生成 ≥75 个 token 才能保持播放流畅。

两个架构后果：

- **流匹配音频模型无法轻易流式传输。** Stable Audio 2.5 和 AudioCraft 2 一次性渲染固定时长的片段。要流式传输，你需要分块处理片段并重叠边界——类似滑动窗口扩散——与编解码器 AR 模型相比增加 100-300ms 的延迟开销。

如果产品是"实时语音对话"或"实时音乐延续"，选择编解码器 AR 路径。如果是"提交后渲染 30 秒片段"，流匹配在质量和总延迟上胜出。

## 延伸阅读

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — 编解码器标准
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 第一个广泛使用的神经音频编解码器
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 2025 年带流匹配的文生音乐

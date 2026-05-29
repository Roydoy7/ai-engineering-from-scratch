# 文字转语音（TTS）——从 Tacotron 到 F5 和 Kokoro

> ASR 把语音转为文字，TTS 把文字转为语音。2026 年的技术栈分三步：文字 → 标记，标记 → Mel 频谱，Mel 频谱 → 波形。每一步都有一个可以跑在笔记本上的默认模型。

**类型：** 构建
**语言：** Python
**前置知识：** 第6阶段第2课（频谱图与 Mel）、第5阶段第9课（Seq2Seq）、第7阶段第5课（完整 Transformer）
**预计时间：** 约75分钟

## 问题背景

你有一个字符串："Please remind me to water the plants at 6 pm."你需要一段 3 秒的自然音频，具有正确的韵律（停顿、重音），正确发出"plants"的元音，还要在 CPU 上 300 ms 内完成，用于实时语音助手。你还需要切换声音，处理代码混合输入（"remind me at 6 pm, daijoubu?"），以及准确发音人名。

现代 TTS 流水线如下：

1. **文本前端**。归一化文本（日期、数字、电子邮件），转为音素或子词标记，预测韵律特征。
2. **声学模型**。文本 → Mel 频谱图。Tacotron 2（2017）、FastSpeech 2（2020）、VITS（2021）、F5-TTS（2024）、Kokoro（2024）。
3. **声码器**。Mel 频谱 → 波形。WaveNet（2016）、WaveRNN、HiFi-GAN（2020）、BigVGAN（2022）、2024 年以后的神经编解码声码器。

到 2026 年，端到端扩散和流匹配模型模糊了声学模型和声码器的边界，但三部分的心智模型在调试时仍然适用。

## 核心概念

**Tacotron 2（2017）**。Seq2seq：字符嵌入 → BiLSTM 编码器 → 位置敏感注意力 → 自回归 LSTM 解码器生成 Mel 帧。速度慢（自回归），在长文本上不稳定。仍被作为基线引用。

**FastSpeech 2（2020）**。非自回归。时长预测器输出每个音素对应几帧 Mel。一次前向，比 Tacotron 快 10 倍。失去了一些自然度（单调对齐），但被广泛部署。

**VITS（2021）**。端到端联合训练编码器 + 基于流的时长 + HiFi-GAN 声码器，使用变分推断。质量高，单一模型。2022–2024 年主流开源 TTS。变体：YourTTS（多说话人零样本）、XTTS v2（2024，Coqui）。

**F5-TTS（2024）**。基于流匹配的扩散 Transformer。自然韵律，仅需 5 秒参考音频即可零样本克隆声音。2026 年开源 TTS 排行榜榜首。3.35 亿参数。

**Kokoro（2024）**。小型模型（8200 万参数），可在 CPU 上运行，实时场景下最佳英语 TTS。仅支持英语闭集词汇，Apache 2.0 协议。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3**。商业 SOTA。ElevenLabs v2.5 的情感标签（"[whispered]"、"[laughing]"）和角色声音在 2026 年主导有声书制作。

### 声码器演进

| 时代 | 声码器 | 延迟 | 质量 |
|------|--------|------|------|
| 2016 | WaveNet | 仅离线 | 发布时 SOTA |
| 2018 | WaveRNN | 约实时 | 良好 |
| 2020 | HiFi-GAN | 100× 实时 | 近人类水平 |
| 2022 | BigVGAN | 50× 实时 | 跨说话人/语言泛化 |
| 2024 | SNAC、DAC（神经编解码器） | 与 AR 模型集成 | 离散 token，比特效率高 |

到 2026 年，大多数"TTS"模型都是从文字到波形的端到端模型，Mel 频谱图是内部表示。

### 评估指标

- **MOS（平均主观意见分）**。1–5 分，众包打分。仍是金标准，但速度极慢。
- **CMOS（比较 MOS）**。A vs B 偏好对比，每次标注的置信区间更窄。
- **UTMOS、DNSMOS**。无参考神经 MOS 预测器，用于排行榜。
- **CER（字符错误率）via ASR**。将 TTS 输出送入 Whisper，计算相对输入文本的 CER，作为可懂度代理指标。
- **SECS（说话人嵌入余弦相似度）**。声音克隆质量评估。

2026 年 LibriTTS test-clean 上的数字：

| 模型 | UTMOS | CER（via Whisper） | 规模 |
|------|-------|-------------------|------|
| 真实录音 | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 3.35 亿 |
| XTTS v2 | 3.81 | 3.5% | 4.70 亿 |
| VITS | 3.62 | 3.1% | 2500 万 |
| Kokoro v0.19 | 3.87 | 1.8% | 8200 万 |
| Parler-TTS Large | 3.76 | 2.8% | 23 亿 |

## 动手实现

### 第一步：音素化输入

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素是通用桥梁。低于 VITS 质量级别时，避免直接输入原始文字。

### 第二步：运行 Kokoro（2026 年 CPU 默认方案）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

离线运行，单文件，8200 万参数。

### 第三步：运行 F5-TTS 进行声音克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

传入 5 秒参考片段及其文本，F5 克隆韵律和音色。

### 第四步：从零实现 HiFi-GAN 声码器

太大无法放入教程脚本，但基本结构如下：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

训练：对抗损失（短窗口判别器）+ Mel 频谱重建损失 + 特征匹配损失。已商品化——使用 `hifi-gan` 仓库或 nvidia-NeMo 的预训练检查点。

### 第五步：完整流水线（伪代码）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 工程应用

2026 年技术栈：

| 场景 | 选型 |
|------|------|
| 实时英语语音助手 | Kokoro（CPU）或 XTTS v2（GPU） |
| 从 5 秒参考克隆声音 | F5-TTS |
| 商业角色声音 | ElevenLabs v2.5 |
| 有声书配音 | ElevenLabs v2.5 或 XTTS v2 + 微调 |
| 低资源语言 | 在 5–20 小时目标语言数据上训练 VITS |
| 表情/情感标签 | ElevenLabs v2.5 或 StyleTTS 2 微调 |

2026 年开源领头羊：**质量优先选 F5-TTS，效率优先选 Kokoro**。除非你是历史学家，否则不要碰 Tacotron。

## 常见陷阱

- **没有文本归一化器**。"Dr. Smith"读成"Doctor"还是"Drive"？"2026"读"twenty twenty six"还是"two zero two six"？在音素化*之前*先归一化。
- **词表外专有名词**。"Ghumare" → "ghyu-mair"？为未知标记部署备用的字素到音素模型。
- **削波**。声码器输出很少削波，但推理时 Mel 缩放不匹配可能超出 ±1.0。始终做 `np.clip(wav, -1, 1)`。
- **采样率不匹配**。Kokoro 输出 24 kHz，下游流水线期望 16 kHz → 重采样，否则会产生混叠。

## 交付物

保存为 `outputs/skill-tts-designer.md`。为给定的声音、延迟和语言目标设计 TTS 流水线。

## 练习

1. **（简单）** 运行 `code/main.py`，从玩具词汇构建音素词典，估计每个音素的时长，打印模拟的"Mel 调度计划"。
2. **（中等）** 安装 Kokoro，用 `af_bella` 和 `am_adam` 声音合成同一句话，比较音频时长和主观质量。
3. **（困难）** 录制 5 秒自己声音的参考片段，用 F5-TTS 克隆，报告参考和克隆输出之间的 SECS 值。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 音素 (Phoneme) | "声音单元" | 抽象声音类别；英语有 39 个（ARPABet） |
| 时长预测器 | "每个音素持续多长" | 非自回归模型输出，每个音素对应的帧数 |
| 声码器 (Vocoder) | "Mel 转波形" | 将 Mel 频谱映射到原始采样的神经网络 |
| HiFi-GAN | "标准声码器" | 基于 GAN，2020–2024 年主流 |
| MOS | "主观质量分" | 人工评分员给出的 1–5 分均值 |
| SECS | "声音克隆指标" | 目标说话人嵌入与输出之间的余弦相似度 |
| F5-TTS | "2024 年开源 SOTA" | 流匹配扩散，零样本克隆 |
| Kokoro | "CPU 英语最优" | 8200 万参数，Apache 2.0 协议 |

## 延伸阅读

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — Seq2seq 基线
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — 端到端基于流的 TTS
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 当前开源 SOTA
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 2026 年仍在使用的声码器
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 年 CPU 友好型英语 TTS

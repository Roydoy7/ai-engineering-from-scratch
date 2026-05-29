# 音频基础——波形、采样与傅里叶变换

> 波形是原始信号，频谱图是表示形式，Mel 特征是机器学习友好的形式。每个现代 ASR 和 TTS 流水线都要走这个梯子，而第一级就是理解采样和傅里叶变换。

**类型：** 学习
**语言：** Python
**前置知识：** 第1阶段第6课（向量与矩阵）、第1阶段第14课（概率分布）
**预计时间：** 约45分钟

## 问题背景

麦克风产生一个随时间变化的声压信号，而神经网络消费的是张量。两者之间夹着一堆惯例，一旦违反就会产生静默 bug：模型训练看起来正常，但词错误率翻倍；TTS 产出一片杂音；语音克隆系统记住了麦克风而不是说话人。

语音系统中的每个 bug 都可以追溯到以下三个问题之一：

1. 数据录制时的采样率是多少？模型期望的是什么？
2. 信号是否发生了混叠（aliasing）？
3. 你在操作的是原始采样值，还是频域表示？

搞清楚这三个问题，第6阶段的其余内容就顺理成章了。搞不清楚，即使是 Whisper-Large-v4 也会产出垃圾。

## 核心概念

**波形（Waveform）**：`[-1.0, 1.0]` 范围内的一维浮点数组，按采样编号索引。换算成秒：`t = n / sr`。16 kHz 下录制的 10 秒音频是一个 160000 个浮点数的数组。

**采样率（sr）**：每秒的采样数。2026 年的常见采样率：

| 采样率 | 用途 |
|--------|------|
| 8 kHz | 电话、传统 VOIP。奈奎斯特频率 4 kHz 会砍掉辅音。避免用于 ASR。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 都消费 16 kHz。 |
| 22.05 kHz | 旧版模型的 TTS 声码器训练。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音频、音乐。 |
| 48 kHz | 影视、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**奈奎斯特-香农定理（Nyquist-Shannon）**：采样率 `sr` 能无歧义表示的最高频率是 `sr/2`，这个边界叫*奈奎斯特频率*。超过奈奎斯特频率的能量会发生*混叠*——折叠到低频区——污染信号。在下采样前始终要进行低通滤波。

**位深（Bit depth）**：16 位 PCM（有符号 int16，范围 ±32767）是通用交换格式。音乐用 24 位，内部 DSP 用 32 位浮点。`soundfile` 等库读取 int16 但以 `[-1, 1]` 范围的 float32 数组形式暴露给用户。

**傅里叶变换（Fourier Transform）**：任何有限信号都是不同频率的正弦波之和。离散傅里叶变换（DFT）对 `N` 个采样点计算 `N` 个复数系数——每个频率 bin 一个。bin k 对应频率 `k · sr / N` Hz，幅度是该频率的振幅，相角是相位。

**FFT（快速傅里叶变换）**：当 `N` 是 2 的幂时，DFT 的 `O(N log N)` 算法。所有音频库在底层都使用 FFT。16 kHz 下 1024 点 FFT 给出 512 个可用频率 bin，覆盖 0-8 kHz，分辨率 15.6 Hz。

**分帧与加窗（Framing + Window）**：我们不对整段音频做 FFT。而是把它切成重叠的*帧*（通常 25 ms 长，10 ms 帧移），对每帧乘以窗函数（汉宁窗、海明窗）以消除边界不连续，再对每帧做 FFT。这就是短时傅里叶变换（STFT）。第2课从这里继续展开。

## 动手实现

### 第一步：读取音频并绘制波形

`code/main.py` 只使用标准库 `wave` 模块以保持无外部依赖。生产代码用 `soundfile` 或 `torchaudio.load`（两者都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 第二步：从第一性原理合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

16 kHz 下 1 秒的 440 Hz 正弦波（音乐会 A 音）是 16000 个浮点数。用 `wave.open(..., "wb")` 以 16 位 PCM 编码写入。

### 第三步：手动计算 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` 复杂度——`N=256` 时足以验证正确性，真实音频用不了。实际代码调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 第四步：找到主频率

幅度峰值的 bin 索引 `k_star` 对应频率 `k_star * sr / N`。对 440 Hz 正弦波运行此步骤，应该在 bin `440 * N / sr` 处看到峰值。

### 第五步：演示混叠

以 10 kHz 采样率（奈奎斯特 = 5 kHz）采样一个 7 kHz 正弦波。7 kHz 高于奈奎斯特，折叠到 `10 − 7 = 3 kHz`。FFT 峰值出现在 3 kHz 处。这是经典的混叠演示，也是每个 DAC/ADC 都要配备砖墙低通滤波器的原因。

## 工程应用

2026 年实际生产技术栈：

| 任务 | 库 | 原因 |
|------|-----|------|
| 读写 WAV/FLAC/OGG | `soundfile`（libsndfile 封装） | 最快、稳定，返回 float32 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠 |
| STFT / Mel | `torchaudio` 或 `librosa` | GPU 友好，融入 PyTorch 生态 |
| 实时流处理 | `sounddevice` 或 `pyaudio` | 跨平台 PortAudio 绑定 |
| 检查文件信息 | `ffprobe` 或 `soxi` | 命令行工具，快速报告 sr/声道/编解码器 |

决策规则：**先匹配采样率，再考虑其他一切**。Whisper 期望 16 kHz 单声道 float32。传入 44.1 kHz 立体声，你会得到看起来像模型 bug 的垃圾输出。

## 交付物

保存为 `outputs/skill-audio-loader.md`。该技能帮助你检查音频输入是否匹配下游模型的期望，并在不匹配时正确地重采样。

## 练习

1. **（简单）** 在 16 kHz 下合成 220 Hz + 440 Hz + 880 Hz 的 1 秒混合信号，运行 DFT，确认三个峰值出现在预期的 bin 处。
2. **（中等）** 以 48 kHz 录制 3 秒人声 WAV，分别用 `torchaudio.transforms.Resample`（带抗混叠）和朴素抽取（每三个样本取一个）下采样到 16 kHz，对两者做 FFT，混叠出现在哪里？
3. **（困难）** 只用 `math` 和第三步的 DFT 从零实现 STFT：帧大小 400，帧移 160，汉宁窗。用 `matplotlib.pyplot.imshow` 绘制幅度图，这就是第2课的频谱图。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 采样率 (Sample rate) | "每秒多少个采样" | ADC 测量信号的频率（Hz） |
| 奈奎斯特 (Nyquist) | "最大可表示频率" | `sr/2`；超过这个频率的能量会混叠 |
| 位深 (Bit depth) | "每个采样的精度" | `int16` = 65536 个级别；`float32` = `[-1, 1]` 中的 24 位精度 |
| DFT | "序列的傅里叶变换" | `N` 个采样 → `N` 个复数频率系数 |
| FFT | "快速 DFT" | `O(N log N)` 算法，要求 `N` 为 2 的幂 |
| Bin（频率格） | "频率列" | `k · sr / N` Hz；分辨率 = `sr / N` |
| STFT | "频谱图的底层" | 随时间的分帧 + 加窗 FFT |
| 混叠 (Aliasing) | "奇怪的频率鬼影" | 超过奈奎斯特的能量折叠到低频 bin |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 采样定理背后的论文
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 免费权威 DSP 教科书
- [librosa 文档 — 音频入门](https://librosa.org/doc/latest/tutorial.html) — 带代码的实践教程
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10 分钟彻底理解频率 bin

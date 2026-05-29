# 频谱图、Mel 尺度与音频特征

> 神经网络消化不好原始波形，但能很好地消化频谱图，Mel 频谱图更是如此。2026 年每个 ASR、TTS 和音频分类器的成败都取决于这一个预处理选择。

**类型：** 构建
**语言：** Python
**前置知识：** 第6阶段第1课（音频基础）
**预计时间：** 约45分钟

## 问题背景

取一段 16 kHz 的 10 秒音频，那是 160000 个 `[-1, 1]` 范围内的浮点数，与"狗叫"或"单词 cat"这样的标签几乎毫无相关性。原始波形包含信息，但以一种模型很难提取的形式存在——相差 100 ms 的两个相同音素，其原始采样点完全不同。

频谱图解决了这个问题。它在人类感知会忽略的时域细节上做压缩（微秒级抖动），同时保留了感知关注的结构（哪些频率有能量，在约 10-25 ms 的时间窗口内）。

Mel 频谱图进一步优化。人类对音高的感知是对数的：100 Hz vs 200 Hz 听起来和 1000 Hz vs 2000 Hz "距离一样远"。Mel 尺度对频率轴做了相应的变换。从 2010 到 2026，Mel 频谱图是语音机器学习中最重要的单一特征。

## 核心概念

**STFT（短时傅里叶变换）**：将波形切成重叠的帧（典型设置：25 ms 窗口，10 ms 帧移，在 16 kHz 下分别是 400 和 160 个采样点），对每帧乘以窗函数（汉宁窗是默认选择，海明窗权衡略有不同），对每帧做 FFT，把幅度频谱叠成形状为 `(n_frames, n_freq_bins)` 的矩阵，这就是频谱图。

**对数幅度（Log-magnitude）**：原始幅度跨越 5-6 个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 来压缩动态范围。所有生产流水线都使用对数幅度，而非原始幅度。

**Mel 尺度**：频率 `f`（Hz）映射到 Mel 值 `m` 的公式：`m = 2595 * log10(1 + f / 700)`。这个映射在 1 kHz 以下近似线性，在 1 kHz 以上近似对数。覆盖 0-8 kHz 的 80 个 Mel bin 是 ASR 的标准输入。

**Mel 滤波器组（Mel filterbank）**：一组在 Mel 尺度上等间距的三角形滤波器。每个滤波器是相邻 FFT bin 的加权和。把 STFT 幅度乘以滤波器组矩阵，一次矩阵乘法就得到 Mel 频谱图。

**对数 Mel 频谱图（Log-mel spectrogram）**：`log(mel_spec + 1e-10)`。Whisper 的输入，Parakeet 的输入，SeamlessM4T 的输入——2026 年通用的音频前端。

**MFCC（梅尔频率倒谱系数）**：对对数 Mel 频谱图做 DCT II 型变换，取前 13 个系数。对特征做去相关并进一步压缩。在约 2015 年 CNN/Transformer 直接处理原始对数 Mel 追上之前，这是主导特征。在说话人识别（x-vector、ECAPA）中仍有使用。

**分辨率权衡**：更大的 FFT = 更好的频率分辨率，但时间分辨率更差。25 ms / 10 ms 是语音 ML 的默认值；音乐用 50 ms / 12.5 ms；瞬态检测（鼓击、爆破音）用 5 ms / 2 ms。

## 动手实现

### 第一步：分帧

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

10 秒 16 kHz 音频，`frame_len=400, hop=160` 得到 998 帧。

### 第二步：汉宁窗

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在做 FFT 前逐元素相乘。消除因在非零端点截断引起的频谱泄漏。

### 第三步：STFT 幅度

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产代码用 `torch.stft` 或 `librosa.stft`（基于 FFT，向量化）。这里的循环是教学用的，在 `code/main.py` 里对短片段运行。

### 第四步：Mel 滤波器组

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

覆盖 0-8 kHz 的 80 个 Mel bin，`n_fft=400`，得到 `(80, 201)` 矩阵。用 `(n_frames, 201)` 的 STFT 幅度乘以转置，得到 `(n_frames, 80)` 的 Mel 频谱图。

### 第五步：对数 Mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代：`librosa.power_to_db`（参考归一化 dB）、`10 * log10(power + eps)`。Whisper 用了更复杂的裁剪 + 归一化方案（见 Whisper 的 `log_mel_spectrogram`）。

### 第六步：MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每个对数 Mel 帧做 DCT，保留前 13 个系数，这就是 MFCC 矩阵。第一个系数通常被丢弃（它编码整体能量）。

## 工程应用

2026 年技术栈：

| 任务 | 特征 |
|------|------|
| ASR（Whisper、Parakeet、SeamlessM4T） | 80 对数 Mel，10 ms 帧移，25 ms 窗口 |
| TTS 声学模型（VITS、F5-TTS、Kokoro） | 80 Mel，5-12 ms 帧移，用于精细时间控制 |
| 音频分类（AST、PANNs、BEATs） | 128 对数 Mel，10 ms 帧移 |
| 说话人嵌入（ECAPA-TDNN、WavLM） | 80 对数 Mel 或原始波形 SSL |
| 音乐（MusicGen、Stable Audio 2） | EnCodec 离散 token（不用 Mel） |
| 关键词识别 | 40 MFCC，用于微型设备 |

经验规则：**如果不是做音乐，从 80 对数 Mel 开始。**任何偏离都需要充分理由。

## 2026 年仍在发货的陷阱

- **Mel 数量不匹配**。训练用 80 Mel，推理用 128 Mel。静默失败。在两端都记录特征形状。
- **上游采样率不匹配**。22.05 kHz 计算的 Mel 和 16 kHz 的看起来不一样。在特征提取*之前*修正采样率。
- **dB vs log**。Whisper 期望对数 Mel，不是 dB-Mel。部分 HF 流水线能自动检测，但你的自定义代码不会。
- **归一化漂移**。训练时按话语归一化，推理时全局归一化。会把词错误率翻倍的生产 bug。
- **padding 引起的泄漏**。对片段末尾做零填充，在末尾帧产生平坦频谱。改用对称填充或复制填充。

## 交付物

保存为 `outputs/skill-feature-extractor.md`。该技能根据给定的模型目标选择特征类型、Mel 数量、帧长/帧移和归一化方式。

## 练习

1. **（简单）** 运行 `code/main.py`，它合成一个扫频信号（200 → 4000 Hz）并打印每帧的 argmax Mel bin，绘图（可选）并确认与扫描一致。
2. **（中等）** 用 `n_mels` ∈ {40, 80, 128} 和 `frame_len` ∈ {200, 400, 800} 重新运行，测量时间轴上的尖峰带宽，哪种组合对扫频信号的分辨率最好？
3. **（困难）** 实现 `power_to_db`，在 AudioMNIST 上用小型 CNN 分类器对比 (a) 原始对数 Mel，(b) `ref=max` 的 dB-Mel，(c) MFCC-13 + 一阶差分 + 二阶差分，报告 top-1 准确率。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 帧 (Frame) | "一个切片" | 送入一次 FFT 的 25 ms 波形片段 |
| 帧移 (Hop) | "步长" | 相邻帧之间的采样间隔，ASR 默认 10 ms |
| 窗函数 (Window) | "汉宁/海明那个东西" | 逐元素的乘数，把帧两端的值渐变为零 |
| STFT | "频谱图生成器" | 分帧 + 加窗 FFT，产出时间 × 频率矩阵 |
| Mel | "扭曲的频率" | 对数感知尺度；`m = 2595·log10(1 + f/700)` |
| 滤波器组 (Filterbank) | "那个矩阵" | 把 STFT 投影到 Mel bin 的三角形滤波器 |
| 对数 Mel (Log-mel) | "Whisper 的输入" | `log(mel_spec + eps)`，2026 年的标准 |
| MFCC | "老派特征" | 对数 Mel 的 DCT，13 个系数，去相关 |

## 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC 论文
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 原始 Mel 尺度论文
- [OpenAI — Whisper 源码 log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 阅读参考实现
- [librosa 特征提取文档](https://librosa.org/doc/main/feature.html) — `mfcc`、`melspectrogram` 及帧/窗口参数的参考
- [NVIDIA NeMo — 音频预处理](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet + Canary 模型的生产级流水线

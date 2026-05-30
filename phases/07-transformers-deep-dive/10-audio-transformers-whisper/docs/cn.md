# 音频 Transformer——Whisper 架构

> 音频是频率随时间变化的图像。Whisper 是一个吃 Mel 频谱图并能说话的 ViT。

**类型：** 学习
**语言：** Python
**前置知识：** 第7阶段第05课（完整 Transformer）、第7阶段第08课（编码器-解码器）、第7阶段第09课（ViT）
**预计时间：** 约45分钟

## 问题背景

在 Whisper（OpenAI，Radford et al. 2022）之前，最先进的自动语音识别（ASR）意味着 wav2vec 2.0 和 HuBERT——自监督特征提取器加微调头。质量高，但数据流水线昂贵，领域脆弱。多语言语音识别需要每个语系单独的模型。

Whisper 下了三个赌注：

1. **在所有数据上训练。** 从互联网抓取 97 种语言共 68 万小时的弱标注音频。没有干净的学术语料，没有音素标签。
2. **多任务单模型。** 一个解码器，通过任务 token 联合训练转录、翻译、语音活动检测、语言识别和时间戳。
3. **标准编码器-解码器 Transformer。** 编码器处理对数 Mel 频谱图。解码器自回归地生成文字 token。没有声码器、没有 CTC、没有 HMM。

结果是：Whisper large-v3 在口音、噪声和零干净标注数据的语言上都表现鲁棒。2026 年它是每个开源语音助手和大多数商业产品的默认语音前端。

## 核心概念

### 第一步——重采样 + 窗口化

音频以 16 kHz 处理。截取/填充到 30 秒。计算对数 Mel 频谱图：80 个 Mel 频段，10 ms 步长 → 约 3000 帧 × 80 个特征。这就是 Whisper 看到的"输入图像"。

### 第二步——卷积主干

两个核大小为 3、步长为 2 的 Conv1D 层将 3000 帧减少到 1500 帧。在不增加太多参数的情况下将序列长度减半。

### 第三步——编码器

在 1500 个时间步上运行 24 层（large 版）Transformer 编码器。正弦位置编码、自注意力、GELU FFN。产生 1500 × 1280 的隐状态。

### 第四步——解码器

24 层 Transformer 解码器。从 BPE 词表中自回归地生成 token，该词表是 GPT-2 词表的超集，加入了几个音频专用特殊 token。

### 第五步——任务 token

解码器提示以控制 token 开头，告诉模型要做什么：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型在这套约定上训练。通过前缀控制任务。这是 2026 年指令微调的等价物，只是应用于语音。

### 第六步——输出

束搜索（宽度 5）配合对数概率阈值。当 `<|notimestamps|>` token 不存在时，每 0.02 秒的音频预测一次时间戳。

### Whisper 规格

| 模型 | 参数量 | 层数 | d_model | 头数 | VRAM（fp16） |
|------|--------|------|---------|------|------------|
| Tiny | 3900万 | 4 | 384 | 6 | ~1 GB |
| Base | 7400万 | 6 | 512 | 8 | ~1 GB |
| Small | 2.44亿 | 12 | 768 | 12 | ~2 GB |
| Medium | 7.69亿 | 24 | 1024 | 16 | ~5 GB |
| Large | 15.5亿 | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 15.5亿 | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 8.09亿 | 32 | 1280 | 20 | ~6 GB（4层解码器） |

Large-v3-turbo（2024）将解码器从 32 层削减到 4 层。解码速度提升 8 倍，WER 下降不到 1 个点。这一解码速度的提升是 Whisper-turbo 成为 2026 年实时语音 Agent 默认选择的原因。

### Whisper 不做什么

- 没有说话人分离（谁在说话）。结合 pyannote 使用。
- 原生不支持实时流式处理——30 秒窗口是固定的。现代包装器（`faster-whisper`、`WhisperX`）通过 VAD + 重叠来补充流式处理。
- 没有超过 30 秒的长程上下文，需要外部分块。实践中效果良好，因为转录很少需要长程上下文。

### 2026 年生态

| 任务 | 模型 | 说明 |
|------|------|------|
| 英语 ASR | Whisper-turbo、Moonshine | Moonshine 在边缘设备上快 4 倍 |
| 多语言 ASR | Whisper-large-v3 | 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 可实现 150 ms 延迟目标 |
| TTS | Piper、XTTS-v2、Kokoro | 编码器-解码器模式，但形如 Whisper |
| 音频 + 语言 | AudioLM、SeamlessM4T | 一个 Transformer 中的文字 token + 音频 token |

## 动手实现

见 `code/main.py`。我们不训练 Whisper——我们构建对数 Mel 频谱图流水线 + 任务 token 提示格式化器。这是你在生产中实际会接触的部分。

### 第一步：合成音频

生成一个以 16 kHz 采样的 440 Hz、持续 1 秒的正弦波。16000 个样本。

### 第二步：对数 Mel 频谱图（简化版）

完整的 Mel 频谱图需要 FFT。我们做一个简化的分帧 + 每帧能量版本，展示流水线而不需要 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧 = 25 ms，步长 = 10 ms。与 Whisper 的窗口化一致。每帧能量代替 Mel 频段用于教学演示。

### 第三步：填充到 30 秒

Whisper 始终处理 30 秒的块。将频谱图填充（或截取）到 3000 帧。

### 第四步：构建提示 token

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是整个任务控制接口。一个 4-token 前缀。

## 工程应用

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快的 OpenAI 兼容写法：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年何时选择 Whisper：**

- 用一个模型处理多语言 ASR
- 对嘈杂、多样化音频进行鲁棒转录
- 研究/原型 ASR——最快的起点

**何时选择其他方案：**

- 边缘设备上的超低延迟流式处理——Moonshine 在相同质量下优于 Whisper
- 需要 <200 ms 延迟的实时对话 AI——使用专用流式 ASR
- 说话人分离——Whisper 不做这个；搭配 pyannote 使用

## 交付物

见 `outputs/skill-asr-configurator.md`。该技能为新的语音应用选择 ASR 模型、解码参数和预处理流水线。

## 练习

1. **（简单）** 运行 `code/main.py`。确认以 10 ms 步长在 16 kHz 的 1 秒信号上帧数约为 100。30 秒：约 3000 帧。
2. **（中等）** 用 `numpy.fft` 构建完整的对数 Mel 频谱图。验证 80 个 Mel 频段在数值误差范围内与 `librosa.feature.melspectrogram(n_mels=80)` 一致。
3. **（困难）** 实现流式推理：将音频分成 10 秒窗口，重叠 2 秒，对每个窗口运行 Whisper，合并转录文本。在一个 5 分钟播客样本上测量词错误率，与单次处理对比。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| Mel 频谱图 (Mel spectrogram) | "音频图像" | 二维表示：一轴为频率频段，另一轴为时间帧；每个单元格为对数能量 |
| 对数 Mel (Log-mel) | "Whisper 看到的" | 经过对数变换的 Mel 频谱图；近似人类对响度的感知 |
| 帧 (Frame) | "一个时间切片" | 25 ms 的样本窗口；以 10 ms 步长重叠 |
| 任务 token (Task token) | "语音的提示前缀" | 解码器提示中的特殊 token，如 `<\|transcribe\|>` / `<\|translate\|>` |
| 语音活动检测 (VAD) | "找到语音" | 在 ASR 之前移除静音的门控；大幅降低成本 |
| CTC | "连接时序分类" | 用于无需对齐训练的经典 ASR 损失；Whisper 不使用它 |
| Whisper-turbo | "小解码器，完整编码器" | large-v3 编码器 + 4 层解码器；解码速度提升 8 倍 |
| Faster-whisper | "生产包装器" | CTranslate2 重新实现；int8 量化；比 OpenAI 参考实现快 4 倍 |

## 延伸阅读

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 论文
- [OpenAI Whisper repo](https://github.com/openai/whisper) — 参考代码 + 模型权重。阅读 `whisper/model.py`，约 400 行从头到尾展示 Conv1D 主干 + 编码器 + 解码器
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 第5–6步描述的束搜索 + 任务 token 逻辑；500 行，完全可读
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 前身；在某些设置下仍是最佳特征
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 生产包装器，比参考实现快 4 倍
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 年边缘友好 ASR，形如 Whisper 但更小
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — 典范微调方案
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现，镜像了本课的架构图

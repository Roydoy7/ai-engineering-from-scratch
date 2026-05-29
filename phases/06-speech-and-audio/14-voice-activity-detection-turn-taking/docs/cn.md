# 声音活动检测与轮次交替——Silero、Cobra 与冲洗技巧

> 每个语音 Agent 都依赖两个决策生存：用户现在在说话吗？他们说完了吗？VAD 回答第一个问题，轮次检测（VAD + 静默挂起 + 语义端点模型）回答第二个问题。任一出错，你的助手要么打断用户，要么永远不闭嘴。

**类型：** 构建
**语言：** Python
**前置知识：** 第6阶段第11课（实时音频处理）、第6阶段第12课（语音助手）
**预计时间：** 约45分钟

## 问题背景

语音 Agent 对每个 20 ms 块做出的三个不同决策：

1. **这帧是语音吗？** — VAD，逐帧二值判断。
2. **用户开始了一个新话语吗？** — 起始检测。
3. **用户说完了吗？** — 端点检测（轮次结束）。

天真的方案（能量阈值）在任何噪声下都会失效——车流声、键盘声、人群喧嚣。2026 年的方案：Silero VAD（开源，深度学习）+ 轮次检测模型（语义端点）+ VAD 校准的静默挂起。

## 核心概念

### 三层 VAD 级联

**第一层：能量门控**。最廉价。RMS 阈值设为 -40 dBFS。过滤明显的静默，但对超过阈值的任何噪声都会触发。

**第二层：Silero VAD**（2020–2026，MIT）。100 万参数，在 6000+ 种语言上训练。单 CPU 线程每 30 ms 块约 1 ms 运行时间。5% FPR 下 TPR 87.7%。开源默认方案。

**第三层：语义轮次检测器**。LiveKit 的轮次检测模型（2024–2026）或自定义小型分类器。区分"句子中间停顿"和"说完了"。使用语言上下文（语调 + 近期词汇），而非仅靠静默。

### 关键参数及其默认值

- **阈值**。Silero 输出概率；在 > 0.5（默认）或 > 0.3（敏感）时判定为语音。阈值越低 → 首词截断越少，误报越多。
- **最短语音时长**。拒绝短于 250 ms 的语音——通常是咳嗽或椅子噪声。
- **静默挂起（端点检测）**。VAD 返回 0 后，等待 500–800 ms 再宣告轮次结束。太短 → 打断用户；太长 → 感觉迟钝。
- **预滚缓冲区**。在 VAD 触发前保留 300–500 ms 的音频，防止"嘿"被截断。

### 冲洗技巧（Kyutai，2025）

流式 STT 模型有预看延迟（Kyutai STT-1B 为 500 ms，STT-2.6B 为 2.5 s）。通常你需要在话语结束后等这么长时间才能得到文字。冲洗技巧：VAD 检测到话语结束时，**向 STT 发送冲洗信号**，强制立即输出。STT 以约 4 倍实时速度处理，因此 500 ms 的缓冲区在约 125 ms 内处理完成。

端到端：125 ms VAD + 冲洗 STT = 对话级延迟。

### 2026 年 VAD 对比

| VAD | 5% FPR 下的 TPR | 延迟 | 协议 |
|-----|----------------|------|------|
| WebRTC VAD（Google，2013） | 50.0% | 30 ms | BSD |
| Silero VAD（2020–2026） | 87.7% | ~1 ms | MIT |
| Cobra VAD（Picovoice） | 98.9% | ~1 ms | 商业 |
| pyannote segmentation | 95% | ~10 ms | 类 MIT |

Silero 是正确的默认选择。Cobra 是合规/精度升级版。纯能量 VAD 在 2026 年的生产中没有立足之地。

## 动手实现

### 第一步：能量门控

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 第二步：Python 中的 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 第三步：轮次结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 第四步：冲洗技巧骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持冲洗才能实现这个效果。Whisper streaming 不支持——它是基于块的，总是等待完整块。

## 工程应用

| 场景 | VAD 选型 |
|------|---------|
| 开源、快速、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 设备端（手机） | Silero VAD ONNX |
| 研究/说话人分离 | pyannote segmentation |
| 零依赖回退 | WebRTC VAD（遗留） |
| 需要高质量轮次结束检测 | Silero + LiveKit 轮次检测器叠加 |

经验规则：除非真的没有其他选择，否则绝不要只用能量 VAD 发货。

## 常见陷阱

- **固定阈值**。安静时工作，嘈杂时失败。要么在设备上校准，要么切换到 Silero。
- **静默挂起太短**。Agent 在句子中间打断用户。500–800 ms 是对话语音的甜点。
- **挂起太长**。感觉迟钝。对目标用户进行 A/B 测试。
- **没有预滚缓冲区**。用户音频的前 200–300 ms 丢失。始终保持滚动预滚缓冲。
- **忽视语义端点检测**。"嗯，让我想想……"包含长停顿。用户不喜欢在思考中途被打断。使用 LiveKit 的轮次检测器或类似方案。

## 交付物

保存为 `outputs/skill-vad-tuner.md`。为给定工作负载选择 VAD 模型、阈值、挂起时长、预滚和轮次检测策略。

## 练习

1. **（简单）** 运行 `code/main.py`，模拟"语音 + 静默 + 语音 + 咳嗽"序列，测试三层 VAD。
2. **（中等）** 安装 `silero-vad`，处理 5 分钟录音，调整阈值以最小化首词截断和误触发，报告精确率/召回率。
3. **（困难）** 构建一个迷你轮次检测器：Silero VAD + 基于最后 10 个词嵌入（使用 sentence-transformers）的 3 层 MLP，在手工标注的轮次结束数据集上训练，F1 比单纯 Silero 提升 10%。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| VAD | "声音检测器" | 逐帧二值判断：这是语音吗？ |
| 轮次检测 | "端点检测" | VAD + 静默挂起 + 语义端点 |
| 静默挂起 | "语音后等待" | 宣告轮次结束前的等待时间，500–800 ms |
| 预滚 | "语音前缓冲" | VAD 触发前保留 300–500 ms 音频 |
| 冲洗技巧 | "Kyutai 的技巧" | VAD → 冲洗 STT → 125 ms 而非 500 ms 延迟 |
| 语义端点 | "他们打算停下来吗？" | 看词汇而非静默的机器学习分类器 |
| 5% FPR 下的 TPR | "ROC 点" | 标准 VAD 基准；Silero 87.7%，WebRTC 50% |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — 参考开源 VAD
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商业精度领头羊
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — 200 ms 以下延迟的工程技巧
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — 生产中的语义端点检测
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 遗留基线
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — 说话人分离级别的分割

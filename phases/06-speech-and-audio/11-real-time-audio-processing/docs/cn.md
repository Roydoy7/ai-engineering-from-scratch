# 实时音频处理

> 批处理流水线处理一个文件，实时流水线在下一个 20 毫秒到来之前处理当前这个。每个对话式 AI、广播工作室和电话机器人都依赖这个延迟预算生存。

**类型：** 构建
**语言：** Python
**前置知识：** 第6阶段第2课（频谱图）、第6阶段第4课（ASR）、第6阶段第7课（TTS）
**预计时间：** 约75分钟

## 问题背景

你想要一个感觉有生命的语音助手。人类对话轮次交替的延迟约为 230 ms（静默到回应）。超过 500 ms 感觉机械，超过 1500 ms 感觉坏掉了。2026 年完整**听 → 理解 → 回应 → 说话**循环的预算：

| 阶段 | 预算 |
|------|------|
| 麦克风 → 缓冲区 | 20 ms |
| VAD | 10 ms |
| ASR（流式） | 150 ms |
| LLM（首 token） | 100 ms |
| TTS（首块） | 100 ms |
| 渲染 → 扬声器 | 20 ms |
| **总计** | **~400 ms** |

Moshi（Kyutai，2024）实现了 200 ms 全双工，GPT-4o-realtime（2024）约为 320 ms，2022 年的级联流水线达到 2500 ms。10 倍改善来自三个技术：(1) 全程流式，(2) 带局部结果的异步流水线，(3) 可中断的生成。

## 核心概念

**帧/块/窗口**。实时音频以固定大小的块流动。常见选择：20 ms（16 kHz 下 320 个采样点）。下游的一切都必须跟上这个节奏。

**环形缓冲区**。固定大小的循环缓冲区。生产者线程写入新帧，消费者线程读取。避免在热路径上分配内存。大小 ≈ 最大延迟 × 采样率；2 秒 16 kHz 的环形缓冲 = 32000 个采样点。

**VAD（声音活动检测）**。在没有人说话时阻断下游工作。Silero VAD 4.0（2024）在 CPU 上每 30 ms 帧的运行时间不到 1 ms。`webrtcvad` 是较旧的替代方案。

**流式 ASR**。音频到来时就开始输出部分转录。Parakeet-CTC-0.6B 流式模式（NeMo，2024）以 320 ms 延迟达到 2–5% WER。Whisper-Streaming（Macháček 等，2023）将 Whisper 分块处理，实现约 2 秒延迟的近流式处理。

**中断**。当用户在助手说话时开口，你必须 (a) 检测到插话，(b) 停止 TTS，(c) 丢弃剩余的 LLM 输出。整个过程在 100 ms 内完成，否则用户会感觉助手听不见。

**WebRTC Opus 传输**。20 ms 帧，48 kHz，自适应码率 8–128 kbps。浏览器和移动端的标准。LiveKit、Daily.co、Pion 是 2026 年构建语音应用的主流技术栈。

**抖动缓冲区**。网络数据包乱序/延迟到达。抖动缓冲区重排并平滑；太小 → 可闻间隙，太大 → 延迟增加。典型值 60–80 ms。

### 常见陷阱

- **线程争抢**。Python GIL + 重型模型可能饿死音频线程。使用 C 回调音频库（sounddevice、PortAudio），让 Python 远离热路径。
- **采样率转换延迟**。流水线内部重采样会增加 5–20 ms。要么在最前端重采样，要么使用零延迟重采样器（PolyPhase、`soxr_hq`）。
- **TTS 预热**。即使是 Kokoro 这样快速的 TTS，第一次请求也有 100–200 ms 的预热时间。在第一次真实调用前，缓存模型并用虚拟请求预热。
- **回声消除**。没有 AEC，TTS 输出会从麦克风重新进入并触发 ASR 转录机器人自己的声音。WebRTC AEC3 是开源默认方案。

## 动手实现

### 第一步：环形缓冲区

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定最大缓冲延迟。16 kHz 下 32000 个采样点 = 2 秒。

### 第二步：VAD 门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产环境替换为 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 第三步：流式 ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 第四步：中断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

依赖异步 I/O 和可取消的 TTS 流式处理。在音频轨道上调用 WebRTC peerconnection.stop() 是标准做法。

## 工程应用

2026 年技术栈：

| 层次 | 选型 |
|------|------|
| 传输层 | LiveKit（WebRTC）或 Pion（Go） |
| VAD | Silero VAD 4.0 |
| 流式 ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM 首 token | Groq、Cerebras、vLLM-streaming |
| 流式 TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 端到端原生方案 | OpenAI Realtime API 或 Moshi |

## 常见陷阱

- **为了"安全"缓冲 500 ms**。缓冲区本身就是你的延迟下限，要尽量缩小。
- **不固定线程优先级**。音频回调跑在低于 UI 线程优先级的线程上 → 负载下出现杂音。
- **TTS 块太小**。小于 200 ms 的块会让声码器瑕疵变得可闻。320 ms 块是甜点。
- **没有抖动缓冲区**。真实网络有抖动，没有平滑处理就会有爆音。
- **单次异常处理**。音频流水线必须崩溃安全，一个异常会终结整个会话。

## 交付物

保存为 `outputs/skill-realtime-designer.md`。设计一个实时音频流水线，给出每个阶段的具体延迟预算。

## 练习

1. **（简单）** 运行 `code/main.py`，模拟环形缓冲区 + 能量 VAD，打印模拟 10 秒流的各阶段延迟。
2. **（中等）** 使用 `sounddevice` 构建一个透传循环，以 20 ms 帧处理麦克风输入，并在每帧打印 VAD 状态。
3. **（困难）** 用 `aiortc` 构建全双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器，用 1 kHz 脉冲测量端到端延迟。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 环形缓冲区 | "循环队列" | 固定大小的无锁（或 SPSC 锁）音频帧 FIFO |
| VAD | "静默门控" | 标记语音与非语音的模型或启发式方法 |
| 流式 ASR | "实时语音转文字" | 音频到来时输出部分文字，有限预看 |
| 抖动缓冲区 | "网络平滑器" | 重排乱序数据包的队列，典型值 60–80 ms |
| AEC | "回声消除" | 减去扬声器到麦克风的反馈路径 |
| 插话 (Barge-in) | "用户中断" | 系统检测到 TTS 播放中用户开口，必须取消播放 |
| 全双工 | "双向同时" | 用户和机器人可同时说话，Moshi 是全双工 |

## 延伸阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 分块近流式 Whisper
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 全双工 200 ms 延迟
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — 生产级音频 Agent 编排
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — 1 ms 以下 VAD，Apache 2.0
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源回声消除

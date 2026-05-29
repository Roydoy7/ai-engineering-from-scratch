# 构建语音助手流水线——第6阶段综合项目

> 将第 01–11 课的所有内容拼接在一起。构建一个能倾听、推理并回应的语音助手。在 2026 年，这是一个已解决的工程问题，而非研究问题——但集成细节决定它能否真正发货。

**类型：** 构建
**语言：** Python
**前置知识：** 第6阶段第04、05、06、07、11课；第11阶段第9课（函数调用）；第14阶段第1课（Agent 循环）
**预计时间：** 约120分钟

## 问题背景

构建一个端到端助手：

1. 采集麦克风输入（16 kHz 单声道）。
2. 检测用户语音的起止点。
3. 流式转录。
4. 将文字传给可调用工具（定时器、天气、日历）的 LLM。
5. 将 LLM 文字流式传给 TTS。
6. 向用户播放音频。
7. 如果用户在回应中途打断，立即停止。

延迟目标：在笔记本 CPU 上，用户话语结束后 800 ms 内发出首个 TTS 音频字节。质量目标：不漏词、静默时无幻觉字幕、无声音克隆泄漏、无提示注入成功。

## 核心概念

### 七个组件

1. **音频采集**。麦克风 → 16 kHz 单声道 → 20 ms 块。Python 中通常用 `sounddevice`，生产环境用原生 AudioUnit/ALSA/WASAPI。
2. **VAD（第11课）**。Silero VAD，阈值 0.5，最短语音 250 ms，静默挂起 500 ms。发出"开始"和"结束"信号。
3. **流式 STT（第4–5课）**。Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API），输出部分和最终文字。
4. **带工具调用的 LLM**。GPT-4o/Claude 3.5/Gemini 2.5 Flash，工具使用 JSON Schema。流式输出 token。
5. **流式 TTS（第7课）**。Kokoro-82M（最快开源）或 Cartesia Sonic（商业）。在 LLM 输出 20 个 token 后启动 TTS。
6. **播放**。音频输出；低带宽网络用 Opus 编码。
7. **中断处理器**。如果 TTS 播放时 VAD 触发，停止播放，取消 LLM，重启 STT。

### 你会踩到的三个失败模式

1. **首词被截**。VAD 启动晚了一拍，用户的"嘿"丢失了。将启动阈值设为 0.3，而不是 0.5。
2. **回应中途中断混乱**。用户中断后 LLM 继续生成，助手在用户说话时也在说话。给 VAD 连上取消 LLM 的逻辑。
3. **静默幻觉**。Whisper 在静默预热帧上输出"Thanks for watching"。始终用 VAD 门控。

### 2026 年生产参考技术栈

| 技术栈 | 延迟 | 协议 | 备注 |
|--------|------|------|------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350–500 ms | 商业 API | 2026 年行业默认 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500–800 ms | 基本开源 | DIY 友好 |
| Moshi（全双工） | 200–300 ms | CC-BY 4.0 | 单一模型，不同架构，见第15课 |
| Vapi / Retell（托管） | 300–500 ms | 商业 | 启动最快，定制有限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 离线 | 开源 | 隐私/边缘端 |

## 动手实现

### 第一步：带分块的麦克风采集（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 第二步：VAD 门控的轮次捕获

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 第三步：流式 STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 第四步：LLM 循环中的工具调用

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 第五步：中断处理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 工程应用

查看 `code/main.py` 获取可运行的模拟实现，它用桩模块将所有七个组件连接在一起，即使没有硬件也能看到流水线形状。真实实现时，用以下内容替换桩模块：

- `silero-vad`（`pip install silero-vad`）
- `deepgram-sdk` 或 `openai-whisper`
- `openai`（`gpt-4o`）或 `anthropic`
- `kokoro` 或 `cartesia`
- `sounddevice` 用于 I/O

## 常见陷阱

- **永久记录 PII**。完整轮次音频在大多数司法管辖区属于 PII。保留 30 天，静态加密。
- **没有插话功能**。用户会中断对话，你的助手必须停止说话。
- **TTS 阻塞**。同步 TTS 会阻塞事件循环，使用异步或单独线程。
- **没有工具调用错误处理**。工具会失败。LLM 必须收到错误信息并重试一次，然后优雅降级。
- **幻觉过滤过猛**。过滤过猛，助手反复说"无法帮助您"；过滤不足，什么都说。在留出集上校准。
- **没有唤醒词选项**。持续监听是隐私隐患。添加唤醒词门控（Porcupine 或 openWakeWord）。

## 交付物

保存为 `outputs/skill-voice-assistant-architect.md`。根据预算、规模、语言和合规限制，产出完整的技术栈规格说明。

## 练习

1. **（简单）** 运行 `code/main.py`，用桩模块模拟一次完整的端到端轮次，打印每个阶段的延迟。
2. **（中等）** 将 STT 桩替换为真实 Whisper 模型，在预录制的 `.wav` 上运行，测量 WER 和端到端延迟。
3. **（困难）** 添加工具调用：实现 `get_weather`（任意 API）和 `set_timer`，通过工具路由 LLM，验证用户说"设置 5 分钟定时器"时正确的函数被触发，且语音回复做出确认。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 轮次 (Turn) | "用户+助手一次交互" | 一次 VAD 界定的用户语音 + 一次 LLM-TTS 回应 |
| 插话 (Barge-in) | "中断" | 助手说话时用户开口，助手停下来 |
| 唤醒词 (Wake word) | "嘿，助手" | 短关键词检测器；Porcupine、Snowboy、openWakeWord |
| 端点检测 (End-pointing) | "轮次结束" | VAD + 最短静默判断用户已说完 |
| 预滚 (Pre-roll) | "语音前缓冲" | 在 VAD 触发前保留 200–400 ms 音频，避免首词被截 |
| 工具调用 (Tool call) | "函数调用" | LLM 输出 JSON，运行时分发执行，结果回传循环 |

## 延伸阅读

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — 生产级参考
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — DIY 友好框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 托管语音原生路径
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — 全双工参考（第15课）
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — 唤醒词门控
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM 函数调用

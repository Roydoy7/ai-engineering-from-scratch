# 压轴项目 03——实时语音助手（ASR 到 LLM 到 TTS）（Capstone 03 — Real-Time Voice Assistant: ASR to LLM to TTS）

> 感觉自然的语音智能体端到端延迟低于 800ms，知道你什么时候停止说话，处理打断，并且可以在不停顿的情况下调用工具。Retell、Vapi、LiveKit Agents 和 Pipecat 都在 2026 年达到了这个标准。它们用相同的形态实现：流式 ASR、轮次检测器、流式 LLM 和流式 TTS，全部通过 WebRTC 连接，在每个跳点都有严格的延迟预算。构建一个，测量 WER 和 MOS 以及误截断率，并在丢包情况下运行它。

**类型：** 压轴项目  
**语言：** Python（智能体 + 管道），TypeScript（Web 客户端）  
**前置知识：** Phase 6（语音和音频）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（智能体）、Phase 17（基础设施）  
**涉及的阶段：** P6 · P7 · P11 · P13 · P14 · P17  
**预计时间：** 30 小时

## 问题所在

语音是 2025-2026 年发展最快的 AI 用户体验类别。技术上限每个季度都在下降。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 都将 800ms 以内的首次音频输出变为可能。标准不仅仅是延迟，还有交互感受：不要打断用户，不要被打断，从句中中断中恢复，在对话中调用工具而不停顿音频，在不稳定的移动网络上继续工作。

你不能通过拼接三个 REST 调用达到这个目标。架构是端到端流水线式流式传输。构建它，失败模式就会变得可见：针对电话音频调整的 VAD 被背景电视触发，等待从未出现的标点的轮次检测器，在发出音频前缓冲 400ms 的 TTS。压轴项目是在负载下逐一修复这些问题，并发布延迟和质量报告。

## 核心概念

管道有五个流式阶段：**音频输入**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（来自 Deepgram Nova-3 或 faster-whisper 的流式部分转录）、**轮次检测**（VAD 加上读取部分转录以寻找完成线索的小型轮次检测器模型）、**LLM**（轮次判断完成后立即流式输出 token）、**TTS**（在第一个 LLM token 的约 200ms 内流式输出音频）。

三个横切关注点。**打断**：当用户在智能体说话时开始说话，TTS 取消，ASR 立即接管。**工具使用**：对话中的函数调用（天气、日历）必须在侧通道上运行而不停顿音频；如果延迟超过 300ms，智能体预填充确认 token（"稍等..."）。**背压**：在丢包情况下，部分转录被保留，VAD 提高语音门限阈值，智能体避免在未确认的消息上说话。

测量标准是定量的。在 15 dB SNR 的 Hamming VAD 基准测试上 WER 低于 8%。在 100 个测量呼叫上首次音频输出 p50 低于 800ms。误截断率低于 3%。TTS MOS 高于 4.2。单个 g5.xlarge 上 50 个并发呼叫。这些数字就是可交付成果。

## 架构

```
浏览器 / Twilio PSTN
        |
        v
   WebRTC / SIP 边缘
        |
        v
  LiveKit Agents 1.0（或 Pipecat 0.0.70）
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5        轮次检测器         侧通道
(Deepgram         (Silero)         (LiveKit)           工具
 Nova-3 /         每 20ms         部分转录          （天气，
 Whisper-v3)      语音门限        完成分数            日历）
   |                   |              |
   +--------+----------+--------------+
            v
        LLM（流式）
     GPT-4o-realtime / Gemini 2.5 Flash /
     级联 Claude Haiku 4.5
            |
            v
        TTS 流式
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     音频返回给呼叫者
            |
            v
   OpenTelemetry 语音追踪 -> Langfuse
```

## 技术栈

- 传输：LiveKit Agents 1.0（WebRTC）加 Twilio PSTN 网关；Pipecat 0.0.70 作为备用框架
- ASR：Deepgram Nova-3（流式，首次部分 300ms 以内）或自托管的 faster-whisper Whisper-v3-turbo
- VAD：Silero VAD v5 加 LiveKit 轮次检测器（读取部分转录的小型 Transformer）
- LLM：OpenAI GPT-4o-realtime（紧密集成）、Gemini 2.5 Flash Live，或级联 Claude Haiku 4.5（流式补全，单独音频路径）
- TTS：Cartesia Sonic-2（最低首字节延迟）、ElevenLabs Flash v3，或自托管的开源 Orpheus
- 工具：FastMCP 侧通道用于天气/日历/预订；如果工具超过 300ms，智能体预发出填充词
- 可观测性：OpenTelemetry 语音 span，Langfuse 语音追踪带音频回放
- 部署：单个 g5.xlarge（24GB VRAM）用于自托管 Whisper + Orpheus；托管 API 用于最低延迟

## 构建它

1. **WebRTC 会话。** 建立一个 LiveKit 房间和一个流式传输麦克风音频的 Web 客户端。在服务器上，附加一个加入房间的智能体工作者。

2. **ASR 流式传输。** 将 20ms PCM 帧送入 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅部分和最终转录。记录每次部分的延迟。

3. **VAD 和轮次检测器。** 在帧流上运行 Silero VAD v5。在语音结束事件时，针对最新部分转录触发 LiveKit 轮次检测器。只有当 VAD 说静默 500ms 且轮次检测器完成分数 > 0.6 时，才提交"轮次完成"。

4. **LLM 流。** 轮次完成时，用运行中的对话加最终转录开始 LLM 调用。流式输出 token。在第一个 token 时，移交给 TTS。

5. **TTS 流。** Cartesia Sonic-2 流式返回音频块。第一个块必须在第一个 LLM token 的 200ms 内离开服务器。将块发送到 LiveKit 房间；客户端通过 WebRTC 抖动缓冲区播放。

6. **打断。** 当 VAD 在 TTS 播放时检测到新的用户语音，立即取消 TTS 流，丢弃剩余的 LLM 输出，并重新武装 ASR。发布 `tts_canceled` span。

7. **工具侧通道。** 将天气和日历注册为函数调用工具。当调用时，并发触发调用；如果它在 300ms 内没有解决，让 LLM 发出"稍等，让我查一下"作为填充词；一旦工具返回就继续。

8. **评估测试框架。** 录制 100 个呼叫。计算 WER（与保留转录对比）、误截断率（用户在句中时 TTS 取消）、首次音频输出 p50、TTS MOS（人工或 NISQA）和抖动丢包测试（丢弃 3% 的数据包）。

9. **负载测试。** 用合成呼叫者在单个 g5.xlarge 上驱动 50 个并发呼叫。测量持续的首次音频输出 p95。

## 使用它

```
呼叫者："明天东京的天气怎么样"
[asr  ] 部分 @280ms: "明天东京"
[asr  ] 部分 @540ms: "明天东京的天气"
[轮次 ] 完成分数 0.82 @820ms；提交
[llm  ] 首个 token @960ms
[工具 ] weather.tokyo tomorrow -> 68/52 多云 @1140ms
[tts  ] 首次音频输出 @1040ms："东京明天将会多云..."
轮次延迟：1040ms 用户停止 -> 音频输出
```

## 交付它

`outputs/skill-voice-agent.md` 是可交付成果。给定一个领域（客户支持、调度或信息亭），它建立一个调整到测量标准的 LiveKit 智能体，带有 ASR/VAD/LLM/TTS 管道。评分标准：

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 端到端延迟 | 在 100 个录制呼叫上 p50 首次音频输出低于 800ms |
| 20 | 轮次转换质量 | 在 Hamming VAD 基准测试上误截断率低于 3% |
| 20 | 工具使用正确性 | 对话中的工具调用返回正确数据且不停顿音频 |
| 20 | 丢包下的可靠性 | 注入 3% 数据包丢失时的 WER 和轮次转换稳定性 |
| 15 | 评估测试框架完整性 | 带公共配置的可重现测量 |
| **100** | | |

## 练习

1. 将 Deepgram Nova-3 换为 g5.xlarge 上的 faster-whisper v3 turbo。测量延迟和 WER 差距。识别 CPU vs GPU 决策重要的地方。

2. 添加打断仲裁策略：用户在工具调用期间打断时智能体做什么？比较三种策略（硬取消、完成工具然后停止、排队下一轮）。

3. 运行对抗性轮次检测器测试：让用户在句中有长时间停顿。调整 VAD 静默阈值和轮次检测器分数阈值，以最低误截断率而不超过 900ms。

4. 通过 Twilio 在 PSTN 上部署同一个智能体。比较 PSTN 和 WebRTC 的首次音频输出。解释抖动缓冲区和编解码器的差异。

5. 为非英语语言（日语、西班牙语）添加语音活动检测。测量 Silero VAD v5 误触发率与语言特定微调的比较。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Turn detection（轮次检测） | "话语结束" | 给定 VAD 静默和部分转录，决定用户已停止说话的分类器 |
| Barge-in（打断） | "打断处理" | 当 VAD 检测到新的用户语音时，在播放中取消 TTS |
| First-audio-out（首次音频输出） | "延迟" | 从用户停止说话到第一个音频数据包离开服务器的时间 |
| VAD | "语音门限" | 将音频帧分类为语音 vs 静默的模型；Silero VAD v5 是 2026 年的默认选择 |
| Jitter buffer（抖动缓冲区） | "音频平滑" | 客户端缓冲区，短暂保留数据包以吸收网络抖动 |
| Filler（填充词） | "确认 token" | 智能体在工具慢时发出的短语，避免静默 |
| MOS | "平均意见分" | 感知语音质量评分；NISQA 是自动化代理 |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 参考 WebRTC 智能体框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 备用 Python 优先流式智能体框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 集成语音模型参考
- [Deepgram Nova-3 文档](https://developers.deepgram.com/docs) — 流式 ASR 参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟 TTS 参考
- [Retell AI 架构](https://docs.retellai.com) — 生产语音智能体架构
- [Vapi.ai 生产栈](https://docs.vapi.ai) — 备用生产参考

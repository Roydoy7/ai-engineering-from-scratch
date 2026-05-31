# 语音智能体：Pipecat 与 LiveKit（Voice Agents: Pipecat and LiveKit）

> 语音智能体是 2026 年的一等生产类别。Pipecat 提供基于帧的 Python 流水线（VAD → STT → LLM → TTS → 传输）。LiveKit Agents 通过 WebRTC 将 AI 模型与用户桥接。顶级技术栈的生产延迟目标在端到端 450-600ms。

**类型：** 学习  
**语言：** Python（标准库）  
**前置知识：** Phase 14 · 01（智能体循环）、Phase 14 · 12（工作流模式）  
**预计时间：** 约 60 分钟

## 学习目标

- 描述 Pipecat 的基于帧的流水线：DOWNSTREAM（源→汇）和 UPSTREAM（控制）。
- 说出典型的语音流水线各阶段以及 Pipecat 支持的传输方式。
- 解释 LiveKit Agents 的两种语音智能体类（MultimodalAgent、VoicePipelineAgent）及各自的适用场景。
- 总结 2026 年生产延迟预期，以及它们如何驱动架构选择。

## 问题所在

语音智能体不是带 TTS 的文本循环。延迟预算非常苛刻（约 600ms），部分音频是默认情况，转换检测是一个模型，传输方式从电话 SIP 到 WebRTC 不等。要么构建基于帧的流水线（Pipecat），要么依赖平台（LiveKit）。

## 核心概念

### Pipecat（pipecat-ai/pipecat）

- 基于帧的 Python 流水线框架。
- `Frame` → `FrameProcessor` 链。
- 两种流向：
  - **DOWNSTREAM** — 源 → 汇（音频输入，TTS 输出）。
  - **UPSTREAM** — 反馈和控制（取消、指标、插话）。
- `PipelineTask` 管理带事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）和用于指标/追踪/RTVI 观察者的生命周期。

典型流水线：

```
VAD (Silero) → STT → LLM（上下文交替 user/assistant） → TTS → 传输
```

传输方式：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 添加结构化对话（状态机）。Pipecat Cloud 是托管运行时。

### LiveKit Agents（livekit/agents）

- 通过 WebRTC 将 AI 模型与用户桥接。
- 关键概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两种语音智能体类：
  - **MultimodalAgent** — 通过 OpenAI Realtime 或等效方式的直接音频。
  - **VoicePipelineAgent** — STT → LLM → TTS 级联；提供文本级控制。
- 通过 Transformer 模型进行语义转换检测。
- 原生 MCP 集成。
- 通过 SIP 支持电话。
- 通过 LiveKit Inference 无需 API 密钥的 50+ 个模型；通过插件支持 200+ 个。

### 商业平台

Vapi（在优化的顶级技术栈上约 450-600ms）和 Retell（跨 180 次测试调用约 600ms 端到端）构建于这些之上。当你想要托管的语音技术栈而不需要 WebRTC 团队时，选择平台。

### 这个模式在哪里出错

- **没有插话处理。** 用户打断；智能体继续说。需要 Pipecat 中的 UPSTREAM 取消帧，LiveKit 中的等效机制。
- **忽略 STT 置信度。** 低置信度的转录被当作圣典喂给 LLM。根据置信度设置门控或请求确认。
- **TTS 句中截断。** 当流水线在话语中途取消时，TTS 需要知道或截断音频。
- **忽略延迟预算。** 每个组件增加 50-200ms。在发布前对你的链求和。

### 2026 年典型延迟

- VAD：20-60ms
- STT 部分结果：100-250ms
- LLM 第一个 token：150-400ms
- TTS 第一段音频：100-200ms
- 传输 RTT：30-80ms

端到端 450-600ms 是顶级。800-1200ms 很常见。超过 1500ms 感觉像损坏了。

## 构建它

`code/main.py` 是一个基于帧的玩具流水线，包含：

- `Frame` 类型（音频、转录、文本、tts_audio、控制）。
- 带有 `process(frame)` 的 `Processor` 接口。
- 五阶段流水线（VAD → STT → LLM → TTS → 传输）作为脚本化处理器。
- 一个 UPSTREAM 取消帧，演示插话。

运行：

```
python3 code/main.py
```

追踪显示正常流程和一次在话语中途停止 TTS 的插话取消。

## 使用它

- **Pipecat** 用于完全控制——自定义处理器、Python 优先、可插拔的提供商。
- **LiveKit Agents** 用于 WebRTC 优先的部署和电话。
- **Vapi / Retell** 用于无需 WebRTC 团队的托管语音智能体。
- **OpenAI Realtime / Gemini Live** 用于直接音频输入/输出（MultimodalAgent）。

## 交付它

`outputs/skill-voice-pipeline.md` 搭建一个 Pipecat 形态的语音流水线，带有 VAD + STT + LLM + TTS + 传输，以及插话处理。

## 练习

1. 在你的玩具流水线中添加指标观察者：统计每阶段每秒帧数。延迟在哪里积累？
2. 实现置信度门控的 STT：低于阈值时，请求"能再说一遍吗？"
3. 添加语义转换检测：简单规则——如果转录以"？"结尾，则转换结束。
4. 阅读 Pipecat 的传输文档。将标准库传输替换为 SmallWebRTCTransport 配置（存根）。
5. 对相同查询，测量 OpenAI Realtime 与 STT+LLM+TTS 级联的延迟。文本级控制带来了多少延迟成本？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Frame（帧） | "事件" | 流水线中的类型化数据单元（音频、转录、文本、控制） |
| Processor（处理器） | "流水线阶段" | 带有 process(frame) 的处理器 |
| DOWNSTREAM | "前向流" | 源到汇：音频输入，语音输出 |
| UPSTREAM | "反馈流" | 控制：取消、指标、插话 |
| VAD | "语音活动检测" | 检测用户何时在说话 |
| Semantic turn detection（语义转换检测） | "智能转换结束" | 基于模型的用户说完的决策 |
| MultimodalAgent | "直接音频智能体" | 音频输入，音频输出；中间没有文本 |
| VoicePipelineAgent | "级联智能体" | STT + LLM + TTS；文本级控制 |

## 延伸阅读

- [Pipecat 文档](https://docs.pipecat.ai/getting-started/introduction) — 基于帧的流水线、处理器、传输
- [LiveKit Agents 文档](https://docs.livekit.io/agents/) — WebRTC + 语音基本元素
- [Vapi](https://vapi.ai/) — 托管语音平台
- [Retell AI](https://www.retellai.com/) — 托管语音，延迟基准测试

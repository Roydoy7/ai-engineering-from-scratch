# 流式语音到语音——Moshi、Hibiki 与全双工对话

> 2024–2026 年重新定义了语音 AI。Moshi 以一个单一模型实现 200 ms 延迟的同时听说。Hibiki 逐块进行语音到语音翻译。两者都放弃了 ASR → LLM → TTS 流水线，改用基于 Mimi 编解码器 token 的统一全双工架构。这是新的参考设计。

**类型：** 学习
**语言：** Python
**前置知识：** 第6阶段第13课（神经音频编解码器）、第6阶段第11课（实时音频处理）、第7阶段第5课（完整 Transformer）
**预计时间：** 约75分钟

## 问题背景

由第 11+12 课构建的每个语音 Agent 都有一个约 300–500 ms 的根本延迟下限：VAD 触发、STT 处理、LLM 推理、TTS 生成。每个阶段都有自己的最小延迟，你可以调优和并行化，但流水线结构本身限制了上限。

Moshi（Kyutai，2024–2026）提出了一个不同的问题：如果没有流水线呢？如果一个模型直接持续地接收音频并输出音频，文字只是一个中间"内部独白"而不是必须的阶段？

答案是**全双工语音到语音**。理论延迟 160 ms（80 ms Mimi 帧 + 80 ms 声学延迟）。单 L4 GPU 实际延迟 200 ms。这是同类最佳流水线语音 Agent 的一半。

## 核心概念

### Moshi 架构

**输入**。两路 Mimi 编解码器流，均为 12.5 Hz × 8 码本：

- 流 1：用户音频（Mimi 编码，持续到来）
- 流 2：Moshi 自己的音频（由 Moshi 生成）

**Transformer**。一个 70 亿参数的 Temporal Transformer 处理两路流和一个文字"内部独白"流。每 80 ms 步骤，它：

1. 消费最新的用户 Mimi token（8 码本）。
2. 消费最近的 Moshi Mimi token（8 码本，已生成的）。
3. 生成下一个 Moshi 文字 token（内部独白）。
4. 通过小型深度 Transformer 生成下一个 Moshi Mimi token（8 码本）。

用户音频、Moshi 音频、Moshi 文字这三路流并行运行。Moshi 可以在说话时听取用户，可以在用户打断时自我中断，可以在不打断主要话语的情况下发出反馈（"嗯哼"）。

**深度 Transformer**。在一帧内，8 个码本不是并行预测的——它们之间有码本间依赖关系。一个小型 2 层"深度 Transformer"在 80 ms 内顺序预测它们。这是 AR 编解码器语言模型的标准因式分解（VALL-E、VibeVoice 也使用此方法）。

### 内部独白文字为何有帮助

没有显式文字，模型必须在声学流中隐式建模语言。Moshi 的洞见：强制它在音频旁边输出文字 token。这个文字流本质上是 Moshi 正在说的内容的文字记录。这改善了语义连贯性，使替换语言模型头更容易，并免费提供文字记录。

### Hibiki：流式语音到语音翻译

相同架构，在翻译对上训练。持续输入源语言音频，输出目标语言音频。Hibiki-Zero（2026 年 2 月）消除了对词级对齐训练数据的需求——使用句子级数据 + GRPO 强化学习进行延迟优化。

初始支持四个语言对，使用约 1000 小时的数据可适配到新语言。

### 更广泛的 Kyutai 技术栈（2026 年）

- **Moshi** — 全双工对话（法语优先，英语支持良好）
- **Hibiki / Hibiki-Zero** — 同声传译
- **Kyutai STT** — 流式 ASR（500 ms 或 2.5 s 预看）
- **Kyutai Pocket TTS** — 1 亿参数 TTS，可在 CPU 上运行（2026 年 1 月）
- **Unmute** — 在公共服务器上组合以上所有组件的完整流水线

L40S GPU 上的吞吐量：以 3 倍实时速度支持 64 个并发会话。

### Sesame CSM——近亲

Sesame CSM（2025）使用类似思路——带 Mimi 编解码器头的 Llama-3 骨干。但 CSM 是单向的（接收上下文 + 文字，生成语音）而非全双工。它是市场上最好的"声音临场感"TTS，与 Moshi 的全双工能力不完全相同。

### 2026 年性能数字

| 模型 | 延迟 | 用途 | 协议 |
|------|------|------|------|
| Moshi | 200 ms（L4） | 英语/法语全双工对话 | CC-BY 4.0 |
| Hibiki | 12.5 Hz 帧率 | 法英双向流式翻译 | CC-BY 4.0 |
| Hibiki-Zero | 相同 | 5 个语言对，无需对齐数据 | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | 上下文条件 TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | 闭源，OpenAI API | 商业 |
| Gemini 2.5 Live | ~350 ms | 闭源，Google API | 商业 |

## 动手实现

### 第一步：接口

Moshi 暴露一个 WebSocket 服务器，持续接收 80 ms 的 Mimi 编码音频块，并持续返回 80 ms 的 Mimi 编码音频块。双向持续进行。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 第二步：全双工循环

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python asyncio 或 Rust futures 是标准传输方式。

### 第三步：训练目标（概念）

对每个 80 ms 帧 `t`：

- 输入：`user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`
- 预测：`moshi_text[t]`，然后 `moshi_mimi[t, codebook_0..7]`

文字在音频之前预测（内部独白），音频在深度 Transformer 内按码本顺序预测。

### 第四步：Moshi 的优势与劣势

Moshi 的优势：

- 低廉硬件上的亚 250 ms 端到端延迟。
- 自然的反馈和中断。
- 无流水线胶水代码。

Moshi 的劣势：

- 工具调用（未为此训练，需要单独的 LLM 路径）。
- 长推理（Moshi 是约 80 亿参数的对话模型，不是 Claude/GPT-4）。
- 小众话题的事实准确性。
- 大多数生产企业用例（2026 年仍使用流水线）。

## 工程应用

| 场景 | 选型 |
|------|------|
| 最低延迟语音伴侣 | Moshi |
| 实时翻译通话 | Hibiki |
| 语音演示/研究 | Moshi、CSM |
| 带工具的企业 Agent | 流水线（第12课），而非 Moshi |
| 上下文中的自定义声音 TTS | Sesame CSM |
| 任意语言语音到语音 | GPT-4o Realtime 或 Gemini 2.5 Live（商业） |

## 常见陷阱

- **工具调用有限**。Moshi 是对话模型，不是 Agent 框架。结合流水线实现工具调用。
- **特定声音条件**。Moshi 使用单一训练的人物角色；克隆需要单独的训练过程。
- **语言覆盖**。法语 + 英语表现出色，其他语言有限。Hibiki-Zero 有所帮助，但仍需训练数据。
- **资源成本**。一个完整的 Moshi 会话占用 GPU 插槽，不适合廉价的共享租户部署模式。

## 交付物

保存为 `outputs/skill-duplex-pipeline.md`。为语音 Agent 工作负载选择流水线 vs 全双工架构，并说明原因。

## 练习

1. **（简单）** 运行 `code/main.py`，以符号化方式模拟双流 + 内部独白架构。
2. **（中等）** 从 HuggingFace 拉取 Moshi，运行服务器，测试一次对话，测量从用户话语结束到 Moshi 回应开始的挂钟延迟。
3. **（困难）** 取第 12 课的流水线 Agent，在 20 个匹配测试话语上对比 P50 延迟 vs Moshi，记录流水线架构在哪些情况下仍然胜出。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 全双工 | "边听边说" | 同一模型上两路音频流同时激活 |
| 内部独白 | "模型的文字流" | Moshi 与音频输出同时输出文字 token |
| 深度 Transformer | "码本间预测器" | 在一个 80 ms 帧内预测 8 个码本的小型 Transformer |
| Mimi | "Kyutai 的编解码器" | 12.5 Hz × 8 码本，语义+声学，驱动 Moshi |
| 流式 S2S | "音频到音频实时" | 逐块翻译/对话，无流水线阶段 |
| 反馈 (Back-channeling) | ""嗯哼"反应" | Moshi 可以在不打断自己轮次的情况下发出小型确认 |

## 延伸阅读

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 原始论文
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 无需对齐数据的流式翻译
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM 规格
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — 安装 + 服务器
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — 闭源商业同类
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — 底层 STT/TTS 框架

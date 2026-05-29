# 语音识别（ASR）——CTC、RNN-T 与注意力机制

> 语音识别就是对每个时间步做音频分类，再用一个懂英语和沉默的序列模型把它们串联起来。CTC、RNN-T 和注意力机制是三种做法，选一种并弄清楚为什么。

**类型：** 构建
**语言：** Python
**前置知识：** 第6阶段第2课（频谱图与 Mel）、第5阶段第8课（文本的 CNN 与 RNN）、第5阶段第10课（注意力机制）
**预计时间：** 约45分钟

## 问题背景

你拿到一段 10 秒的 16 kHz 音频片段，想得到一个字符串："turn on the kitchen lights"。难点在于结构性的：音频帧和字符之间不是一一对应的。"okay"这个词可能耗时 200 ms 也可能耗时 1200 ms。沉默穿插其中。有的音素比其他的更长。输出 token 的数量事先未知。

三种方案解决这个问题：

1. **CTC（联结时序分类）**。在每帧输出包含特殊*空白符*的 token 概率。解码时折叠连续重复并去掉空白符。非自回归，速度快。wav2vec 2.0、MMS 使用此方案。
2. **RNN-T（循环神经网络转录器）**。联合网络结合编码器帧和前序 token 预测下一个 token。可流式处理。Google 端侧 ASR、NVIDIA Parakeet 使用此方案。
3. **注意力编码器-解码器**。编码器将音频压缩为隐状态，解码器通过交叉注意力自回归地生成 token。Whisper、SeamlessM4T 使用此方案。

2026 年，LibriSpeech test-clean 上的 SOTA 词错误率为 1.4%（Parakeet-TDT-1.1B，NVIDIA）和 1.58%（Whisper-Large-v3-turbo）。指标差异微乎其微，但部署差异极大。

## 核心概念

**CTC 直觉**。让编码器输出 `T` 帧级别的分布，覆盖 `V+1` 个 token（V 个字符 + 空白符）。对于长度 `U < T` 的目标字符串 `y`，任何折叠后等于 `y` 的帧对齐方式都算数。CTC 损失对所有这些对齐方式求和。推理时：逐帧 argmax，折叠重复，去掉空白符。

优点：非自回归、可流式、零预看。缺点：*条件独立假设*——每帧预测相互独立，没有内部语言模型。解决方法是通过束搜索或浅融合接入外部语言模型。

**RNN-T 直觉**。增加了一个*预测网络*，用于嵌入 token 历史，以及一个*联合网络*，将预测状态和编码器帧合并为 `V+1` 上的联合分布（多出的 1 是空发射/不发射）。显式建模了 CTC 忽略的条件依赖关系。可流式处理，因为每步只依赖过去的帧和 token。

优点：可流式 + 内部语言模型。缺点：训练更复杂，内存开销更大（三维损失格），RNN-T 损失核本身就是一个完整的库类别。

**注意力编码器-解码器**。编码器（6-32 层 Transformer）处理对数 Mel 帧，解码器（6-32 层 Transformer）通过交叉注意力指向编码器输出，自回归生成 token。没有对齐约束——注意力可以看向音频中的任意位置。除非限制注意力范围（分块 Whisper-Streaming，2024），否则不可流式。

优点：离线 ASR 质量最高，用标准 seq2seq 工具链易于训练。缺点：自回归延迟与输出长度成正比，不经工程改造无法流式处理。

### 词错误率（WER）：唯一核心指标

**词错误率** = `(S + D + I) / N`，其中 S=替换数，D=删除数，I=插入数，N=参考词数。等价于词级别的莱文斯坦编辑距离，越低越好。WER 超过 20% 通常不可用；低于 5% 对朗读语音达到人类水平。2026 年标准基准数字：

| 模型 | LibriSpeech test-clean | LibriSpeech test-other | 规模 |
|------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 11 亿参数 |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 8.09 亿 |
| Canary-1B Flash | 1.48% | 2.87% | 10 亿 |
| Seamless M4T v2 | 1.7% | 3.5% | 23 亿 |

以上都是编码器-解码器或 RNN-T 架构。纯 CTC 系统（wav2vec 2.0）在 test-clean 上约为 1.8–2.1%。

## 动手实现

### 第一步：CTC 贪心解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：折叠连续重复，去掉空白符。示例：`a a _ _ a b b _ c` → `a a b c`。

### 第二步：CTC 束搜索

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产环境使用带语言模型融合的前缀树束搜索，这里是概念骨架。

### 第三步：词错误率

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 第四步：用 Whisper 进行推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026 年最强通用 ASR 的一行代码。在 24 GB GPU 上以约 20 倍实时速度运行。

### 第五步：用 Parakeet 或 wav2vec 2.0 流式识别

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式 ASR 需要分块编码器注意力和跨块状态传递，请使用支持此功能的库（NeMo 支持 Parakeet，`transformers` pipeline 支持 `chunk_length_s` 参数）。

## 工程应用

2026 年技术栈：

| 场景 | 选型 |
|------|------|
| 英文、离线、最高质量 | Whisper-large-v3-turbo |
| 多语言、鲁棒性强 | SeamlessM4T v2 |
| 流式、低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘端、移动端、延迟 <500 ms | 量化 Whisper-Tiny 或 Moonshine（2024） |
| 长音频 | Whisper + 基于 VAD 的分块（WhisperX） |
| 领域专用（医疗、法律） | 微调 wav2vec 2.0 + 领域语言模型融合 |

## 2026 年仍在发货的陷阱

- **没有 VAD**。在静默音频上运行 Whisper 会产生幻觉（"Thanks for watching!"）。始终用 VAD 做过滤。
- **字符/词/子词 WER 混淆**。报告归一化后的词级 WER（小写、去除标点）。
- **语言识别漂移**。Whisper 的自动语言识别会把嘈杂片段误判为日语或威尔士语；确定语言时强制指定 `language="en"`。
- **长音频不分块**。Whisper 的输入窗口为 30 秒，超过此长度的音频使用 `chunk_length_s=30, stride=5`。

## 交付物

保存为 `outputs/skill-asr-picker.md`。根据给定的部署目标选择模型、解码策略、分块方案和语言模型融合方式。

## 练习

1. **（简单）** 运行 `code/main.py`，它对手动构造的 CTC 输出做贪心解码，并计算相对参考答案的词错误率。
2. **（中等）** 正确实现第二步中的前缀树束搜索（处理空白符合并规则），在 10 个合成样本上与贪心解码进行对比。
3. **（困难）** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 上运行 `whisper-large-v3-turbo`，计算前 100 句话的词错误率，与已发布数字对比。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| CTC | "带空白符的损失" | 对所有帧到 token 对齐方式求边缘概率；非自回归 |
| RNN-T | "流式损失" | CTC + 下一 token 预测器；建模词序 |
| 注意力编码器-解码器 | "Whisper 那种" | 编码器 + 交叉注意力解码器；离线质量最佳 |
| WER（词错误率） | "你上报的那个数" | 词级别的 `(S+D+I)/N` |
| 空白符 (Blank) | "空标记" | CTC 中的特殊 token，表示"本帧不发射" |
| 语言模型融合 | "外部语言模型" | 束搜索时加权叠加语言模型对数概率 |
| VAD | "静默过滤器" | 声音活动检测器，裁掉非语音段 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 论文
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 论文
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022 年经典论文，2024 年 v3-turbo 扩展
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026 年 Open ASR 排行榜榜首
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 覆盖 25+ 模型的实时基准排行榜

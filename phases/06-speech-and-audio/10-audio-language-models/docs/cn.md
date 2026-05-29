# 音频语言模型——Qwen2.5-Omni、Audio Flamingo 与 GPT-4o Audio

> 2026 年音频语言模型可对语音、环境声和音乐进行综合推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上与 GPT-4o Audio 持平，Audio Flamingo Next 在 LongAudioBench 上超越 Gemini 2.5 Pro。开源与闭源之间的差距已基本消弭——除了多音频任务，所有模型在该任务上的表现几乎与随机相当。

**类型：** 学习
**语言：** Python
**前置知识：** 第6阶段第4课（ASR）、第12阶段第3课（视觉语言模型）、第7阶段第10课（音频 Transformer）
**预计时间：** 约45分钟

## 问题背景

你有 5 秒的音频：狗叫，有人喊"stop!"，然后沉默。有用的问题跨越多个维度：

- **转录**。"说了什么？"——ASR 的领域。
- **语义推理**。"这个人有危险吗？"——需要联合理解狗叫 + 喊叫 + 沉默。
- **音乐推理**。"主旋律由什么乐器演奏？"
- **长音频检索**。"这位老师在 90 分钟讲座的哪里讲解了梯度下降？"

用一个提示回答所有这些问题的单一模型就是**音频语言模型**（LALM / ALM）。与纯 ASR 不同：LALM 生成自由形式的自然语言答案，而不仅仅是文字记录。

## 核心概念

### 三组件模板

2026 年每个 LALM 都有相同的骨架：

1. **音频编码器**。Whisper 编码器、BEATs、CLAP、WavLM，或各模型自定义编码器。
2. **投影器**。线性层或 MLP，将音频编码器特征桥接到 LLM 的词嵌入空间。
3. **LLM**。基于 Llama/Qwen/Gemma 的解码器，接收交错的文字和音频 token，生成文字。

训练阶段：

- **阶段一**。冻结编码器和 LLM，仅在 ASR/字幕数据上训练投影器。
- **阶段二**。在指令遵循音频任务（问答、推理、音乐理解）上全量/LoRA 微调。
- **阶段三（可选）**。增加语音输入/输出，添加语音解码器。Qwen2.5-Omni 和 AF3-Chat 支持此功能。

### 2026 年模型全景

| 模型 | 骨干 | 音频编码器 | 输出模态 | 可用性 |
|------|------|-----------|---------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | 自定义 + Whisper | 文字 + 语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | 自定义 | 文字 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文字 | NVIDIA 非商业 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文字 | NVIDIA 非商业 |
| SALMONN | Vicuna | Whisper + BEATs | 文字 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 文字 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 文字 | Apache-2.0 |
| Gemini 2.5 Flash/Pro（闭源） | Gemini | 专有 | 文字 + 语音 | API |
| GPT-4o Audio（闭源） | GPT-4o | 专有 | 文字 + 语音 | API |

### 基准现实检验（2026 年）

**MMAU-Pro**。1800 个问答对，覆盖语音/声音/音乐/混合类别，包含多音频子集。

| 模型 | 总体 | 语音 | 声音 | 音乐 | 多音频 |
|------|------|------|------|------|--------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench SOTA | — | — | — | — |

**多音频那一列对所有人来说都很难看**。四选一题的随机水平 = 25%，大多数模型就在这附近徘徊。LALM 在比较两段音频时仍然力不从心。

### 2026 年 LALM 的有用场景

- **呼叫中心录音合规审计**。"客服人员是否提到了必要的披露声明？"
- **无障碍服务**。向听障用户描述声音事件（不仅仅是转录）。
- **内容审核**。检测暴力语言 + 威胁语气 + 背景环境。
- **播客/会议分章**。语义摘要，而不仅仅是说话人轮次。
- **音乐曲库分析**。"找出所有 B 段有转调的曲目。"

### 尚不适用的场景

- 精细的乐理分析（和弦级别以下）。
- 长对话中的说话人归因推理（超过 10 分钟后明显退化）。
- 多音频比较（22–26% 几乎与随机相当）。
- 实时流式推理（大多数模型是离线批处理）。

## 动手实现

### 第一步：查询 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 第二步：投影器模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就这样。投影器通常是 1–3 个线性层。在 ASR 对（音频 → 文字）上训练它，是第一阶段的预训练任务。

### 第三步：在 MMAU/LongAudioBench 上评测

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

按类别（语音/声音/音乐/多音频）分别报告，聚合数字会掩盖模型的失败之处。

## 工程应用

| 任务 | 2026 年选型 |
|------|-----------|
| 自由形式音频问答（开源） | Qwen2.5-Omni-7B |
| 开源长音频最优 | Audio Flamingo Next |
| 闭源最优 | Gemini 2.5 Pro |
| 语音输入/输出 Agent | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2（专门的 AF-CLAP） |
| 呼叫中心审计 | Gemini 2.5 Pro via API，结合策略文档做 RAG |

## 常见陷阱

- **过度信任多音频**。如果你的任务需要"哪段音频有 X"，随机水平的表现是真实存在的。
- **长音频退化**。超过 10 分钟后，大多数模型的说话人归因会崩溃。先做分离（第6课），再做摘要。
- **静默时幻觉**。使用 Whisper 编码器的 LALM 继承了同样的问题，要用 VAD 过滤。
- **基准数字挑选**。厂商博客文章只展示最佳类别。自己跑 MMAU-Pro 多音频子集。

## 交付物

保存为 `outputs/skill-alm-picker.md`。为给定的音频理解任务选择 LALM、基准子集和输出模态（文字 vs 语音）。

## 练习

1. **（简单）** 运行 `code/main.py`，演示玩具投影器模式以及（音频嵌入、文字 token）→ 输出 token 的假 LALM 路由。
2. **（中等）** 在 100 个 MMAU-Pro 语音条目上评测 Qwen2.5-Omni-7B，与论文报告的数字对比。
3. **（困难）** 构建一个最简音频字幕基线：BEATs 编码器 + 2 层投影器 + 冻结的 Llama-3.2-1B，仅在 AudioCaps 上微调投影器，在 Clotho-AQA 上与 SALMONN 对比。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| LALM | "音频 ChatGPT" | 音频编码器 + 投影器 + LLM 解码器 |
| 投影器 (Projector) | "适配器" | 将音频特征映射到 LLM 嵌入空间的小型 MLP |
| MMAU | "那个基准" | 1 万个音频问答对，覆盖语音、声音、音乐 |
| MMAU-Pro | "更难的 MMAU" | 1800 个多音频/重推理问题 |
| LongAudioBench | "长格式评测" | 多分钟音频配语义查询 |
| 语音输入/输出 | "原生语音" | 模型直接接收和生成语音，无需文字绕路 |

## 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 参考架构
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 语音输入语音输出
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 开源长音频领头羊
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 双编码器先驱
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — 2026 年实时排名

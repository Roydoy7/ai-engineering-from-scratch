# Whisper——架构与微调

> Whisper 是一个 30 秒窗口的 Transformer 编码器-解码器，在 68 万小时多语言弱监督音频-文本对上训练而成。一个架构，多种任务，覆盖 99 种语言，鲁棒性强。2026 年 ASR 参考模型。

**类型：** 构建
**语言：** Python
**前置知识：** 第6阶段第4课（ASR）、第5阶段第10课（注意力机制）、第7阶段第5课（完整 Transformer）
**预计时间：** 约75分钟

## 问题背景

Whisper 于 2022 年 9 月由 OpenAI 发布，是第一个以商品化形式发货的 ASR 模型：粘贴音频，得到文本，99 种语言，鲁棒抗噪，笔记本上就能运行。到 2024 年，OpenAI 推出了 Large-v3 和 Turbo 变体；到 2026 年，Whisper 已成为从播客转录到语音助手再到 YouTube 字幕的默认基线。

但 Whisper 不是一个可以永远当黑盒用的流水线。领域偏移会让它失效——技术术语、说话人口音、专有名词、短片段、静默。你需要了解：

1. 它内部到底是什么。
2. 如何正确地给它提供分块、流式或长音频。
3. 何时需要微调，以及怎么做。

## 核心概念

**架构**。标准 Transformer 编码器-解码器。

- 输入：30 秒对数 Mel 频谱图，80 个 Mel bin，10 ms 帧移 → 3000 帧。短于 30 秒的片段做零填充，长于 30 秒的做分块。
- 编码器：卷积下采样（步长 2）+ `N` 个 Transformer 块。Large-v3：32 层，1280 维，20 个注意力头。
- 解码器：`N` 个 Transformer 块，包含因果自注意力 + 交叉注意力指向编码器输出。与编码器规模相同。
- 输出：基于 51,865 个词条的 BPE 词表。

Large-v3 共 15.5 亿参数。Turbo 使用 4 层解码器（原为 32 层），延迟降低 8 倍，词错误率仅上升不到 1%。

**提示格式**。Whisper 是一个多任务模型，通过解码器提示中的特殊 token 引导：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — 语言标签，控制翻译与转录行为。
- `<|transcribe|>` 或 `<|translate|>` — 将任意语言输入翻译为英文输出，或原文转录。
- `<|notimestamps|>` — 跳过词级时间戳（更快）。

正是这个提示格式让一个模型能完成多种任务。把 `<|en|>` 改成 `<|fr|>` 就能转录法语。

**30 秒窗口**。所有输入固定为 30 秒。超长片段需要分块；短片段做填充。窗口原生不支持流式——这就是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**对数 Mel 归一化**。`(log_mel - mean) / std`，均值和标准差来自 Whisper 自己的训练语料。你*必须*使用 Whisper 的预处理（`whisper.audio.log_mel_spectrogram`），而不是 `librosa.feature.melspectrogram`。

### 2026 年的各版本

| 版本 | 参数量 | 延迟（A100） | WER（LibriSpeech-clean） |
|------|--------|-------------|------------------------|
| Tiny | 3900 万 | 1× 实时 | 5.4% |
| Base | 7400 万 | 1× | 4.1% |
| Small | 2.44 亿 | 1× | 3.0% |
| Medium | 7.69 亿 | 1× | 2.7% |
| Large-v3 | 15.5 亿 | 2× | 1.8% |
| Large-v3-turbo | 8.09 亿 | 8× | 1.58% |
| Whisper-Streaming（2024） | 15.5 亿 | 流式 | 2.0% |

### 微调

2026 年标准工作流：

1. 收集目标领域的 10–100 小时带对齐文本的音频。
2. 用 `transformers.Seq2SeqTrainer` 配合 `generate_with_loss` 回调训练。
3. 参数高效：在注意力层的 `q_proj`、`k_proj`、`v_proj` 上使用 LoRA，GPU 内存降低 4 倍，词错误率损失不超过 0.3%。
4. 数据少于 10 小时时冻结编码器，仅微调解码器。
5. 始终使用 Whisper 自己的分词器和提示格式，绝不替换分词器。

社区结果：在 20 小时医疗口述数据上微调 Medium，医疗词汇的 WER 从 12% 降至 4.5%。在 4 小时冰岛语数据上微调 Turbo，WER 从 18% 降至 6%。

## 动手实现

### 第一步：开箱即用运行 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

你应该始终覆盖的关键默认值：`temperature=0.0`（采样默认为 0.0 → 0.2 → 0.4 … 的回退链）、`condition_on_previous_text=False`（防止级联幻觉）以及 `no_speech_threshold=0.6`（静默检测）。

### 第二步：长音频分块处理

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 增加了：(1) Silero VAD 过滤，(2) 通过 wav2vec 2.0 做词级对齐，(3) 通过 `pyannote.audio` 做说话人分离。2026 年生产级转录的主力工具。

### 第三步：用 LoRA 微调

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

之后使用标准 Trainer 训练循环，每 1000 步保存检查点，在留出集上计算 WER 进行评估。

### 第四步：观察每一层学到了什么

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

用热力图可视化——你会看到对角线对齐：解码器的每个步骤扫过编码器帧。这条对角线就是 Whisper 对词时间戳的感知。

## 工程应用

2026 年技术栈：

| 场景 | 选型 |
|------|------|
| 通用英文、离线 | 通过 `whisperx` 使用 Large-v3-turbo |
| 移动端/边缘端 | 量化 Whisper-Tiny（int8）或 Moonshine |
| 多语言长音频 | 通过 `whisperx` 使用 Large-v3 + 说话人分离 |
| 低资源语言 | 用 LoRA 微调 Medium 或 Turbo |
| 流式（2 秒延迟） | Whisper-Streaming 或 Parakeet-TDT |
| 词级时间戳 | WhisperX（通过 wav2vec 2.0 强制对齐） |

`faster-whisper`（CTranslate2 后端）是 2026 年最快的 CPU+GPU 推理运行时——比原版快 4 倍，输出完全相同。

## 2026 年仍在发货的陷阱

- **静默时产生幻觉**。Whisper 在字幕数据上训练，会输出"Thanks for watching!"、"Subscribe!"、歌词等内容。调用前始终用 VAD 过滤。
- **`condition_on_previous_text` 级联**。一次幻觉会污染后续所有窗口。除非需要跨块的流畅性，否则设为 `False`。
- **短片段填充**。2 秒的片段填充到 30 秒后，会在尾部静默中产生幻觉。使用 `pad=False` 或 VAD 过滤。
- **错误的 Mel 统计**。使用 librosa 的 Mel 而不是 Whisper 的，会产生近乎随机的输出。请使用 `whisper.audio.log_mel_spectrogram`。

## 交付物

保存为 `outputs/skill-whisper-tuner.md`。为给定领域设计 Whisper 微调或推理流水线。

## 练习

1. **（简单）** 运行 `code/main.py`，它对 Whisper 风格的提示进行分词，计算解码形状预算，并打印 10 分钟音频的分块计划。
2. **（中等）** 安装 `faster-whisper`，转录一段 10 分钟的播客，对比人工转录计算词错误率，尝试 `language="auto"` vs 强制 `language="en"`。
3. **（困难）** 使用 HF `datasets` 选择一个 Whisper 表现较差的语言（如乌尔都语），在 2 小时数据上用 LoRA 微调 Medium 2 个 epoch，报告词错误率变化。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 30 秒窗口 | "Whisper 的限制" | 硬输入上限；超出则分块 |
| SOT | "转录开始" | `<\|startoftranscript\|>` 启动解码器提示 |
| 时间戳 token | "时间对齐" | 51K 词表中每 0.02 秒偏移对应一个特殊 token |
| Turbo | "快速版" | 4 层解码器，快 8 倍，词错误率几乎不变 |
| WhisperX | "长音频封装器" | VAD + Whisper + wav2vec 对齐 + 说话人分离 |
| LoRA 微调 | "高效微调" | 在注意力层加低秩适配器，仅训练约 0.3% 的参数 |
| 幻觉 | "静默失败" | Whisper 从噪声/静默中生成流畅英文 |

## 延伸阅读

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — 原始架构与训练方案
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — 4 层解码器，8 倍加速
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 长音频、词级对齐、说话人分离
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 后端，快 4 倍
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — 标准 LoRA/全量微调教程

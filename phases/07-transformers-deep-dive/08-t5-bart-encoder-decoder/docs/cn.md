# T5、BART——编码器-解码器模型

> 编码器理解。解码器生成。把它们放在一起，你就得到了一个为输入 → 输出任务而生的模型：翻译、摘要、改写、转录。

**类型：** 学习
**语言：** Python
**前置知识：** 第7阶段第05课（完整 Transformer）、第7阶段第06课（BERT）、第7阶段第07课（GPT）
**预计时间：** 约45分钟

## 问题背景

仅解码器的 GPT 和仅编码器的 BERT 各自为不同目标精简了 2017 年的架构。但许多任务天然就是输入-输出型：

- 翻译：英语 → 法语
- 摘要：5000 token 的文章 → 200 token 的摘要
- 语音识别：音频 token → 文字 token
- 结构化抽取：散文 → JSON

对于这些任务，编码器-解码器是最自然的选择。编码器生成源端的密集表示；解码器生成输出，在每一步交叉关注该表示。训练使用输出侧的偏移一位损失。与 GPT 相同的损失，只是以编码器输出为条件。

两篇论文定义了现代的最佳实践：

1. **T5**（Raffel et al. 2019）。"文本到文本迁移 Transformer"。每个 NLP 任务都重新定义为文本输入、文本输出。单一架构、单一词表、单一损失。在掩码片段预测上预训练（在输入中破坏片段，在输出中解码它们）。
2. **BART**（Lewis et al. 2019）。"双向自回归 Transformer"。去噪自编码器：以多种方式破坏输入（乱序、掩码、删除、旋转），让解码器重构原文。

2026 年，编码器-解码器格式在输入结构重要的场景中继续存在：

- Whisper（语音 → 文字）
- Google 的翻译栈
- 一些具有明确上下文-编辑结构的代码补全/修复模型
- 用于结构化推理任务的 Flan-T5 及其变体

仅解码器赢得了聚光灯，但编码器-解码器从未消失。

## 核心概念

### 前向循环

```
源 token ─▶ 编码器 ─▶ (N_src, d_model)  ──┐
                                           │
目标 token ─▶ 解码器块                     │
              ├─▶ 带掩码自注意力           │
              ├─▶ 交叉注意力 ◀─────────────┘
              └─▶ FFN
             ↓
           下一个 token 的 logit
```

关键在于，编码器对每个输入只运行一次。解码器自回归地运行，但在每一步都交叉关注*相同的*编码器输出。缓存编码器输出是长输入的免费加速。

### T5 预训练——片段破坏

随机选择输入的片段（平均长度 3 个 token，总计 15%）。用唯一哨兵替换每个片段：`<extra_id_0>`、`<extra_id_1>` 等。解码器只输出被破坏的片段加其哨兵前缀：

```
源：The quick <extra_id_0> fox jumps <extra_id_1> dog
目标：<extra_id_0> brown <extra_id_1> over the lazy
```

比预测整个序列的信号更廉价。在 T5 论文的消融实验中，与 MLM（BERT）和前缀语言模型（UniLM）相当。

### BART 预训练——多噪声去噪

BART 尝试五种噪声函数：

1. Token 掩码
2. Token 删除
3. 文本填充（掩码一个片段，解码器插入正确长度）
4. 句子排列
5. 文档旋转

文本填充 + 句子排列的组合在下游任务上产生了最佳结果。解码器总是重构原文。BART 的输出是完整序列，而非只有被破坏的片段——所以预训练计算量高于 T5。

### 推理

与 GPT 相同的自回归生成。贪心/束搜索/top-p 采样均适用。束搜索（宽度 4–5）是翻译和摘要的标准，因为输出分布比对话更窄。

### 2026 年如何选择

| 任务 | 用编码器-解码器？ | 原因 |
|------|----------------|------|
| 翻译 | 是，通常 | 明确的源序列；固定输出分布；束搜索有效 |
| 语音到文字 | 是（Whisper） | 输入模态与输出不同；编码器塑造音频特征 |
| 对话/推理 | 否，用仅解码器 | 没有持久的"输入"——对话就是序列 |
| 代码补全 | 通常否 | 带长上下文的仅解码器赢；Qwen 2.5 Coder 等代码模型是仅解码器 |
| 摘要 | 两者均可 | BART、PEGASUS 曾优于早期仅解码器基线；现代仅解码器 LLM 已与之持平 |
| 结构化抽取 | 两者均可 | T5 很简洁，因为"文本→文本"可以吸收任何输出格式 |

自 ~2022 年以来的趋势：仅解码器接管了原本属于编码器-解码器的任务，原因是：(a) 经过指令微调的仅解码器 LLM 通过提示泛化到任何任务；(b) 一种架构比两种更容易扩展；(c) RLHF 假定使用解码器。编码器-解码器在输入模态不同（语音、图像）或束搜索质量重要的场景中坚守阵地。

## 动手实现

见 `code/main.py`。我们为玩具语料库实现 T5 风格的片段破坏——本课最有用的单个片段，因为它出现在此后的每个编码器-解码器预训练方案中。

### 第一步：片段破坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """选择总计约 mask_rate 的 token 的片段。返回（破坏后输入，目标）。"""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式遵循 T5 约定：`<sent0> span0 <sent1> span1 ...`。破坏后的输入将未更改的 token 与片段位置的哨兵 token 交错排列。

### 第二步：验证往返重建

给定破坏后的输入和目标，重建原始句子。如果你的破坏是可逆的，前向传播定义是清晰的。这是一个正确性检查——真实训练不会这样做，但测试代价低，能捕获片段记录中的差一错误。

### 第三步：BART 噪声

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并展示结果。

## 工程应用

HuggingFace 参考写法：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 的技巧：任务名称放入输入文本。同一个模型处理数十个任务，因为每个任务都是文本输入、文本输出。2026 年这一模式已被指令微调的仅解码器模型所推广，但 T5 率先将其规范化。

## 交付物

见 `outputs/skill-seq2seq-picker.md`。该技能根据输入-输出结构、延迟和质量目标，为新任务选择编码器-解码器还是仅解码器。

## 练习

1. **（简单）** 运行 `code/main.py`，对一个 30-token 句子应用片段破坏，验证将非哨兵源 token 与解码的目标片段拼接后能重现原文。
2. **（中等）** 实现 BART 的 `text_infill` 噪声：用单个 `<mask>` token 替换随机片段，解码器必须推断正确的片段长度和内容。展示一个示例。
3. **（困难）** 在小型英语 → 猪拉丁语语料库（200 对）上微调 `flan-t5-small`。在留出的 50 对集合上测量 BLEU。与在相同数据和相同计算下微调 `Llama-3.2-1B` 进行比较。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 编码器-解码器 (Encoder-decoder) | "Seq2seq Transformer" | 两个堆栈：用于输入的双向编码器，用于输出的带交叉注意力的因果解码器 |
| 交叉注意力 (Cross-attention) | "源端与目标端交流的地方" | 解码器的 Q × 编码器的 K/V。编码器信息进入解码器的唯一途径 |
| 片段破坏 (Span corruption) | "T5 的预训练技巧" | 用哨兵 token 替换随机片段；解码器输出这些片段 |
| 去噪目标 (Denoising objective) | "BART 的游戏" | 对输入应用噪声函数，训练解码器重构干净序列 |
| 哨兵 token (Sentinel token) | "`<extra_id_N>` 占位符" | 在源端标记被破坏片段并在目标端重新标记的特殊 token |
| Flan | "指令微调的 T5" | 在 1800 多个任务上微调的 T5；使编码器-解码器在指令跟随上具有竞争力 |
| 束搜索 (Beam search) | "解码策略" | 每步保留前 k 个部分序列；翻译/摘要的标准做法 |
| 教师强制 (Teacher forcing) | "训练时的输入" | 训练时，向解码器提供真实的前一个输出 token，而非采样的 token |

## 延伸阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026 年典范编码器-解码器
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 参考实现

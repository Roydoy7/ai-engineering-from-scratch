# 序列到序列模型

> 两个 RNN 假装自己是翻译器。它们碰到的瓶颈，正是注意力机制存在的原因。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第8课（文本 CNN + RNN）、第3阶段第11课（PyTorch 入门）
**预计时间：** 约75分钟

## 问题背景

分类任务是将可变长度序列映射到单个标签；翻译任务则是将可变长度序列映射到另一个可变长度序列。输入和输出属于不同词汇表，可能是不同语言，长度也不一定对齐。

seq2seq 架构（Sutskever、Vinyals、Le，2014）用一个刻意简单的方案解决了这个问题：两个 RNN。一个读取源句子并生成一个固定大小的上下文向量，另一个读取这个向量并逐个 token 生成目标句子。跟第8课写的代码几乎一样，只是拼接方式不同。

学习这个有两个原因。第一，上下文向量瓶颈是 NLP 中最具教学价值的失败案例，它解释了注意力机制和 Transformer 擅长做什么的全部动机。第二，训练方法（教师强制、计划采样、推理时的集束搜索）至今仍适用于包括 LLM 在内的每一个现代生成系统。

## 核心概念

**编码器**：读取源句子的 RNN。最终隐藏状态是**上下文向量**——整个输入的固定大小摘要。理论上不丢失任何源信息。

**解码器**：另一个 RNN，用上下文向量初始化。每一步以上一步生成的 token 作为输入，输出目标词汇表上的概率分布。采样或取 argmax 选下一个 token，再反馈回去，重复，直到生成 `<EOS>` 或达到最大长度。

**训练**：每个解码步骤上的交叉熵损失，对整个序列求和，对两个网络标准的 BPTT。

**教师强制（Teacher Forcing）**：训练时，解码器在第 `t` 步的输入是位置 `t-1` 的**真实** token，而不是解码器自己的上一步预测。这会让训练更稳定；没有它，早期的错误会级联，模型永远学不会。但推理时只能用模型自己的预测，因此训练和推理之间始终存在分布差距，这个差距叫**暴露偏差（Exposure Bias）**。

**瓶颈所在**：编码器对源句子学到的一切都必须被压进那一个上下文向量。长句子丢失细节，罕见词被模糊化，词序颠倒（chat noir vs. black cat）只能靠死记硬背，而非计算得出。

注意力机制（第10课）通过让解码器查看**每一步**编码器的隐藏状态而不仅仅是最后一步来解决这个问题。这就是它全部的意义所在。

## 动手实现

### 第一步：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的形状是 `[batch, seq_len, hidden_dim]`——每个输入位置一个隐藏状态。`hidden` 的形状是 `[1, batch, hidden_dim]`——最后一步的状态。第8课说"对输出做池化用于分类"，这里我们把最后的隐藏状态当上下文向量，忽略每步的输出。

### 第二步：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器每次只调用一步。输入：一批单个 token 和当前隐藏状态。输出：下一个 token 的词汇表 logit 和更新后的隐藏状态。

### 第三步：带教师强制的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

两个重要参数：`ignore_index=0` 跳过填充 token 上的损失；`teacher_forcing_ratio` 控制每步使用真实 token 还是模型预测的概率，从 1.0（完全教师强制）开始，在训练过程中逐渐退火到约 0.5，以缩小暴露偏差。

### 第四步：推理循环（贪心解码）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码每步选最高概率的 token。它有偏航风险：一旦提交了一个 token，就无法撤回。**集束搜索（Beam Search）** 同时保留 top-k 个候选序列，最后选得分最高的完整序列。集束宽度 3-5 是标准配置。

### 第五步：演示瓶颈

在玩具复制任务上训练模型：源序列 `[a, b, c, d, e]`，目标序列 `[a, b, c, d, e]`。增加序列长度，观察精度：

```
seq_len=5   复制精度: 98%
seq_len=10  复制精度: 91%
seq_len=20  复制精度: 62%
seq_len=40  复制精度: 23%
```

单个 GRU 隐藏状态无法无损地记住 40 个 token 的输入。信息在每个编码器步骤都存在，但解码器只看得到最后那一步。注意力直接解决了这个问题。

## 工程应用

PyTorch 有 `nn.Transformer` 和基于 `nn.LSTM` 的 seq2seq 模板。Hugging Face 的 `transformers` 库提供了在数十亿 token 上训练好的完整编码器-解码器模型（BART、T5、mBART、NLLB）。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器已经用 Transformer 替换了 RNN。高层结构（编码器、解码器、逐 token 生成）与 2014 年的 seq2seq 论文完全相同，只是每个模块内部的机制不同。

### 什么时候还会用到基于 RNN 的 seq2seq

新项目几乎不会用。有几种特定例外：

- **流式翻译**：逐 token 消费输入，内存占用有界的场景。
- **设备端文本生成**：Transformer 内存开销太高时。
- **教学目的**：理解编码器-解码器瓶颈，是理解 Transformer 为什么胜出的最快路径。

### 暴露偏差及其缓解方案

- **计划采样（Scheduled Sampling）**：在训练期间逐渐退火教师强制比率，让模型学会从自己的错误中恢复。
- **最小风险训练（Minimum Risk Training）**：用句子级 BLEU 分数代替 token 级交叉熵训练，更接近你真正想优化的目标。
- **强化学习微调**：用评估指标奖励序列生成器，现代 LLM 的 RLHF 就用了这种方法。

以上三种方法同样适用于基于 Transformer 的生成系统。

## 交付物

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 练习

1. **（简单）** 实现玩具复制任务：在源序列等于目标序列的输入输出对上训练 GRU seq2seq，测量在长度 5、10、20 上的精度，复现瓶颈现象。
2. **（中等）** 添加集束宽度为 3 的集束搜索解码，在小型平行语料上对比贪心解码和集束搜索的 BLEU，记录集束搜索在哪些地方占优（通常是最后几个 token）、在哪些地方没有区别。
3. **（困难）** 在 1 万对改写数据集上微调 `facebook/bart-base`，将微调后模型的集束-4 输出与基础模型对比，报告 BLEU 并手工挑选 10 个定性样例。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 编码器 (Encoder) | "输入 RNN" | 读取源序列，产生逐步隐藏状态和最终上下文向量 |
| 解码器 (Decoder) | "输出 RNN" | 用上下文向量初始化，逐个生成目标 token |
| 上下文向量 (Context vector) | "摘要向量" | 编码器最终隐藏状态，固定大小，注意力要解决的瓶颈 |
| 教师强制 (Teacher forcing) | "用真实 token" | 训练时喂入真实的上一步 token，稳定学习过程 |
| 暴露偏差 (Exposure bias) | "训练-测试差距" | 模型训练时用真实 token，从未练习从自己的错误中恢复 |
| 集束搜索 (Beam search) | "更好的解码" | 每步同时保留 top-k 个候选序列，而非贪心提交 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 原始 seq2seq 论文，四页
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 引入 GRU 和编码器-解码器框架的论文
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 注意力论文，读完本课立刻去读这篇
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 可运行的 seq2seq + 注意力代码

# 文本 CNN 与 RNN

> 卷积学习 n-gram，循环保留记忆。两者都被注意力超越了，却在受限硬件上仍然重要。

**类型：** 构建
**语言：** Python
**前置知识：** 第3阶段第11课（PyTorch 入门）、第5阶段第3课（词嵌入）、第4阶段第2课（从零实现卷积）
**预计时间：** 约75分钟

## 问题背景

TF-IDF 和 Word2Vec 产生的平铺向量忽略了词序。建立在它们之上的分类器无法区分"狗咬人"和"人咬狗"。词序有时才是真正的信号所在。

在 Transformer 到来之前，两类架构填补了这个空缺。

**文本卷积网络（TextCNN）**：对词嵌入序列应用一维卷积。宽度为 3 的滤波器是可学习的三元组检测器，它跨越三个词并输出一个分数。叠加不同宽度（2、3、4、5）来检测多尺度模式，再经过最大池化得到固定长度的表示。扁平、并行、速度快。

**循环神经网络（RNN、LSTM、GRU）**：逐个处理 token，维护一个向前传递信息的隐藏状态。顺序执行，有记忆，能处理可变长度输入。从 2014 年到 2017 年主导了序列建模，然后注意力机制出现了。

本课从零实现这两种架构，并指出是什么缺陷催生了注意力机制。

## 核心概念

**TextCNN**（Kim，2014）：token 先经过嵌入层，宽度为 `k` 的一维卷积在连续 `k` 个词的嵌入上滑动，生成一张特征图。全局最大池化从特征图中取最强激活值，再将多个滤波器宽度的最大池化结果拼接起来，最后送入分类头。

为什么有效：一个滤波器就是一个可学习的 n-gram 检测器。最大池化与位置无关，所以"not good"无论出现在评论的开头还是中间，都会触发同一个特征。三种滤波器宽度各 100 个，就给你 300 个学到的 n-gram 检测器。训练是并行的，没有时间步上的顺序依赖。

**RNN**：在每个时间步 `t`，隐藏状态 `h_t = f(W * x_t + U * h_{t-1} + b)`。`W`、`U`、`b` 在时间步之间共享。时刻 `T` 的隐藏状态是整个前缀序列的摘要。对于分类任务，可以对 `h_1 ... h_T` 做池化（最大值、均值或取最后一步）。

普通 RNN 有梯度消失问题。**LSTM** 增加了门控机制，决定遗忘什么、存储什么、输出什么，从而在长序列上稳定梯度。**GRU** 将 LSTM 简化为两个门；参数更少，效果相当。

**双向 RNN** 正向跑一个 RNN，反向再跑一个，拼接两者的隐藏状态。每个 token 的表示同时看到左右上下文，对序列标注任务不可或缺。

## 动手实现

### 第一步：PyTorch 实现 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 将形状从 `[batch, seq_len, embed_dim]` 转为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 把中间轴当作通道。池化后的输出是固定大小的，与输入长度无关。

### 第二步：LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

这里对整个序列取最大池化，而不是取最后一步的隐藏状态。对于分类任务，最大池化通常优于取最后状态，因为在长序列末尾的信息往往主导最后的隐藏状态，而中间的关键信息会被稀释。

### 第三步：梯度消失演示（直觉理解）

没有门控的普通 RNN 无法学习长距离依赖。考虑一个玩具任务：预测 token `A` 是否出现在序列中任何位置。如果 `A` 在位置 1 而序列长度为 100，损失的梯度就需要经过 99 次循环权重的乘法才能流回去。如果权重小于 1，梯度消失；如果大于 1，梯度爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# 权重=0.9，步数100时：
#   0.9 ^ 100 ≈ 2.7e-5
# 从步骤100到步骤1的梯度实际上已经为零。
```

LSTM 用一条**细胞状态**来解决这个问题，它在网络中以加法方式流动（遗忘门对其做乘法缩放，但梯度仍能沿"高速公路"流动）。GRU 用更少的参数实现类似效果。两者都能在 100 步以上的序列中稳定训练。

### 第四步：为什么这仍然不够

即便有了 LSTM，三个问题依然存在：

1. **顺序瓶颈**：在长度为 1000 的序列上训练 RNN 需要 1000 个串行的前向/反向步骤，无法在时间维度上并行化。
2. **编码器-解码器中的固定大小上下文向量**：解码器只能看到编码器最终隐藏状态，整个输入被压缩进这一个向量里。长输入会丢失细节。第9课会直接讨论这个问题。
3. **长距离依赖的精度天花板**：LSTM 比普通 RNN 好，但要在 200+ 步中准确传递特定信息仍然很吃力。

注意力机制解决了这三个问题。Transformer 彻底丢掉了循环结构。第10课是转折点。

## 工程应用

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 已经可以直接用于生产。训练代码是标准套路。

Hugging Face 提供预训练嵌入，可以直接插入作为输入层：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

适用场景速查：

- **边缘端/设备端推理**：TextCNN + GloVe 嵌入比 Transformer 小 10-100 倍。部署目标是手机时，这套方案首选。
- **流式/在线分类**：RNN 每次处理一个 token；Transformer 需要完整序列。实时处理流入文本时，LSTM 仍然占优。
- **快速基线建立**：新任务快速迭代，TextCNN 在 CPU 上 5 分钟就能跑完。
- **数据量有限的序列标注**：BiLSTM-CRF（第6课）对于 1k-10k 标注句子的 NER 仍是生产级架构。

其他场景一律上 Transformer。

## 交付物

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 练习

1. **（简单）** 在自己构造的三分类玩具数据集上训练 TextCNN，验证使用多种滤波器宽度（2、3、4）的平均 F1 优于单一宽度（3）。
2. **（中等）** 为 LSTM 分类器分别实现最大池化、均值池化和取最后状态三种方式，在小型数据集上比较效果，记录哪种池化方式胜出并分析原因。
3. **（困难）** 结合第6课和本课，构建 BiLSTM-CRF NER 标注器，在 CoNLL-2003 上训练，与仅用 CRF 的基线和 BERT 微调方案对比训练时间、内存占用和 F1。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| TextCNN | 文本 CNN | 对词嵌入做全局最大池化的一维卷积堆叠，Kim（2014） |
| RNN（循环神经网络） | 循环网络 | 每个时间步更新隐藏状态：`h_t = f(W x_t + U h_{t-1})` |
| LSTM（长短期记忆） | 门控 RNN | 增加输入门/遗忘门/输出门和细胞状态，在长序列上训练稳定 |
| GRU（门控循环单元） | 简化版 LSTM | 两个门替代三个门，参数更少，精度相当 |
| 双向（Bidirectional） | 双向处理 | 正向 + 反向 RNN 拼接，每个 token 同时看到两侧上下文 |
| 梯度消失（Vanishing gradient） | 训练信号消失 | 普通 RNN 中权重<1的反复乘法，导致早期步骤的梯度趋近于零 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 论文，八页，易读
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 论文，出乎意料地清晰
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 让 LSTM 变得人人可懂的那组图解

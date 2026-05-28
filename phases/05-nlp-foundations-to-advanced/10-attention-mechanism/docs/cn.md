# 注意力机制——突破性进展

> 解码器不再眯眼盯着一个压缩的摘要，而是开始看完整的源序列。此后的一切都是注意力加工程。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第9课（序列到序列模型）
**预计时间：** 约45分钟

## 问题背景

第9课以一次有据可查的失败告终：在玩具复制任务上训练的 GRU 编码器-解码器，在长度 5 时精度 89%，到长度 80 时接近随机猜测。原因是结构性的，不是训练 bug：编码器所了解的一切都必须塞进一个固定大小的隐藏状态里，解码器看不到其他任何东西。

Bahdanau、Cho 和 Bengio 在 2014 年发表了一个三行的解决方案：不再只给解码器最终的编码器状态，而是保留每一步的编码器状态。在每个解码步骤，计算编码器状态的加权平均，权重的含义是"解码器现在需要多关注编码器位置 `i`？"这个加权平均就是上下文，它在每个解码步骤都会改变。

这就是全部的想法。Transformer 在此基础上扩展。自注意力把它应用到单一序列上。多头注意力并行运行多个头。但 2014 年的版本已经打破了瓶颈，理解了它之后，向 Transformer 的转变只是工程问题，不是概念问题。

## 核心概念

在每个解码步骤 `t`：

1. 用前一步的解码器隐藏状态 `s_{t-1}` 作为**查询（Query）**。
2. 把它与每个编码器隐藏状态 `h_1, ..., h_T` 打分，每个编码器位置产生一个标量。
3. 对分数做 softmax，得到注意力权重 `α_{t,1}, ..., α_{t,T}`，总和为 1。
4. 上下文向量 `c_t = Σ α_{t,i} * h_i`，即编码器状态的加权平均。
5. 解码器以 `c_t` 加上前一步的输出 token 为输入，生成下一个 token。

加权平均才是重点。当解码器需要翻译"Je"为"I"时，它把编码器在"Je"上的状态权重调高，其他调低。当需要"not"时，它把"pas"的权重调高。上下文向量在每一步都在重塑。

## 形状（第一次踩坑的地方）

每个注意力实现第一次都在这里出错。慢慢读。

| 变量 | 形状 | 备注 |
|------|------|------|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 若为 BiLSTM，`d_h = 2 * d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 一个向量 |
| 注意力分数 `e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | 对所有 `i` 做 softmax 后 |
| 上下文向量 `c_t` | `(d_h,)` | 与编码器状态形状相同 |

**Bahdanau（加性）评分**：`e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`

- `s_{t-1}` 形状 `(d_s,)`，`h_i` 形状 `(d_h,)`。
- `W_a` 形状 `(d_attn, d_s)`，`U_a` 形状 `(d_attn, d_h)`。
- tanh 内部的和形状为 `(d_attn,)`。
- `v_α` 形状 `(d_attn,)`，与它做内积后压缩为标量。**这就是 `v_α` 的作用**——不是什么魔法，它就是把注意力维度向量投影成标量分数的那个东西。

**Luong（乘性）评分**，三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`，要求 `d_s == d_h`，硬约束，编码器为双向时跳过。
- `general`：`e_{t,i} = s_t^T * W * h_i`，`W` 形状 `(d_s, d_h)`，去掉了维度相等的约束。
- `concat`：本质上和 Bahdanau 形式一样，通常不用前两种更便宜的情况下才用。

**一个值得点名的 Bahdanau / Luong 陷阱**：Bahdanau 用 `s_{t-1}`（生成当前词**之前**的解码器状态），Luong 用 `s_t`（生成**之后**的状态）。混淆它们会产生微妙错误的梯度，极难调试。选一篇论文，坚守它的约定。

## 动手实现

### 第一步：加性（Bahdanau）注意力

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上面的形状表检查：`encoder_states` 形状 `(T_enc, d_h)`，`projected_enc` 形状 `(T_enc, d_attn)`，`projected_dec` 形状 `(d_attn,)` 且会广播，`combined` 形状 `(T_enc, d_attn)`，`scores` 形状 `(T_enc,)`，`weights` 形状 `(T_enc,)`，`context` 形状 `(d_h,)`。

### 第二步：Luong dot 和 general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

各三行。这就是 Luong 论文落地的原因——大多数任务精度相当，代码少很多。

### 第三步：一个具体的数值示例

给定三个编码器状态（大致对应"cat"、"sat"、"mat"）和一个与第一个最对齐的解码器状态，注意力分布会集中在位置 0。如果解码器状态转向与最后一个对齐，注意力就会移到位置 2，上下文向量随之跟踪变化。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行获胜。把解码器状态改为靠近第三个编码器状态，再观察权重如何移动。就这么多。注意力就是显式的对齐。

### 第四步：这为什么是通往 Transformer 的桥梁

把上面的语言翻译成 Q/K/V：

- **Query（查询）** = 解码器状态 `s_{t-1}`
- **Key（键）** = 编码器状态（用来打分的目标）
- **Value（值）** = 编码器状态（加权求和的对象）

在经典注意力中，键和值是同一个东西。自注意力将它们分开：你可以用不同的可学习投影矩阵把序列对自身做查询，Key 和 Value 来自不同投影。多头注意力用不同的可学习投影并行运行多次。Transformer 把这整套叠加很多层，并丢掉了 RNN。

数学是一样的，形状也是一样的。从 Bahdanau 注意力到缩放点积注意力的概念跳跃，基本上只是符号问题。

## 工程应用

PyTorch 和 TensorFlow 直接提供注意力模块。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是一个 Transformer 注意力层：5 个位置的查询批次，10 个位置的键/值批次，每个 128 维，8 个头。`output` 是经上下文增强后的新查询，`weights` 是可以可视化的 5×10 对齐矩阵。

### 经典注意力仍然重要的场合

- **教学目的**：单头、单层、基于 RNN 的版本让每个概念都清晰可见。
- **Transformer 放不下的设备端序列任务**。
- **阅读 2014-2017 年的论文**：不了解 Bahdanau 的约定就会看错。
- **机器翻译中的细粒度对齐分析**：即使在 Transformer 模型上，原始注意力权重也是可解释性工具，阅读它们需要知道它们是什么。

### 注意力权重作为解释的陷阱

注意力权重看起来可解释。它们是在各位置上加和为 1 的权重，可以绘图，高值意味着"关注了这里"，审稿人很喜欢。

但它们并没有看起来那么可解释。Jain 和 Wallace（2019）表明，在某些任务中，注意力分布可以被打乱并替换为任意替代值，而不改变模型预测。永远不要在没有消融实验或反事实检查的情况下，把注意力权重作为推理证据来报告。

## 交付物

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 练习

1. **（简单）** 实现 `softmax` 掩码，让编码器中的填充 token 注意力权重为零，在包含可变长度序列的批次上测试。
2. **（中等）** 为 Luong `general` 形式添加多头注意力：把 `d_h` 分成 `n_heads` 组，每头独立运行注意力，再拼接结果，验证单头情况与之前实现一致。
3. **（困难）** 在第9课的玩具复制任务上训练带 Bahdanau 注意力的 GRU 编码器-解码器，绘制精度随序列长度的变化曲线，与无注意力基线对比，应该能看到随长度增加差距扩大，证明注意力解除了瓶颈。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 注意力 (Attention) | "看东西" | 值序列的加权平均，权重由查询-键相似度计算 |
| 查询、键、值 (Query, Key, Value) | QKV | 三个投影：Q 提问，K 是匹配目标，V 是返回内容 |
| 加性注意力 (Additive attention) | Bahdanau | 前馈评分：`v^T tanh(W q + U k)` |
| 乘性注意力 (Multiplicative attention) | Luong dot/general | 评分为 `q^T k` 或 `q^T W k`，更便宜，大多数任务精度相当 |
| 对齐矩阵 (Alignment matrix) | "那张好看的图" | 注意力权重构成的 `(T_dec, T_enc)` 网格，读它可以看模型关注了什么 |

## 延伸阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 原始论文
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 三种评分变体及对比
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 可解释性警告
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — 带 PyTorch 的可运行逐步讲解

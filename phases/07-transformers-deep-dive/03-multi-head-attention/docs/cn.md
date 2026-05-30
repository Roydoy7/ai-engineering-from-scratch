# 多头注意力

> 单头注意力一次学习一种关系，八头就能学八种。头是免费的，多要几个。

**类型：** 构建
**语言：** Python
**前置知识：** 第7阶段第02课（从零实现自注意力）
**预计时间：** 约75分钟

## 问题背景

单个自注意力头只能计算一个注意力矩阵。该矩阵捕获一种关系——通常是在训练信号上最小化损失的那种。如果你的数据同时包含主谓一致、共指消解、长距离语篇结构和句法分块，单个头会把它们混入一个 softmax 分布，丢失一半信号。

2017 年 Vaswani 论文给出的解法：并行运行多个注意力函数，每个都有自己的 Q、K、V 投影，然后拼接输出。每个头在维度为 `d_model / n_heads` 的较小子空间中操作。参数总量不变，表达能力提升。

多头注意力是 2026 年每个 Transformer 的默认配置。唯一的争论在于用*多少*头，以及键和值是否共享投影（分组查询注意力、多查询注意力、多头潜在注意力）。

## 核心概念

**拆分。** 取形状为 `(N, d_model)` 的 `X`，将其投影为各自形状为 `(N, d_model)` 的 Q、K、V。重塑为 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`，再转置为 `(n_heads, N, d_head)`。

**并行计算注意力。** 在每个头内运行缩放点积注意力。每个头产生 `(N, d_head)`。各头在嵌入的不同子空间上操作，注意力计算期间互不干扰。

**拼接并投影。** 将各头堆叠回 `(N, d_model)`，乘以形状为 `(d_model, d_model)` 的学习输出矩阵 `W_o`。`W_o` 是各头得以混合的地方。

**为什么有效。** 每个头可以专门化而不必与其他头争夺表示预算。2019–2024 年的探针研究发现了不同的头分工：位置头、关注前一个 token 的头、复制头、命名实体头、归纳头（上下文学习的底层机制）。

**2026 年的变体谱系：**

| 变体 | Q 头数 | K/V 头数 | 使用者 |
|------|--------|---------|--------|
| 多头注意力 (MHA) | N | N | GPT-2、BERT、T5 |
| 多查询注意力 (MQA) | N | 1 | PaLM、Falcon |
| 分组查询注意力 (GQA) | N | G（如 N/8） | Llama 2 70B、Llama 3+、Qwen 2+、Mistral |
| 多头潜在注意力 (MLA) | N | 压缩为低秩 | DeepSeek-V2、V3 |

GQA 是现代默认方案，因为它将 KV 缓存内存减少了 `N/G` 倍，同时保持接近完整的质量。MLA 更进一步，将 K/V 压缩到潜在空间，计算时再投影回来——花费 FLOP，节省更多内存。

## 动手实现

### 第一步：从单头注意力拆分出多头

取第02课的 `SelfAttention`，用拆分/拼接包装它。详见 `code/main.py` 的 NumPy 实现；核心逻辑是：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次 reshape，一次 transpose。没有循环。这正是 PyTorch 在 `nn.MultiheadAttention` 内部所做的。

### 第二步：对每个头运行缩放点积注意力

每个头获得 Q、K、V 各自的切片。注意力变成批量矩阵乘：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真实硬件上，`Qh @ Kh.transpose(...)` 就是一次 `bmm`。GPU 看到的是一个形状为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` 的批量矩阵乘。增加头数是免费的。

### 第三步：分组查询注意力变体

只有键和值的投影发生变化。Q 获得 `n_heads` 组；K 和 V 获得 `n_kv_heads < n_heads` 组，并通过重复来匹配 Q：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

推理时这可以节省内存，因为 KV 缓存中只存 `n_kv_heads` 份，而不是 `n_heads` 份。Llama 3 70B 使用 64 个查询头配 8 个 KV 头——缓存缩小了 8 倍。

### 第四步：探究每个头学到了什么

在一个短句上运行 4 头的 MHA。对每个头打印 `(N, N)` 注意力矩阵。你会看到即使是随机初始化，不同的头也会捕捉不同的结构——这部分是信号，部分是子空间中的旋转对称性。

## 工程应用

在 PyTorch 中，单行写法：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 的 GQA 写法：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention 在 CUDA 上自动分发 Flash Attention。
# GQA 时，Q 形状为 (B, n_heads, N, d_head)，K,V 形状为 (B, n_kv_heads, N, d_head)。
# PyTorch 自动处理重复。
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**应该用多少头？** 2026 年生产模型的经验法则：

| 模型规模 | d_model | n_heads | d_head |
|---------|---------|---------|--------|
| 小型（约1.25亿） | 768 | 12 | 64 |
| 基础（约3.5亿） | 1024 | 16 | 64 |
| 大型（约10亿） | 2048 | 16 | 128 |
| 前沿（约700亿） | 8192 | 64 | 128 |

`d_head` 几乎总是 64 或 128。它是单个头"能看到多少"的单位。低于 32，头会开始与缩放因子 `sqrt(d_head)` 产生冲突；高于 256，则失去"许多小专家"的优势。

## 交付物

见 `outputs/skill-mha-configurator.md`。该技能根据参数预算、序列长度和部署目标，为新的 Transformer 推荐头数、KV 头数和投影策略。

## 练习

1. **（简单）** 取 `code/main.py` 中的 MHA，在 `d_model=64` 固定的情况下将 `n_heads` 从 1 改到 16。在合成复制任务的微型单层模型上绘制损失曲线。更多的头有帮助、停滞还是有害？
2. **（中等）** 实现 MQA（所有查询头共享一个 KV 头）。测量参数量相比完整 MHA 下降了多少。计算在 N=2048 推理时 KV 缓存大小缩小了多少。
3. **（困难）** 实现迷你版多头潜在注意力：将 K、V 压缩为秩为 `r` 的潜在向量，在 KV 缓存中存潜在向量，注意力时解压。在什么 `r` 值下，缓存内存低于完整 MHA 的 1/8，同时验证集困惑度保持在 1 位以内？

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 头 (Head) | "单个注意力回路" | 一组维度为 `d_head = d_model / n_heads` 的 Q/K/V 投影，有自己的注意力矩阵 |
| d_head（头维度） | "头的维度" | 每个头的隐藏宽度；生产中几乎总是 64 或 128 |
| 拆分/合并 (Split/Combine) | "reshape 技巧" | `(N, d_model) ↔ (n_heads, N, d_head)` 的 reshape+transpose |
| W_o（输出投影） | "输出投影矩阵" | 拼接各头后应用的 `(d_model, d_model)` 矩阵；各头在此混合 |
| MQA（多查询注意力） | "单 KV 头" | 所有查询头共享单一 K/V 投影，KV 缓存最小，质量略有损失 |
| GQA（分组查询注意力） | "Llama 2 以来的默认方案" | `n_kv_heads < n_heads`，通过重复匹配 Q |
| MLA（多头潜在注意力） | "DeepSeek 的技巧" | K/V 压缩为低秩潜在向量，注意力时解压 |
| 归纳头 (Induction head) | "上下文学习背后的回路" | 一对头，检测之前的出现并复制其后续内容 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 原始多头注意力规范
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA 论文
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 训练后将 MHA 转换为 GQA 的方法
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA 及其相比 MHA/GQA 在缓存内存上的优势
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 关于头实际学到什么的机制分析

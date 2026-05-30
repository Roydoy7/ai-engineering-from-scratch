# 位置编码——正弦编码、RoPE、ALiBi

> 注意力是置换不变的。"猫坐在垫子上"和"垫子在上坐猫"在没有位置信号的情况下会产生相同的输出。三种算法来修复这个问题——每种对"位置"的含义下了不同的赌注。

**类型：** 构建
**语言：** Python
**前置知识：** 第7阶段第02课（自注意力）、第7阶段第03课（多头注意力）
**预计时间：** 约45分钟

## 问题背景

缩放点积注意力是顺序盲的。注意力矩阵 `softmax(Q K^T / √d) V` 由成对相似度计算得出。打乱 `X` 的行，输出的行会以相同方式打乱。注意力内部没有任何东西关心位置。

对于词袋模型而言这不是 bug。但对于语言、代码、音频、视频——任何顺序承载意义的场景——这是致命的。

解决方案是以某种方式将位置注入嵌入。三个时代的答案：

1. **绝对正弦编码**（Vaswani 2017）。在嵌入中加入位置的 `sin/cos`。简单，无需学习，但超出训练长度后外推效果差。
2. **RoPE——旋转位置嵌入**（Su 2021）。按与位置成比例的角度旋转 Q 和 K 向量。直接在点积中编码*相对*位置。2026 年的主流方案。
3. **ALiBi——带线性偏置的注意力**（Press 2022）。完全跳过嵌入技巧；根据距离在注意力分数上添加每头线性惩罚。长度外推效果极佳。

截至 2026 年，几乎所有前沿开源模型都使用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型使用 ALiBi 或其现代变体。绝对正弦编码已成历史。

## 核心概念

### 绝对正弦编码

预计算形状为 `(max_len, d_model)` 的固定矩阵 `PE`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在注意力之前令 `X' = X + PE[:N]`。每个维度是不同频率的正弦波。模型从相位模式中学习读取位置。超出 `max_len` 则失效：模型从未被告知位置 2048 处会发生什么（如果只看过 0–2047）。

### RoPE

旋转 Q 和 K 向量（而非嵌入）。对于一对维度 `(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head)，base 默认为 10000
```

对位置 `pos_k` 处的键应用相同旋转。点积 `q'_m · k'_n` 变为仅关于 `(m - n)` 的函数。也就是说：**注意力分数仅取决于相对距离**，尽管旋转是基于绝对位置计算的。绝妙的技巧。

扩展 RoPE：可以缩放 `base`（NTK-aware、YaRN、LongRoPE）以在不重新训练的情况下外推到更长上下文。Llama 3 就是这样将上下文从 8K 扩展到 128K 的。

### ALiBi

跳过嵌入技巧，直接对注意力分数添加偏置：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是与头相关的斜率（如 `1 / 2^(8·h/H)`）。距离近的 token 得到加成；距离远的 token 受到惩罚。无需训练开销。论文显示其长度外推优于正弦编码，在原始训练长度上与 RoPE 相当。

### 2026 年如何选择

| 变体 | 外推能力 | 训练代价 | 使用者 |
|------|---------|---------|--------|
| 绝对正弦编码 | 差 | 免费 | 原始 Transformer、早期 BERT |
| 学习型绝对编码 | 无 | 极小 | GPT-2、GPT-3 |
| RoPE | 良好（配合缩放） | 免费 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 极佳 | 微调阶段 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 极佳 | 免费 | BLOOM、MPT、Baichuan |

RoPE 胜出，因为它能嵌入注意力而不改变架构、编码相对位置，且其 `base` 超参数为长上下文微调提供了清晰的调节旋钮。

## 动手实现

### 第一步：正弦编码

见 `code/main.py`。4 行代码：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一个注意力层之前将其加到嵌入矩阵上。

### 第二步：对 Q、K 应用 RoPE

RoPE 对 Q 和 K 就地操作。对每对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键：对位置 `m` 处的 Q 和位置 `n` 处的 K 应用相同函数。它们的点积在每对坐标上都会多出一个 `cos((m-n)·θ_i)` 因子。注意力就这样免费学到了相对位置。

### 第三步：ALiBi 斜率与偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # softmax 前加到注意力分数上
```

将 `bias[h]` 加到头 `h` 的 `(seq_len, seq_len)` 注意力分数矩阵上，然后再做 softmax。

### 第四步：验证 RoPE 的相对距离性质

任取两个随机向量 `a, b`。分别按 `(pos_a, pos_b)` 旋转，再按 `(pos_a + k, pos_b + k)` 旋转。两次点积必须在浮点误差范围内一致。这就是 RoPE 的全部意义——它对绝对偏移不变，只有相对间距才重要。

## 工程应用

PyTorch 2.5+ 在 `torch.nn.functional` 中内置了 RoPE 工具。大多数生产代码使用 `flash_attn` 或 `xformers`，RoPE 在注意力核中应用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年的长上下文技巧：**

- **NTK-aware 插值。** 将 `base` 缩放为 `base * (scale_factor)^(d/(d-2))`，用于从 4K 扩展到 16K+。
- **YaRN。** 更智能的插值，在长上下文上保持注意力熵。Llama 3.1 128K 使用它。
- **LongRoPE。** 微软 2024 年的方法，用进化搜索为每个维度选择缩放因子。Phi-3-Long 使用它。
- **位置插值 + 微调。** 直接按扩展系数缩小位置，然后用 1–5B token 微调。出人意料地有效。

## 交付物

见 `outputs/skill-positional-encoding-picker.md`。该技能根据目标上下文长度、外推需求和训练预算，为新模型选择编码策略。

## 练习

1. **（简单）** 以 `max_len=512, d=128` 将正弦 `PE` 矩阵绘制为热力图。确认"维度索引越大，条纹越宽"的模式。
2. **（中等）** 实现 NTK-aware RoPE 缩放。在长度 256 的序列上训练一个小型语言模型，然后在有无缩放的情况下分别在长度 1024 上测试。测量困惑度。
3. **（困难）** 在同一个注意力模块中实现 ALiBi 和 RoPE。在长度 512 的复制任务上训练 4 层 Transformer。测试时外推到 2048。比较退化程度。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 位置编码 (Positional encoding) | "告诉注意力顺序" | 任何编码位置并加入嵌入或注意力的信号 |
| 正弦编码 (Sinusoidal) | "原始方案" | 以几何频率的 `sin/cos` 加入嵌入；无法外推 |
| RoPE（旋转位置嵌入） | "旋转嵌入" | 按位置相关角度旋转 Q、K；点积编码相对距离 |
| ALiBi（带线性偏置的注意力） | "线性偏置技巧" | 在注意力分数上加 `-m·\|i-j\|`；不需要嵌入，外推能力极佳 |
| base（RoPE 的旋钮） | "RoPE 的调节参数" | RoPE 中的频率缩放器；增大可在推理时扩展上下文 |
| NTK-aware（NTK 感知） | "一种 RoPE 缩放技巧" | 重新缩放 `base`，防止上下文扩展时高频维度被过度压缩 |
| YaRN | "更精细的方案" | 每维度插值+外推，保持注意力熵 |
| 外推 (Extrapolation) | "超出训练长度仍有效" | 位置方案能否在超过训练 `max_len` 时给出正确输出 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始正弦编码
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 最先进的 RoPE 缩放
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 长上下文论文
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Phi-3-Long 使用的微软方法
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 每种 RoPE 缩放方案的生产级实现

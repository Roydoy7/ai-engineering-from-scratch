# 完整 Transformer——编码器 + 解码器

> 注意力是主角。其他一切——残差、归一化、前馈网络、交叉注意力——都是让你堆叠深度的脚手架。

**类型：** 构建
**语言：** Python
**前置知识：** 第7阶段第02课（自注意力）、第7阶段第03课（多头注意力）、第7阶段第04课（位置编码）
**预计时间：** 约75分钟

## 问题背景

单个注意力层是特征提取器，而非完整模型。每层一次矩阵乘对语言而言容量不足。你需要深度——而没有正确"管道"的深度会崩溃。

2017 年 Vaswani 论文封装了六个设计决策，将单个注意力层变成可堆叠的块。此后每一个 Transformer——仅编码器（BERT）、仅解码器（GPT）、编码器-解码器（T5）——都继承了同一个骨架。2026 年这些块已经精化（RMSNorm、SwiGLU、Pre-norm、RoPE），但骨架完全相同。

本课讲骨架。后续课程对其专门化——第06课讲编码器，第07课讲解码器，第08课讲编码器-解码器。

## 核心概念

### 六个组成部分

1. **嵌入 + 位置信号。** Token → 向量。通过 RoPE（现代）或正弦编码（经典）注入位置。
2. **自注意力。** 每个位置关注其他所有位置。在解码器中带掩码。
3. **前馈网络 (FFN)。** 按位置应用的两层 MLP：`W_2 · activation(W_1 · x)`。默认扩展比 4×。
4. **残差连接。** `x + sublayer(x)`。没有它，梯度在 ~6 层后消失。
5. **层归一化。** `LayerNorm` 或 `RMSNorm`（现代）。稳定残差流。
6. **交叉注意力（仅解码器）。** 查询来自解码器，键和值来自编码器输出。

### 编码器块（BERT、T5 编码器使用）

```
x → LN → MHA(自注意力) → + → LN → FFN → + → 输出
                         ^              ^
                         |              |
                         └─── 残差 ────┘
```

编码器是双向的。不带掩码。所有位置可以看到所有位置。

### 解码器块（GPT、T5 解码器使用）

```
x → LN → MHA(带掩码自注意力) → + → LN → MHA(交叉注意力) → + → LN → FFN → + → 输出
```

解码器每个块有三个子层。中间那个——交叉注意力——是信息从编码器流向解码器的唯一通道。在纯解码器架构（GPT）中，交叉注意力被省略，只剩带掩码自注意力 + FFN。

### Pre-norm vs Post-norm

原始论文：`x + sublayer(LN(x))` vs `LN(x + sublayer(x))`。Post-norm 在 2019 年前后失去青睐——在没有仔细预热的情况下难以训练深层网络。Pre-norm（子层*之前* `LN`）是 2026 年的默认方案：Llama、Qwen、GPT-3+、Mistral 都使用它。

### 2026 年现代化的块

Vaswani 2017 使用 LayerNorm + ReLU。现代堆栈两者都替换了。生产块实际的样子：

| 组件 | 2017 | 2026 |
|------|------|------|
| 归一化 | LayerNorm | RMSNorm |
| FFN 激活 | ReLU | SwiGLU |
| FFN 扩展比 | 4× | 2.6×（SwiGLU 使用三个矩阵，总参数量相当） |
| 位置 | 绝对正弦编码 | RoPE |
| 注意力 | 完整 MHA | GQA（或 MLA） |
| 偏置项 | 有 | 无 |

RMSNorm 省去了 LayerNorm 的均值中心化（少一次减法），节省计算量，且经验上至少同样稳定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM 和 Qwen 论文中始终以约 0.5 点困惑度优于 ReLU/GELU FFN。

### 参数量

对于 `d_model = d`、FFN 扩展比 `r` 的单个块：

- MHA：`4 · d²`（Q、K、V、O 投影）
- FFN（SwiGLU）：`3 · d · (r · d)` ≈ `3rd²`
- 归一化层：可忽略

在 `d = 4096, r = 2.6, layers = 32`（大致为 Llama 3 8B）时，合计：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B 参数/层 × 32 ≈ 7B`（加上嵌入和输出头）。与公开数字吻合。

## 动手实现

### 第一步：基础构件

使用第03课中的小型 `Matrix` 类（已复制到本文件以保持独立性）：

- `layer_norm(x, eps=1e-5)` — 减均值，除以标准差
- `rms_norm(x, eps=1e-6)` — 除以 RMS。不减均值
- `gelu(x)` 和 `silu(x) * W3 x`（SwiGLU）
- `ffn_swiglu(x, W1, W2, W3)`
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`

完整接线见 `code/main.py`。

### 第二步：接线 2 层编码器和 2 层解码器

将它们堆叠起来。将编码器输出传入每个解码器交叉注意力层。在输出投影前加一个最终的 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 第三步：在玩具示例上运行前向传播

输入一个 6-token 的源序列和一个 5-token 的目标序列。验证输出形状为 `(5, vocab)`。不需要训练——本课关注架构，而非损失。

### 第四步：换入 RMSNorm + SwiGLU

用 RMSNorm 和 SwiGLU 替换 LayerNorm 和 ReLU-FFN。确认形状仍然匹配。这就是 2026 年现代化所需的一次函数替换。

## 工程应用

PyTorch/TF 参考实现：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但 2026 年大多数生产代码都自己实现块，原因是：

- Flash Attention 在注意力内部调用，不通过 `nn.MultiheadAttention`
- GQA / MLA 不在标准库参考实现中
- RoPE、RMSNorm、SwiGLU 不是 PyTorch 的默认配置

HF `transformers` 有干净的参考块值得阅读：`modeling_llama.py` 是 2026 年典范的仅解码器块，约 500 行，值得通读一遍。

**编码器 vs 解码器 vs 编码器-解码器——如何选择：**

| 需求 | 选型 | 示例 |
|------|------|------|
| 分类、嵌入、文本上的问答 | 仅编码器 | BERT、DeBERTa、ModernBERT |
| 文本生成、对话、代码、推理 | 仅解码器 | GPT、Llama、Claude、Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | 编码器-解码器 | T5、BART、Whisper |

仅解码器赢得了语言任务，因为它扩展最干净，既能理解又能生成。当输入有明确的"源序列"身份时（翻译、语音识别、结构化任务），编码器-解码器仍然最佳。

## 交付物

见 `outputs/skill-transformer-block-reviewer.md`。该技能根据 2026 年默认标准审查新的 Transformer 块实现，标记缺失部分（pre-norm、RoPE、RMSNorm、GQA、FFN 扩展比）。

## 练习

1. **（简单）** 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 下统计 encoder_block 的参数量。通过实现该块并使用 `sum(p.numel() for p in block.parameters())` 验证。
2. **（中等）** 从 post-norm 切换到 pre-norm。初始化两者，在随机输入经过 12 层堆叠后测量激活范数。Post-norm 的激活应该爆炸；Pre-norm 的应该保持有界。
3. **（困难）** 在玩具复制任务（反转复制 `x`）上实现 4 层编码器-解码器。训练 100 步，报告损失。换入 RMSNorm + SwiGLU + RoPE——损失是否下降？

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 块 (Block) | "一个 Transformer 层" | 归一化+注意力+归一化+FFN 的堆叠，包裹在残差连接中 |
| 残差 (Residual) | "跳跃连接" | `x + f(x)` 输出；使梯度流过深层堆栈 |
| Pre-norm（前归一化） | "先归一化，后计算" | 现代方案：`x + sublayer(LN(x))`。无需预热即可训练更深 |
| RMSNorm | "没有均值的 LayerNorm" | 除以 RMS；少一次操作，经验稳定性相当 |
| SwiGLU | "大家都换过去的 FFN" | `Swish(W1 x) ⊙ W3 x → W2`。在语言模型困惑度上优于 ReLU/GELU |
| 交叉注意力 (Cross-attention) | "解码器如何看到编码器" | MHA，Q 来自解码器，K/V 来自编码器输出 |
| FFN 扩展比 (FFN expansion) | "中间 MLP 有多宽" | 隐藏层与 d_model 的比值，通常 4（LayerNorm）或 2.6（SwiGLU） |
| 无偏置 (Bias-free) | "去掉 +b 项" | 现代堆栈在线性层中省略偏置；困惑度略有提升，模型更小 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始块规范
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 为什么 pre-norm 在深层网络中优于 post-norm
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 论文
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 2026 年典范仅解码器块

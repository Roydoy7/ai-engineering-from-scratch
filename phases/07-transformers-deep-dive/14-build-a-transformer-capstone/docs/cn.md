# 从零构建 Transformer——综合项目

> 十三节课。一个模型。没有捷径。

**类型：** 构建
**语言：** Python
**前置知识：** 第7阶段第01课至第13课。不要跳过。
**预计时间：** 约120分钟

## 问题背景

你读过每一篇论文。你已经实现了注意力、多头拆分、位置编码、编码器和解码器块、BERT 和 GPT 损失、MoE、KV 缓存。现在把它们在一个真实任务上整合运行。

综合项目：在字符级语言建模任务上端到端训练一个小型仅解码器 Transformer。它读莎士比亚，生成新的莎士比亚。它足够小，可以在笔记本电脑上 10 分钟内训练完。它足够正确，换上更大的数据集和更长的训练就能得到真正的语言模型。

这是本课程的"nanoGPT"。它并不原创——Karpathy 2023 年的 nanoGPT 教程是每个学生至少写一次的参考实现。我们借鉴其结构，围绕我们涵盖的内容重新组织。

## 核心概念

架构，带注释：

```
输入 token (B, N)
   │
   ▼
token 嵌入 + 位置嵌入  ◀── 第04课（RoPE 选项）
   │
   ▼
┌──── 块 × L ────────────────────┐
│  RMSNorm                       │  ◀── 第05课
│  MultiHeadAttention（因果）     │  ◀── 第03课 + 第07课（因果掩码）
│  残差                           │
│  RMSNorm                       │
│  SwiGLU FFN                    │  ◀── 第05课
│  残差                           │
└─────────────────────────────── ┘
   │
   ▼
最终 RMSNorm
   │
   ▼
lm_head（与 token 嵌入矩阵绑定）
   │
   ▼
logits (B, N, V)
   │
   ▼
偏移一位的交叉熵              ◀── 第07课
```

### 我们提供什么

- `GPTConfig` — 配置所有超参数的单一入口
- `MultiHeadAttention` — 因果、批量，带可选的 Flash 风格路径（PyTorch 的 `scaled_dot_product_attention`）
- `SwiGLUFFN` — 现代 FFN
- `Block` — pre-norm、残差包装的注意力 + FFN
- `GPT` — 嵌入、堆叠块、语言模型头、generate()
- 带 AdamW、余弦学习率、梯度裁剪的训练循环
- 莎士比亚文本上的字符级分词器

### 我们不提供什么

- RoPE——在第04课概念性实现。这里为简单起见使用学习型位置嵌入。练习要求你换入 RoPE。
- 生成时的 KV 缓存——每个生成步骤对完整前缀重新计算注意力。更慢但更简单。练习要求你添加 KV 缓存。
- Flash Attention——PyTorch 2.0+ 在输入匹配时自动分发；我们使用 `F.scaled_dot_product_attention`。
- MoE——每块单个 FFN。你在第11课看到了 MoE。

### 目标指标

在 Mac M2 笔记本上，一个 4 层、4 头、d_model=128 的 GPT 在 `tinyshakespeare.txt` 上训练 2000 步：

- 训练损失在约 6 分钟内从 ~4.2（随机）收敛到 ~1.5
- 采样输出看起来像莎士比亚风格：古语词汇、换行、"ROMEO:" 等专有名词涌现
- 验证损失（留出最后 10% 的文本）与训练损失紧密跟踪；在这个规模/预算下没有过拟合

## 动手实现

本课使用 PyTorch。安装 `torch`（CPU 版本即可）。见 `code/main.py`。脚本处理：

- 若缺失则下载 `tinyshakespeare.txt`（或读取本地副本）
- 字节级字符分词器
- 90/10 训练/验证集划分
- 支持的硬件上带 bf16 自动转换的训练循环
- 训练完成后采样

### 第一步：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个独特字符。微型词表。词表大小只需 4 字节。没有 BPE，没有分词器烦恼。

### 第二步：模型

见 `code/main.py`。块就是第05课的教科书内容——pre-norm、RMSNorm、SwiGLU、因果 MHA。4/4/128 配置的参数量：约 80 万。

### 第三步：训练循环

取随机一批长度为 256 的 token 窗口。前向传播。偏移一位交叉熵。反向传播。AdamW 步骤。记录。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 第四步：采样

给定提示，反复前向传播，从 top-p logit 中采样，追加，继续。500 个 token 后停止。

### 第五步：阅读输出

2000 步之后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚。但是莎士比亚风格的。对于约 80 万参数和 6 分钟的笔记本电脑训练，这是明显的胜利。

## 工程应用

这个综合项目是一个参考架构。三个扩展可以将其升级为真正有用的东西：

1. **换分词器。** 使用 BPE（如 `tiktoken.get_encoding("cl100k_base")`）。词表大小从 65 跳到约 50000。模型容量需要相应扩大。
2. **在更大语料库上训练。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。在单张 A100 上用 100 亿 token 训练 1.25 亿参数的 GPT 需要约 24 小时。
3. **添加 RoPE + KV 缓存 + Flash Attention。** 下面的练习逐步指导你完成每个步骤。

最终得到一个生成流利英语的 1.25 亿参数 GPT。不是前沿模型。但相同的代码路径——只是更大——就是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年训练研究检查点所使用的。

## 交付物

见 `outputs/skill-transformer-review.md`。该技能根据之前所有 13 课对从零开始的 Transformer 实现进行正确性审查。

## 练习

1. **（简单）** 运行 `code/main.py`。验证你的训练模型最后一步的验证损失低于 2.0。将 `max_steps` 从 2000 改为 5000——验证损失还在继续改善吗？
2. **（中等）** 用 RoPE 替换学习型位置嵌入。在 `MultiHeadAttention` 内部对 Q 和 K 应用旋转。训练并验证验证损失至少同样低。
3. **（中等）** 在采样循环中实现 KV 缓存。有无缓存分别生成 500 个 token。笔记本电脑上挂钟时间应提升 5–20 倍。
4. **（困难）** 为模型添加第二个头，预测下下个 token（MTP——DeepSeek-V3 的多 token 预测）。联合训练。有帮助吗？
5. **（困难）** 将每块的单个 FFN 替换为 4 个专家的 MoE。路由器 + top-2 路由。观察在活跃参数量相同的情况下验证损失如何变化。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| nanoGPT | "Karpathy 的教程仓库" | 最小化仅解码器 Transformer 训练代码，约 300 行；典范参考实现 |
| tinyshakespeare | "标准玩具语料库" | 约 1.1 MB 文本；2015 年以来每个字符语言模型教程都使用它 |
| 绑定嵌入 (Tied embeddings) | "共享输入/输出矩阵" | 语言模型头权重 = token 嵌入矩阵的转置；节省参数，提升质量 |
| bf16 自动转换 (bf16 autocast) | "训练精度技巧" | 用 bf16 运行前向/反向传播，用 fp32 保持优化器状态；2021 年后的标准 |
| 梯度裁剪 (Gradient clipping) | "防止尖刺" | 将全局梯度范数上限设为 1.0；防止训练爆炸 |
| 余弦学习率调度 (Cosine LR schedule) | "2020 年后的默认" | 学习率线性预热，然后余弦形状衰减到峰值的 10% |
| MFU（模型 FLOP 利用率） | "Model FLOP Utilization" | 实际 FLOP / 理论峰值；2026 年稠密模型 40%、MoE 30% 是不错的成绩 |
| 验证损失 (Val loss) | "留出损失" | 模型从未见过的数据上的交叉熵；过拟合检测器 |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 经典带注释实现

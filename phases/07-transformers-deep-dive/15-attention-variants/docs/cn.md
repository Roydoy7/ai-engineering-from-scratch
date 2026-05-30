# 注意力变体——滑动窗口、稀疏、差分

> 全注意力是一个圆。每个 token 看到每个 token，内存付出代价。四种变体弯曲了圆的形状，找回了一半的代价。

**类型：** 构建
**语言：** Python
**前置知识：** 第7阶段第02课（自注意力）、第7阶段第03课（多头）、第7阶段第12课（KV 缓存 / Flash Attention）
**预计时间：** 约60分钟

## 问题背景

全注意力在序列长度上需要 `O(N²)` 的内存和 `O(N²)` 的计算。对于 128K 上下文的 Llama 3 70B，每层有 160 亿个注意力条目，乘以 80 层。Flash Attention（第12课）隐藏了 `O(N²)` 的激活内存，但不改变算术代价——每个 token 仍然关注其他所有 token。

三类变体改变了注意力矩阵本身的拓扑：

1. **滑动窗口注意力（SWA）。** 每个 token 只关注固定窗口内的邻居，而不是完整的前缀。内存和计算降至 `O(N · W)`，其中 `W` 是窗口大小。Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long 都使用此方案。
2. **稀疏/块注意力。** 只有选定的 `(i, j)` 对会被打分；其余被强制为零权重。Longformer、BigBird、OpenAI 稀疏 Transformer。
3. **差分注意力。** 用独立的 Q/K 投影计算两个注意力图，然后相减。消除将权重泄漏到前几个 token 的"注意力汇聚"现象。微软的 DIFF Transformer（2024 年）。

这些变体可以共存。2026 年的前沿模型通常混合使用：大多数层是 SWA-1024，每隔五层是全局完整注意力，还有一些差分头用于清理检索。Gemma 3 的 5:1 SWA-全局比例是当前的教科书默认值。

## 核心概念

### 滑动窗口注意力（SWA）

位置 `i` 处的查询只关注 `[i - W, i]`（因果 SWA）或 `[i - W/2, i + W/2]`（双向）范围内的位置。窗口外的 token 在分数矩阵中得到 `-inf`。

```
完整因果：               滑动窗口（W=4）：
位置 0-7                 位置 0-7，W=4
    0 1 2 3 4 5 6 7          0 1 2 3 4 5 6 7
0 | x                  0 |  x
1 | x x                1 |  x x
2 | x x x              2 |  x x x
3 | x x x x            3 |  x x x x
4 | x x x x x          4 |    x x x x
5 | x x x x x x        5 |      x x x x
6 | x x x x x x x      6 |        x x x x
7 | x x x x x x x x    7 |          x x x x
```

对于 `N = 8192` 和 `W = 1024`，分数矩阵期望有 1024 × 8192 个非零行——减少了 8 倍。

**SWA 使 KV 缓存缩小。** 每层只需保留最后 `W` 个 token 的 K 和 V。对于类似 Gemma-3 的配置（1024 窗口，128K 上下文），KV 缓存缩小 128 倍。

**质量代价。** 纯 SWA Transformer 在长距离检索上表现不佳。解决方案：将 SWA 层与全注意力层交错使用。Gemma 3 使用 5:1 的 SWA:全局比例。Mistral 7B 使用因果 SWA 堆栈，信息通过重叠窗口"向前流动"——每层将有效感受野扩展 `W`，`L` 层之后模型可以关注 `L × W` 个 token 之前。

### 稀疏/块注意力

预先选择 `N × N` 的稀疏模式。三种典范形状：

- **局部 + 步长（OpenAI 稀疏 Transformer）。** 关注最后 `W` 个 token，加上之前每隔 `stride` 个 token。以 `O(N · sqrt(N))` 的计算同时捕获局部和长距离信息。
- **Longformer / BigBird。** 局部窗口 + 一小组全局 token（如 `[CLS]`），全局 token 关注所有人，所有人也关注全局 token，加上随机稀疏链接。在相同质量下经验上支持 2 倍上下文。
- **原生稀疏注意力（DeepSeek，2025 年）。** 学习哪些 `(Q, K)` 块重要；在核层面跳过零块。与 FlashAttention 兼容。

稀疏注意力是一个核工程的问题。数学很简单（掩码分数矩阵）；收益来自于从不将零条目加载到 SRAM。FlashAttention-3 和 2026 年的 FlexAttention API 使自定义稀疏模式在 PyTorch 中成为一等公民。

### 差分注意力（DIFF Transformer，2024 年）

普通注意力有"注意力汇聚"问题：softmax 强制每行之和为 1，所以不想特别关注任何东西的 token 会把权重倾倒在第一个 token（或前几个）上。这偷走了本应给真实内容的容量。

差分注意力通过计算**两个**注意力图并相减来解决这个问题：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是学习到的标量（通常为 0.5–0.8）。A1 捕获真实内容权重；A2 捕获汇聚。相减消除汇聚，将权重重新分配给相关 token。

报告结果（微软 2024 年）：困惑度降低 5–10%，在相同训练长度下有效上下文延长 1.5–2 倍，针尖在干草堆中的检索更精准。

### 变体对比

| 变体 | 计算量 | KV 缓存 | 质量 vs 全注意力 | 生产使用 |
|------|--------|---------|----------------|---------|
| 全注意力 | O(N²) | 每层 O(N) | 基线 | 所有模型的默认层 |
| SWA（窗口1024） | O(N·W) | 每层 O(W) | 困惑度 -0.1，配合全局层效果好 | Gemma 2/3、Phi-3-Long |
| 局部+步长稀疏 | O(N·√N) | 混合 | 类似 SWA | OpenAI 稀疏 Transformer、Longformer |
| BigBird（局部+全局+随机） | 近似 O(N) | 混合 | 在 2 倍上下文下与全注意力持平 | 早期长上下文 BERT |
| 原生稀疏（DeepSeek-V3.2） | O(N · 活跃比例) | O(N) | 困惑度差距在 0.05 以内 | DeepSeek-V3.2，2025 年 |
| 差分注意力 | O(2·N²) | O(2N) | 困惑度降低 5–10% | DIFF Transformer，2026 年初期模型 |

## 动手实现

见 `code/main.py`。我们实现一个因果掩码比较器，在玩具序列上并排展示全注意力、SWA、局部+步长和差分注意力。

### 第一步：完整因果掩码（基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

来自第07课的基线。下三角；对角线以上权重为零。

### 第二步：滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

只有一个参数——`window`。当 `window >= n` 时，你恢复了完整因果注意力。当 `window = 1` 时，每个 token 只关注自身。

### 第三步：局部+步长稀疏掩码

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密集局部窗口加上每隔 `stride` 个 token 一直追溯到序列开始。感受野随额外层的增加以对数步长增长。

### 第四步：差分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力传播，用学习到的混合系数相减。代码中我们比较单注意力和差分注意力的注意力汇聚热力图，观察汇聚的消失。

### 第五步：KV 缓存大小

在 `N = 131072` 下打印每个变体每层的缓存大小。SWA 和稀疏变体降低 10–100 倍。差分注意力翻倍。有意识地计算你的内存账单。

## 工程应用

2026 年生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 以 5:1 混合 SWA（window=1024）和全局层。
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 的 FlexAttention 接受掩码函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这会编译为自定义 Triton 核。对于常见模式，速度在 FlashAttention-3 的 10% 以内，且掩码函数是 Python 可调用的。

**何时选择哪种：**

- **纯全注意力** — 所有层直到约 16K 上下文，或检索质量至关重要时
- **SWA + 全局混合** — 长上下文（>32K），训练和推理都受内存限制。2026 年 32K 以上的默认方案
- **稀疏块注意力** — 自定义核，自定义模式。保留用于专门工作负载（检索、音频）
- **差分注意力** — 任何注意力汇聚污染有害的工作负载（长上下文 RAG、针尖在干草堆中的检索）

## 交付物

见 `outputs/skill-attention-variant-picker.md`。该技能根据目标上下文长度、检索需求和训练/推理计算配置，为新模型选择注意力拓扑。

## 练习

1. **（简单）** 运行 `code/main.py`。验证 `window=4` 的 SWA 将每行窗口外的所有内容归零。验证 `window=n` 与完整因果注意力逐位相同。
2. **（中等）** 在第07课综合项目之上实现 `window=1024` 的因果 SWA。在 tinyshakespeare 上训练 1000 步。验证损失相比全注意力退化了多少？峰值内存减少了多少？
3. **（困难）** 在综合项目模型中实现 Gemma-3 风格的 5:1 层混合（5 个 SWA，1 个全局）。在相同参数量下与纯 SWA 和纯全局基线对比损失、内存和生成质量。
4. **（困难）** 实现每头学习 `λ` 的差分注意力。在合成检索任务（1 根针，2000 个干扰项）上训练。测量相比单注意力基线（参数量相同）的检索准确率。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 滑动窗口注意力 (SWA) | "局部注意力" | 每个查询关注最后 `W` 个 token；KV 缓存缩小到 `O(W)` |
| 有效感受野 | "模型能看多远" | 在窗口为 `W` 的 `L` 层 SWA 堆栈中，最多 `L × W` 个 token |
| Longformer / BigBird | "局部+全局+随机" | 带少量始终关注全局 token 的稀疏模式；早期长上下文方案 |
| 原生稀疏注意力 | "DeepSeek 的核技巧" | 学习块级稀疏性；在核层面跳过零块，同时保持质量 |
| 差分注意力 | "两图相减" | DIFF Transformer：从第一个注意力图中减去学习到的 `λ` 倍第二个注意力图，以消除注意力汇聚 |
| 注意力汇聚 (Attention sink) | "权重流向 token 0" | Softmax 归一化强制行之和为 1；无信息的查询将权重倾倒在位置 0 |
| FlexAttention | "掩码即 Python" | PyTorch 2.5+ API，将任意掩码函数编译成 FlashAttention 形状的核 |
| 层类型混合 (Layer type mix) | "5:1 SWA-全局" | 在堆栈中交错稀疏和全注意力层，以较低内存保持质量 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 典范滑动窗口 + 全局 token 论文
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — 局部 + 全局 + 随机
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI 的局部+步长模式
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:全局混合
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 5:1 混合，window=1024，现为教科书默认
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer 论文
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2 的学习稀疏注意力
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — 工程应用中掩码即可调用模式的 API 参考

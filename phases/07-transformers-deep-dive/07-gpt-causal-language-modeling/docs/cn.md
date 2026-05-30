# GPT——因果语言建模

> BERT 两侧都看。GPT 只看过去。三角掩码是现代 AI 中影响最深远的一行代码。

**类型：** 构建
**语言：** Python
**前置知识：** 第7阶段第02课（自注意力）、第7阶段第05课（完整 Transformer）、第7阶段第06课（BERT）
**预计时间：** 约75分钟

## 问题背景

语言模型回答一个问题：给定前 `t-1` 个 token，第 `t` 个 token 的概率分布是什么？在这个信号——下一个 token 预测——上训练，就得到一个能逐 token 生成任意文本的模型。

为了在整个序列上并行进行端到端训练，你需要每个位置的预测只依赖于更早的位置。否则模型会通过看答案轻松作弊。

因果掩码做到了这一点。它是一个上三角的 `-inf` 矩阵，在 softmax 之前加到注意力分数上。softmax 之后，那些位置变为 0。每个位置只能关注自身和更早的位置。因为你将其一次性应用于整个序列，一次前向传播就能得到 N 个并行的下一个 token 预测。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2024）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——它们都是具有相同核心循环的仅解码器因果 Transformer。只是更大、更好的数据和更好的 RLHF。

## 核心概念

### 掩码

给定长度为 `N` 的序列，构建一个 `N × N` 矩阵：

```
M[i, j] = 0       若 j <= i
M[i, j] = -inf    若 j > i
```

在 softmax 之前将 `M` 加到原始注意力分数上。`exp(-inf) = 0`，所以被掩码的位置贡献零权重。注意力矩阵的每一行是仅对之前位置的概率分布。

实现代价：一次 `torch.tril()` 调用。计算时间：纳秒级。对该领域的影响：一切。

### 并行训练，串行推理

训练：对整个 `(N, d_model)` 序列做一次前向传播，计算 N 个交叉熵损失（每个位置一个），求和，反向传播。沿序列方向并行。这就是为什么 GPT 训练可以扩展——你在一次 GPU 传播中处理一批 100 万个 token。

推理：逐 token 生成。输入 `[t1, t2, t3]`，得到 `t4`。输入 `[t1, t2, t3, t4]`，得到 `t5`。以此类推。KV 缓存（第12课）保存 `t1…tn` 的隐状态，避免每步重新计算。但推理时的串行深度 = 输出长度。这是自回归税，也是为什么解码是所有 LLM 延迟的瓶颈。

### 损失——偏移一位

给定 token `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`。求和。这是整个序列的交叉熵。

你听说过的每一个 Transformer 语言模型都在这个损失上训练。预训练、微调、SFT——相同的损失，不同的数据。

### 解码策略

训练后，采样选择的影响比人们想象的更大。

| 方法 | 做什么 | 何时使用 |
|------|--------|---------|
| 贪心 (Greedy) | 每步取最大值 | 确定性任务、代码补全 |
| 温度 (Temperature) | logit 除以 T 后采样 | 创意任务，T 越高多样性越强 |
| Top-k | 只从概率最高的 k 个 token 中采样 | 消除低概率长尾 |
| Top-p（核采样） | 从累积概率 ≥ p 的最小集合中采样 | 2020 年后的默认方案；适应分布形状 |
| Min-p | 保留 `p > min_p * max_p` 的 token | 2024 年后；比 top-p 更好地拒绝长尾 |
| 投机解码 | 草稿模型提出 N 个 token，大模型验证 | 延迟降低 2–3 倍，质量相同 |

2026 年，对于开放权重模型，min-p + 温度 0.7 是合理的默认配置。投机解码是所有生产推理栈的标配。

### "GPT 配方"为何有效

1. **仅解码器。** 无编码器开销。每层一次注意力 + FFN 传播。
2. **规模化。** 1.24 亿 → 15 亿 → 1750 亿 → 万亿。Chinchilla 缩放定律（第13课）告诉你如何分配计算量。
3. **上下文学习。** 在 60 亿–130 亿参数左右出现。模型无需微调就能遵循少样本示例。
4. **RLHF。** 基于人类偏好的后训练将原始预训练文本转化为对话助手。
5. **Pre-norm + RoPE + SwiGLU。** 大规模稳定训练。

核心架构自 GPT-2 以来变化不大。所有有趣的事情都发生在数据、规模和后训练上。

## 动手实现

### 第一步：因果掩码

见 `code/main.py`。一行代码：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 之前将其加到注意力分数上。这就是整个机制。

### 第二步：2 层类 GPT 模型

堆叠两个解码器块（带掩码自注意力 + FFN，无交叉注意力）。添加 token 嵌入、位置编码和反嵌入（与 token 嵌入矩阵绑定——GPT-2 以来的标准技巧）。

### 第三步：端到端的下一个 token 预测

在 20 个 token 的玩具词表上，在每个位置产生 logit。对偏移一位的目标计算交叉熵损失。不使用梯度——这是前向传播的正确性检查。

### 第四步：采样

实现贪心、温度、top-k、top-p、min-p。在固定提示上运行每种方法并比较输出。一个采样函数大约 10 行。

## 工程应用

PyTorch，2026 年惯例写法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

在底层，`generate()` 运行前向传播，提取最后位置的 logit，采样下一个 token，拼接，然后重复。每个生产 LLM 推理栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都实现了同一个循环，只是有大量优化——批量预填充、连续批处理、KV 缓存分页、投机解码。

**GPT vs BERT，各一句话：** GPT 预测 `P(x_t | x_{<t})`。BERT 预测 `P(x_masked | x_unmasked)`。损失决定模型能否生成。

## 交付物

见 `outputs/skill-sampling-tuner.md`。该技能为新的生成任务选择采样参数，并在需要确定性解码时发出标记。

## 练习

1. **（简单）** 运行 `code/main.py`，验证 softmax 后的因果注意力矩阵是下三角的。抽查：第 3 行只应在第 0–3 列有权重。
2. **（中等）** 实现宽度为 4 的束搜索。在 10 个短提示上比较束搜索-4 与贪心的困惑度。束搜索总是赢吗？（提示：对翻译通常是，对开放式对话则不然。）
3. **（困难）** 实现投机解码：用 2 层模型作为草稿，用 6 层模型作为验证器。在 100 个长度 64 的补全上测量挂钟加速比。确认输出与验证器的贪心解码匹配。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 因果掩码 (Causal mask) | "三角形" | 上三角 `-inf` 矩阵，加到注意力分数上，使位置 `i` 只能看到 `≤ i` 的位置 |
| 下一个 token 预测 | "损失" | 模型分布对每个位置真实下一个 token 的交叉熵 |
| 自回归 (Autoregressive) | "逐个生成" | 将输出反馈为输入；并行性只在训练时存在，生成时没有 |
| Logit | "softmax 前的分数" | 语言模型头在 softmax 之前的原始输出；采样在这上面进行 |
| 温度 (Temperature) | "创意旋钮" | logit 除以 T；T→0 = 贪心，T→∞ = 均匀分布 |
| Top-p（核采样） | "核采样" | 将分布截断为累积概率 ≥ p 的最小集合；从剩余部分采样 |
| Min-p | "比 top-p 更好" | 保留 `p ≥ min_p × max_p` 的 token；根据分布尖锐度自适应截断 |
| 投机解码 (Speculative decoding) | "草稿 + 验证" | 小模型提出 N 个 token；大模型并行验证 |
| 教师强制 (Teacher forcing) | "训练技巧" | 训练时输入真实的前一个 token，而非模型的预测。所有 seq2seq 语言模型的标准做法 |

## 延伸阅读

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 与上下文学习
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 投机解码论文
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 典范因果语言模型参考代码

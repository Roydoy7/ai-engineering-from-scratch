# Transformer 前的文本生成——N-gram 语言模型

> 一个词越令模型惊讶，模型就越差。困惑度把惊讶量化成一个数字，平滑让这个数字保持有限。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第1课（文本处理）、第2阶段第14课（朴素贝叶斯）
**预计时间：** 约45分钟

## 问题背景

在 Transformer 之前，在 RNN 之前，在词嵌入之前，语言模型通过统计一个词跟在前面 n-1 个词后面的频率来预测下一个词："the cat"后跟"sat"出现 47 次，后跟"jumped"出现 12 次，后跟"refrigerator"出现 0 次，归一化得到概率分布。

这就是 n-gram 语言模型。从 1980 年到 2015 年，每个语音识别器、每个拼写检查器、每个基于短语的机器翻译系统都靠它运行。当你需要廉价的设备端语言建模时，它今天仍然有效。

有趣的问题是怎么处理从未见过的 n-gram。基于原始计数的模型对它没见过的任何东西赋概率为零，这是灾难性的，因为句子很长，几乎每个长句子都包含至少一个未见过的序列。五十年的平滑研究解决了这个问题，Kneser-Ney 平滑是最终成果，现代深度学习继承了这个经验主义传统。

## 核心概念

**N-gram 概率**：`P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（通常 3 用三元组，4 用四元组），从计数估计：

```text
P(w | 上下文) = count(上下文, w) / count(上下文)
```

**零计数问题**：训练集中未见过的任何 n-gram 得到概率零。2007 年对 Brown 语料库的研究发现，即使是四元组模型也有 30% 的保留测试四元组在训练集中未见过。不做平滑就无法在任何真实文本上评估。

**平滑方法，按复杂度排列：**

1. **拉普拉斯（加一平滑）**：对每个计数加 1，简单，但对稀有事件很糟糕。
2. **Good-Turing**：根据频率的频率，把概率质量从高频事件重新分配给未见事件。
3. **插值**：将 n-gram、(n-1)-gram 等估计用可调权重组合。
4. **回退（Backoff）**：如果 n-gram 计数为零，退而使用 (n-1)-gram。Katz 回退对此做了形式化。
5. **绝对折扣**：从所有计数中减去固定折扣 `D`，重新分配给未见事件。
6. **Kneser-Ney**：绝对折扣加上对低阶模型的聪明选择：使用**延续概率**（一个词出现在多少种不同上下文中）而非原始频率。

Kneser-Ney 的洞见很深刻："San Francisco"是常见二元组，"Francisco"几乎只在"San"后出现。朴素绝对折扣给"Francisco"很高的一元组概率（因为计数高）。而 Kneser-Ney 注意到"Francisco"只出现在一种上下文中，相应地降低了它的延续概率。结果：以"Francisco"结尾的陌生二元组得到合理的低概率。

**评估：困惑度（Perplexity）**。在保留测试集上，每词平均负对数似然的指数。越低越好。困惑度为 100 意味着模型的困惑程度相当于在 100 个词中均匀随机选择。

```text
困惑度 = exp(- (1/N) * Σ log P(w_i | 上下文_i))
```

## 动手实现

### 第一步：三元组计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是分词后的句子列表，输出是 n-gram 计数和上下文计数。`<s>` 和 `</s>` 是句子边界标记。

### 第二步：拉普拉斯平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

对每个计数加 1。平滑了，但对未见事件分配了过多质量，同时也损害了罕见已知事件。

### 第三步：Kneser-Ney（二元组，插值版）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三个核心部分：`continuation_prob` 捕捉"这个词出现在多少种不同上下文中？"（Kneser-Ney 的核心创新）；`lambda_prev` 是折扣释放的质量，用于权重回退；最终概率是折扣后的主项加上加权延续项。

### 第四步：采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例采样，每个随机种子产生不同输出。如果想要类似集束搜索的输出，每步取 argmax（贪心），并加一个温度参数控制随机性。

### 第五步：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。在 Brown 语料库上，调优好的四元组 KN 模型困惑度约为 140；Transformer 语言模型在同一测试集上可达 15-30。差距约 10 倍，这就是整个领域转向的原因。

## 工程应用

- **经典 NLP 教学**：能得到最清晰的平滑、MLE 和困惑度概念理解。
- **KenLM**：生产 n-gram 库，在延迟敏感的语音和 MT 系统中用作重评分器。
- **设备端自动补全**：键盘里的三元组模型，现在还在用。
- **基线建立**：在宣称你的神经语言模型好之前，始终先计算 n-gram LM 困惑度。如果你的 Transformer 没有大幅超过 KN，说明有问题。

## 交付物

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 练习

1. **（简单）** 在 1000 句莎士比亚语料上训练三元组语言模型，生成 20 个句子，它们会局部合理但全局不连贯——这是经典演示效果。
2. **（中等）** 在保留的莎士比亚测试集上计算 KN 模型的困惑度，与拉普拉斯平滑对比，应该能看到 KN 将困惑度降低 30-50%。
3. **（困难）** 构建三元组拼写纠错器：给定拼写错误的词和它的上下文，生成候选纠正并按语言模型上下文概率排序，在 Birkbeck 拼写语料库（公开）上评估。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| N-gram | "词序列" | n 个连续 token 的序列 |
| 平滑 (Smoothing) | "避免零概率" | 重新分配概率质量，使未见事件获得非零概率 |
| 困惑度 (Perplexity) | "语言模型质量指标" | 保留数据上 `exp(-平均对数概率)`，越低越好 |
| 回退 (Backoff) | "退而使用更短上下文" | 三元组计数为零时使用二元组，Katz 回退对此形式化 |
| Kneser-Ney | "n-gram 最佳平滑" | 绝对折扣加上低阶模型的延续概率 |
| 延续概率 (Continuation probability) | "KN 特有的概念" | 按一个词出现的上下文数量加权的 `P(w)`，而非原始计数 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — n-gram 语言模型和平滑的权威处理
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — 确立 Kneser-Ney 为最佳 n-gram 平滑器的论文
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — 原始 KN 论文
- [KenLM](https://kheafield.com/code/kenlm/) — 快速生产级 n-gram 语言模型，2026 年在延迟敏感应用中仍在使用

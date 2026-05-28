# GloVe、FastText 与子词嵌入

> Word2Vec 为每个词训练了一个嵌入；GloVe 分解了共现矩阵；FastText 嵌入了词的碎片；BPE 则架起了通往 Transformer 的桥梁。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第3课（从零实现 Word2Vec）
**预计时间：** 约45分钟

## 问题背景

Word2Vec 留下了两个未解问题。

首先，有另一条并行研究路线：直接对共现矩阵进行分解（LSA、HAL），而非做在线 skip-gram 更新。Word2Vec 的迭代方式是本质上更优的吗？还是差异只是两种方法处理计数方式的产物？**GloVe** 给出了答案：使用精心设计损失函数的矩阵分解，效果与 Word2Vec 相当或更好，训练成本还更低。

其次，两种方法对从未见过的词都束手无策。`Zoomer-approved`、`dogecoin`、上周才造出来的任何专有名词、罕见词根的每种屈折形式。**FastText** 通过嵌入字符 n-gram 解决了这个问题：一个词是其各部分（包括词素）之和，所以即使是未登录词也能得到合理的向量。

第三，Transformer 出现后，问题再次转移。词级词汇表最多约一百万条；而真实语言比这更开放。**字节对编码（BPE）**及其变体通过学习频繁子词单元的词汇表解决了这个问题，覆盖一切可能的词。所有现代 LLM 的所有分词器都是子词分词器。

本课讲解这三种方法，并说明何时选择哪种。

## 核心概念

**GloVe（全局向量）**：构建词-词共现矩阵 `X`，其中 `X[i][j]` 是词 `j` 出现在词 `i` 上下文中的次数。训练向量使得 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`，对损失加权避免高频词对占主导。

**FastText**：一个词 = 其字符 n-gram 之和加上词本身。`where` 变成 `<wh, whe, her, ere, re>, <where>`。词向量是这些组成成分向量之和。训练方式与 Word2Vec 相同。好处：未见过的词（`whereupon`）可以从已知 n-gram 中组合得到。

**BPE（字节对编码）**：从单个字节（或字符）组成的词汇表开始，统计语料库中每个相邻对的频率，将最高频的对合并为新 token，重复 `k` 次。结果：一个 `k + 256` 个 token 的词汇表，其中高频序列（`ing`、`tion`、`the`）是单个 token，罕见词被拆成熟悉的片段，任何句子都能分词。

## 动手实现

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

两个值得命名的要点：权重函数 `f(x) = (x/x_max)^alpha` 降权高频词对（如 `(the, and)`），避免它们主导损失。最终嵌入是 `W`（中心词）和 `W_tilde`（上下文词）两个表的和——这是一个已发表的技巧，通常比只用一个表效果更好。

### FastText：感知子词的嵌入

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词用其 n-gram 集合表示（通常 3 到 6 个字符）。词嵌入是其 n-gram 嵌入之和。在 skip-gram 训练中，用这个替换 Word2Vec 使用的单一向量。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于未见过的词，只要其部分 n-gram 是已知的，就仍能得到向量。`whereupon` 与 `where` 共享 `<wh`、`her`、`ere`、`<where`，因此两者在空间中靠得很近。

### BPE：学习子词词汇表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一次迭代合并最常见的相邻对。足够多的迭代后，高频子串（`low`、`est`、`tion`）成为单个 token，罕见词被干净地拆分。

真正的 GPT / BERT / T5 分词器学习 3 万到 10 万次合并。结果：任何文本都能分词为已知 ID 的有界长度序列，永远不会出现未登录词。

## 工程应用

实践中，你几乎不会自己训练这些模型，而是加载预训练检查点：

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Transformer 时代的 BPE 风格子词分词：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀标记词边界（GPT-2 的约定）。每个现代分词器都是 BPE 变体、WordPiece（BERT）或 SentencePiece（T5、LLaMA）。

### 如何选择

| 情景 | 选择 |
|------|------|
| 通用预训练词向量，不需要处理未登录词 | GloVe 300d |
| 通用预训练词向量，需要处理拼写错误/新词/形态丰富的语言 | FastText |
| 任何要输入 Transformer 的内容（训练或推理） | 使用模型附带的分词器，永远不要替换 |
| 从零训练自己的语言模型 | 先在语料上训练 BPE 或 SentencePiece 分词器 |
| 使用线性模型的生产文本分类 | 仍然用 TF-IDF，见第2课 |

## 交付物

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 练习

1. **（简单）** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`，计算两个 n-gram 集合的 Jaccard 重叠度。你应该看到大量共享片段（`pla`、`lay`、`play`），这正是 FastText 能在形态变体之间迁移的原因。
2. **（中等）** 扩展 `learn_bpe`，跟踪词汇表增长。绘制每次合并后"每语料字符的 token 数"曲线，你应该看到起初快速压缩，逐渐趋近于每词约 2-3 个 token。
3. **（困难）** 在莎士比亚全集上训练 1000 次合并的 BPE，比较常用词与罕见专有名词的分词结果，测量合并前后每词平均 token 数，写下让你惊讶的地方。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 共现矩阵 (Co-occurrence matrix) | "词-词频率表" | `X[i][j]` = 词 `j` 出现在词 `i` 窗口中的次数 |
| 子词 (Subword) | "词的片段" | 字符 n-gram（FastText）或学习到的 token（BPE/WordPiece/SentencePiece） |
| BPE（字节对编码） | "字节对编码" | 迭代合并最高频相邻对，直到词汇表达到目标大小 |
| OOV（未登录词） | "词表外的词" | 模型从未见过的词，Word2Vec/GloVe 无法处理，FastText 和 BPE 可以 |
| 字节级 BPE | "基于原始字节的 BPE" | GPT-2 的方案，词汇表从 256 个字节开始，因此永远不会有未登录词 |

## 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 论文，七页，仍是最好的损失函数推导
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText 论文
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 将 BPE 引入现代 NLP 的论文
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE、WordPiece 和 SentencePiece 在实践中的实际差异

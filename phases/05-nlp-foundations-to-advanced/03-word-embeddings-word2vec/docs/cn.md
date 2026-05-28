# 词嵌入 — 从零实现 Word2Vec

> 词由其伴侣定义。在这个思想上训练一个浅层网络，几何结构就自然涌现。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第2课（BoW + TF-IDF）、第3阶段第3课（从零实现反向传播）
**预计时间：** 约75分钟

## 问题背景

TF-IDF 知道 `dog` 和 `puppy` 是不同的词，却不知道它们含义几乎相同。在 `dog` 上训练的分类器无法泛化到关于 `puppy` 的评论。你可以通过列出同义词来临时解决，但这对罕见词、领域术语以及你没有预料到的每种语言都会失效。

你想要一种表示方式，让 `dog` 和 `puppy` 在空间中靠得很近，让 `king - man + woman` 落在 `queen` 附近，让在 `dog` 上训练的模型能免费将部分信号迁移到 `puppy`。

Word2Vec 给了我们这个空间。两层神经网络，万亿 token 的训练，2013 年发表。架构简单得有些令人尴尬，结果却重塑了 NLP 整整十年。

## 核心概念

**分布假设**（Firth，1957）："你可以通过一个词的伴侣了解它。" 如果两个词出现在相似的上下文中，它们很可能含义相近。

Word2Vec 有两种变体，都利用了这个思想：

- **Skip-gram**：给定中心词，预测周围的词。窗口大小为 2 时，`cat -> (the, sat, on)`。
- **CBOW（连续词袋）**：给定周围的词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练较慢，但对罕见词处理更好，成为了默认选择。

网络只有一个没有非线性激活的隐藏层。输入是词汇表上的独热向量，输出是词汇表上的 softmax。训练后丢弃输出层，隐藏层的权重就是嵌入。

```
one-hot(中心词) ── W ──▶ 隐藏层 (d 维) ── W' ──▶ softmax(词汇表)
                              ^
                              这就是嵌入
```

关键技巧：对 10 万个词计算 softmax 代价极高。Word2Vec 使用**负采样**将其转化为二分类任务——预测"这个上下文词是否出现在这个中心词附近？"每个训练对只采样少量负样本（不共现的词），而不是在整个词汇表上计算 softmax。

## 动手实现

### 第一步：从语料库生成训练对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口内的每个（中心词，上下文词）对都是一个正训练样本。

### 第二步：嵌入表

两个矩阵。`W` 是中心词嵌入表（你保留的那个），`W'` 是上下文词表（通常丢弃，有时与 `W` 取平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小随机初始化。词汇表 1 万、维度 100 是现实中的规模；教学用途下，50 词汇 × 16 维足以看到几何结构。

### 第三步：负采样目标函数

对每个正样本对（中心词，上下文词），从词汇表中随机采样 `k` 个词作为负样本。训练模型使正样本的点积 `W[center] · W'[context]` 高，负样本的点积低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

魔法公式：对正样本对的逻辑损失（希望 sigmoid 接近 1）加上对负样本对的逻辑损失（希望 sigmoid 接近 0）。梯度流向两个表。如果想彻底理解，拿纸笔推导一遍原始论文中的完整推导。

### 第四步：在玩具语料上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大型语料上训练足够多的 epoch 后，共享上下文的词会有相似的中心嵌入。在玩具语料上能隐约看到这个效果，在数十亿 token 上会看到显著的效果。

### 第五步：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 Google News 300d 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。不是因为模型知道什么是皇室，而是因为向量 `(king - man)` 捕捉到了类似"皇室"的概念，加到 `woman` 上就落在了皇室-女性区域附近。

## 工程应用

从零实现 Word2Vec 是教学用途。生产 NLP 使用 `gensim`：

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

真正工作中，你几乎不会自己训练 Word2Vec，而是下载预训练向量：

- **GloVe** — 斯坦福的共现矩阵分解方法。提供 50d、100d、200d、300d 检查点，通用覆盖范围好。第4课专门讲 GloVe。
- **fastText** — Facebook 的 Word2Vec 扩展，嵌入字符 n-gram，能处理未登录词。第4课讲解。
- **Google News 预训练 Word2Vec** — 300d，300 万词汇表，2013 年发布，至今每天仍有大量下载。

### Word2Vec 在 2026 年仍然获胜的场景

- 轻量级领域特定检索——在笔记本电脑上一小时内训练医学摘要，得到通用模型无法捕捉的专业向量。
- 类比式特征工程——`gender_vector = mean(man - woman 词对)`，从其他词中减去它得到性别中性轴，仍在公平性研究中使用。
- 可解释性——100d 足够小，可以通过 PCA 或 t-SNE 可视化，真实看到聚类形成。
- 任何必须在设备端无 GPU 运行的场景——Word2Vec 查找只是一次行读取。

### Word2Vec 失败的地方

多义词墙。`bank` 只有一个向量，"河岸"和"银行"共享它；`table`（表格 vs 家具）也共享。下游分类器无法从向量中区分含义。

上下文嵌入（ELMo、BERT、此后所有 Transformer）解决了这个问题，通过根据周围上下文为每次词语出现生成不同的向量。这就是从 Word2Vec 到 BERT 的跨越：从静态到上下文相关。第7阶段讲解 Transformer。

未登录词是另一个失败点。如果训练数据中没有出现过 `Zoomer-approved`，Word2Vec 就没有这个向量，也没有回退机制。fastText 通过子词组合解决了这个问题（第4课）。

## 交付物

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## 练习

1. **（简单）** 在小型语料上（20 句关于猫和狗的句子）运行训练循环。200 个 epoch 后，验证 `nearest(vocab, W, W[vocab["cat"]])` 在前 3 名中返回 `dog`。如果没有，增加 epoch 数或扩大词汇量。
2. **（中等）** 添加高频词的下采样。频率超过 `10^-5` 的词以与其频率成正比的概率从训练对中丢弃。测量这对罕见词相似度的影响。
3. **（困难）** 在 20 Newsgroups 语料上训练模型。计算两个偏差轴：`he - she` 和 `doctor - nurse`。将职业词投影到两个轴上，报告偏差差距最大的职业。这正是公平性研究人员使用的探针类型。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 词嵌入 (Word embedding) | "词向量" | 从上下文中学到的密集低维（通常 100-300）表示 |
| Skip-gram | "Word2Vec 技巧" | 从中心词预测上下文词，比 CBOW 慢，对罕见词效果更好 |
| 负采样 (Negative sampling) | "训练捷径" | 用对 k 个随机词的二分类替代完整词汇表上的 softmax |
| 静态嵌入 (Static embedding) | "每词一个向量" | 无论上下文如何，同一个向量，多义词问题的根源 |
| 上下文嵌入 (Contextual embedding) | "上下文相关向量" | 根据周围词为每次出现生成不同向量，Transformer 的产出 |
| OOV（未登录词） | "词表外的词" | 训练时未见过的词，Word2Vec 无法为其生成向量 |

## 延伸阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — 负采样论文，简短易读
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 梯度推导最清晰的版本，原论文数学感觉密集时可参考
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — 真正有效的生产训练设置

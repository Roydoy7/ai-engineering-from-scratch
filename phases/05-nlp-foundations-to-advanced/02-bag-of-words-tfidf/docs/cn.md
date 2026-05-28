# 词袋模型、TF-IDF 与文本表示

> 先计数，再思考。2026 年，TF-IDF 在定义明确的任务上仍能胜过嵌入模型。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第1课（文本处理）、第2阶段第2课（从零实现线性回归）
**预计时间：** 约75分钟

## 问题背景

模型需要数字，你有的是字符串。

每个 NLP 流水线都要回答同一个问题：如何将可变长度的 token 序列转化为分类器能消费的固定大小向量？这个领域找到的第一个答案是最蠢但能用的方案——数词，做向量。

这个向量承载的生产 NLP 工作比任何嵌入模型都多。垃圾邮件过滤、主题分类器、日志异常检测、搜索排名（BM25 之前）、第一波情感分析、学术 NLP 基准的第一个十年。2026 年的从业者在处理窄化分类任务时仍然首先使用它。它快速、可解释，而且在词语出现与否决定一切的任务上，往往与 4 亿参数嵌入模型不相上下。

本课从零构建词袋模型，然后是 TF-IDF，再展示 scikit-learn 如何用三行代码完成同样的工作，最后点出让你转向嵌入模型的失败场景。

## 核心概念

**词袋模型（BoW）** 丢弃顺序信息。对每篇文档，统计每个词汇表中词出现的次数。向量长度等于词汇表大小，第 `i` 位是第 `i` 个词的计数。

**TF-IDF** 对 BoW 重新加权。出现在每篇文档中的词没有信息量，降权；在整个语料库中罕见但在某篇文档中频繁出现的词是信号，加权。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是文档中的词频，`df` 是文档频率（包含该词的文档数），`N` 是文档总数。`log` 使无处不在的词的权重保持有界。

关键特性：两者都产生具有可解释坐标轴的稀疏向量。你可以查看训练好的分类器权重，读出哪些词将文档推向各个类别。而 768 维 BERT 嵌入无法做到这一点。

## 动手实现

### 第一步：构建词汇表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分词后的文档列表（任何词级分词器都行）。输出：`{词: 索引}` 字典。插入顺序稳定，词索引 0 是第一篇文档中第一个出现的词。惯例各异；scikit-learn 按字母顺序排序。

### 第二步：词袋模型

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档，列是词汇表索引。条目 `[i][j]` 是"词 `j` 在文档 `i` 中出现的次数"。第1篇文档中 `cat` 出现了两次，第0篇文档中 `ran` 出现了零次。

### 第三步：词频和文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

两个值得命名的平滑技巧：`(n+1)/(d+1)` 避免 `log(x/0)`；末尾的 `+1` 确保出现在每篇文档中的词 IDF 仍为 1 而非 0，与 scikit-learn 的默认行为一致。其他实现使用原始的 `log(N/df)`，两者都可以，平滑版本更友好。

### 第四步：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三篇文档，五个词汇（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在全部三篇中，IDF 低；`dog` 只出现在一篇，IDF 高。向量是稀疏的（大多数条目很小），有区分性的词会凸显出来。

### 第五步：L2 归一化

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

不归一化的话，较长的文档会得到更大的向量，在相似度计算中占主导。L2 归一化将每篇文档投影到单位超球面上。此后行与行之间的余弦相似度就等于点积。

## 工程应用

scikit-learn 提供了生产级版本：

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 一次完成分词、建词汇表和 BoW。`TfidfVectorizer` 额外添加 IDF 加权和 L2 归一化。两者都返回稀疏矩阵。10 万篇文档时，稠密版本放不进内存——在分类器要求稠密矩阵之前保持稀疏。

改变一切的参数：

| 参数 | 效果 |
|------|------|
| `ngram_range=(1, 2)` | 包含二元组。通常提升分类性能。 |
| `min_df=2` | 去掉在少于 2 篇文档中出现的词。清理噪声数据的词汇表。 |
| `max_df=0.95` | 去掉在超过 95% 文档中出现的词。无需硬编码停用词列表即可近似去除停用词。 |
| `stop_words="english"` | scikit-learn 内置停用词列表。取决于任务——情感分析**不应该**去掉否定词。 |
| `sublinear_tf=True` | 用 `1 + log(tf)` 代替原始 `tf`。当某词在一篇文档中重复多次时有帮助。 |

### TF-IDF 仍然获胜的场景（截至 2026 年）

- 垃圾邮件检测、主题标签、日志异常标记——词语是否出现才是关键，语义细微差异无所谓。
- 低数据情况（数百个标注样本）。TF-IDF 加逻辑回归无需预训练成本。
- 任何对延迟敏感的场景。TF-IDF 加线性模型能在微秒内响应，而通过 Transformer 嵌入一篇文档需要 10-100ms。
- 必须解释预测结果的系统——检查分类器系数，正权重最高的词就是原因。

### TF-IDF 失败的场景

语义盲目性失败。考虑这两篇文档：

- "The movie was not good at all."（这部电影一点都不好。）
- "The movie was excellent."（这部电影很出色。）

一个是负面评价，一个是正面评价。它们的 TF-IDF 重叠词汇恰好是 `{the, movie, was}`。词袋分类器必须死记硬背"not good 附近的 not 词会翻转标签"。有足够数据时能学会，但永远不如理解句法的模型优雅。

另一个失败：推理时的未登录词。在 IMDb 影评上训练的 BoW 模型对 `Zoomer-approved` 完全不知所措，如果这个 token 从未出现在训练数据中。子词嵌入（第4课）能处理这个问题，TF-IDF 不行。

### 混合方案：TF-IDF 加权嵌入

2026 年中等数据分类的务实默认方案：用 TF-IDF 权重作为词嵌入的注意力。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

从嵌入中获得语义能力，从 TF-IDF 中获得对稀有词的强调。分类器在池化后的向量上训练。在约 5 万条标注样本以下的情感、主题和意图分类中，这比单独使用两者中任何一个都好。

## 交付物

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 练习

1. **（简单）** 在 L2 归一化后的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得分为 1.0，词汇表不重叠的文档得分为 0.0。
2. **（中等）** 为 `bag_of_words` 添加 n-gram 支持。参数 `n` 生成 n-gram 的计数。测试 `n=2` 在 `["the", "cat", "sat"]` 上产生 `["the cat", "cat sat"]` 的二元组计数。
3. **（困难）** 用 GloVe 100d 向量（下载一次，缓存）构建上面的 TF-IDF 加权嵌入混合方案。在 20 Newsgroups 数据集上比较与纯 TF-IDF 和纯均值池化嵌入的分类精度，报告各方案在哪里获胜。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| BoW（词袋） | "词频向量" | 一篇文档中词汇表各词的计数，丢弃顺序 |
| TF（词频） | "词出现频率" | 一个词在文档中的计数，可选地除以文档长度 |
| DF（文档频率） | "文档出现次数" | 至少包含该词一次的文档数量 |
| IDF（逆文档频率） | "逆文档频率" | `log(N / df)` 平滑后的值，降权无处不在的词 |
| 稀疏向量 | "大多是零" | 词汇表通常有 1 万到 10 万个词，大多不出现在任意给定文档中 |
| 余弦相似度 | "向量夹角" | L2 归一化向量的点积，1 表示相同，0 表示正交 |

## 延伸阅读

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 权威 API 参考，加上每个参数的说明
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 让 TF-IDF 成为十年默认方案的论文
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 年视角：老方法何时获胜及原因

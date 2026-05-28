# 主题建模——LDA 与 BERTopic

> LDA：文档是主题的混合，主题是词的分布。BERTopic：文档在嵌入空间聚类，聚类就是主题。目标相同，分解方式不同。

**类型：** 学习
**语言：** Python
**前置知识：** 第5阶段第2课（BoW + TF-IDF）、第5阶段第3课（Word2Vec）
**预计时间：** 约45分钟

## 问题背景

你有 1 万张客服工单、5 万篇新闻文章、或者 20 万条推文。你需要知道这个集合讲的是什么，但不能一篇一篇读。你没有标注好的类别，甚至不知道有多少个类别。

主题建模在不需要监督的情况下回答这个问题：给它一个语料库，得到一组连贯的主题，以及每篇文档在这些主题上的分布。

两个算法家族占主导。**LDA**（2003）把每篇文档视为潜在主题的混合，每个主题是词的分布，推理是贝叶斯式的。在需要混合成员主题分配和可解释词级概率分布的场景中，它仍然活跃在生产环境。

**BERTopic**（2020）用 BERT 编码文档，用 UMAP 降维，用 HDBSCAN 聚类，再用基于类的 TF-IDF 提取主题词。它在短文本、社交媒体和任何语义相似度比词重叠更重要的场合胜出。每篇文档只得到一个主题，这对长文本内容是个局限。

本课为两者都建立直觉，并说明在给定语料库时该选哪个。

## 核心概念

**LDA 生成故事**：每个主题是词的分布，每篇文档是主题的混合。生成文档中的一个词：从文档的主题混合中采样一个主题，再从那个主题的词分布中采样一个词。推理是反向的：给定观测到的词，推断每篇文档的主题分布和每个主题的词分布。折叠吉布斯采样或变分贝叶斯来完成计算。

LDA 的关键输出：
- `doc_topic`：矩阵 `(n_docs, n_topics)`，每行和为 1（文档的主题混合）。
- `topic_word`：矩阵 `(n_topics, vocab_size)`，每行和为 1（主题的词分布）。

**BERTopic 流水线**：
1. 用句子 Transformer（如 `all-MiniLM-L6-v2`）编码每篇文档，得到 384 维向量。
2. 用 UMAP 将维度降至约 5 维，BERT 嵌入维度太高，不适合直接聚类。
3. 用 HDBSCAN 聚类，基于密度，产生大小不等的聚类和"离群点"标签。
4. 对每个聚类，在该聚类的文档上计算基于类的 TF-IDF，提取最高权重词。

输出是每篇文档一个主题（加上 -1 离群点标签），可选地通过 HDBSCAN 的概率向量给出软成员度。

## 动手实现

### 第一步：用 scikit-learn 实现 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：去除停用词，`min_df` 和 `max_df` 过滤罕见词和无处不在的词，使用 CountVectorizer（而非 TfidfVectorizer），因为 LDA 期望原始计数。

### 第二步：BERTopic（生产用法）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

`Topic != -1` 过滤掉 BERTopic 的离群点桶（HDBSCAN 无法归类的文档）。`min_topic_size` 控制 HDBSCAN 的最小聚类大小，BERTopic 库的默认值是 10，本示例明确设为 15。对于超过 1 万文档的语料库，请增大到 50 或 100。

### 第三步：评估

两种方法都输出主题词，问题是这些词是否连贯。

- **主题连贯性（c_v）**：对顶级词对的归一化逐点互信息（NPMI）在滑动窗口上下文中求和，将分数聚合成主题向量，再通过余弦相似度对比。越高越好。使用 `gensim.models.CoherenceModel` 并设 `coherence="c_v"`。
- **主题多样性**：所有主题顶级词中唯一词的比例，越高越好（主题不重叠）。
- **定性检查**：读每个主题的顶级词，它们是否描述了一个真实的事物？人类判断仍然是最后一道防线。

## 如何选择

| 情况 | 选择 |
|------|------|
| 短文本（推文、评论、标题） | BERTopic |
| 有主题混合的长文档 | LDA |
| 无 GPU / 计算资源有限 | LDA 或 NMF |
| 需要文档级多主题分布 | LDA |
| 与 LLM 集成做主题标注 | BERTopic（直接支持） |
| 资源受限的边缘部署 | LDA |
| 最高语义连贯性 | BERTopic |

最重要的实践考量是文档长度。BERT 嵌入会截断；LDA 的计数对任意长度都有效。对于超过嵌入模型上下文长度的文档，要么分块后聚合，要么用 LDA。

## 工程应用

2026 年技术栈：

- **BERTopic**：短文本和语义重要场景的默认选择。
- **`gensim.models.LdaModel`**：生产用的经典 LDA，成熟、经过实战检验。
- **`sklearn.decomposition.LatentDirichletAllocation`**：实验用的简便 LDA。
- **NMF（非负矩阵分解）**：LDA 的快速替代，在短文本上质量相当。
- **Top2Vec**：与 BERTopic 设计类似，社区较小但在某些基准上表现良好。
- **FASTopic**：更新、在非常大的语料库上比 BERTopic 快。
- **基于 LLM 的标注**：先做任意聚类，再用模型提示给每个聚类命名。

## 交付物

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 练习

1. **（简单）** 在 20 Newsgroups 数据集上用 5 个主题拟合 LDA，打印每个主题的前 10 个词，手工给每个主题贴标签，看看算法是否找到了真实的类别。
2. **（中等）** 在同一 20 Newsgroups 子集上拟合 BERTopic，对比发现的主题数量、顶级词和定性连贯性与 LDA 的差异，哪种方法更清晰地呈现出真实类别？
3. **（困难）** 计算你的语料库上 LDA 和 BERTopic 各自的 c_v 连贯性，分别用 5、10、20、50 个主题运行，绘制连贯性与主题数的关系曲线，报告哪种方法在不同主题数下更稳定。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 主题 (Topic) | "语料库谈论的事情" | 词的概率分布（LDA）或相似文档的聚类（BERTopic） |
| 混合成员 (Mixed membership) | "文档属于多个主题" | LDA 给每篇文档分配所有主题上的分布 |
| UMAP | "降维" | 保留局部结构的流形学习，BERTopic 中使用 |
| HDBSCAN | "密度聚类" | 找到大小不等的聚类，为离群点产生"噪声"标签（-1） |
| c_v 连贯性 | "主题质量指标" | 顶级主题词在滑动窗口内的平均逐点互信息 |

## 延伸阅读

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 论文
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 论文
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 引入 c_v 等指标的论文
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — 生产参考，示例优秀

# 信息检索与搜索

> BM25 精准但脆，密集检索撒网广但漏关键词。混合检索是 2026 年的默认方案。其余一切都是调优。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第2课（BoW + TF-IDF）、第5阶段第4课（GloVe、FastText、子词）
**预计时间：** 约75分钟

## 问题背景

用户输入"有人撒谎骗钱会怎样"，期望找到真正覆盖这种情况的法条："印度刑法第 420 条"。关键词搜索会完全漏掉（没有共享词汇），语义搜索如果嵌入模型没有在法律文本上训练过也会漏掉。真实的搜索必须两者都能处理。

信息检索是每个 RAG 系统、每个搜索栏、每个文档站点模糊查找背后的流水线。2026 年生产中有效的架构不是单一方法，而是互补方法的链式组合，每一步都在捕捉前一步的失败。

本课实现每个模块，并说明每个模块捕捉哪种失败。

## 核心概念

四层架构，按需选取：

1. **稀疏检索（BM25）**：快速，精准匹配，语义上一塌糊涂。在倒排索引上运行，在百万文档上每次查询 10ms 以内。擅长找法条引用、产品编码、错误信息、命名实体。
2. **密集检索**：将查询和文档编码成向量，做最近邻搜索。能捕捉同义改写和语义相似度，但对差一个字符的精确关键词匹配无能为力。使用 FAISS 或向量数据库时每次查询 50-200ms。
3. **融合**：合并稀疏和密集的排名列表。**互反排名融合（RRF）** 是简便默认方案，因为它忽略原始分数（两者量纲不同），只使用排名位置。当你知道某个信号在特定领域更主导时，加权融合是可选项。
4. **交叉编码器重排**：取融合结果的 top-30，用交叉编码器（查询+文档一起，对每对打分）。保留 top-5。交叉编码器每对比双编码器慢，但精度高得多，只对 top-30 运行可以摊平成本。

2026 年基准显示三路检索（BM25 + 密集 + 学习稀疏如 SPLADE）优于两路，但需要为学习稀疏索引搭建基础设施。对大多数团队来说，两路加交叉编码器重排是最佳平衡点。

## 动手实现

### 第一步：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个值得了解的参数：`k1=1.5` 控制词频饱和，越高对词重复越看重；`b=0.75` 控制长度归一化，0 忽略文档长度，1 完全归一化。这两个默认值来自原始论文 Robertson 的推荐，几乎不需要调整。

### 第二步：用双编码器做密集检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

对嵌入做 L2 归一化后，点积等于余弦相似度。`all-MiniLM-L6-v2` 是 384 维，速度快，对大多数英语检索任务足够强。多语言工作用 `paraphrase-multilingual-MiniLM-L12-v2`，最高精度用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 第三步：互反排名融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 常数来自原始 RRF 论文。`k` 越高，排名差异的贡献越平坦；越低，高排名越主导。60 是已发布的默认值，很少需要调整。

### 第四步：混合搜索加重排

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三阶段组合：BM25 找词汇匹配，密集检索找语义匹配，RRF 在不需要分数校准的情况下合并两个排名，交叉编码器把查询-文档对放在一起重新打分捕捉双编码器错过的细粒度相关性，保留 top-5。

### 第五步：评估

| 指标 | 含义 |
|------|------|
| Recall@k | 对于正确文档存在的查询，有多少比例在 top-k 中？ |
| MRR（平均互反排名） | 第一个相关文档排名的 1/rank 的平均值 |
| nDCG@k | 考虑相关度等级，不只是二元相关/不相关 |

对于 RAG 而言，检索器的 **Recall@k** 是最重要的数字。如果正确文档不在检索集中，阅读器就无法回答。

调试技巧：对于失败的查询，对比稀疏和密集排名的差异。如果一个找到了正确文档而另一个没有，要么是词汇不匹配（修复：补充缺失的那半）要么是语义歧义（修复：更好的嵌入或重排序器）。

## 工程应用

2026 年技术栈：

| 规模 | 技术选型 |
|------|---------|
| 1k-10w 文档 | 内存中 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF，无需单独数据库 |
| 10w-1000w 文档 | FAISS 或 pgvector 做密集检索 + Elasticsearch/OpenSearch 做 BM25，并行运行 |
| 1000w+ 文档 | 支持混合检索的 Qdrant / Weaviate / Vespa / Milvus，加 top-30 交叉编码器重排 |
| 最高质量前沿 | 三路（BM25 + 密集 + SPLADE）+ ColBERT 后交互重排 |

无论选哪种，都要给评估预留资源。先基准测试检索召回率，再基准测试端到端 RAG 精度。检索器遗漏的内容，阅读器无法弥补。

### 2026 年生产 RAG 的经验教训

- **80% 的 RAG 失败源于数据摄取和分块，而非模型**。团队花几周时间换 LLM 和调提示词，而检索器每隔三次查询就悄悄返回错误上下文。先修分块。
- **分块策略比分块大小更重要**。固定大小切割会破坏表格、代码和嵌套标题。句子感知是默认方案；语义或基于 LLM 的分块对技术文档和产品手册更划算。
- **父子文档模式**：检索小的"子"块获得精确度。当来自同一父节的多个子块出现时，换入父块以保留上下文。这能持续提升答案质量，而不需要重新训练。
- **k_rerank=3 通常是最优**。超过 3 块之后每增加一块都增加 token 成本和生成延迟，而不提升答案质量。如果你的情况下 k=8 仍然优于 k=3，说明重排序器性能不足。
- **HyDE / 查询扩展**：从查询生成假设性答案，嵌入那个答案，再检索。弥合短问题和长文档之间的措辞差距，无需训练即可免费提升精度。
- **上下文预算控制在 8K token 以内**。频繁触碰这个上限意味着重排序器阈值过松。
- **版本化一切**：提示词、分块规则、嵌入模型、重排序器。任何漂移都会悄无声息地破坏答案质量。在忠实度、上下文精确率和未回答率上设 CI 门控，在用户看到问题之前阻断回归。
- **三路检索（BM25 + 密集 + 学习稀疏如 SPLADE）** 在 2026 年基准上优于两路，尤其对混合专有名词和语义的查询。基础设施支持 SPLADE 索引时就部署。

根据 2026 年行业测量，正确的检索设计可将幻觉率降低 70-90%。大多数 RAG 性能提升来自更好的检索，而非模型微调。

## 交付物

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习

1. **（简单）** 在 500 文档语料上实现上面的 `hybrid_search`，测试 20 个查询，对比 BM25 单独、密集检索单独和混合检索在 Recall@5 上的差异。
2. **（中等）** 添加 MRR 计算：对每个有已知正确文档的测试查询，找到该文档在 BM25、密集检索和混合检索排名中的位置，报告各方法的 MRR。
3. **（困难）** 用 MultipleNegativesRankingLoss（Sentence Transformers）在你选择的领域上微调密集编码器，用 500 个查询-文档对构建训练集，对比微调前后的召回率。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| BM25 | "关键词搜索" | Okapi BM25，按词频、IDF 和文档长度给文档打分 |
| 密集检索 (Dense retrieval) | "向量搜索" | 把查询和文档编码成向量，找最近邻 |
| 双编码器 (Bi-encoder) | "嵌入模型" | 独立编码查询和文档，查询时速度快 |
| 交叉编码器 (Cross-encoder) | "重排序模型" | 将查询和文档一起编码，慢但精准 |
| RRF（互反排名融合） | "排名融合" | 通过对 `1/(k + rank)` 求和合并两个排名 |
| Recall@k | "检索指标" | 相关文档出现在 top-k 中的查询比例 |

## 延伸阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — 权威的 BM25 论文
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，经典双编码器
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 填补密集检索空白的学习稀疏检索器
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 论文
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — 后交互检索

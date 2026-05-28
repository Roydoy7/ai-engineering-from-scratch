# 嵌入模型深度解析

> Word2Vec 给你每个词一个向量。现代嵌入模型给你每个段落一个向量——跨语言、稀疏加密集加多向量三合一，尺寸灵活。选错了，你的 RAG 40% 的时间检索到错误内容。

**类型：** 学习
**语言：** Python
**前置知识：** 第5阶段第3课（Word2Vec）、第5阶段第14课（信息检索）
**预计时间：** 约60分钟

## 问题背景

你的 RAG 系统有 40% 的时间检索到错误的段落。罪魁祸首很少是向量数据库或提示词——通常是嵌入模型。

2026 年选择嵌入模型需要在五个维度上权衡：

1. **密集 vs 稀疏 vs 多向量**。每个段落一个向量、每个 token 一个向量，还是稀疏加权词袋？
2. **语言覆盖**。单语言英语模型在纯英语任务上仍然领先，多语言模型在混合语料库时胜出。
3. **上下文长度**。512 token vs 8192 vs 32768——实际有效容量通常是标称最大值的 60-70%。
4. **维度预算**。全精度 3072 维 = 每个向量 12 KB。1 亿个向量，存储费用约 1300 美元/月。Matryoshka 截断可以节省 4 倍空间。
5. **开放 vs 托管**。开放权重意味着你掌控整个技术栈和数据；托管则用控制权换取始终最新的模型。

本课把这些权衡说清楚，让你凭证据选模型，而不是跟风选上个季度流行的东西。

## 核心概念

**密集嵌入（Dense Embedding）**：每个段落一个向量（通常 384-3072 维），用余弦相似度按语义距离排名段落。代表模型：OpenAI `text-embedding-3-large`、BGE-M3 密集模式、Voyage-3。默认首选。

**稀疏嵌入（Sparse Embedding）**：SPLADE 风格。Transformer 对词汇表中每个 token 预测一个权重，然后大部分置零，结果是一个大小为 |vocab| 的稀疏向量。能捕捉词汇匹配（类似 BM25），但权重是端到端学习的。对关键词密集的查询效果强。

**多向量（Multi-Vector，晚交互）**：ColBERTv2、Jina-ColBERT。每个 token 一个向量。用 MaxSim 打分：对查询中每个 token，找文档中最相似的 token，把这些分数求和。存储和打分成本更高，但在长查询和特定领域语料库上表现更好。

**BGE-M3：三合一**。单个模型同时输出密集、稀疏和多向量表示，每种模式都能独立查询，得分通过加权求和融合。2026 年希望从一个检查点获得灵活性时的默认选择。

**Matryoshka 表示学习**：模型被训练为：向量的前 N 维本身就是一个有用的独立嵌入。把 1536 维向量截断到 256 维，付出约 1% 的精度损失，换取 6 倍存储节省。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+ 均支持。

### MTEB 排行榜讲了一半的故事

Massive Text Embedding Benchmark——2022 年推出时覆盖 8 种任务类型共 56 个任务，MTEB v2 扩展到 100+ 个任务。2026 年初，Gemini Embedding 2 领跑检索（67.71 MTEB-R），Cohere embed-v4 领跑通用评测（65.2 MTEB），BGE-M3 领跑开放权重多语言（63.0）。排行榜是必要参考但不够充分——始终在你的领域做基准测试。

### 三层模式

| 用途 | 方案 |
|------|------|
| 快速初筛 | 密集双编码器（BGE-M3、text-3-small） |
| 召回率提升 | 稀疏（SPLADE、BGE-M3 稀疏）+ RRF 融合 |
| top-50 精排 | 多向量（ColBERTv2）或交叉编码器重排序 |

大多数生产系统三层都会用到。

## 动手实现

### 第一步：基线——用 Sentence-BERT 做密集嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 使点积等于余弦相似度。始终设置它。

### 第二步：Matryoshka 截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后重新归一化。Nomic v1.5、OpenAI text-3 和 Voyage-4 经过专门训练，前几个维度截断几乎无损。非 Matryoshka 模型（原始 Sentence-BERT）截断后性能会急剧下降。

### 第三步：BGE-M3 多功能模式

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

三个索引，一次推理。得分融合：

```python
dense_score = ... # 对 dense_vecs 做余弦相似度
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在你的领域数据上调整这些权重。

### 第四步：用 MTEB 在自定义任务上评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在**具有代表性的**任务子集上运行候选模型。不要只信排行榜名次——你的领域才是决定性因素。

### 第五步：从零手写余弦相似度

见 `code/main.py`。使用平均哈希技巧嵌入（纯标准库）。无法与 Transformer 嵌入竞争，但展示了核心形状：分词 → 向量 → 归一化 → 点积。

## 陷阱

- **查询和文档用同一模型路径**。部分模型（Voyage、Jina-ColBERT）使用非对称编码——查询和文档走不同路径。始终检查模型卡。
- **遗漏前缀**。`bge-*` 系列模型查询时需要加前缀 `"Represent this sentence for searching relevant passages: "`，忘了会有 3-5 个点的召回率差距。
- **Matryoshka 截得太狠**。1536 → 256 通常安全，1536 → 64 则不行。要在评估集上验证。
- **上下文截断**。大多数模型对超过最大长度的输入静默截断。长文档需要分块（见第23课）。
- **忽略延迟尾部**。MTEB 分数隐藏了 p99 延迟。600M 参数模型可能比 335M 模型多 2 分，但每次查询成本高 3 倍。

## 工程应用

2026 年技术栈：

| 情况 | 选择 |
|------|------|
| 纯英语、快速、API 接入 | `text-embedding-3-large` 或 `voyage-3-large` |
| 开放权重、英语 | `BAAI/bge-large-en-v1.5` |
| 开放权重、多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文（32k+） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| 仅 CPU 部署 | Nomic Embed v2（137M 参数，MoE 架构） |
| 存储受限 | Matryoshka 截断 + int8 量化 |
| 关键词密集查询 | 加 SPLADE 稀疏，与密集 RRF 融合 |

2026 年模式：从 BGE-M3 或 text-3-large 开始，在你的领域用 MTEB 评估，如果领域特定模型赢出超过 3 个点则换用。

## 交付物

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## 练习

1. **（简单）** 用 `bge-small-en-v1.5` 对 100 个句子编码，分别取全维（384）和 Matryoshka 128 维，在 10 个查询上测量 MRR 下降幅度。
2. **（中等）** 在你领域的 500 个段落上对比 BGE-M3 的密集、稀疏和 ColBERT 三种模式，哪种在 recall@10 上最好？RRF 融合能不能超过最好的单一模式？
3. **（困难）** 在你领域最重要的 2 个任务上用 MTEB 跑三个候选模型，报告 MTEB 分、100 个查询批次的 p99 延迟和每百万次查询的费用，选出 Pareto 最优的那个。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 密集嵌入 (Dense embedding) | "那个向量" | 每段文本一个固定大小的向量，用余弦相似度排名 |
| 稀疏嵌入 (Sparse embedding) | "学到的 BM25" | 词汇表中每个 token 一个权重，大部分为零，端到端训练 |
| 多向量 (Multi-vector) | "ColBERT 风格" | 每个 token 一个向量，用 MaxSim 打分，索引更大但召回率更好 |
| Matryoshka | "俄罗斯套娃技巧" | 前 N 维本身就是一个有效的更小嵌入 |
| MTEB | "那个基准" | Massive Text Embedding Benchmark——发布时 56 个任务，v2 扩展到 100+ |
| BEIR | "检索基准" | 18 个零样本检索任务，常用于评测跨领域鲁棒性 |
| 非对称编码 (Asymmetric encoding) | "查询和文档路径不同" | 模型对查询和文档使用不同的投影层 |

## 延伸阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — 双编码器论文
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — 排行榜论文
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — 三合一统一模型
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — 维度梯度训练目标
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — 晚交互在生产中的应用
- [MTEB 排行榜（Hugging Face）](https://huggingface.co/spaces/mteb/leaderboard) — 实时排名

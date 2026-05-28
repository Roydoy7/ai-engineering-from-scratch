# RAG 分块策略

> 分块配置对检索质量的影响与嵌入模型的选择一样大（Vectara NAACL 2025）。分块做错了，再好的重排序也救不了你。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第14课（信息检索）、第5阶段第22课（嵌入模型）
**预计时间：** 约60分钟

## 问题背景

你把一份 50 页的合同放进 RAG 系统。用户问："终止条款是什么？"检索器返回了封面页。为什么？因为模型在 512 token 的块上训练，而终止条款在 20 页之后，跨了一个换页符，没有局部关键词把它和查询联系起来。

解决方案不是"换个更好的嵌入模型"，而是分块策略。多大？要不要重叠？在哪里切？要不要带上下文？

2026 年 2 月的基准测试给出了令人惊讶的结果：

- Vectara 2026 研究：递归 512 token 分块以 69% vs 54% 的准确率击败语义分块。
- SPLADE + Mistral-8B 在 Natural Questions 上：重叠带来的可测量收益为零。
- 上下文悬崖：响应质量在约 2500 token 上下文处急剧下降。

"显而易见"的答案（语义分块、20% 重叠、1000 token）往往是错的。本课建立对六种策略的直觉，并告诉你何时选哪种。

## 核心概念

**固定分块（Fixed）**：每 N 个字符或 token 切一刀。最简单的基线。会断在句子中间。压缩率好，连贯性差。

**递归分块（Recursive）**：LangChain 的 `RecursiveCharacterTextSplitter`。先尝试在 `\n\n` 切，再 `\n`，再 `. `，再空格，层层回退。2026 年的默认选择。

**语义分块（Semantic）**：对每个句子编码，计算相邻句子的余弦相似度，在相似度低于阈值处切割。保持话题连贯。速度慢，有时会产生 40 token 的碎片，损害检索质量。

**句子分块（Sentence）**：在句子边界切割，每块一句或 N 句的滑动窗口。在约 5k token 以内，效果与语义分块相当，但成本只有后者的一小部分。

**父子文档分块（Parent-Document）**：存储小的子块用于检索，同时存储更大的父块用于上下文。按子块检索，返回父块。容错性好：即便子块质量一般，父块内容仍然合理。

**晚分块（Late Chunking，2024）**：先在 token 级别对整个文档编码，再把 token 嵌入聚合成块嵌入。保留跨块的上下文。需要长上下文嵌入器（BGE-M3、Jina v3）。计算成本更高。

**上下文检索（Contextual Retrieval，Anthropic，2024）**：在索引前，用 LLM 为每个块生成一段描述其在文档中位置的摘要（"本块是终止条款 3.2 节……"），把这段摘要前置到块内容前。Anthropic 自己的基准测试显示检索质量提升 35-50%。索引成本高。

### 超越默认配置的核心规则

把块大小和查询类型匹配起来：

| 查询类型 | 块大小 |
|---------|--------|
| 事实性（"CEO 叫什么名字？"） | 256-512 token |
| 分析性 / 多跳推理 | 512-1024 token |
| 整节理解 | 1024-2048 token |

来自 NVIDIA 2026 基准测试。块要足够大，能容纳答案和局部上下文；足够小，让检索器的 top-K 聚焦在答案本身而非上下文噪声。

## 动手实现

### 第一步：固定和递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 第二步：语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

在你的领域数据上调整 `threshold`。太高 → 产生碎片；太低 → 产生一个巨大的块。

### 第三步：父子文档分块

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键细节：对父块去重。多个子块可能映射到同一父块，重复返回会浪费上下文窗口。

### 第四步：上下文检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

对上下文化后的块建立索引。查询时，额外的位置信号让检索质量显著提升。

### 第五步：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

始终做基准测试。适合你语料库的"最佳"策略可能和任何博客文章说的都不一样。

## 陷阱

- **只用事实性查询评估分块**。多跳查询会给出截然不同的排名结果。用按查询类型分层的评估集。
- **语义分块没有最小块限制**。会产生 40 token 的碎片，损害检索。始终设置 `min_tokens`。
- **把重叠当成惯例**。2026 年研究发现重叠通常带来零收益，却让索引成本翻倍。要测量，不要假设。
- **没有最小/最大限制**。5 个 token 或 5000 个 token 的块都会破坏检索。设置上下界。
- **跨文档分块**。永远不要让一个块横跨两个文档。始终按文档分块，再合并。

## 工程应用

2026 年技术栈：

| 情况 | 策略 |
|------|------|
| 首次构建，语料库未知 | 递归，512 token，无重叠 |
| 事实性 QA | 递归，256-512 token |
| 分析性 / 多跳推理 | 递归，512-1024 token + 父子文档 |
| 大量交叉引用（合同、论文） | 晚分块或上下文检索 |
| 对话 / 多轮语料 | 轮次级别分块 + 发言者元数据 |
| 短文本（推文、评论） | 一文档一块 |

从递归 512 开始，在 50 个查询的评估集上测量 recall@5，再从那里调优。

## 交付物

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 练习

1. **（简单）** 用 fixed(512, 0)、recursive(512, 0) 和 recursive(512, 100) 三种方式对一份 20 页文档分块，比较块数量和边界质量。
2. **（中等）** 在 5 篇文档上构建 30 个查询的评估集，分别测量递归、语义和父子文档分块的 recall@5，哪种赢了？结果和博客文章说的一致吗？
3. **（困难）** 实现上下文检索，测量相对于基线递归分块的 MRR 提升，同时报告索引成本（LLM 调用次数）和精度收益的对比。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 块 (Chunk) | "文档的一片" | 被嵌入、索引和检索的子文档单元 |
| 重叠 (Overlap) | "安全边距" | 相邻块共享的 N 个 token，2026 年基准测试中通常无用 |
| 语义分块 (Semantic chunking) | "智能分块" | 在相邻句子嵌入相似度下降处切割 |
| 父子文档 (Parent-document) | "两级检索" | 检索小子块，返回大父块 |
| 晚分块 (Late chunking) | "先嵌入再分块" | 在 token 级别嵌入全文，再聚合成块向量 |
| 上下文检索 (Contextual retrieval) | "Anthropic 的技巧" | 索引前在每个块前面加 LLM 生成的位置摘要 |
| 上下文悬崖 (Context cliff) | "2500 token 壁" | RAG 中在约 2500 token 上下文处观察到的质量骤降（2026 年 1 月） |

## 延伸阅读

- [LangChain — Recursive Character Splitting 文档](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产中的默认方案
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — 分块和嵌入选择同等重要
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 晚分块论文
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — LLM 生成上下文前缀，检索质量提升 35-50%
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 按查询类型划分的块大小建议

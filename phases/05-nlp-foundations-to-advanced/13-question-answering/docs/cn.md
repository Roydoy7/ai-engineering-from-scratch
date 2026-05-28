# 问答系统

> 三种系统塑造了现代问答：抽取式找到答案片段，检索增强把答案扎根于文档，生成式直接产出答案。每一个现代 AI 助手都是这三者的混合。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第11课（机器翻译）、第5阶段第10课（注意力机制）
**预计时间：** 约75分钟

## 问题背景

用户输入"第一代 iPhone 是什么时候发布的？"，期望的回答是"2007 年 6 月 29 日"，而不是"苹果的历史悠久而丰富"，也不是孤零零的"2007"。要的是直接、有依据、正确的答案。

过去十年间，三种架构主导了问答任务：

- **抽取式 QA**：给定问题和一段已知包含答案的文章，在文章中找到答案片段的起止索引。SQuAD 是经典基准。
- **开放域 QA**：没有给定文章，先检索相关文章，再抽取或生成答案。这是当今每个 RAG 流水线的基石。
- **生成式 / 封闭书本 QA**：大语言模型从参数记忆中回答，无需检索。推理最快，事实最不可靠。

2026 年的趋势是混合方式：检索最好的几段文章，再提示生成模型基于这些文章给出答案。这就是 RAG，第14课深入讲检索部分。本课搭建问答部分。

## 核心概念

**抽取式**：用 Transformer（BERT 家族）对问题和文章一起编码，训练两个预测答案起止 token 索引的分类头。损失是在有效位置上的交叉熵，输出是文章中的一个片段。设计上永远不产生幻觉，也设计上无法处理文章答不了的问题。

**检索增强（RAG）**：两个阶段。首先，检索器从语料库中找到 top-k 段文章；然后，阅读器（抽取式或生成式）用这些文章生成答案。检索器-阅读器分离让两者可以独立训练和评估。现代 RAG 通常在两者之间加一个重排序器。

**生成式**：仅解码器的 LLM（GPT、Claude、Llama）从学到的权重中回答，无检索步骤。对常见知识表现优秀，对罕见或最新事实则灾难性失效。幻觉率与预训练数据中事实出现的频率成反比。

## 动手实现

### 第一步：用预训练模型做抽取式 QA

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 在 SQuAD 2.0 上训练，包含无法回答的问题。默认情况下，`question-answering` 流水线即使空答案得分最高也会返回得分最高的片段，**不会**自动返回空答案。要获得明确的"无答案"行为，需在流水线调用中传入 `handle_impossible_answer=True`，此时只有空答案得分超过所有片段得分时才返回空答案。无论哪种情况，都要检查 `score` 字段。

### 第二步：检索增强流水线（简版）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段流水线：密集检索器（Sentence-BERT）通过语义相似度找到相关文章，抽取式阅读器（RoBERTa-SQuAD）从合并的文章中抽取答案片段。适用于小型语料库；对于百万文档级别，使用 FAISS 或向量数据库。

### 第三步：带 RAG 的生成式问答

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示词模式很重要。明确告诉模型基于上下文作答，且在上下文不足时返回"不知道"，相比朴素提示可以将幻觉率降低 40-60%。更精细的模式还会加上引用来源、置信度评分和结构化提取。

### 第四步：反映真实世界的评估

SQuAD 使用**精确匹配（EM）**和**token 级 F1**。EM 在归一化后（小写化、去标点、去冠词）做严格匹配，要么完全一致要么得 0 分。F1 在预测和参考的 token 重叠上计算，给部分信用。两者都低估了改写的贡献："June 29, 2007"和"June 29th, 2007"通常得 0 EM（序数词打破了归一化），但重叠 token 仍然给出可观的 F1。

生产 QA 的评估维度：

- **答案准确率**（LLM 裁判或人工裁判，因为指标无法捕捉语义等价）。
- **引用准确率**：引用的文章段落是否真的支持答案？用字符串匹配生成引用和检索段落之间的关系，很容易自动检查。
- **拒答校准**：当答案不在检索到的文章中时，系统是否正确地说"不知道"？测量误自信率。
- **检索召回率**：在评估阅读器之前，先测量检索器是否把正确的文章放入了 top-k。阅读器无法弥补缺失的文章。

### RAGAS：2026 年的生产评估框架

`RAGAS` 是专为 RAG 系统设计的评估框架，是 2026 年的发布默认选择。它在不需要黄金参考答案的情况下对四个维度打分：

- **忠实度（Faithfulness）**：答案中的每个声明是否来自检索到的上下文？通过基于 NLI 的蕴含关系衡量。这是主要的幻觉指标。
- **答案相关性（Answer relevance）**：答案是否回答了问题？通过从答案生成假设性问题并与真实问题对比来衡量。
- **上下文精确率（Context precision）**：检索到的文本块中，有多少比例真正相关？低精确率意味着提示中有噪声。
- **上下文召回率（Context recall）**：检索到的集合是否包含所有必要信息？低召回率意味着阅读器无法成功。

无参考评分让你可以在实时生产流量上评估，无需人工标注的黄金答案。对于精确匹配指标无效的开放性问题，在上面叠加 LLM-as-judge。

`pip install ragas`，接入你的检索器和阅读器，每个查询得到四个标量，对回归做告警。

## 工程应用

2026 年技术栈：

| 使用场景 | 推荐方案 |
|---------|---------|
| 给定文章，找答案片段 | `deepset/roberta-base-squad2` |
| 固定语料库，不接受封闭书本 | RAG：密集检索器 + LLM 阅读器 |
| 实时文档库 | RAG + 混合检索（BM25 + 密集）+ 重排序器（第14课） |
| 对话式 QA（追问） | 带对话历史的 LLM + 每轮 RAG |
| 高事实性、受监管领域 | 权威语料库上的抽取式，绝不单独使用生成式 |

抽取式 QA 在 2026 年已经不流行，因为带 LLM 的 RAG 能处理更多情况。但在需要原文引用的场合它仍然是主流：法律研究、法规遵从、审计工具。

## 交付物

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习

1. **（简单）** 在 10 段维基百科文章上搭建上面的 SQuAD 抽取式流水线，手工设计 10 个问题，测量正确率，文章和问题干净的话应该能看到 7-9 个正确。
2. **（中等）** 添加拒答分类器：当最高检索得分低于阈值（如余弦相似度 0.3）时，返回"不知道"而非调用阅读器，在保留集上调优阈值。
3. **（困难）** 在自选的 1 万文档语料上构建 RAG 流水线，实现混合检索（BM25 + 密集）加 RRF 融合（见第14课），对比有无混合检索步骤的答案准确率，记录哪类问题受益最大。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 抽取式 QA (Extractive QA) | "找答案片段" | 在给定文章中预测答案的起止索引 |
| 开放域 QA (Open-domain QA) | "在语料库上问答" | 无给定文章，需要先检索再回答 |
| RAG（检索增强生成） | "检索后生成" | 检索器 + 阅读器流水线 |
| SQuAD | "经典基准" | 斯坦福问答数据集，用 EM + F1 评估 |
| 幻觉 (Hallucination) | "凭空捏造" | 阅读器输出中不被检索上下文支持的内容 |
| 拒答校准 (Refusal calibration) | "知道何时闭嘴" | 系统在无法回答时正确说"不知道" |

## 延伸阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 基准论文
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，问答的经典密集检索器
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 命名 RAG 的论文
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 全面的 RAG 综述

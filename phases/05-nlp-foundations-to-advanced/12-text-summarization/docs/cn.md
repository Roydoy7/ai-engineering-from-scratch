# 文本摘要

> 抽取式系统告诉你文档说了什么，生成式系统告诉你作者想表达什么。不同的任务，不同的坑。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第2课（BoW + TF-IDF）、第5阶段第11课（机器翻译）
**预计时间：** 约75分钟

## 问题背景

一篇 2000 字的新闻落入你的 feed，你需要 120 字的摘要。你可以从文章中挑出三句最重要的话（抽取式），或者用自己的话重写内容（生成式）。两者都叫摘要，却完全是两个不同的问题。

抽取式摘要是一个排名问题：给每句话打分，返回前 k 句。输出永远合乎语法，因为是原文照搬。风险是遗漏分散在文章各处的内容。

生成式摘要是一个生成问题：Transformer 以输入为条件生成新文本。输出流畅而压缩，但可能产生源文中没有的幻觉事实。风险是自信地捏造。

本课实现这两种方式，各自讲清楚它们特有的失败模式。

## 核心概念

**抽取式**：把文章当作一个图，节点是句子，边是相似度。对图运行 PageRank（或类似算法），根据与其他句子的连接程度给每句话打分，得分最高的句子构成摘要。经典实现是 **TextRank**（Mihalcea 和 Tarau，2004）。

**生成式**：在文档-摘要对上微调 Transformer 编码器-解码器（BART、T5、Pegasus）。推理时，模型通过交叉注意力读取文档并逐 token 生成摘要。Pegasus 的间隙句子预训练目标使其无需太多微调就能在摘要任务上表现优异。

用 **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation）评估：ROUGE-1 和 ROUGE-2 计算一元和二元 n-gram 重叠，ROUGE-L 计算最长公共子序列。越高越好，但 40 ROUGE-L 算"良好"，50 算"优秀"。每篇论文都会报告全部三个指标。使用 `rouge-score` 包。

## 动手实现

### 第一步：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

两点值得说明：相似度函数使用对数归一化的词重叠，这是原始 TextRank 变体；TF-IDF 向量的余弦相似度也可以。阻尼系数 0.85 和迭代次数是 PageRank 的默认值。

### 第二步：用 BART 做生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail 语料上微调，开箱即可生成新闻风格摘要。其他领域（科学论文、对话、法律）请用对应的 Pegasus 检查点，或在目标数据上微调。

### 第三步：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

始终启用词干化。不启用的话，"running"和"run"算不同的词，ROUGE 会低估匹配数。

### 超越 ROUGE——2026 年的摘要评估

ROUGE 主导摘要评估二十年了，但 2026 年单靠它已经不够。对 NLG 论文的大规模元分析表明：

- **BERTScore**（上下文嵌入相似度）在 2023 年前后获得广泛采用，现在大多数摘要论文都与 ROUGE 一起报告。
- **BARTScore** 把评估当作生成任务：用预训练 BART 给定源文本对摘要打分。
- **MoverScore**（上下文嵌入上的推土机距离）在 2025 年摘要基准中排名最高，因为它比 ROUGE 更好地捕捉语义重叠。
- **FactCC** 和基于 QA 的事实性检测在 2021-2023 年常用，现在通常被 **G-Eval** 取代——G-Eval 是一种使用链式推理打分一致性、连贯性、流畅性和相关性的 GPT-4 提示链。
- **G-Eval** 及类似 LLM 裁判方法在评分标准设计良好时与人类判断的一致率约 80%。

生产推荐：报告 ROUGE-L 用于历史对比，BERTScore 用于语义重叠，G-Eval 用于连贯性和事实性。用 50-100 条人工标注摘要进行校准。

### 第四步：事实性问题

生成式摘要容易产生幻觉。抽取式摘要由于原文照搬，幻觉风险低得多，但如果源句子被去除上下文、已过时或引用顺序混乱，仍可能产生误导。这是生产系统在合规相关内容上仍然偏好抽取式方法的最大原因。

需要了解的幻觉类型：

- **实体替换**：源文说"张三"，摘要说"李四"。
- **数字漂移**：源文说"2.5 万"，摘要说"2500 万"。
- **极性翻转**：源文说"拒绝了提案"，摘要说"接受了提案"。
- **事实捏造**：源文没有提到 CEO，摘要说 CEO 批准了。

有效的评估方法：

- **FactCC**：在源句和摘要句之间的蕴含关系上训练的二分类器，预测是否符合事实。
- **基于 QA 的事实性**：对 QA 模型提问，答案应在源文中存在，如果摘要支持不同的答案则标记。
- **实体级 F1**：比较源文和摘要中的命名实体，仅出现在摘要中的实体是可疑的。

对于事实性重要的面向用户内容（新闻、医疗、法律、金融），抽取式是更安全的默认选择。生成式方法需要在流程中加入事实性检查。

## 工程应用

2026 年技术栈：

| 使用场景 | 推荐方案 |
|---------|---------|
| 新闻，3-5 句英语摘要 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或调优 T5 |
| 多文档、长篇 | 任何 32k+ 上下文的 LLM，加提示词 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式，天然低幻觉风险 | TextRank 或 `sumy` 的 LSA/LexRank |

2026 年在计算资源不受限时，长上下文 LLM 通常比专用模型更好。权衡是成本和可复现性；专用模型输出更一致。

## 交付物

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 练习

1. **（简单）** 对 5 篇新闻文章运行 TextRank，将前 3 句与参考摘要对比，测量 ROUGE-L，CNN/DailyMail 风格文章应该能看到 30-45 的 ROUGE-L。
2. **（中等）** 实现实体级事实性检测：用 spaCy 从源文和摘要中抽取命名实体，计算摘要中源文实体的召回率和摘要实体对应源文的精确率。高精确率低召回率意味着安全但简短；低精确率意味着存在幻觉实体。
3. **（困难）** 在 50 篇 CNN/DailyMail 文章上对比 BART-large-CNN 与 LLM（Claude 或 GPT-4），报告 ROUGE-L、事实性（实体 F1）和每条摘要的成本，记录各自的优势场景。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 抽取式 (Extractive) | "挑句子" | 从源文中原文返回句子，绝不产生幻觉 |
| 生成式 (Abstractive) | "重写" | 以源文为条件生成新文本，可能产生幻觉 |
| ROUGE | "摘要指标" | 系统输出与参考摘要之间的 n-gram/LCS 重叠 |
| TextRank | "基于图的抽取式" | 在句子相似度图上运行 PageRank |
| 事实性 (Factuality) | "内容正确性" | 摘要中的声明是否有源文支持 |
| 幻觉 (Hallucination) | "捏造内容" | 摘要中源文不支持的内容 |

## 延伸阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 抽取式经典论文
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 论文
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus 及间隙句子预训练目标
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 论文
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 事实性全景论文

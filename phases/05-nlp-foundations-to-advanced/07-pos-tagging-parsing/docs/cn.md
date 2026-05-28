# 词性标注与句法分析

> 语法曾经一度不流行。后来每个 LLM 流水线都需要验证结构化提取，它又回来了。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第1课（文本处理）、第2阶段第14课（朴素贝叶斯）
**预计时间：** 约45分钟

## 问题背景

第1课承诺词形还原需要词性标签。不知道 `running` 是动词，词形还原器就无法将其还原为 `run`；不知道 `better` 是形容词，就无法还原为 `good`。

这个承诺背后藏着一整个子领域。词性标注为每个 token 分配语法类别；句法分析恢复句子的树形结构：哪个词修饰哪个词，哪个动词支配哪些论元。经典 NLP 花了二十年精炼这两者，然后深度学习把它们压缩成预训练 Transformer 上的 token 分类任务，研究界就转向了。

但应用界没有。每个结构化提取流水线底层仍然使用词性标注和依存树。LLM 生成的 JSON 根据语法约束进行验证；问答系统用依存分析分解查询；机器翻译质量评估器检查分析树的对齐。

值得了解。本课介绍标注集、基线方法，以及你该停止自己实现转而调用 spaCy 的时机。

## 核心概念

**词性标注**为每个 token 打上语法类别标签。**宾州树库（PTB）**标注集是英语默认方案，36 个标签，区分让普通读者觉得繁琐的细节：`NN` 单数名词、`NNS` 复数名词、`NNP` 单数专有名词、`VBD` 过去式动词、`VBZ` 第三人称单数现在时动词，等等。**通用依存（UD）**标注集更粗粒度（17 个标签）且语言无关，成为跨语言工作的默认方案。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法分析**产生树形结构。两种主要风格：

- **成分分析**：名词短语、动词短语、介词短语相互嵌套，输出是以非终结符类别（NP、VP、PP）为节点、以词为叶节点的树。
- **依存分析**：每个词有一个它所依存的中心词，用语法关系标记，输出是每条边都是（中心词，从属词，关系）三元组的树。

依存分析在 2010 年代胜出，因为它能干净地泛化到各种语言，尤其是词序自由的语言。

```
running 是 ROOT
cats 是 running 的 nsubj（名词性主语）
were 是 running 的 aux（助动词）
at 是 running 的 prep（介词修饰）
3pm 是 at 的 pobj（介词宾语）
```

## 动手实现

### 第一步：最高频标签基线

最笨但能用的词性标注器：对每个词，预测它在训练集中最常见的标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 语料库上，这个基线能达到约 85% 精度。不够好，但这是任何认真的模型不该低于的下界。

### 第二步：二元组 HMM 标注器

对序列的联合概率建模：

```
P(标签序列, 词序列) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两张表：转移概率（给定前一个标签的当前标签概率）和发射概率（给定标签的词概率）。从计数估计两者，加拉普拉斯平滑，用维特比算法（在标签格上动态规划）解码。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

二元组 HMM 在 Brown 上达到约 93% 精度。从 85% 到 93% 的提升主要来自转移概率——模型学到 `DET NOUN` 常见，`NOUN DET` 罕见。

### 第三步：为什么现代标注器能超越这个

转移 + 发射概率是局部的。它们无法捕捉 `saw` 在"I bought a saw"中是名词，在"I saw the movie"中是动词。带任意特征（后缀、词形、前后词、词本身）的 CRF 能达到约 97%。BiLSTM-CRF 或 Transformer 能达到约 98% 以上。

这个任务的天花板由标注者分歧决定。人类标注者在宾州树库上约 97% 的时候意见一致。超过 98% 的模型可能是在测试集上过拟合了。

### 第四步：依存分析概要

从零实现完整的依存分析超出了本课范围；权威教科书处理见 Jurafsky 和 Martin。需要了解的两个经典家族：

- **基于转移的分析器**（arc-eager、arc-standard）像 shift-reduce 分析器：读取 token，将其压栈，应用创建弧的规约操作。贪心解码速度快，经典实现是 MaltParser，现代神经版本是 Chen 和 Manning 的转移分析器。
- **基于图的分析器**（Eisner 算法、Dozat-Manning 双仿射）对每条可能的中心词-从属词边打分，选择最大生成树，速度较慢但更准确。

大多数应用工作直接调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下往上读 `dep` 列，句子的语法结构就自然浮现。

## 工程应用

每个生产 NLP 库都将词性标注和依存分析作为标准流水线的一部分。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）：快速、准确，与分词 + NER + 词形还原集成。`token.tag_`（PTB）、`token.pos_`（UD）、`token.dep_`（依存关系）。
- **Stanford NLP（stanza）**：斯坦福接替 CoreNLP 的方案，60 多种语言上的最新水平。
- **trankit**：基于 Transformer，UD 精度高。
- **NLTK**：`pos_tag`，可用，速度慢，较旧，适合教学。

### 2026 年仍然重要的场景

- **词形还原**：第1课需要词性才能正确还原词形，始终如此。
- **验证 LLM 输出的结构化提取**：验证生成的句子遵守语法约束（如主谓一致、必要修饰语）。
- **方面级情感**：依存分析告诉你哪个形容词修饰哪个名词。
- **查询理解**："movies directed by Wes Anderson starring Bill Murray"通过分析分解为结构化约束。
- **跨语言迁移**：UD 标签和依存关系是语言中立的，支持对新语言进行零样本结构分析。
- **低计算流水线**：如果无法部署 Transformer，词性 + 依存分析 + 词典能走得出人意料地远。

## 交付物

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 练习

1. **（简单）** 在小型标注语料（如 NLTK 的 Brown 子集）上使用最高频标签基线，测量在保留句子上的精度，验证约 85% 的结果。
2. **（中等）** 训练上面的二元组 HMM，报告逐标签精确率/召回率，找出 HMM 最容易混淆的标签对。
3. **（困难）** 使用 spaCy 的依存分析从 1000 句样本中提取主-谓-宾三元组，在 50 个手动标注三元组上评估，记录提取失败的位置（通常是被动语态、并列结构和省略主语）。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 词性标签 (POS tag) | "词的类型" | 语法类别，PTB 有 36 个，UD 有 17 个 |
| 宾州树库 (Penn Treebank) | "标准标注集" | 英语专用，区分动词时态和名词单复数的细粒度方案 |
| 通用依存 (Universal Dependencies) | "多语言标注集" | 比 PTB 更粗粒度，语言中立，跨语言工作的默认方案 |
| 依存分析 (Dependency parse) | "句子树" | 每个词有一个中心词，每条边有一个语法关系 |
| 维特比 (Viterbi) | "动态规划" | 给定发射和转移概率，找到最高概率的标签序列 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — 词性标注和分析的权威教科书处理
- [Universal Dependencies project](https://universaldependencies.org/) — 每个多语言分析器使用的跨语言标注集和树库集合
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token` 上每个属性的实用参考
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 将神经分析器带入主流的论文

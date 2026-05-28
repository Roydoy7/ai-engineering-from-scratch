# 共指消解

> "她给他打电话。他没接。医生在吃午饭。"三个指代，两个人，没有名字。共指消解就是搞清楚"谁是谁"。

**类型：** 学习
**语言：** Python
**前置知识：** 第5阶段第6课（命名实体识别）、第5阶段第7课（词性标注与句法分析）
**预计时间：** 约60分钟

## 问题背景

从一篇 300 字的文章中提取所有对苹果公司的提及。说"Apple"时很简单，说"这家公司""他们""库比蒂诺的科技巨头""乔布斯的公司"时就难了。不把这些提及解析到同一实体，NER 流水线会漏掉 60-80% 的提及。

共指消解把所有指向同一真实世界实体的表达链接到一个簇中。它是表层 NLP（NER、句法分析）和下游语义任务（信息抽取、问答、摘要、知识图谱）之间的粘合剂。

2026 年为什么重要：

- **摘要**："CEO 宣布……"和"Tim Cook 宣布……"——摘要应该用有名字的版本。
- **问答**："她打给谁了？"需要先解析"她"。
- **信息抽取**：如果知识图谱把"PER1 创立了苹果"和"乔布斯创立了苹果"当成两条独立记录，就是错的。
- **跨文档信息抽取**：跨多篇文章合并关于同一事件的提及，是跨文档共指消解。

## 核心概念

**任务定义**：输入一篇文档，输出一组提及（文本跨度）的簇，每个簇指向一个实体。

**提及类型**：

- **命名实体**："Tim Cook"
- **名词短语**："CEO"、"这家公司"
- **代词**："他"、"她"、"他们"、"它"
- **同位语**："Tim Cook，苹果公司 CEO，"

**主要架构**：

1. **基于规则（Hobbs，1978）**：利用语法规则基于句法树解析代词。是个好基线，在代词上出奇地难以超越。
2. **提及对分类器**：对每对提及 (m_i, m_j) 预测是否共指，再通过传递闭包聚类。2016 年前的标准方法。
3. **提及排名**：对每个提及，对候选先行词（包括"无先行词"）排名，取最高分。
4. **基于跨度的端到端（Lee et al., 2017）**：Transformer 编码器枚举所有候选跨度，预测提及得分，再对每个跨度预测先行词概率，贪心聚类。现代默认方法。
5. **生成式（2024+）**：提示 LLM："列出文本中所有代词及其先行词。"在简单情况下效果好，长文档和罕见指代物上表现差。

**评估指标**：五个标准指标（MUC、B³、CEAF、BLANC、LEA）——因为没有任何单一指标能完整衡量聚类质量。通常报告前三者的平均值，称为 CoNLL F1。2026 年 CoNLL-2012 上的最新水平：约 83 F1。

**已知困难情况**：

- 指向数页前引入实体的限定描述。
- 桥接回指（"车轮"→ 前面提到的那辆车）。
- 中文、日语等语言中的零回指。
- 前向指代（代词出现在先行词之前）："**她**走进来时，Mary 笑了。"

## 动手实现

### 第一步：预训练神经共指（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在较长文档上的典型输出：
- 簇 1: [Apple, The company, they]
- 簇 2: [new products]

### 第二步：基于规则的代词解析器（教学示例）

见 `code/main.py`（纯标准库实现）：

1. 提取提及：命名实体（大写跨度）、代词（词典查找）、限定描述（"the X"）。
2. 对每个代词，查看前 K 个提及，按以下标准打分：
   - 性别/数一致性（启发式）
   - 近邻优先（更近的得分更高）
   - 句法角色（主语优先）
3. 链接得分最高的先行词。

无法与神经模型竞争，但它展示了搜索空间和端到端模型必须做出的决策。

### 第三步：用 LLM 做共指消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

注意两种失败模式：第一，LLM 过度合并（把指向两个不同人的"他"和"她"合并到一起）；第二，LLM 在长文档中静默漏掉提及。始终用跨度偏移检查来验证。

### 第四步：评估

标准 CoNLL-2012 脚本计算 MUC、B³、CEAF-φ4，并报告平均值。对于内部评估，先从保留测试集上的跨度级精确率和召回率开始，再加上提及链接 F1。

## 陷阱

- **单例爆炸**。部分系统把每个提及都报告为独立的单元素簇。B³ 对这种情况较为宽容，MUC 会惩罚它。始终同时检查三个指标。
- **长上下文中的代词**。文档超过 2000 token 时，性能下降约 15 F1。小心分块。
- **性别假设**。硬编码的性别规则在非二元指代物、机构、动物上会失败。使用学习模型或中性打分策略。
- **LLM 在长文档上漂移**。单次 API 调用无法可靠地跨 50+ 段落做提及聚类。使用滑动窗口 + 合并方案。

## 工程应用

2026 年技术栈：

| 情况 | 选择 |
|------|------|
| 英语，单文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP 神经共指 |
| 多语言 | 在 OntoNotes 或多语言 CoNLL 上训练的 SpanBERT / XLM-R |
| 跨文档事件共指 | 专门的端到端模型（2025-26 SOTA） |
| 快速 LLM 基线 | GPT-4o / Claude + 结构化输出共指提示词 |
| 生产对话系统 | 基于规则的兜底 + 神经主力 + 关键槽位人工审核 |

2026 年常见的集成模式：先运行 NER，再运行共指消解，再把共指簇合并进 NER 实体。下游任务看到的是每个簇一个实体，而不是每个提及一个实体。

## 交付物

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 练习

1. **（简单）** 在 5 段手工编写的段落上运行 `code/main.py` 中的基于规则的解析器，对比地面真值，测量提及链接准确率。
2. **（中等）** 用预训练神经共指模型处理一篇新闻文章，把模型输出的簇和你的手工标注对比，找出失败案例。
3. **（困难）** 构建共指增强的 NER 流水线：先跑 NER，再通过共指簇合并，在 100 篇文章上测量实体覆盖率相对于纯 NER 的提升。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 提及 (Mention) | "一个引用" | 指向某实体的文本跨度（专名、代词、名词短语） |
| 先行词 (Antecedent) | "它指的是什么" | 后续提及所共指的更早提及 |
| 簇 (Cluster) | "实体的所有提及" | 所有指向同一真实世界实体的提及的集合 |
| 回指 (Anaphora) | "向前查找" | 后续提及指向更早的（"他"→"John"） |
| 前向指代 (Cataphora) | "向后查找" | 更早的提及指向后来的（"当他到达时，John……"） |
| 桥接 (Bridging) | "隐式引用" | "我买了辆车，车轮坏了。"（那辆车的车轮） |
| CoNLL F1 | "排行榜上的分数" | MUC、B³、CEAF-φ4 F1 分的平均值 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 第26章 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 权威教科书章节
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — 基于跨度的端到端方法
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 改善共指消解的预训练方法
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — 标准基准测试
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 基于规则的经典论文

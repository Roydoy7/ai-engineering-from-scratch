# 关系抽取与知识图谱构建

> NER 找到了实体，实体链接锚定了它们，关系抽取找到实体之间的边。知识图谱就是节点、边及其来源的总和。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第6课（命名实体识别）、第5阶段第25课（实体链接）
**预计时间：** 约60分钟

## 问题背景

一位分析师读到："Tim Cook became CEO of Apple in 2011."（Tim Cook 于 2011 年成为苹果公司 CEO。）这句话包含四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取（Relation Extraction，RE）把自由文本转化为结构化三元组 `(主体, 关系, 客体)`。在语料库上聚合三元组就得到知识图谱。有了知识图谱，就有了 RAG、分析或合规审计的推理基础。

2026 年的问题：LLM 抽取关系热情高涨，但也热情过头了——它们会幻觉出原文根本没有支持的三元组。没有来源信息，你根本分不清真实三元组和听起来很合理的虚构内容。2026 年的答案是 AEVS 风格的"锚定-抽取-验证-补充"流水线。

## 核心概念

**三元组形式**：`(主体实体, 关系类型, 客体实体)`。关系可以来自封闭本体（Wikidata 属性、FIBO、UMLS），也可以是开放集合（OpenIE 风格，什么都行）。

**三种抽取方法**：

1. **规则/模式方法**：Hearst 模式："X such as Y" → `(Y, isA, X)`，加上手工编写的正则表达式。脆而精确，可解释。
2. **有监督分类器**：给定句子中两个实体提及，从固定集合中预测关系类型。在 TACRED、ACE、KBP 上训练。2015-2022 年的标准方法。
3. **生成式 LLM**：提示模型输出三元组。开箱即用，但需要来源验证，否则会产生看起来合理的垃圾。

**AEVS（锚定-抽取-验证-补充，2026）**：当前的幻觉缓解框架：

- **锚定（Anchor）**：识别每个实体跨度和关系短语跨度，记录精确的字符位置。
- **抽取（Extract）**：生成与锚定跨度关联的三元组。
- **验证（Verify）**：把每个三元组的元素逐一匹配回源文本，拒绝任何不被支持的三元组。
- **补充（Supplement）**：一次覆盖检查，确保没有锚定跨度被遗漏。

幻觉率大幅下降。需要更多计算，但结果可审计。

**开放 vs 封闭的权衡**：

- **封闭本体**：固定属性列表（如 Wikidata 的 11000+ 个属性）。可预测、可查询、难以捏造。
- **开放 IE**：任何动词短语都成为一种关系。召回率高，精确率低，难以查询。

生产知识图谱通常混合使用：用开放 IE 做探索性发现，然后在合并进主图之前把关系规范化到封闭本体。

## 动手实现

### 第一步：基于模式的抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

完整的玩具抽取器见 `code/main.py`。Hearst 模式在领域特定流水线中仍然在用，因为它们是可调试的。

### 第二步：有监督关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一个 seq2seq 关系抽取器：输入文本，输出三元组，使用 Wikidata 属性 ID。在远程监督数据上微调，是标准的开放权重基线。

### 第三步：带锚定的 LLM 提示抽取

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

验证每个返回的跨度是否与源文本匹配，拒绝任何 `text[start:end] != triple_entity` 的三元组。这就是 AEVS"验证"步骤的最简形式。

### 第四步：规范化到封闭本体

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by"（主客体互换）
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # 丢弃未映射的开放关系，或路由到人工审核
```

规范化通常占整个工程工作量的 60-80%，要为此预留时间。

### 第五步：构建小型图并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每个 RAG-over-KG 系统的原子单元。用 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强图存储来扩展规模。

## 陷阱

- **共指消解先于 RE**。"He founded Apple"——RE 需要知道"he"是谁。先运行共指消解（第24课）。
- **实体规范化**。"Apple Inc"和"Apple"必须解析到同一节点。先做实体链接（第25课）。
- **幻觉三元组**。LLM 会输出文本根本不支持的三元组，必须执行跨度验证。
- **关系规范化漂移**。开放 IE 关系不一致（"was born in"、"came from"、"is a native of"），如果不折叠到规范 ID，知识图谱就无法查询。
- **时态错误**。"Tim Cook is CEO of Apple"——现在正确，2005 年不正确。许多关系是有时间限制的，需要用限定符（Wikidata 中的 `P580` 开始时间、`P582` 结束时间）。
- **领域不匹配**。REBEL 在维基百科上训练，法律、医学和科学文本通常需要领域微调的 RE 模型。

## 工程应用

2026 年技术栈：

| 情况 | 选择 |
|------|------|
| 快速生产，通用领域 | REBEL 或 LlamaPred + Wikidata 规范化 |
| 领域特定（生医、法律） | SciREX 风格领域微调 + 自定义本体 |
| LLM 提示，可审计输出 | AEVS 流水线：锚定→抽取→验证→补充 |
| 大量新闻 IE | 模式方法 + 有监督混合 |
| 从零构建知识图谱 | 开放 IE + 人工规范化 |
| 时态知识图谱 | 带限定符抽取（开始/结束时间、时间点） |

集成模式：NER → 共指消解 → 实体链接 → 关系抽取 → 本体映射 → 图加载。每个阶段都是潜在的质量关卡。

## 交付物

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 练习

1. **（简单）** 在 5 个新闻句子上运行 `code/main.py` 中的模式抽取器，手工检查精确率。
2. **（中等）** 对相同句子使用 REBEL（或小型 LLM），比较两者抽取的三元组，哪个精确率更高？哪个召回率更高？
3. **（困难）** 构建 AEVS 流水线：用 LLM 抽取 + 验证跨度是否与源文本匹配，在 50 个维基百科风格的句子上测量验证步骤前后的幻觉率。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 三元组 (Triple) | "主谓宾" | `(s, r, o)` 元组，是知识图谱的原子单元 |
| 开放 IE (Open IE) | "抽取任何关系" | 开放词汇的关系短语，召回率高，精确率低 |
| 封闭本体 (Closed ontology) | "固定 schema" | 有限的关系类型集合（Wikidata、UMLS、FIBO） |
| 规范化 (Canonicalization) | "全部归一化" | 把表层名称/关系映射到规范 ID |
| AEVS | "有来源的抽取" | 锚定-抽取-验证-补充流水线（2026 年） |
| 来源 (Provenance) | "真相来源链接" | 每个三元组携带文档 ID + 字符跨度指向其来源 |
| 远程监督 (Distant supervision) | "廉价标注" | 把文本与现有知识图谱对齐，生成训练数据 |

## 延伸阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — 远程监督论文
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE 主力工具
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 联合信息抽取
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 年幻觉缓解设计
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 规范图查询

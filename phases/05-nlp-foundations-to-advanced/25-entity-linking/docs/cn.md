# 实体链接与消歧

> NER 找到了"Paris"。实体链接要判断：是法国巴黎？是 Paris Hilton？是德克萨斯州的 Paris？还是特洛伊王子帕里斯？不链接就是歧义，知识图谱就乱了。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第6课（命名实体识别）、第5阶段第24课（共指消解）
**预计时间：** 约60分钟

## 问题背景

有一句话："Jordan beat the press."（乔丹击败了媒体/报刊。）你的 NER 把"Jordan"标为 PERSON，很好。但是哪个 Jordan？

- Michael Jordan（篮球运动员）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（伯克利 ML 教授——这个混淆在 ML 论文里确实发生过）？
- Jordan（约旦这个国家）？
- Jordan（希伯来语名字）？

实体链接（Entity Linking，EL）把每个提及解析到知识库中的唯一条目：Wikidata、Wikipedia、DBpedia，或你自己的领域知识库。这个任务分两步：

1. **候选生成（Candidate Generation）**：给定"Jordan"，哪些知识库条目是合理的候选？
2. **消歧（Disambiguation）**：结合上下文，哪个候选是正确答案？

两步都可以通过机器学习来做，都有标准基准测试。整体流水线框架十年来基本稳定——改变的只是消歧器的质量。

## 核心概念

**候选生成**：给定提及的表层形式（"Jordan"），在别名索引中查找候选。维基百科别名词典覆盖了大多数命名实体："JFK"→ 约翰·F·肯尼迪、杰奎琳·肯尼迪、JFK 机场、电影 JFK。典型的索引每个提及返回 10-30 个候选。

**消歧：三种方法**：

1. **先验 + 上下文（Milne & Witten，2008）**：`P(实体|提及) × 上下文相似度(实体, 文本)`。效果好、速度快、无需训练。
2. **基于嵌入（ESS / REL / BLINK）**：对提及+上下文编码，对每个候选描述编码，取最大余弦值。2020-2024 年的默认方法。
3. **生成式（GENRE，2021；LLM 版本，2023+）**：逐 token 解码实体的规范名称，并约束到有效实体名称的字典树（trie），确保输出一定是有效的知识库 ID。

**端到端 vs 流水线**：现代模型（ELQ、BLINK、ExtEnD、GENRE）在一次推理中完成 NER + 候选生成 + 消歧。但生产环境中流水线系统仍然占主流，因为可以独立替换每个组件。

### 两个关键测量指标

- **提及召回率（候选生成）**：在多少金标准提及中，正确的知识库条目出现在候选列表里。这是整个流水线性能的下界。
- **消歧准确率 / F1**：在候选列表正确的前提下，top-1 预测有多少次是对的。

始终同时报告两个指标。一个在 80% 候选召回上消歧准确率 99% 的系统，整体流水线准确率只有 80%。

## 动手实现

### 第一步：从维基百科重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

维基百科别名数据：约 1800 万（别名，实体）对。从 Wikidata 数据快照下载，以倒排索引形式存储。

### 第二步：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard 重叠只是演示用的。生产中用嵌入向量的余弦相似度替换（见 `code/main.py` 第二步中的 Transformer 版本）。

### 第三步：基于嵌入（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

索引阶段：把知识库中每个实体嵌入一次。查询阶段：把提及+上下文嵌入一次，与候选池做点积，取最大值。

### 第四步：生成式实体链接（概念）

GENRE 逐字符解码实体的维基百科标题。约束解码（见第20课）保证只有有效标题才能输出，与知识库支持的字典树紧密集成。现代的后继是 REL-GEN 和带结构化输出的 LLM 提示 EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合白名单（Outlines `choice` 约束），这是 2026 年最简单的可发布 EL 流水线。

### 第五步：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准 EL 基准：1393 篇路透社文章，34k 个提及，使用维基百科实体。报告库内准确率（`P@1`）和库外 NIL 检测率。

## 陷阱

- **NIL 处理**。部分提及在知识库中没有对应条目（新兴实体、无名人士）。系统必须预测 NIL，而不是猜一个错误实体。NIL 单独评估。
- **提及边界错误**。上游 NER 漏掉部分跨度（"美国银行"被标注为只有"银行"）。实体链接召回率因此下降。
- **热门度偏差**。训练好的系统倾向于过度预测高频实体。一篇 ML 论文中提到"Michael I. Jordan"，往往会被错误地链接到篮球运动员。
- **跨语言实体链接**。把中文文本中的提及链接到英文维基百科实体，需要多语言编码器或翻译步骤。
- **知识库过期**。新公司、新事件、新人物不在去年的维基百科快照里。生产流水线需要定期刷新机制。

## 工程应用

2026 年技术栈：

| 情况 | 选择 |
|------|------|
| 通用英语 + 维基百科 | BLINK 或 REL |
| 跨语言，知识库为维基百科 | mGENRE |
| LLM 友好，每天提及量少 | 提示 Claude/GPT-4 + 候选列表 + 约束 JSON |
| 领域特定知识库（医疗、法律） | 定制 BERT + 知识库感知检索 + 在领域 AIDA 风格数据上微调 |
| 极低延迟 | 仅用出现次数先验（Milne-Witten 基线） |
| 研究 SOTA | GENRE / ExtEnD / 生成式 LLM-EL |

2026 年常见的生产模式：NER → 共指消解 → 对每个提及做 EL → 把共指簇合并为每个簇一个规范实体。输出是文档中每个实体一个知识库 ID，而不是每个提及一个。

## 交付物

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 练习

1. **（简单）** 在 `code/main.py` 中对 10 个有歧义的提及（Paris、Jordan、Apple）实现先验+上下文消歧器，手工标注正确实体，测量准确率。
2. **（中等）** 用句子 Transformer 对 50 个有歧义的提及编码，同时对每个候选的描述编码，对比基于嵌入的消歧和 Jaccard 上下文重叠的准确率。
3. **（困难）** 构建一个 1000 实体的领域知识库（如你公司的员工和产品），端到端实现 NER + EL，在 100 个保留句上测量精确率和召回率。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 实体链接 (Entity Linking, EL) | "链接到维基百科" | 把提及映射到知识库中的唯一条目 |
| 候选生成 (Candidate generation) | "它可能是谁？" | 为一个提及返回一组合理的知识库候选 |
| 消歧 (Disambiguation) | "选出正确的那个" | 利用上下文对候选打分，选出赢家 |
| 别名索引 (Alias index) | "查找表" | 从表层形式到候选实体的映射 |
| NIL | "不在知识库里" | 明确预测没有任何知识库条目与提及匹配 |
| 知识库 (KB) | "知识库" | Wikidata、Wikipedia、DBpedia，或你的领域知识库 |
| AIDA-CoNLL | "那个基准" | 1393 篇路透社文章，附有金标准实体链接 |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 先验+上下文方法的奠基论文
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 基于嵌入的主流方法
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 带约束解码的生成式 EL
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — 基准测试论文
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 开放生产级技术栈

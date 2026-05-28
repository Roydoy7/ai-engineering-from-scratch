# 多语言 NLP

> 一个模型，100+ 种语言，其中大多数语言零训练数据。跨语言迁移是 2020 年代的实用奇迹。

**类型：** 学习
**语言：** Python
**前置知识：** 第5阶段第4课（GloVe、FastText、子词）、第5阶段第11课（机器翻译）
**预计时间：** 约45分钟

## 问题背景

英语有数十亿条标注样本，乌尔都语有几千条，迈蒂利语几乎没有。任何服务全球用户的实用 NLP 系统都必须在任务特定训练数据不存在的长尾语言上工作。

多语言模型通过同时在多种语言上训练一个模型来解决这个问题。共享表示让模型能把在高资源语言中学到的技能迁移到低资源语言。在英语情感分析上微调这个模型，它就能开箱即用地产出令人惊讶的良好乌尔都语情感预测。这就是零样本跨语言迁移，它重塑了 NLP 走向世界的方式。

本课梳理权衡、经典模型，以及一个让初涉多语言工作的团队最容易踩的坑：选择用于迁移的源语言。

## 核心概念

**共享词汇表**：多语言模型使用在所有目标语言的文本上训练的 SentencePiece 或 WordPiece 分词器。词汇表是共享的：同一个子词单元在相关语言中表示同一个词素。英语和意大利语中的"anti-"得到相同的 token。

**共享表示**：在多种语言上做掩码语言建模预训练的 Transformer 学会了：不同语言中语义相似的句子产生相似的隐藏状态。mBERT、XLM-R 和 NLLB 都表现出这种特性。英语"cat"的嵌入与法语"chat"和西班牙语"gato"的嵌入聚集在一起，完整句子的嵌入也是如此。

**零样本迁移**：在一种语言（通常是英语）的标注数据上微调模型，推理时在模型支持的任何其他语言上运行，无需目标语言标签。对类型学上相近的语言效果强，对距离较远的语言效果较弱。

**少样本微调**：在目标语言中添加 100-500 个标注样本，分类任务的准确率会跳升到英语基线的 95-98%。这是多语言 NLP 中性价比最高的单一杠杆。

## 主要模型

| 模型 | 年份 | 覆盖 | 备注 |
|------|------|------|------|
| mBERT | 2018 | 104 种语言 | 在维基百科上训练，第一个实用的多语言 LM，低资源语言较弱 |
| XLM-R | 2019 | 100 种语言 | 在 CommonCrawl（远大于维基百科）上训练，设定跨语言基线，基础版 270M，大版 550M |
| XLM-V | 2023 | 100 种语言 | XLM-R 配 100 万 token 词汇表（而非 25 万），低资源语言更好 |
| mT5 | 2020 | 101 种语言 | 用于多语言生成的 T5 架构 |
| NLLB-200 | 2022 | 200 种语言 | Meta 的翻译模型，包含 55 种低资源语言 |
| BLOOM | 2022 | 46 种语言 + 13 种编程语言 | 多语言训练的开放 176B LLM |
| Aya-23 | 2024 | 23 种语言 | Cohere 的多语言 LLM，阿拉伯语、印地语、斯瓦希里语表现强 |

按使用场景选取：分类任务默认用 XLM-R-base，生成任务根据翻译还是开放生成选 mT5 或 NLLB，LLM 风格工作配 Aya-23 或 Claude 加明确的多语言提示词。

## 源语言决策（2026 年研究进展）

大多数团队默认用英语作为微调源语言，最新研究（2026 年）表明这往往是错的。

语言相似度比原始语料库大小更能预测迁移质量。对斯拉夫语目标，德语或俄语往往优于英语；对印度次大陆语目标，印地语往往优于英语。**qWALS** 相似度指标（2026 年，基于世界语言结构图谱特征）将此量化；**LANGRANK**（Lin et al., ACL 2019）是更早的方法，从语言相似度、语料库大小和语系关联的组合中排序候选源语言。

实践规则：如果你的目标语言有类型学上接近的高资源亲属，先在那上面微调，再与英语微调对比。

## 动手实现

### 第一步：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，相同的 API。在 NLI 数据上训练的 XLM-R 通过蕴含技巧很好地迁移到分类任务。

### 第二步：多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译结果在嵌入空间中落在很近的位置，不同的英语句子落得较远。这正是跨语言检索、聚类和相似度任务能工作的原因。

### 第三步：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100-500 个目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全的默认值。更高的学习率会导致多语言对齐崩溃，你会得到一个只会英语的模型。

## 真正有效的评估

- **在保留集上按语言分别报告准确率**，不要聚合，聚合会隐藏长尾。
- **与单语言基线对比**：对于数据足够的语言，从头训练的单语言模型有时会胜过多语言模型，要测试。
- **实体级测试**：目标语言中的命名实体，多语言模型对远离拉丁字母的字符集往往分词很弱。
- **跨语言一致性**：相同含义的两种语言应产生相同预测，测量差距。

## 工程应用

2026 年技术栈：

| 任务 | 推荐方案 |
|------|---------|
| 分类，100 种语言 | XLM-R-base（约 270M）微调后 |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200 种语言 | `facebook/nllb-200-distilled-600M`（见第11课） |
| 多语言生成 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言 NLP | XLM-V 或在相近高资源语言上做领域特定微调 |

如果性能重要，始终预留目标语言微调的预算。零样本是起点，不是终点。

### 分词税（低资源语言会出什么问题）

多语言模型在所有语言间共享一个分词器。该词汇表在以英语、法语、西班牙语、中文、德语为主的语料上训练。对于主导集之外的任何语言，三种税会悄悄叠加：

- **生育率税**：低资源语言文本每个词被分成的 token 数量远多于英语。一个印地语句子可能需要英语等效句子 3-5 倍的 token，这 3-5 倍吃掉你的上下文窗口、训练效率和延迟。
- **变体恢复税**：每一个拼写错误、变音符号变体、Unicode 归一化不匹配或大小写变化，都成为嵌入空间中冷启动的无关序列，模型无法学习母语使用者视为显而易见的正字法对应关系。
- **容量溢出税**：前两种税消耗上下文位置、层深度和嵌入维度，实际用于推理的空间系统性地比同一模型给高资源语言的要少。

实践症状：你的模型在印地语上训练正常，损失曲线看起来对，评估困惑度看起来合理，而生产输出却微妙地出错——形态在句子中间崩溃，罕见的屈折形式无法恢复。**你无法靠增加数据量来逃脱一个坏分词器。**

缓解措施：选择对目标语言有良好覆盖的分词器（XLM-V 的 100 万 token 词汇表是直接解决方案）；在训练前在保留目标文本上验证分词生育率；对真正长尾的书写系统使用字节级兜底（SentencePiece `byte_fallback=True`，或 GPT-2 风格的字节级 BPE），确保不会有 OOV。

## 交付物

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 练习

1. **（简单）** 用零样本分类流水线在英语、法语、印地语、阿拉伯语各运行 10 个句子，报告每种语言的准确率，法语应该强，印地语还行，阿拉伯语会参差不齐。
2. **（中等）** 用 `paraphrase-multilingual-MiniLM-L12-v2` 在小型混合语言语料上构建跨语言检索器：用英语查询，检索任意语言的文档，测量 Recall@5。
3. **（困难）** 对印地语分类任务比较英语源和印地语源微调，两种方式各用 500 个目标语言样本做少样本微调，报告哪个源产生更好的印地语准确率及差距有多大——这是 LANGRANK 论点的微缩版。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 多语言模型 (Multilingual model) | "一个模型多种语言" | 跨语言共享词汇表和参数 |
| 跨语言迁移 (Cross-lingual transfer) | "在一种语言训练，在另一种语言运行" | 在源语言微调，在无目标语言标签的情况下评估目标语言 |
| 零样本 (Zero-shot) | "无目标语言标签" | 不在目标语言上微调就做迁移 |
| 少样本 (Few-shot) | "少量目标语言标签" | 使用 100-500 个目标语言样本进行微调 |
| mBERT | "第一个多语言 LM" | 104 种语言的 BERT，在维基百科上预训练 |
| XLM-R | "标准跨语言基线" | 100 种语言的 RoBERTa，在 CommonCrawl 上预训练 |
| NLLB | "Meta 的 200 语言 MT" | 不落下任何语言，包含 55 种低资源语言 |

## 延伸阅读

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R 论文
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 开创跨语言迁移研究方向的分析论文
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 论文
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Cohere 的多语言 LLM Aya
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK 源语言论文

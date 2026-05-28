# 命名实体识别

> 把名字提取出来。听起来简单，直到你遇到模糊边界、嵌套实体和领域术语。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第2课（BoW + TF-IDF）、第5阶段第3课（词嵌入）
**预计时间：** 约75分钟

## 问题背景

"Apple sued Google over its iPhone search deal in the US."（苹果因 iPhone 搜索协议在美国起诉谷歌。）五个实体：Apple（机构）、Google（机构）、iPhone（产品）、search deal（也许算）、US（地理位置）。好的 NER 系统能提取所有这些并附上正确类型；差的系统会遗漏 iPhone，把水果苹果和苹果公司混淆，把"US"标注为人名。

NER 是每个结构化提取流水线底层的主力工具。简历解析、合规日志扫描、医疗记录匿名化、搜索查询理解、聊天机器人响应的接地、法律合同提取——你永远看不见它，却总是依赖它。

本课沿着经典路径（基于规则、HMM、CRF）走向现代方案（BiLSTM-CRF，然后是 Transformer）。每一步都解决前一步的特定局限，这个模式本身就是教训。

## 核心概念

**BIO 标注**（或 BILOU）将实体提取转化为序列标注问题。将每个 token 标注为 `B-类型`（实体开始）、`I-类型`（实体内部）或 `O`（实体外）。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多 token 实体链式连接：`New B-GPE`、`York I-GPE`、`City I-GPE`。理解 BIO 的模型可以提取任意长度的文本片段。

架构演进：

- **基于规则**：正则表达式 + 词典查找。对已知实体精确率高，对新实体零覆盖。
- **HMM**：隐马尔可夫模型。给定标签的 token 发射概率，加上标签间的转移概率，Viterbi 解码，在标注数据上训练。
- **CRF**：条件随机场。类似 HMM 但是判别式的，因此可以混合任意特征（词形、大小写、邻近词）。2026 年低资源部署中仍是经典生产主力。
- **BiLSTM-CRF**：用神经特征替代手工特征。LSTM 双向读取句子，CRF 层在上面强制一致的标签序列。
- **基于 Transformer**：微调带 token 分类头的 BERT，精度最好，计算最多。

## 动手实现

### 第一步：BIO 标注辅助函数

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 第二步：手工特征

对于经典（非神经）NER，特征是核心。常用的有：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`，`word_shape("USA-2024")` 返回 `XXX-dddd`。大写模式对专有名词信号很强。

### 第三步：简单的基于规则 + 词典基线

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产词典有数百万条目，从 Wikipedia 和 DBpedia 爬取。覆盖率不错，消歧（苹果公司 vs 水果苹果）很糟糕——这就是统计模型获胜的原因。

### 第四步：CRF 步骤（概要，非完整实现）

50 行从零实现 CRF 在没有概率论基础的情况下不具启发性，改用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 是 L1 和 L2 正则化。`all_possible_transitions=True` 让模型学到非法序列（如 `O` 后跟 `I-ORG`）的概率很低，这就是 CRF 在不需要你写约束的情况下强制 BIO 一致性的方式。

### 第五步：BiLSTM-CRF 的增益

特征变为可学习的。输入：token 嵌入（GloVe 或 fastText），LSTM 双向读取句子，拼接的隐藏状态通过 CRF 输出层。CRF 仍然强制标签序列一致性；LSTM 用学到的特征替代手工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRF 层使用 `torchcrf.CRF`（pip install pytorch-crf）。相比手工 CRF 的提升是可测量的，但除非你有数万条标注句子，提升幅度比预期要小。

## 工程应用

spaCy 开箱即提供生产级 NER：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` 被标注为 `ORG` 而非 `PRODUCT`——spaCy 小模型的产品实体覆盖较弱。大模型（`en_core_web_lg`）更好，Transformer 模型（`en_core_web_trf`）更好。

Hugging Face 的基于 BERT 的 NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 将连续的 B-X、I-X token 合并成一个片段。不设置则得到 token 级标签，需要自己合并。

### 基于 LLM 的 NER（2026 年选项）

零样本和少样本 LLM NER 现在在许多领域与微调模型竞争，在标注数据稀少时显著更好。

- **零样本提示**：给 LLM 一个实体类型列表和示例 schema，要求 JSON 输出，开箱即用，在新领域上精度适中。
- **ZeroTuneBio 风格提示**：将任务分解为候选提取 → 含义解释 → 判断 → 重新验证的多阶段提示（非一次性），在生物医学 NER 上大幅提升精度，同样的模式适用于法律、金融、科学领域。
- **RAG 动态提示**：从小型标注种子集中检索最相似的标注样本，每次推理动态构建少样本提示。2026 年基准测试中，这让 GPT-4 生物医学 NER F1 比静态提示高 11-12%。
- **按实体类型分解**：对于长文档，一次提取所有实体类型的单次调用随长度增加会降低召回率；改为每种实体类型单独跑一遍提取，推理成本更高，精度大幅提升，是临床病历和法律合同的标准模式。

2026 年生产建议：在收集训练数据之前先建立 LLM 零样本基线，通常 F1 已经够好，永远不需要微调。

### 经典 NER 仍然获胜的场景

即使有 LLM 可用，经典 NER 在以下情况更合适：
- 延迟预算低于 50ms
- 有数千个标注样本，需要 98% 以上 F1
- 领域有稳定本体，预训练 CRF 或 BiLSTM 迁移效果好
- 监管约束要求本地部署的非生成式模型

### 失败场景

- **领域迁移**：在 CoNLL 数据上训练的 NER 应用到法律合同上，表现还不如词典，需要在你的领域上微调。
- **嵌套实体**："Bank of America Tower" 同时是机构和建筑。标准 BIO 无法表示重叠片段，需要嵌套 NER（多轮或基于片段的模型）。
- **长实体**："United States Federal Deposit Insurance Corporation"，token 级模型有时会切断，使用 `aggregation_strategy` 或后处理。
- **稀疏类型**：医学 NER 标签如 DRUG_BRAND、ADVERSE_EVENT、DOSE，通用模型无从知晓，Scispacy 和 BioBERT 是起点。

## 交付物

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 练习

1. **（简单）** 实现 `bio_to_spans`（`spans_to_bio` 的逆函数），在 10 句话上验证往返一致性。
2. **（中等）** 在 CoNLL-2003 英语 NER 数据集上训练上面的 sklearn-crfsuite CRF，用 `seqeval` 报告逐实体 F1，典型结果约 84 F1。
3. **（困难）** 在领域特定 NER 数据集（医疗、法律或金融）上微调 `distilbert-base-cased`，与 spaCy 小模型对比，记录数据泄露检查并写下让你惊讶的地方。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| NER | "提取名字" | 将 token 片段标注类型（人名、机构、地点、日期……） |
| BIO | "标注方案" | `B-X` 开始，`I-X` 延续，`O` 表示外部 |
| BILOU | "更好的 BIO" | 添加 `L-X`（最后）、`U-X`（单词）以获得更清晰的边界 |
| CRF | "结构化分类器" | 对标签间转移建模，而非仅对发射概率建模，强制有效序列 |
| 嵌套 NER (Nested NER) | "重叠实体" | 一个片段是另一个片段的不同实体，BIO 无法表示 |
| 实体级 F1 | "正确的 NER 指标" | 预测片段必须与真值片段完全匹配，token 级 F1 会高估精度 |

## 延伸阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 论文，经典必读
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 介绍了成为标准的 token 分类模式
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents` 和 `Span` 上每个属性的实用参考
- [seqeval](https://github.com/chakki-works/seqeval) — 正确的指标库，始终使用它

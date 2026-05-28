# 文本处理 — 分词、词干提取、词形还原

> 语言是连续的，模型是离散的。预处理是连接两者的桥梁。

**类型：** 构建
**语言：** Python
**前置知识：** 第2阶段第14课（朴素贝叶斯）
**预计时间：** 约45分钟

## 问题背景

模型读不了"The cats were running."，它读的是整数。

每个 NLP 系统都从同样三个问题开始：一个词从哪里开始？词的词根是什么？什么时候应该把"run"、"running"、"ran"视为同一件事，什么时候又应该区分它们？

分词出了错，模型就在垃圾上训练。如果你的分词器把 `don't` 当成一个 token，而把 `do n't` 当成两个，训练分布就会分裂。如果你的词干提取器把 `organization` 和 `organ` 归并到同一个词干，主题建模就完蛋了。如果你的词形还原器需要词性上下文但你没有传入，动词就会被当作名词处理。

本课从零开始构建这三个预处理步骤，再演示 NLTK 和 spaCy 如何实现同样的工作，让你看清各自的权衡。

## 核心概念

三个操作，各有职责和失败模式。

**分词（Tokenization）** 将字符串切分成 token。"token"故意模糊，因为合适的粒度取决于任务。经典 NLP 用词级，Transformer 用子词级，没有空格的语言用字符级。

**词干提取（Stemming）** 用规则裁掉后缀。快速、激进、粗糙。`running -> run`，`organization -> organ`。后者就是失败模式。

**词形还原（Lemmatization）** 利用语法知识将词还原为词典形式。速度较慢，结果准确，需要查找表或形态分析器。`ran -> run`（需要知道"ran"是"run"的过去式）；`better -> good`（需要知道比较级形式）。

经验法则：当速度重要、能容忍噪声时用词干提取（搜索索引、粗粒度分类）；当含义重要时用词形还原（问答、语义搜索、任何用户会读到的内容）。

## 动手实现

### 第一步：基于正则的词级分词器

最简单的实用分词器在非字母数字字符处切分，同时保留标点符号作为独立 token。不完美、不最终，但一行就能运行。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

按优先级排列的三个模式：带可选内部撇号的词（`don't`、`it's`）；纯数字；任何单个非空白非字母数字字符作为独立 token（标点符号）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

注意失败模式：`3pm` 被拆成 `['3', 'pm']`，因为字母序列和数字序列交替出现。对大多数任务够用，但 URL、邮箱、话题标签都会出错。生产环境中，在通用模式前加入特定模式。

### 第二步：Porter 词干提取器（仅第 1a 步）

完整的 Porter 算法有五个阶段的规则。仅第 1a 步就覆盖了最常见的英语后缀，足以说明模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

从上往下读规则。`ies -> i` 规则是 `ponies -> poni` 而非 `pony` 的原因。真正的 Porter 算法有第 1b 步来修正这个问题。规则之间存在竞争，前面的规则优先。顺序比任何单条规则都重要。

### 第三步：基于查找表的词形还原器

正式的词形还原需要形态学知识。一个适合教学的版本使用小型词形表和回退机制。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个案例是关键教学时刻。`watched` 不在我们的表中，我们的回退只处理 `ing`。真正的词形还原需要覆盖 `ed`、不规则动词、比较级形容词、发音变化的复数（`children -> child`）。这就是生产系统使用 WordNet、spaCy 形态分析器或完整形态分析工具的原因。

### 第四步：串联成流水线

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的一环是词性标注器。第5阶段第7课（词性标注）会构建一个。现在默认把所有词当 `NOUN` 处理，并承认这一局限。

## 工程应用

NLTK 和 spaCy 提供了生产级版本，各自几行代码搞定。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 能处理缩写、Unicode 和正则会遗漏的边缘情况。`PorterStemmer` 运行全部五个阶段。`WordNetLemmatizer` 需要将词性标签从 NLTK 的宾州树库方案转换为 WordNet 的缩写方案。上面的转换代码是大多数教程跳过的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 将整个流水线隐藏在 `nlp(text)` 背后。分词、词性标注和词形还原全部运行。大规模处理比 NLTK 快，开箱准确度更高。代价是难以替换单个组件。

### 如何选择

| 情景 | 选择 |
|------|------|
| 教学、研究、需要替换组件 | NLTK |
| 生产、多语言、速度优先 | spaCy |
| Transformer 流水线（反正要用模型自带的分词器） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 两个没人警告你的失败模式

大多数教程只讲算法就停了。有两件事会咬到真实的预处理流水线，但几乎从未被提及。

**可复现性漂移。** NLTK 和 spaCy 在版本之间会改变分词和词形还原的行为。spaCy 2.x 中产生 `['do', "n't"]` 的代码，在 3.x 中可能产生 `["don't"]`。你的模型是在一种分布上训练的，推理现在运行在另一种分布上。精度悄悄下降，没人知道为什么。在 `requirements.txt` 中锁定库版本，写一个预处理回归测试，固定 20 个样本句子的预期分词结果，每次升级都运行它。

**训练/推理不匹配。** 训练时用激进预处理（小写、去停用词、词干提取），部署时处理原始用户输入，性能暴跌。这是最常见的生产 NLP 故障。如果训练时做了预处理，推理时必须运行完全相同的函数。将预处理作为函数封装在模型包内，而不是让服务团队在 notebook 单元格里重写一遍。

## 交付物

一个可复用的提示词，帮助工程师在不读三本教科书的情况下选择预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 练习

1. **（简单）** 扩展 `tokenize`，将 URL 保留为单个 token。测试：`tokenize("Visit https://example.com today.")` 应产生一个 URL token。
2. **（中等）** 实现 Porter 第 1b 步：若单词包含元音且以 `ed` 或 `ing` 结尾，则去除后缀，并处理双辅音规则（`hopping -> hop`，而非 `hopp`）。
3. **（困难）** 构建一个使用 WordNet 作为查找表、但当 WordNet 没有词条时回退到 Porter 词干提取器的词形还原器。在一个带标注语料上测量其精度，对比纯 WordNet 和纯 Porter。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| Token（词元） | "一个词" | 模型消费的任意单元，可以是词、子词、字符或字节 |
| Stem（词干） | "词的词根" | 基于规则的后缀裁剪结果，不一定是真实的词 |
| Lemma（词形） | "词典形式" | 你会查词典的那种形式，需要语法上下文才能正确计算 |
| POS tag（词性标签） | "词性" | 名词、动词、形容词等类别，词形还原准确性依赖于它 |
| Morphology（形态学） | "词形规则" | 词如何根据时态、数量、格等变化，词形还原依赖于此 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，迄今最清晰的阐述
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真实流水线的接线方式
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你还没想到的分词边缘情况

# 自然语言推理与文本蕴含

> 给定"猫在垫子上"，那"有动物在表面上"是真的吗？NLI 把这个判断变成一个三类分类问题，推动了零样本分类和 RAG 忠实度检验的工业应用。

**类型：** 学习
**语言：** Python
**前置知识：** 第5阶段第8课（CNN/RNN）、第5阶段第13课（问答系统）
**预计时间：** 约45分钟

## 问题背景

你构建了一个 RAG 系统。它检索到相关文档，生成了一个听起来很合理的答案。但那个答案真的来自文档吗？还是模型在瞎编？你需要一种方法来检验：这段文字是否*蕴含*于那段上下文？

这正是自然语言推理（NLI）解决的问题。给定一个前提和一个假设，NLI 模型判断：假设是从前提中蕴含出来的，还是与前提矛盾的，还是既不蕴含也不矛盾（中性）？

这个看似学术的任务有三个直接的生产应用：零样本文本分类（无需任何标注数据对新类别打分）、RAG 忠实度验证（用作 RAGAS 中忠实度维度的后端）、事实核查（某个说法是否被来源支持？）。

## 核心概念

**三类判断**：

| 类别 | 含义 | 示例 |
|------|------|------|
| 蕴含（Entailment） | 前提为真则假设必为真 | 前提："我的狗在叫。" 假设："有动物在叫。" |
| 矛盾（Contradiction） | 前提为真则假设必为假 | 前提："我的狗在叫。" 假设："这里没有动物。" |
| 中性（Neutral） | 前提无法决定假设的真假 | 前提："我的狗在叫。" 假设："我的狗叫 Buddy。" |

**架构**：BERT 风格的编码器把前提和假设拼接成单个序列：`[CLS] 前提 [SEP] 假设 [SEP]`。`[CLS]` token 的向量接一个线性头，输出三类 softmax。交叉注意力让模型能同时关注前提和假设的所有位置——这比把两段文字分开编码再比较要强很多。

**主要数据集**：

| 数据集 | 规模 | 特点 |
|--------|------|------|
| SNLI | 570k 对 | 众包自图片描述，干净但分布偏窄 |
| MultiNLI | 433k 对 | 跨 10 个体裁（口语、政府文件、小说等），泛化性更好 |
| ANLI | 约 16k 对，分三轮 | 对抗性构造，专门针对当前 SOTA 模型找难例，训练后鲁棒性更强 |

**零样本分类原理**：把标签名称改写成假设模板（"This text is about {label}."），把原始文本当前提，对每个候选标签计算蕴含得分，取蕴含得分最高的标签。整个过程不需要任何针对目标分类任务的标注数据。

## 动手实现

### 第一步：NLI 推理基础

```python
from transformers import pipeline

nli = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    top_k=None,
)

premise = "The new policy reduced costs by 30% while maintaining service quality."
hypothesis = "The policy was cost-effective."

scores = nli(f"{premise}</s></s>{hypothesis}")
print(scores)
# [{'label': 'ENTAILMENT', 'score': 0.94}, {'label': 'NEUTRAL', 'score': 0.05}, ...]
```

注意 `bart-large-mnli` 用 `</s></s>` 作为前提和假设的分隔符（BART 的特殊格式）；DeBERTa 系模型用 `[SEP]`。使用前确认文档。

### 第二步：零样本分类

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The quarterly report shows a 15% increase in subscription revenue."
candidate_labels = ["finance", "sports", "technology", "politics"]

result = classifier(text, candidate_labels)
print(result["labels"][0], result["scores"][0])
# finance  0.91
```

`zero-shot-classification` pipeline 封装了假设模板逻辑（默认模板："This example is {label}."），你也可以通过 `hypothesis_template` 参数自定义。

```python
result = classifier(
    text,
    candidate_labels,
    hypothesis_template="This document discusses {}.",
)
```

模板措辞对得分影响显著——后文陷阱部分会详细说明。

### 第三步：RAG 忠实度检验

这是 2026 年最常见的生产用法——判断生成的答案是否真的被检索到的上下文所支持：

```python
from transformers import pipeline

nli = pipeline("text-classification", model="facebook/bart-large-mnli", top_k=None)


def is_faithful(answer: str, context: str, threshold: float = 0.5) -> bool:
    scores = nli(f"{context}</s></s>{answer}")
    score_map = {s["label"]: s["score"] for s in scores}
    return score_map.get("ENTAILMENT", 0.0) >= threshold


context = "The Eiffel Tower is located in Paris, France, and was completed in 1889."
faithful_answer = "The Eiffel Tower was built in the 19th century."
hallucinated_answer = "The Eiffel Tower is in London."

print(is_faithful(faithful_answer, context))    # True
print(is_faithful(hallucinated_answer, context)) # False
```

RAGAS 的忠实度维度就是对这个模式的系统化版本：把生成的答案拆成若干陈述句，分别用 NLI 检查每条陈述是否被上下文蕴含，取蕴含比例作为忠实度得分。

### 第四步：多条假设批量评分

```python
def score_hypotheses(premise, hypotheses):
    results = {}
    for hyp in hypotheses:
        scores = nli(f"{premise}</s></s>{hyp}")
        score_map = {s["label"]: s["score"] for s in scores}
        results[hyp] = score_map
    return results


premise = "Revenue grew 20% year-over-year to $5.2B, driven by cloud services."
hypotheses = [
    "The company's cloud business is growing.",
    "Revenue declined last year.",
    "Total revenue exceeded $5 billion.",
]
for hyp, scores in score_hypotheses(premise, hypotheses).items():
    print(f"E={scores['ENTAILMENT']:.2f} C={scores['CONTRADICTION']:.2f} | {hyp}")
```

## 陷阱

### 假设捷径（Hypothesis-Only Shortcut）

SNLI/MultiNLI 数据集中存在标注偏差。某些假设单凭措辞就能预测标签，完全不需要看前提。例如，包含"从不（never）"的假设往往是矛盾，包含"某些（some）"的假设往往是蕴含。

实验结果：只给模型假设（不给前提），在 SNLI 上仍能达到约 60% 的准确率。这意味着这些模型没有真正在做推理，而是在利用数据集偏差。ANLI 是为了修复这个问题而设计的。

### 词汇重叠启发式

前提和假设共享的词越多，模型越倾向于预测蕴含——哪怕实际上是矛盾。例如：

- 前提："没有学生通过了考试。"
- 假设："一些学生通过了考试。"

两句话大量共享词汇，但语义是矛盾的。未经 ANLI 训练的模型有时会错判为中性或蕴含。

### 文档长度退化

大多数 NLI 模型在短到中等长度的句子对上训练。当上下文变长（比如一整段文档而非一两句话），性能可能下降超过 20 个 F1 点。

解决方案：把长文档切成句子或段落，分别对每一块做 NLI，取最高蕴含得分；或者使用专门为文档级 NLI 设计的模型（如 `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`）。

### 模板敏感性

零样本分类中，假设模板的措辞变化会导致得分偏移超过 10 个百分点：

```
"This text is about {}."          → accuracy 84%
"This example is {}."             → accuracy 77%
"The topic of this text is {}."   → accuracy 81%
```

建议：在目标领域的小样本上比较几种模板，选最稳定的；或者直接用 SetFit 在少量标注样本上微调，完全绕开模板敏感性问题。

### 领域迁移

在新闻报道上微调的 NLI 模型迁移到法律合同或医学文献时性能会明显下降，因为领域词汇和蕴含模式差异很大。生产部署前始终在目标领域数据上做评估。

## 工程应用

2026 年技术栈：

| 用途 | 推荐模型 |
|------|---------|
| 通用零样本分类 | `facebook/bart-large-mnli` |
| 高精度 NLI / RAG 忠实度 | `microsoft/deberta-v3-large-mnli` |
| 多数据集强鲁棒性 | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 NLI | `joeddav/xlm-roberta-large-xnli` |
| 轻量级推理 | `cross-encoder/nli-MiniLM2-L6-H768` |

选择原则：如果精度优先选 DeBERTa；如果延迟优先或边缘部署选 MiniLM；如果多语言场景选 XLM-R。

## 交付物

保存为 `outputs/skill-nli-faithfulness.md`：

```markdown
---
name: nli-faithfulness-checker
description: Use NLI to verify RAG answer faithfulness and build zero-shot classifiers.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, rag, zero-shot]
---

Given a RAG system or classification task, output:

1. NLI model selection. DeBERTa-v3-large-mnli for accuracy, MiniLM for latency, XLM-R for multilingual.
2. Faithfulness check design. Split generated answers into atomic claims. NLI-check each claim against retrieved context. Report entailment ratio.
3. Zero-shot classification setup. Hypothesis template (test at least 3 variants on held-out set). Candidate label design (use natural language descriptions, not class codes).
4. Evaluation plan. Faithfulness: precision/recall on a set of known-faithful and known-hallucinated answer pairs. Classification: accuracy vs supervised baseline on 100-shot eval set.

Refuse to deploy NLI-based faithfulness checks without validating on domain-specific examples — out-of-domain transfer can drop 20+ F1 points. Flag hypothesis templates that haven't been compared against alternatives on target data.
```

## 练习

1. **（简单）** 对 10 对前提-假设使用 `facebook/bart-large-mnli`，覆盖三类（蕴含、矛盾、中性），确认模型判断是否符合直觉，找出至少一个明显的错误案例。
2. **（中等）** 构建一个零样本新闻分类器（体育/财经/科技/政治），对 50 条真实新闻标题分类，对比三种不同假设模板的准确率，量化模板敏感性。
3. **（困难）** 实现一个 RAG 忠实度流水线：用 Wikipedia 段落检索 10 个问题的上下文，让 GPT/Claude 生成答案，用 NLI 对每条答案评分，手动验证 NLI 得分与实际忠实度的一致性，计算准确率。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 自然语言推理（NLI） | "文本蕴含任务" | 三类分类：前提是否蕴含/矛盾/中性于假设 |
| 蕴含（Entailment） | "前提支持假设" | 前提为真则假设必为真 |
| 矛盾（Contradiction） | "前提否定假设" | 前提为真则假设必为假 |
| 中性（Neutral） | "前提无法决定" | 前提与假设既无蕴含也无矛盾关系 |
| 假设模板 | "把标签改写成句子" | 零样本分类中把类别名称改写为自然语言陈述 |
| 忠实度（Faithfulness） | "答案有来源吗？" | 生成内容是否被检索上下文所蕴含，是 RAGAS 核心指标之一 |
| ANLI | "对抗性 NLI" | 专门针对 SOTA 模型的弱点构造的困难样本，训练后鲁棒性更强 |

## 延伸阅读

- [Bowman et al. (2015). A Large Annotated Corpus for Learning Natural Language Inference](https://arxiv.org/abs/1508.05326) — SNLI 论文
- [Williams, Nangia, Bowman (2018). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI 论文
- [Nie et al. (2020). Adversarial NLI: A New Benchmark for Natural Language Understanding](https://arxiv.org/abs/1910.14599) — ANLI 论文
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — 把 NLI 用于零样本分类的开创性论文
- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — NLI 忠实度在 RAG 评估中的系统化应用

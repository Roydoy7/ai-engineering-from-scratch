# LLM 评估——RAGAS、DeepEval、G-Eval

> 精确匹配和 F1 会错过语义等价。人工评审无法扩展。LLM 作为裁判是生产的答案——只要你做好足够的校准让这个数字可信。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第13课（问答系统）、第5阶段第14课（信息检索）
**预计时间：** 约75分钟

## 问题背景

你的 RAG 系统回答："June 29th, 2007."  
金标准参考答案是："June 29, 2007."  
精确匹配打 0 分。F1 打约 75%。人类会打 100%。

现在乘以 10000 个测试用例，再乘以每次对检索器、分块策略、提示词或模型的改动。你需要一个能理解语义的评估器，它要能廉价地大规模运行，对回归问题诚实，并且能暴露正确的失败模式。

2026 年有三个框架主导这个问题。

- **RAGAS**：检索增强生成评估（Retrieval-Augmented Generation ASsessment）。四个 RAG 指标（忠实度、答案相关性、上下文精确率、上下文召回率），后端使用 NLI + LLM 裁判。有研究支撑，轻量级。
- **DeepEval**：LLM 界的 pytest。包含 G-Eval、任务完成度、幻觉、偏见等指标，天然融入 CI/CD。
- **G-Eval**：一种方法（也是 DeepEval 中的一个指标）：LLM 裁判 + 思维链，自定义评分标准，输出 0-1 分。

三者都依赖 LLM 裁判。本课建立对该方法的直觉，以及让它变得可信所需的信任层。

## 核心概念

**LLM 裁判**：用一个 LLM 替代静态指标，让它根据评分标准对输出打分。给定 `(查询, 上下文, 答案)`，提示一个裁判 LLM："对忠实度打 0-1 分。"返回分数。

为什么有效：LLM 以极低的成本逼近人类判断。GPT-4o-mini 约 0.003 美元/个案例，1000 个样本的回归评估不到 5 美元。

为什么会静默失败：

1. **裁判偏差**：裁判偏好更长的答案，偏好与自己同一模型家族的答案，偏好与提示词风格匹配的答案。
2. **JSON 解析失败**：坏 JSON → NaN 分 → 被静默排除在汇总之外。RAGAS 用户对此深有体会。用 try/except + 明确的失败模式来把守。
3. **版本间漂移**：升级裁判模型会改变所有指标。冻结裁判模型和版本。

**RAG 四维指标**：

| 指标 | 问题 | 后端实现 |
|------|------|---------|
| 忠实度 (Faithfulness) | 答案中的每个声明是否来自检索到的上下文？ | 基于 NLI 的蕴含检验 |
| 答案相关性 (Answer Relevance) | 答案是否回答了问题？ | 从答案生成假设性问题，与真实问题比较 |
| 上下文精确率 (Context Precision) | 检索到的块中有多少比例是相关的？ | LLM 裁判 |
| 上下文召回率 (Context Recall) | 检索是否返回了所有必要内容？ | LLM 裁判，对照金标准答案 |

**G-Eval**：定义自定义标准："答案是否引用了正确的来源？"框架自动展开成思维链评估步骤，然后打 0-1 分。适用于 RAGAS 不覆盖的领域特定质量维度。

**校准**：在有人工标注的样本上验证相关性之前，不要相信裁判的原始分数。运行 100 个手工标注的样本，绘制裁判分 vs 人工分的散点图，计算 Spearman 相关系数。如果 rho < 0.7，评分标准需要改进。

## 动手实现

### 第一步：用 NLI 做忠实度评估（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` 是任意可调用对象：提示字符串 -> 生成字符串
# 示例: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

把答案分解成原子声明，对每条声明用 NLI 检查是否被检索上下文蕴含，忠实度 = 被支持的比例。

### 第二步：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder 是任意实现了 .encode(texts, normalize_embeddings=True) -> ndarray 的模型
# 例如: encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示的问题与实际问的问题不同，相关性就会降低。

### 第三步：G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤就是评分标准。明确的步骤比隐性的"打 0-1 分"提示更稳定。

### 第四步：CI 质量关卡

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

作为 pytest 文件发布。在每个 PR 上运行。回归时阻止合并。

### 第五步：从零实现玩具评估器

见 `code/main.py`。纯标准库近似实现忠实度（答案声明与上下文的词重叠）和相关性（答案词与问题词的重叠）。不适合生产，但展示了核心形状。

## 陷阱

- **没有校准**。与人工标注相关性只有 0.3 的裁判就是噪声。发布前必须做校准运行。
- **自我评估**。用同一个 LLM 既生成又裁判，分数会虚高 10-20%。用不同模型家族做裁判。
- **成对判断中的位置偏差**。裁判偏好先呈现的那个选项。始终随机化顺序并正反向各跑一次。
- **原始汇总隐藏失败**。平均分 0.85 往往隐藏了 5% 的灾难性失败。始终检查分数分布的底部分位数。
- **黄金数据集腐烂**。没有版本控制、随时间漂移的评估集会破坏纵向比较。对每次变更打标签。
- **LLM 成本**。大规模时，裁判调用主导成本。用满足校准阈值的最便宜模型：GPT-4o-mini、Claude Haiku、Mistral-small。

## 工程应用

2026 年技术栈：

| 用途 | 框架 |
|------|------|
| RAG 质量监控 | RAGAS（4 个指标） |
| CI/CD 回归关卡 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 中的 G-Eval |
| 线上流量实时监控 | RAGAS 无参考模式 |
| 人工抽查 | LangSmith 或 Phoenix + 标注 UI |
| 红队测试 / 安全评估 | Promptfoo + DeepEval |

典型技术栈：RAGAS 用于监控，DeepEval 用于 CI，G-Eval 用于新维度。三者都跑，它们的分歧往往很有价值。

## 交付物

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习

1. **（简单）** 在 10 个包含已知幻觉的 RAG 样本上使用 RAGAS，验证忠实度指标能否捕捉到每个幻觉。
2. **（中等）** 手工标注 50 个问答答案的正确性（0-1），用 G-Eval 打分，测量裁判分与人工分之间的 Spearman 相关系数。
3. **（困难）** 用 DeepEval 构建 pytest CI 关卡，故意让检索器退化，验证关卡会失败，并对最低 10% 的分数添加底部分位数警报。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| LLM 裁判 (LLM-as-judge) | "用 LLM 打分" | 提示裁判模型根据评分标准对输出打 0-1 分 |
| RAGAS | "那个 RAG 指标库" | 开源评估框架，包含 4 个无参考 RAG 指标 |
| 忠实度 (Faithfulness) | "答案有来源支持吗？" | 被检索上下文蕴含的答案声明比例 |
| 上下文精确率 (Context Precision) | "检索到的块是否相关？" | top-K 块中实际有用的比例 |
| 上下文召回率 (Context Recall) | "检索找到所有内容了吗？" | 被检索块支持的金标准答案声明比例 |
| G-Eval | "自定义 LLM 裁判" | 评分标准 + 思维链评估步骤 + 0-1 分 |
| 校准 (Calibration) | "信任但要验证" | 裁判分与人工分之间的 Spearman 相关系数 |

## 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文
- [DeepEval 文档](https://deepeval.com/docs/metrics-introduction) — 开放生产级技术栈
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 偏差、校准与局限性
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 整合 RAGAS、DeepEval、Phoenix 的统一框架

# 长上下文评估——NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣传 1000 万 token 的上下文。在 100 万 token 时，8 针 MRCR 降至 26.3%。宣传的 ≠ 可用的。长上下文评估告诉你实际在使用的模型的真实容量。

**类型：** 学习
**语言：** Python
**前置知识：** 第5阶段第13课（问答系统）、第5阶段第23课（分块策略）
**预计时间：** 约60分钟

## 问题背景

你有一份 200 页的合同，模型声称有 100 万 token 的上下文。你把合同粘进去问："终止条款是什么？"模型给出了答案——但答的是封面页的内容，因为终止条款在 12 万 token 深处，超过了模型实际关注的范围。

这就是 2026 年的上下文容量差距。规格表说 100 万或 1000 万 token，实际可用的只有 60-70%，而"可用"的含义还取决于任务类型：

- **检索（草堆中找单针）**：前沿模型在宣称的最大长度内几乎完美。
- **多跳/聚合**：在大多数模型上，超过约 12.8 万 token 后急剧退化。
- **对分散事实的推理**：最先失效的任务类型。

长上下文评估就是测量这些维度。本课介绍相关基准测试，说明每个基准实际测量什么，以及如何为你的领域构建自定义针测试。

## 核心概念

**草堆中找针（NIAH，2023）**：把一个事实（"魔法词是 pineapple"）放在可控深度的长上下文中，让模型检索它，扫描深度 × 长度的组合。这是最早的长上下文基准测试。前沿模型现在已经在这个基准上饱和了，它是必要但不充分的基线。

**RULER（NVIDIA，2024）**：横跨 4 个类别的 13 种任务：检索（单键/多键/多值）、多跳追踪（变量追踪）、聚合（常用词频率）、问答。上下文长度可配置（4k 到 128k+）。能揭露那些在 NIAH 饱和但在多跳任务上失败的模型——2024 年发布时，声称支持 32k+ 上下文的 17 个模型中，只有一半在 32k 时保持了质量。

**LongBench v2（2024）**：503 道多选题，上下文 8k-200 万词，六种任务类别：单文档 QA、多文档 QA、长上下文学习、长对话、代码库、长结构化数据。用于测量真实世界长上下文行为的生产基准。

**MRCR（多轮共指消解）**：大规模多轮共指。有 8 针、24 针、100 针三种变体。暴露模型能在注意力退化前同时跟踪多少个事实。

**NoLiMa**："非词汇针"。针和查询没有字面重叠，检索需要一步语义推理，比 NIAH 更难。

**HELMET**：拼接多篇文档，从任意一篇中提问，测试选择性注意力。

**BABILong**：把 bAbI 推理链嵌入无关信息的草堆中，测试草堆中的推理能力，而不仅仅是检索能力。

### 实际应该报告什么

- **宣传的上下文窗口**：规格表上的数字。
- **有效检索长度**：NIAH 在某阈值（如 90%）下通过时的长度。
- **有效推理长度**：多跳或聚合在该阈值下通过时的长度。
- **退化曲线**：准确率 vs 上下文长度，按任务类型分别绘制。

规格表上应给出两个数字：检索有效长度和推理有效长度。通常推理有效长度是宣传窗口的 25-50%。

## 动手实现

### 第一步：为你的领域定制 NIAH

见 `code/main.py`。框架如下：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

扫描 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}，绘制热力图，这就是目标模型的 NIAH 评分卡。

### 第二步：多针变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

"三个魔法词是什么？"这类问题需要同时检索全部三个。单针成功不能预测多针成功。

### 第三步：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要串联三个赋值语句。前沿模型在 12.8 万 token 时通常在这里降到 50-70% 的准确率。

### 第四步：在你的技术栈上运行 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

按类别报告准确率。汇总分数会隐藏任务级别的巨大差异。

## 陷阱

- **只做 NIAH 评估**。在 100 万 token 时通过 NIAH，并不能说明多跳能力。始终运行 RULER 或自定义多跳测试。
- **深度采样不均匀**。许多实现只测试 depth=0.5。务必测试 depth=0、0.25、0.5、0.75、1.0——"迷失在中间"效应是真实存在的。
- **针和草堆有词汇重叠**。如果针和填充文本共享关键词，检索就变得简单了。使用 NoLiMa 风格的无重叠针。
- **忽略延迟**。100 万 token 的提示预填充需要 30-120 秒。同时测量首 token 时间和准确率。
- **供应商自报数字**。OpenAI、Google、Anthropic 都发布自己的评分。始终在你的使用场景上独立重跑。

## 工程应用

2026 年技术栈：

| 情况 | 基准测试 |
|------|---------|
| 快速健全性检查 | 3 个深度 × 3 个长度的自定义 NIAH |
| 生产模型选型 | RULER（13 个任务）在目标长度 |
| 真实世界 QA 质量 | LongBench v2 单文档 QA 子集 |
| 多跳推理 | BABILong 或自定义变量追踪 |
| 对话 / 多轮 | MRCR 8 针在目标长度 |
| 模型升级回归测试 | 固定内部 NIAH + RULER 测试集，每次新模型都跑 |

生产经验法则：在未做 NIAH + 至少 1 个推理任务的情况下，不要信任任何上下文窗口声明。

## 交付物

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 练习

1. **（简单）** 构建一个 3 深度（0.25、0.5、0.75）× 3 长度（1k、4k、16k）的 NIAH，在任意模型上运行，绘制通过率的 3×3 热力图。
2. **（中等）** 添加 3 针变体，测量每个长度下全部 3 根针的检索成功率，对比同等长度下单针通过率的差距。
3. **（困难）** 构造一个变量追踪任务（X1 → X2 → X3，3 跳），嵌入 6.4 万 token 的填充文本，在 3 个前沿模型上测量准确率，报告每个模型的有效推理长度。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| NIAH (草堆中找针) | "草堆中找针" | 把一个事实植入填充文本，让模型检索它 |
| RULER | "NIAH 升级版" | 横跨检索/多跳/聚合/QA 的 13 种任务类型 |
| 有效上下文 (Effective context) | "真实容量" | 准确率仍高于阈值时的最大长度 |
| 迷失在中间 (Lost in the middle) | "深度偏差" | 模型对长输入中间位置的内容关注不足 |
| 多针 (Multi-needle) | "同时找多个事实" | 多个植入点，测试注意力的平衡能力，而不仅仅是检索 |
| MRCR | "多轮共指消解" | 8、24 或 100 针共指；暴露注意力饱和点 |
| NoLiMa | "非词汇针" | 针和查询没有字面重叠，检索需要语义推理 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 原始 NIAH 仓库
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 多任务基准论文
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 真实世界长上下文评测
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更难的针
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — 草堆中的推理测试
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 深度偏差论文

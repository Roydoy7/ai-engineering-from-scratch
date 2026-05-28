# 结构化输出与约束解码

> 让 LLM 返回 JSON，大多数时候能拿到 JSON。在生产中，"大多数"就是问题所在。约束解码在采样前修改 logit，把"大多数"变成"始终"。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第17课（聊天机器人）、第5阶段第19课（子词分词）
**预计时间：** 约60分钟

## 问题背景

分类器向 LLM 提示："返回 {positive, negative, neutral} 中的一个。"模型返回"情感是正面的——这篇评论极其正面，因为顾客明确表示……"。你的解析器崩溃了，分类器的 F1 变成 0.0。

自由形式生成不是合同，只是建议。生产系统需要合同。

2026 年存在三个层次：

1. **提示词工程**：好好说。"只返回 JSON 对象。"在前沿模型上有约 80% 的成功率，小模型更低。
2. **原生结构化输出 API**：OpenAI `response_format`、Anthropic 工具使用、Gemini JSON 模式。在支持的 schema 上可靠，但供应商锁定。
3. **约束解码**：在每个生成步骤修改 logit，使模型**无法**生成无效 token。设计上 100% 有效，适用于任何本地模型。

本课为这三种方法都建立直觉，并说明何时选哪种。

## 核心概念

**约束解码的工作原理**：在每个生成步骤，LLM 在完整词汇表（约 10 万 token）上产生 logit 向量。一个 *logit 处理器* 位于模型和采样器之间，计算给定目标语法（JSON Schema、正则表达式、上下文无关文法）中当前位置哪些 token 有效，并将所有无效 token 的 logit 设为负无穷。剩余有效 token 上的 softmax 把概率质量只分配给有效续写。

2026 年的实现方案：

- **Outlines**：将 JSON Schema 或正则表达式编译成有限状态机（FSM），每个 token 获得 O(1) 的有效下一 token 查询。基于 FSM，递归 schema 需要展平。
- **XGrammar / llguidance**：上下文无关文法引擎，处理递归 JSON Schema，解码开销接近零。OpenAI 在 2025 年结构化输出实现中致谢了 llguidance。
- **vLLM 引导解码**：内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`，通过 Outlines、XGrammar 或 lm-format-enforcer 后端实现。
- **Instructor**：基于 Pydantic 的任意 LLM 封装，验证失败时自动重试。跨供应商，但不修改 logit，依赖重试和结构化输出感知提示词。

### 反直觉的结果

约束解码通常比无约束生成**更快**。原因有二：第一，它缩小了下一 token 的搜索空间；第二，聪明的实现对强制 token（脚手架，如 `{"name": "` ——每个字节都是确定的）完全跳过 token 生成。

### 让你付出代价的陷阱

字段顺序很重要。把 `answer` 放在 `reasoning` 之前，模型在思考之前就提交了答案。JSON 有效，但答案是错的，没有验证能捕捉到这个问题。

```json
// 错误
{"answer": "yes", "reasoning": "because ..."}

// 正确
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 字段顺序是逻辑，不是格式问题。

## 动手实现

### 第一步：从零实现正则约束生成

核心思路（约 30 行）：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 跟踪迄今为止满足了文法的哪些部分。`valid_tokens(state, tokenizer)` 计算词汇表中哪些 token 能在不离开接受路径的情况下推进 FSM。

### 第二步：用 Outlines 做 JSON Schema 约束

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零验证错误，永远。FSM 让无效输出变得不可达。

### 第三步：Instructor 做跨供应商 Pydantic 约束

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

机制不同。Instructor 不接触 logit，而是将 schema 格式化进提示词，解析输出，验证失败时重试（默认 3 次）。适用于任何供应商，重试会增加延迟和成本，跨供应商可移植性是其卖点。

### 第四步：原生供应商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务器端约束解码。在支持的 schema 上可靠性与 Outlines 相当，无需本地模型管理，但锁定到该供应商。

## 陷阱

- **递归 schema**：Outlines 把递归展平为固定深度，树形结构输出（嵌套评论、AST）需要 XGrammar 或 llguidance（基于 CFG）。
- **超大枚举**：1 万选项的枚举编译缓慢或超时，改用检索器：先预测 top-k 候选，再约束到这些候选中。
- **文法过于严格**：强制 `date: "YYYY-MM-DD"` 正则，模型对缺失日期就无法输出"unknown"，于是捏造一个日期，允许 `null` 或一个哨兵值。
- **过早提交**：见上面的字段顺序陷阱，始终把推理放在前面。
- **无 schema 的供应商 JSON 模式**：纯 JSON 模式只保证 JSON 语法有效，不保证对你的用例有效，始终提供完整 schema。

## 工程应用

2026 年技术栈：

| 情况 | 选择 |
|------|------|
| OpenAI/Anthropic/Google 模型，简单 schema | 原生供应商结构化输出 |
| 任意供应商，Pydantic 工作流，可接受重试 | Instructor |
| 本地模型，需要 100% 有效性，扁平 schema | Outlines（FSM） |
| 本地模型，递归 schema | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM 引导解码 |
| 批处理，可接受重试 | Instructor + 最便宜的模型 |

## 交付物

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## 练习

1. **（简单）** 不用约束解码，提示一个小型开放权重模型（如 Llama-3.2-3B）返回 `Review(sentiment, confidence, evidence_span)`，在 100 条评论上测量能成功解析为有效 JSON 的比例。
2. **（中等）** 在同一语料上使用 Outlines JSON 模式，对比合规率、延迟和语义准确率。
3. **（困难）** 从零实现用于电话号码的正则约束解码器（`\d{3}-\d{3}-\d{4}`），在 1000 个样本上验证无效输出为 0。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 约束解码 (Constrained decoding) | "强制有效输出" | 在每个生成步骤掩蔽无效 token 的 logit |
| Logit 处理器 (Logit processor) | "约束执行器" | 函数：`(logits, state) -> masked_logits` |
| FSM（有限状态机） | "FSM" | 编译后的文法表示，O(1) 有效下一 token 查询 |
| CFG（上下文无关文法） | "CFG" | 处理递归的文法，比 FSM 更慢但表达能力更强 |
| Schema 字段顺序 | "有关系吗？" | 有——第一个字段就会提交；始终把推理放在答案之前 |
| 引导解码 (Guided decoding) | "vLLM 的叫法" | 同一概念，集成到推理服务器中 |
| JSON 模式 (JSON mode) | "OpenAI 早期版本" | 保证 JSON 语法有效，**不**保证符合 schema |

## 延伸阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines 论文
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — 快速基于 CFG 的约束解码
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 推理服务器集成
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API 参考和注意事项
- [Instructor library](https://python.useinstructor.com/) — 跨供应商的 Pydantic + 重试方案
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 对 6 种约束解码框架的基准测试

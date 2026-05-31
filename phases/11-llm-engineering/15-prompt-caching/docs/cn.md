# 提示词缓存与上下文缓存（Prompt Caching and Context Caching）

> 你的系统提示词有 4,000 个 token，RAG 上下文有 20,000 个 token，两者随每次请求一起发送，也随每次请求一起计费。提示词缓存让服务商在他们那边将这段前缀保持热状态，复用时只收取正常价格的 10%。用好的话，可以将推理成本降低 50–90%，首 token 延迟降低 40–85%。

**类型：** 构建  
**语言：** Python  
**前置知识：** Phase 11 · 01（提示词工程）、Phase 11 · 05（上下文工程）、Phase 11 · 11（缓存与成本）  
**预计时间：** 约 60 分钟

## 问题所在

一个代码智能体在对话的每一轮都向 Claude 发送同一个 15,000-token 的系统提示词。以 $3/M 输入 token 计，二十轮对话仅输入成本就达 $0.90——这还不算用户消息本身。乘以每天 10,000 次对话，账单就因这段从不变化的文字达到了 $9,000/天。

你无法缩减提示词，否则会损害质量。你也无法不发送它——模型每轮都需要它。唯一的出路，是停止为服务商已经见过的前缀支付全额费用。

这条出路就是提示词缓存。Anthropic 于 2024 年 8 月上线了这项功能（2025 年又新增了 1 小时扩展 TTL 变体），OpenAI 在同年晚些时候实现了自动化，Google 随 Gemini 1.5 推出了显式上下文缓存，三家厂商现在都将其作为前沿模型的一等功能提供。

## 核心概念

![提示词缓存：写一次，低价读多次](../assets/prompt-caching.svg)

**工作机制。** 当一次请求的前缀与最近一次请求匹配时，服务商直接使用上次运行留下的 KV 缓存，而不重新编码这些 token。你第一次写入时支付少量溢价，此后每次命中都享有大幅折扣。

**2026 年三家服务商的实现方式。**

| 服务商 | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL | 最小可缓存量 |
|--------|----------|----------|----------|----------|-------------|
| Anthropic | 在内容块上显式标注 `cache_control` | 输入费用 9 折优惠（节省 90%） | 25% 附加费 | 5 分钟（可扩展至 1 小时） | 1,024 token（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | 输入费用 5 折优惠（节省 50%） | 无 | 最长 1 小时（尽力而为） | 1,024 token |
| Google（Gemini） | 显式 `CachedContent` API | 按存储计费；读取约为正常价格的 25% | 每 token·小时存储费 | 用户自定（默认 1 小时） | 4,096 token（Flash），32,768（Pro） |

**不变的规律。** 三家服务商都只缓存前缀。若两次请求之间有任何 token 不同，第一个差异 token 之后的所有内容都会缓存未命中。把**稳定**的部分放在顶部，**可变**的部分放在底部。

### 对缓存友好的布局

```
[系统提示词]       <-- 缓存这里
[工具定义]         <-- 缓存这里
[少样本示例]       <-- 缓存这里
[检索到的文档]     <-- 若重复使用则缓存，否则不缓存
[对话历史]         <-- 缓存到最后一轮
[当前用户消息]     <-- 永不缓存（每次都不同）
```

违反这个顺序——把用户消息放在系统提示词前面，或将动态检索结果插入少样本示例之间——缓存就永远不会命中。

### 收支平衡计算

Anthropic 25% 的写入溢价意味着一个缓存块至少需要被读取两次才能净节省成本。1 次写 + 1 次读的平均成本为正常的 0.675 倍（节省 32%）；1 次写 + 10 次读的平均成本为正常的 0.205 倍（节省 80%）。经验法则：对任何预计在 TTL 内至少复用 3 次的内容启用缓存。

## 动手构建

### 第一步：使用显式标记的 Anthropic 提示词缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告诉 Anthropic 将该块存储 5 分钟。在此窗口内复用会命中；过期后重新写入。

**响应中的用量字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # 按 1.25x 计费
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # 按 0.1x 计费
```

在 CI 中同时检查这两个字段——如果 `cache_read_input_tokens` 跨请求始终为零，说明你的缓存键在漂移。

### 第二步：1 小时扩展 TTL

对于长时运行的批处理任务，默认的 5 分钟 TTL 可能在任务间隙过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 的写入溢价是 2 倍（比基准高 50% 而非 25%），但对于任何复用同一前缀超过 5 次的批处理，很快就能回本。

### 第三步：OpenAI 自动缓存

OpenAI 不需要任何配置。超过 1,024 token 且与最近请求匹配的任何前缀都会自动享受 50% 折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # 长且稳定
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # 享受折扣的部分
```

同样的缓存友好布局规则同样适用。有两件事会破坏 OpenAI 的缓存，但不会破坏 Anthropic 的：更改 `user` 字段（用作缓存键的一部分）和对工具重新排序。

### 第四步：Gemini 显式上下文缓存

Gemini 将缓存视为你创建并命名的一等对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 在缓存存活期间按 token·小时收取存储费，读取时约为正常输入价格的 25%。当你需要在多天的多个会话中复用同一个超大提示词时，这种模式最为合适。

### 第五步：在生产中监测命中率

`code/main.py` 中提供了一个模拟三服务商的记账器，追踪写入/读取/未命中计数并计算每千次请求的综合成本。在部署时设置目标命中率门槛——大多数生产环境的 Anthropic 部署在预热后应能看到 >80% 的读取比例。

## 2026 年仍在出货的陷阱

- **在顶部放动态时间戳。** 在系统提示词顶部写 `"Current time: 2026-04-22 15:30:02"`——每次请求都会缓存未命中。把时间戳移到缓存断点以下。
- **工具重新排序。** 以稳定的顺序序列化工具——部署间的字典随机化会破坏所有命中。
- **自由文本的近似重复。** `"You are helpful."` 与 `"You are a helpful assistant."` ——一个字节的差异就是完全未命中。
- **块太小。** Anthropic 强制要求 1,024-token 的最小块（Haiku 为 2,048）。更小的块会静默地不被缓存。
- **盲目的成本仪表盘。** 将"输入 token"拆分为已缓存与未缓存两类。否则流量下降看起来像是缓存收益。

## 如何使用

2026 年的缓存技术选型：

| 场景 | 选择 |
|------|------|
| 有稳定 10k+ 系统提示词、多轮对话的智能体 | Anthropic `cache_control`，5 分钟 TTL |
| 复用同一前缀超过 30 分钟的批处理任务 | Anthropic，设置 `ttl: "1h"` |
| GPT-5 上的无服务器端点，无定制基础设施 | OpenAI 自动缓存（只需保持前缀稳定且足够长） |
| 对超大代码/文档语料的多天复用 | Gemini 显式 `CachedContent` |
| 跨服务商降级 | 在各服务商之间保持可缓存前缀布局一致，使任意命中都有效 |

与语义缓存（Phase 11 · 11）结合用于用户消息层：提示词缓存处理**token 完全相同**的复用，语义缓存处理**语义相同**的复用。

## 输出产物

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 练习

1. **简单。** 使用 5,000-token 系统提示词与 Claude 进行一次 10 轮对话。不加 `cache_control` 跑一遍，再加上跑一遍。分别报告两次的输入 token 账单。
2. **中等。** 编写一个测试工具，给定提示词模板和请求日志，计算各服务商（Anthropic 5m、Anthropic 1h、OpenAI 自动、Gemini 显式）的预期命中率和节省金额。
3. **困难。** 构建一个布局优化器：给定一个提示词和一组标注了 `stable=True/False` 的字段，在不丢失信息的前提下，将提示词重写为在最大缓存友好位置设置单一缓存断点。在真实的 Anthropic 端点上验证。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 提示词缓存（Prompt caching） | "让长提示词变便宜" | 对匹配前缀复用服务商侧 KV 缓存；重复输入 token 享受 50–90% 折扣。 |
| `cache_control` | "Anthropic 的标记" | 内容块属性，声明"直到这里的内容可以缓存"；`{"type": "ephemeral"}`。 |
| 缓存写入（Cache write） | "支付溢价" | 首次填充缓存的请求；在 Anthropic 按约 1.25 倍输入价格计费，在 OpenAI 免费。 |
| 缓存读取（Cache read） | "享受折扣" | 后续匹配前缀的请求；按 10%（Anthropic）、50%（OpenAI）、约 25%（Gemini）计费。 |
| TTL | "能活多久" | 缓存保持热状态的时间；Anthropic 默认 5 分钟（可扩展至 1 小时），OpenAI 尽力最长 1 小时，Gemini 用户自定。 |
| 扩展 TTL（Extended TTL） | "1 小时 Anthropic 缓存" | `{"type": "ephemeral", "ttl": "1h"}`；写入溢价 2 倍，但对批量复用超过 5 次的场景很快回本。 |
| 前缀匹配（Prefix match） | "为什么缓存未命中" | 只有从头到断点的每个 token 完全一致时才会命中。 |
| 上下文缓存（Context caching，Gemini） | "那个显式的" | Google 的有名、按存储计费的缓存对象；最适合大型语料的多天复用。 |

## 延伸阅读

- [Anthropic — 提示词缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1h TTL、收支平衡表格。
- [OpenAI — 提示词缓存](https://platform.openai.com/docs/guides/prompt-caching) — 自动前缀匹配。
- [Google — 上下文缓存](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 与存储定价。
- [Anthropic 工程博客 — 面向长上下文工作负载的提示词缓存](https://www.anthropic.com/news/prompt-caching) — 包含延迟数据的原始发布文章。
- Phase 11 · 05（上下文工程） — 如何切分提示词以使缓存生效。
- Phase 11 · 11（缓存与成本） — 将提示词缓存与用户消息层的语义缓存配合使用。
- [Pope 等，"Efficiently Scaling Transformer Inference"（2022）](https://arxiv.org/abs/2211.05102) — 提示词缓存所暴露的 KV 缓存内存模型；解释了为什么重读已缓存前缀比重新计算便宜约 10 倍。
- [Agrawal 等，"SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills"（2023）](https://arxiv.org/abs/2308.16369) — Prefill 是提示词缓存所跳过的阶段；本文解释了为什么缓存命中时 TTFT 会大幅下降而 TPOT 不受影响。
- [Leviathan 等，"Fast Inference from Transformers via Speculative Decoding"（2023）](https://arxiv.org/abs/2211.17192) — 提示词缓存与推测解码、Flash Attention、MQA/GQA 并列，同为压低推理成本曲线的杠杆；阅读本文了解其他三种方式。

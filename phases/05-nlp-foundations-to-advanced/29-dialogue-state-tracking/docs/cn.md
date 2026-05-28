# 对话状态追踪

> "我想要北边一家便宜的餐厅……不，改成中等价位……再加上意大利菜。"三轮对话，三次状态更新。DST 让槽位-值字典保持同步，才能让预订成功完成。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第17课（聊天机器人）、第5阶段第20课（结构化输出）
**预计时间：** 约75分钟

## 问题背景

在任务导向的对话系统中，用户的目标被编码成一组槽位-值对：`{cuisine: italian, area: north, price: moderate}`。每轮用户话语都可以添加、修改或删除一个槽位。系统必须读取整段对话，正确输出当前状态。

一个槽位出错，系统就会预订错误的餐厅、排错航班或刷错卡。DST 是用户所说的话和后端实际执行之间的关键枢纽。

2026 年在 LLM 时代为什么仍然重要：

- 合规敏感领域（银行、医疗、航空预订）需要确定性的槽位值，而不是自由形式生成。
- 工具调用智能体在调用 API 之前仍然需要解析槽位。
- 多轮纠错比看起来更难："不，改成周四吧。"

现代流水线：经典 DST 概念 + LLM 提取器 + 结构化输出护栏。

## 核心概念

**任务结构**：一个 schema 定义了领域（餐厅、酒店、出租车）及其槽位（菜系、区域、价格、人数）。每个槽位可以为空，可以填入封闭集合中的值（价格：{cheap, moderate, expensive}），也可以是自由形式的值（名称："The Copper Kettle"）。

**两种 DST 形式**：

- **分类式**：对每个（槽位，候选值）对，预测是/否。适用于封闭词汇槽位。2020 年前的标准方法。
- **生成式**：给定对话，以自由文本形式生成槽位值。适用于开放词汇槽位。现代默认方法。

**评估指标**：联合目标准确率（JGA）——所有槽位都正确的轮次比例。全对才算对。2026 年 MultiWOZ 2.4 排行榜最高约 83%。

**主要架构**：

1. **基于规则（槽位正则 + 关键词）**：对窄域应用是强基线，可调试。
2. **TripPy / BERT-DST**：基于复制的生成，使用 BERT 编码。前 LLM 时代的标准方法。
3. **LDST（LLaMA + LoRA）**：带领域-槽位提示的指令调优 LLM，在 MultiWOZ 2.4 上达到 ChatGPT 级别的质量。
4. **无本体（2024-26）**：跳过 schema，直接生成槽位名称和值，适用于开放领域。
5. **提示词 + 结构化输出（2024-26）**：LLM + Pydantic schema + 约束解码。5 行代码，生产可用。

### 经典失败模式

- **跨轮共指**："就选第一个选项吧。"需要解析哪个是"第一个"。
- **覆盖 vs 追加**：用户说"加意大利菜"，是替换菜系还是追加？
- **隐式确认**："好的没问题"——这是接受了预订提议吗？
- **纠错**："不，改成晚上 7 点。"必须更新时间，但不能清空其他槽位。
- **指向上一个系统话语的共指**："就那个。"哪个"那个"？

## 动手实现

### 第一步：基于规则的槽位抽取器

见 `code/main.py`。正则表达式 + 同义词词典能覆盖窄域中约 70% 的规范话语：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

超出规范词汇就会失效，但对确定性的槽位确认很有效。

### 第二步：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变量：

- 永远不要重置用户没有提及的槽位。
- 明确的否定（"算了，不限菜系了"）必须清空。
- 用户纠错（"其实……"）必须覆盖，而非追加。

### 第三步：带结构化输出的 LLM 驱动 DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证返回有效的状态对象。无需正则表达式，无 schema 不匹配，无幻觉槽位。

### 第四步：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统在多少比例的轮次中能答对所有槽位？MultiWOZ 2.4 上，2026 年最强系统约 80-83%。在你的窄域词汇上，你的系统应该超过这个，否则直接用 LLM 基线就行了。

### 第五步：处理纠错

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到纠错时，覆盖上一次更新的槽位，而不是追加。没有 LLM 帮助很难做对。现代模式：始终让 LLM 从历史对话重新生成完整状态，而不是增量更新——这样自然就能处理纠错。

## 陷阱

- **全历史重新生成的成本**。每轮都让 LLM 重新生成状态会消耗 O(n²) 总 token。要截断历史或汇总较早的轮次。
- **Schema 漂移**。事后添加新槽位会破坏旧训练数据，要对 schema 做版本控制。
- **大小写敏感**。"Italian" vs "italian" vs "ITALIAN"——到处做归一化。
- **隐式继承**。如果用户之前指定了"4 人"，新的预订时间请求不应该清空人数。始终传递完整历史。
- **自由形式 vs 封闭集合**。姓名、时间和地址需要自由形式槽位；菜系和区域是封闭的。Schema 中同时使用两者。

## 工程应用

2026 年技术栈：

| 情况 | 方案 |
|------|------|
| 窄域（一两个意图） | 基于规则 + 正则表达式 |
| 宽域，有标注数据 | LDST（LLaMA + LoRA，在 MultiWOZ 风格数据上微调） |
| 宽域，无标注，生产就绪 | LLM + Instructor + Pydantic schema |
| 语音 / 口头输入 | ASR + 文本归一化 + LLM-DST |
| 多领域预订流程 | Schema 引导 LLM + 每个领域独立 Pydantic 模型 |
| 合规敏感 | 基于规则为主，LLM 兜底 + 确认流程 |

## 交付物

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 练习

1. **（简单）** 在 `code/main.py` 中为 3 个槽位（cuisine、area、price）构建基于规则的状态追踪器，在 10 段手工设计的对话上测试，测量 JGA。
2. **（中等）** 在相同数据集上使用 Instructor + Pydantic + 小型 LLM，对比 JGA，检查最难的几轮。
3. **（困难）** 同时实现两种方法并做路由：基于规则为主，当基于规则抽取的槽位少于 2 个时 LLM 兜底，测量联合 JGA 和每轮推理成本。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| DST（对话状态追踪） | "对话状态追踪" | 在多轮对话中维护槽位-值字典 |
| 槽位 (Slot) | "用户意图的单元" | 后端需要的命名参数（菜系、日期） |
| 领域 (Domain) | "任务领域" | 餐厅、酒店、出租车——各自有一组槽位 |
| JGA（联合目标准确率） | "联合目标准确率" | 所有槽位都正确的轮次比例，全对才算对 |
| MultiWOZ | "那个基准" | 多领域 WOZ 数据集，标准 DST 评测集 |
| 无本体 DST (Ontology-free DST) | "无 schema" | 直接生成槽位名称和值，无固定列表 |
| 纠错 (Correction) | ""其实……"" | 覆盖之前已填槽位的轮次 |

## 延伸阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 标准基准数据集
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA 指令调优用于 DST
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 基于复制的 DST 主力工具
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — 基于 EM 的无监督任务导向对话
- [MultiWOZ 排行榜](https://github.com/budzianowski/multiwoz) — 标准 DST 结果

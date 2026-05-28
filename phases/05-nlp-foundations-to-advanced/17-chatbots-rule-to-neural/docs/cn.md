# 聊天机器人——从规则到神经网络再到 LLM 智能体

> ELIZA 用模式匹配回复。DialogFlow 映射意图。GPT 从权重中回答。Claude 运行工具并验证结果。每个时代都解决了上一个时代最严重的失败。

**类型：** 学习
**语言：** Python
**前置知识：** 第5阶段第13课（问答系统）、第5阶段第14课（信息检索）
**预计时间：** 约75分钟

## 问题背景

用户说"我想改签我的航班"。系统必须弄清楚用户想要什么，缺少哪些信息，如何获取，以及如何完成操作。然后用户说"等等，如果我取消呢？"系统必须记住上下文，切换任务，并保持状态。

对 ML 系统来说，对话很难。输入是开放式的，输出必须在多轮对话中保持连贯，系统可能需要对外部世界采取行动（改签、扣款），每一步错误对用户都是可见的。

聊天机器人架构经历了四个范式，每个范式都是因为前一个失败得太过明显而被引入。本课按顺序介绍它们。2026 年的生产格局是最后两者的混合。

## 核心概念

**基于规则（ELIZA、AIML、DialogFlow）**：手工编写的模式匹配用户输入并产生响应。意图分类器路由到预定义流程，槽位填充状态机收集所需信息。在它被设计的狭窄范围内效果极好，一出范围立刻失败。在银行认证、航空预订等不容忍幻觉的安全关键领域仍然在用。

**检索式**：类似 FAQ 的系统，对每对（话语，回复）编码，运行时编码用户消息并检索最近的存储回复。就像 Zendesk 经典的"相似文章"功能。比规则更好地处理改写，无需生成，所以不会幻觉。

**神经式（seq2seq）**：在对话日志上训练的编码器-解码器，从头生成回复。流畅，但容易产生通用输出（"我不知道"）和事实漂移。从来不能可靠地切题。这就是谷歌、Facebook 和微软在 2016-2019 年的聊天机器人都令人失望的原因。

**LLM 智能体**：被封装在循环中的语言模型，负责规划、调用工具、验证结果。不是带长提示词的聊天机器人，而是一个智能体循环：规划 → 调用工具 → 观察结果 → 决定下一步。检索优先的接地（RAG）防止幻觉，工具调用让它真正能做事。这是 2026 年的架构。

四个范式不是顺序替换的关系。2026 年的生产聊天机器人会通过全部四种：基于规则处理认证和破坏性操作，检索处理 FAQ，神经生成提供自然的措辞，LLM 智能体处理模糊的开放式查询。

## 动手实现

### 第一步：基于规则的模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20 行的 ELIZA。反射技巧（"我感到悲伤"→"你为什么感到悲伤？"）是 Weizenbaum 1966 年经典心理治疗师演示的核心，至今仍有启发意义。

### 第二步：检索式（FAQ）

以下示例代码需要 `pip install sentence-transformers`（会拉取 torch）。本课的可运行 `code/main.py` 使用标准库的 Jaccard 相似度代替，无需外部依赖。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒答是关键设计选择。如果最佳匹配不够接近，返回 `None` 并让系统升级处理。

### 第三步：神经生成（基线）

使用小型指令调优的编码器-解码器（FLAN-T5）或微调的对话模型。2026 年单独使用在生产中不可行（矛盾、偏离主题、事实胡说），但在混合系统中为自然措辞服务。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 第四步：LLM 智能体循环

2026 年的生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

三件事值得命名：工具是 LLM 可以调用的可执行函数；当 LLM 返回最终答案而非工具调用时循环终止；步数预算防止在模糊任务上无限循环。

真实生产还会增加：检索优先接地（每次 LLM 调用前注入相关文档）、护栏（拒绝在未确认的情况下执行破坏性操作）、可观测性（记录每一步）和评估（自动检查智能体行为符合规格）。

### 第五步：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式：确定性规则处理任何破坏性操作，检索处理固定 FAQ，LLM 智能体处理其他一切。这就是 2026 年客服系统的实际发布形态。

## 工程应用

2026 年技术栈：

| 使用场景 | 架构 |
|---------|------|
| 预订、支付、认证 | 基于规则的状态机 + 槽位填充 |
| 客服 FAQ | 对精选答案做检索 |
| 开放式帮助对话 | LLM 智能体 + RAG + 工具调用 |
| 内部工具/IDE 助手 | LLM 智能体 + 工具调用（搜索、读、写） |
| 陪伴/角色聊天机器人 | 带人格系统提示词的调优 LLM + 知识检索 |

生产中始终使用混合路由。没有任何单一架构能很好地处理每类请求，路由层本身通常是一个小型意图分类器。

## 仍在发货的失败模式

- **自信地捏造**：LLM 智能体声称完成了一个实际未完成的操作。缓解：验证结果、记录工具调用、永远不允许 LLM 在没有成功工具返回的情况下声称已完成某事。
- **提示词注入（Prompt Injection）**：用户插入覆盖系统提示词的文本。在 OWASP 2025 年 LLM 应用 Top 10 中排第一（LLM01）。两种形式：直接注入（粘贴到对话中）和间接注入（隐藏在智能体读取的文档、邮件或工具输出中）。

  攻击成功率因场景而异：在通用工具使用和编程基准的前沿模型上约 0.5-8.5%，针对 AI 编程智能体的自适应攻击和脆弱的编排架构可达约 84%。生产级 CVE 包括 EchoLeak（CVE-2025-32711，CVSS 9.3）——Microsoft 365 Copilot 中的零点击数据渗漏漏洞，由攻击者控制的电子邮件触发。

  缓解措施：在整个循环中把用户输入视为不可信；工具调用前做消毒；将工具输出与主提示词隔离；使用"规划-验证-执行（PVE）"模式，智能体先规划，执行每个操作前对照计划验证（这可以阻止工具结果注入新的计划外操作）；对破坏性操作要求用户确认；对工具权限应用最小特权原则。

  再多的提示词工程也无法完全消除这种风险，需要外部运行时防御层（LLM Guard、白名单验证、语义异常检测）。

- **目标蔓延**：工具调用返回了相关但无关紧要的信息，智能体偏离任务。缓解：收窄工具契约，保持系统提示词聚焦，增加偏离任务率的评估。
- **无限循环**：智能体不断调用同一工具。缓解：步数预算、工具调用去重、LLM 裁判判断"我们是否在取得进展"。
- **上下文窗口耗尽**：长对话把最早的轮次推出上下文。缓解：总结较旧的轮次、通过相似度检索相关的历史轮次，或使用长上下文模型。

## 交付物

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 练习

1. **（简单）** 用上面的基于规则方式为咖啡店点单机器人实现 10 个模式，测试边缘情况：双份点单、修改、取消、意图不明确。
2. **（中等）** 构建混合 FAQ + LLM 兜底系统：针对某 SaaS 产品提供 50 条固定 FAQ 答案，用检索文档站点的 LLM 做兜底，在 100 个真实支持问题上测量拒答率和准确率。
3. **（困难）** 用三个工具（搜索、读取用户数据、发送邮件）实现上面的智能体循环，用 50 个测试场景（包含提示词注入尝试）运行评估，报告偏离任务率、任务失败率和任何注入成功案例。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 意图 (Intent) | "用户想要什么" | 分类标签（book_flight、reset_password），路由到对应处理器 |
| 槽位 (Slot) | "一条信息" | 机器人需要的参数（日期、目的地），槽位填充是一系列追问 |
| RAG | "检索后生成" | 检索相关文档，再以此接地 LLM 的回复 |
| 工具调用 (Tool call) | "函数调用" | LLM 发出包含名称和参数的结构化调用，运行时执行后返回结果 |
| 智能体循环 (Agent loop) | "规划-行动-验证" | 交替运行 LLM 调用和工具调用直到任务完成的控制器 |
| 提示词注入 (Prompt injection) | "用户攻击提示词" | 试图覆盖系统提示词的恶意输入 |

## 延伸阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 原始基于规则的聊天机器人论文
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — 谷歌的晚期神经聊天机器人论文，LLM 智能体接管之前
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 命名智能体循环模式的论文
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024 年生产指南，2026 年仍然适用
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — 提示词注入论文
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 将提示词注入列为首要安全威胁的排名
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 间接提示词注入的标志性零点击数据渗漏 CVE，写访问智能体为何需要运行时防御的参考案例

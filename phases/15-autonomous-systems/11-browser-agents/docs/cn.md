# 浏览器智能体与长时域网络任务（Browser Agents and Long-Horizon Web Tasks）

> ChatGPT 智能体（2025 年 7 月）将 Operator 和深度研究合并为一个浏览器/终端智能体，在 BrowseComp 上创下 68.9% 的最高水平。OpenAI 于 2025 年 8 月 31 日关闭了 Operator——产品层的整合。Anthropic 收购 Vercept 将 Claude Sonnet 在 OSWorld 上从低于 15% 提升到 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修复了原始 WebArena 中 11.3 个百分点的假阴性率，并发布了 258 个任务的困难子集。这些数字是真实的。攻击面也是：OpenAI 的准备负责人公开表示，对浏览器智能体的间接提示词注入"不是一个可以完全修补的 bug"。2025-2026 年有记录的攻击：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks）和 Perplexity Comet 中的一键劫持。

**类型：** 学习  
**语言：** Python（标准库，间接提示词注入攻击面模型）  
**前置知识：** Phase 15 · 10（权限模式）、Phase 15 · 01（长时域智能体）  
**预计时间：** 约 45 分钟

## 问题所在

浏览器智能体是一个读取不可信内容并采取后果性动作的长时域智能体。智能体访问的每个页面都是用户没有写的输入。每个页面上的每个表单都是潜在的命令通道。2025-2026 年的攻击语料库表明这不是假设：Tainted Memories 让攻击者通过精心制作的页面将恶意指令绑定到智能体的记忆中；HashJack 在智能体访问的 URL 片段中隐藏命令；Perplexity Comet 劫持在一次点击中发生。

防御态势令人不安。OpenAI 的准备负责人说出了安静的部分：间接提示词注入"不是一个可以完全修补的 bug"。这是因为攻击存在于智能体的读取与行动边界，这在架构上是模糊的——模型读取的每个 token 在原则上都可以被读取为指令。

本课命名攻击面，命名基准格局（BrowseComp、OSWorld、WebArena-Verified），并对最小的间接提示词注入场景进行建模，以便你可以推理第 14 和 18 课中的真实防御。

## 核心概念

### 2026 格局，每个系统一段话

**ChatGPT 智能体（OpenAI）。** 2025 年 7 月推出。统一了 Operator（浏览）和深度研究（多小时研究）。2025 年 8 月 31 日关闭了独立的 Operator。BrowseComp 最高水平 68.9%；OSWorld 和 WebArena-Verified 上有强劲数字。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 收购 Vercept 专注于计算机使用能力。将 Claude Sonnet 在 OSWorld 上从 <15% 提升到 72.5%。Claude Computer Use 作为工具 API 发布。

**带浏览器使用的 Gemini 3 Pro（DeepMind）。** 浏览器使用集成发布计算机使用控制；FSF v3（2026 年 4 月，第 20 课）专门追踪 ML R&D 领域的自主性。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复了一个有据可查的问题：原始 WebArena 有约 11.3% 的假阴性率（被标记为失败但实际上已解决的任务）。经过验证的发布使用人工策划的成功标准重新评分，并添加了 258 个任务的困难子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp 与 OSWorld 与 WebArena 的比较

| 基准 | 测量什么 | 时域 |
|------|---------|------|
| BrowseComp | 在开放网络上在时间压力下找到特定事实 | 分钟 |
| OSWorld | 智能体操作完整桌面（鼠标、键盘、shell） | 数十分钟 |
| WebArena-Verified | 在模拟站点中的事务性网络任务 | 分钟 |
| 困难子集 | 具有多页状态转换的 WebArena-Verified 任务 | 数十分钟 |

不同的轴。高 BrowseComp 分数说明智能体找到了事实；它不说明智能体能预订航班。OSWorld 分数更接近"它在我的桌面上有效"。WebArena-Verified 更接近"它能完成一个流程"。任何生产决策都需要与任务分布匹配的基准。

### 攻击面，命名

1. **间接提示词注入。** 不可信页面内容包含指令。智能体读取它们。智能体执行它们。公开示例：2024 年 Kai Greshake 等人，2025 年 Tainted Memories 论文，2026 年 HashJack（Cato Networks）。
2. **URL 片段/查询注入。** 爬取的 URL 的 `#fragment` 或查询字符串包含命令。从不可见地渲染；仍然在智能体的上下文中。
3. **记忆绑定攻击。** 页面指示智能体写入持久记忆（第 12 课涵盖持久状态）。下一个会话，记忆触发有效载荷，没有可见触发器。
4. **针对已认证会话的 CSRF 形状攻击。** Tainted Memories 类：智能体登录了某处；攻击者的页面发出智能体用用户 cookie 执行的状态改变请求。
5. **一键劫持。** 视觉上无害的按钮附带智能体跟随的有效载荷。Comet 类。
6. **智能体主机表面中的内容安全策略漏洞。** 渲染和工具层本身可以是攻击向量；浏览器中的智能体中的浏览器堆栈很宽。

### 为什么"无法完全修补"

攻击与智能体的能力同构。智能体必须读取不可信内容才能完成其工作。智能体读取的任何内容都可能包含指令。智能体遵循的任何指令都可能与用户的实际请求不一致。防御（信任边界、分类器、工具允许列表、后果性动作的 HITL）提高了攻击成本并减少了爆炸半径。它们不关闭这个类别。

这与 Löb 定理（第 8 课）的推理模式相同：智能体不能证明下一个 token 是安全的；它只能建立一个不安全 token 更容易被检测的系统。

### 实际发布的防御态势

- **读/写边界。** 读取从不是后果性的。写入（提交表单、发布内容、调用有副作用的工具）需要如果发起内容来自信任边界之外的新鲜人工批准。
- **每任务工具允许列表。** 智能体可以浏览；除非为任务明确启用了该工具，否则它不能发起电汇。第 13 课涵盖预算。
- **会话隔离。** 浏览器智能体会话仅使用范围凭据运行。没有生产认证，没有个人电子邮件。每个 HTTP 请求的日志保留以供审计。
- **内容清理器。** 获取的 HTML 在被连接到模型上下文之前去除已知不良模式。（减少简单攻击；不能阻止复杂有效载荷。）
- **后果性动作的 HITL。** 提案再提交模式（第 15 课）。
- **记忆上的金丝雀 token。** 如果记忆条目触发，用户会看到它（第 14 课）。

## 使用它

`code/main.py` 针对三个合成页面对一个微型浏览器智能体运行进行建模。一个页面是良性的，一个在可见文本中有直接提示词注入 blob，一个有 URL 片段注入（不可见但在智能体的上下文中）。脚本显示（a）朴素智能体会做什么，（b）读/写边界捕获什么，（c）清理器捕获什么，（d）两者都捕获不了什么。

## 交付它

`outputs/skill-browser-agent-trust-boundary.md` 范围界定了提议的浏览器智能体部署：它触及的信任区域、它被授权写入什么，以及在第一次运行之前必须有哪些防御措施。

## 练习

1. 运行 `code/main.py`。识别清理器捕获但读/写边界不捕获的攻击，以及只有读/写边界捕获的攻击。

2. 扩展清理器以检测一类 HashJack 风格的 URL 片段注入。在具有合法片段的良性 URL 上测量假阳性率。

3. 选择你知道的一个真实浏览器智能体工作流（例如"预订航班"）。列出每次读取和每次写入。标记哪些写入需要 HITL 以及原因。

4. 阅读 WebArena-Verified ICLR 2026 论文。识别原始 WebArena 评分不可靠的一类任务，并解释经过验证的子集如何解决它。

5. 为浏览器智能体设置设计一个记忆金丝雀。你会存储什么，在哪里，什么触发警报？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Indirect prompt injection（间接提示词注入） | "坏页面文本" | 智能体读取的页面中的不可信内容包含智能体执行的指令 |
| Tainted Memories（被污染的记忆） | "记忆攻击" | 智能体将攻击者提供的指令写入持久记忆；下一个会话触发 |
| HashJack | "URL 片段攻击" | 隐藏在 URL 片段/查询字符串中的有效载荷在智能体的上下文中但不可见渲染 |
| One-click hijack（一键劫持） | "坏按钮" | 可见的功能附带智能体执行的后续有效载荷 |
| BrowseComp | "网络搜索基准" | 在开放网络上找到特定事实；分钟级时域 |
| OSWorld | "桌面基准" | 完整 OS 控制；多步骤 GUI 任务 |
| WebArena-Verified | "固定网络任务基准" | ServiceNow 重新评分的 WebArena，带困难子集 |
| Read/write boundary（读/写边界） | "副作用门" | 读取从不后果性；如果内容不在信任范围内，写入需要新鲜批准 |

## 延伸阅读

- [OpenAI — 介绍 ChatGPT 智能体](https://openai.com/index/introducing-chatgpt-agent/) — Operator 和深度研究的合并；BrowseComp 最高水平。
- [OpenAI — 计算机使用智能体](https://openai.com/index/computer-using-agent/) — Operator 谱系和成为 ChatGPT 智能体的架构。
- [Zhou 等 — WebArena](https://webarena.dev/) — 原始基准。
- [WebArena-Verified（OpenReview）](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 固定子集论文。
- [Anthropic — 实践中测量智能体自主性](https://www.anthropic.com/research/measuring-agent-autonomy) — 包括计算机使用智能体的攻击面讨论。

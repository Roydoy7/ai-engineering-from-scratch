# 群聊与发言者选择（Group Chat and Speaker Selection）

> AutoGen GroupChat 和 AG2 GroupChat 在 N 个智能体之间共享一个对话；选择器函数（LLM、轮询或自定义）决定谁下一个发言。这是涌现式多智能体对话的原型——智能体不知道自己在静态图中的角色，它们只是对共享池做出反应。AutoGen v0.2 的 GroupChat 语义在 AG2 分支中得以保留；AutoGen v0.4 将其重写为事件驱动的 Actor 模型。微软于 2026 年 2 月将 AutoGen 置于维护模式，并将其与 Semantic Kernel 合并为 Microsoft Agent Framework（RC 版于 2026 年 2 月发布）。GroupChat 原语在 AG2 和 Microsoft Agent Framework 中都得以延续——学一次，处处可用。

**类型：** 学习 + 构建  
**语言：** Python（标准库）  
**前置知识：** Phase 16 · 04（原语模型）  
**预计时间：** 约 60 分钟

## 问题所在

静态图（LangGraph）在工作流已知的情况下很好用。但真实的对话不是静态的：有时程序员会向审查者提问，有时会向研究者提问，有时会向作者提问。将每一种可能的交接都硬编码会产生边数爆炸。你需要的是*让智能体对共享池做出反应*，再由某个函数决定谁下一个发言。

这正是 AutoGen GroupChat 所做的。

## 核心概念

### 结构形态

```
              ┌─── 共享池 ────┐
              │  m1  m2  m3 ...│
              └─────────┬──────┘
                        │（所有人读取所有消息）
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    智能体A 智能体B  智能体C  智能体D  选择器
                                          │
                                          ▼
                                "下一个发言者 = C"
```

每个智能体看到每一条消息。每一轮都会调用选择器函数来决定谁下一个发言。

### 三种选择器风格

**轮询（Round-robin）。** 固定循环，确定性。N 的线性扩展，但忽略上下文——即便当前话题是法务审查，程序员也会轮到发言。

**LLM 选择。** 调用一个 LLM，它读取最近的消息池并返回最佳的下一个发言者。感知上下文，但较慢：每轮都额外增加一次 LLM 调用。这是 AutoGen 的默认方式。

**自定义。** 一个包含任意逻辑的 Python 函数。典型做法：LLM 选择加上备用规则（例如，"程序员发言后总是让验证者发言"）。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有选择器。当一个智能体完成发言，管理者调用选择器，选择器返回下一个智能体。循环持续直到终止条件触发。

### 终止

三种常见模式：

- **最大轮数。** 总发言轮数的硬性上限。
- **"TERMINATE"令牌。** 智能体可以发出一条哨兵消息；管理者在检测到该消息时停止。
- **目标达成检查。** 每轮运行一个轻量级验证者，完成后停止对话。

### AutoGen → AG2 分裂与 Microsoft Agent Framework 合并

2025 年初，微软开始围绕事件驱动的 Actor 模型对 AutoGen（v0.4）进行大规模重写。社区将 AutoGen v0.2 的 GroupChat 语义分支为 AG2，保留了早期采用者已集成的 API。

2026 年 2 月，微软宣布 AutoGen 进入维护模式，事件驱动的 Actor 模型合并进了 **Microsoft Agent Framework**（RC 于 2026 年 2 月发布，现已与 Semantic Kernel 合并）。GroupChat 概念在两条路线中都得以延续；实现细节有所不同。AG2 是 v0.2 兼容代码的首选上游。

### GroupChat 适用的场景

- **涌现式对话。** 你不想预先连线每一种可能的下一个发言者。
- **角色混合任务。** 程序员询问研究者，研究者询问档案管理员，档案管理员再询问程序员。流程不是有向无环图。
- **探索性问题求解。** 想象成"头脑风暴会议"，而非"流水线"。

### 失效场景

- **严格确定性。** LLM 选择器可能不一致。相同提示词，不同运行，不同的下一个发言者。
- **谄媚级联。** 智能体向最有把握说话的那个人靠拢。在提示词中明确反制。
- **上下文膨胀。** 每个智能体读取每条消息；10 轮后上下文会变得非常庞大。使用投影（第 15 课）来限定视图范围。
- **热门发言者。** 一个智能体因选择器偏好其专业领域而主导对话。在选择器中引入发言平衡特性。

### 群聊与监督者的对比

相同的原语，不同的默认设置：

- 监督者：一个智能体规划，其余执行。选择器是"问规划者下一步做什么"。
- 群聊：所有智能体是平等的同伴；选择器是一个作用于共享池的函数。

两者都使用第 04 课的四个原语。群聊默认采用 LLM 选择的编排和全池共享状态。

## 构建它

`code/main.py` 使用标准库从零开始实现了一个 GroupChat。三个智能体（程序员、审查者、管理者），轮询和 LLM 选择两种变体，以及基于 `TERMINATE` 令牌的终止。

演示打印对话记录，以及两种变体的选择器决策追踪。

运行：

```
python3 code/main.py
```

## 使用它

`outputs/skill-groupchat-selector.md` 为给定任务配置 GroupChat 选择器——轮询 vs LLM 选择 vs 自定义，以及选择器应使用哪些输入（最近消息、智能体专业领域、发言轮次计数）。

## 交付它

检查清单：

- **最大轮数上限。** 始终设置。典型任务设为 10-20 轮。
- **发言平衡指标。** 追踪每个智能体的发言轮次；当不平衡超过阈值时触发警报。
- **终止令牌。** `TERMINATE` 或专用的验证者智能体。
- **投影或范围化记忆。** 约 10 条消息后，考虑给每个智能体只提供范围化的视图，以防止上下文膨胀。
- **选择器日志记录。** 对于 LLM 选择变体，同时记录选择器的输入和其选择。否则调试将无从下手。

## 练习

1. 运行 `code/main.py`。对比轮询和 LLM 选择下的对话。在每种模式下，哪个智能体占主导？
2. 在选择器中添加"每个智能体最多发言次数"规则。它对对话记录有什么影响？
3. 实现目标达成终止：当审查者返回"approved"时停止。在达到轮数上限之前，它触发的频率如何？
4. 阅读 AutoGen 稳定文档中关于 GroupChat 的部分（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。识别 `GroupChatManager` 使用的默认选择器。
5. 阅读 AG2 仓库（https://github.com/ag2ai/ag2），对比其 v0.2 GroupChat 与 v0.4 事件驱动版本。v0.4 具体新增了哪个属性（吞吐量、容错性、可组合性）？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| GroupChat（群聊） | "智能体在一个聊天室里" | 共享消息池 + 选择器函数。AutoGen / AG2 的原语。 |
| Speaker selection（发言者选择） | "谁下一个说话" | 选取下一个智能体的函数。轮询、LLM 选择或自定义。 |
| GroupChatManager（群聊管理者） | "会议主持人" | AutoGen 组件，持有选择器并循环执行各轮发言。 |
| ConversableAgent（可对话智能体） | "基础智能体" | AutoGen 基类；一个可以发送和接收消息的智能体。 |
| Termination token（终止令牌） | "停止"暗语 | 哨兵字符串（通常是 `TERMINATE`），触发后结束对话。 |
| Hot speaker（热门发言者） | "一个智能体占主导" | 选择器持续选取同一个智能体的失效模式。 |
| Context bloat（上下文膨胀） | "池无限增长" | 每个智能体读取所有先前消息；上下文随轮次增长。 |
| Projection（投影） | "范围化视图" | 共享池中角色特定的视图，用于防止上下文膨胀。 |

## 延伸阅读

- [AutoGen 群聊文档](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 参考实现
- [AG2 仓库](https://github.com/ag2ai/ag2) — 社区版 AutoGen v0.2 的延续
- [Microsoft Agent Framework 文档](https://microsoft.github.io/agent-framework/) — 合并后的继任者，RC 版于 2026 年 2 月发布
- [AutoGen v0.4 发布说明](https://microsoft.github.io/autogen/stable/) — 事件驱动 Actor 模型重写详情

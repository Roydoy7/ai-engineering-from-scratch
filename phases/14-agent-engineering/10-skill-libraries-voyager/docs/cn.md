# 技能库与终身学习（Voyager）（Skill Libraries and Lifelong Learning — Voyager）

> Voyager（Wang 等人，TMLR 2024）将可执行代码视为技能。技能是命名的、可检索的、可组合的，并通过环境反馈加以精炼。这是 Claude Agent SDK 技能、skillkit 和 2026 年技能库模式的参考架构。

**类型：** 构建  
**语言：** Python（标准库）  
**前置知识：** Phase 14 · 07（MemGPT）、Phase 14 · 08（Letta 块）  
**预计时间：** 约 75 分钟

## 学习目标

- 说出 Voyager 的三个组件——自动课程、技能库、迭代提示机制——及各自的作用。
- 解释为什么 Voyager 将动作空间设定为代码而非原始命令。
- 用标准库实现一个带有注册、检索、组合和失败驱动精炼的技能库。
- 将 Voyager 的模式映射到 2026 年的 Claude Agent SDK 技能和 skillkit 生态系统。

## 问题所在

每次会话都从头重建所有能力的智能体会犯三个错误：

1. **浪费 token。** 每个任务都重复引发相同的推理。
2. **丢失进展。** 会话 A 中学到的纠正不会传递到会话 B。
3. **在长序列组合上失败。** 复杂任务需要能力层次结构；单次提示无法表达它们。

Voyager 的答案：将每个可复用的能力视为存储在库中的一段命名代码，可通过相似性检索，可与其他技能组合，并通过执行反馈进行精炼。

## 核心概念

### 三个组件

Voyager（arXiv:2305.16291）围绕以下三者构建智能体：

1. **自动课程（Automatic curriculum）。** 一个好奇心驱动的提议者根据智能体当前的技能集和环境状态选择下一个任务。探索是自底向上的。
2. **技能库（Skill library）。** 每个技能是可执行代码。任务成功时添加新技能。技能通过查询到描述的相似性被检索。
3. **迭代提示机制（Iterative prompting mechanism）。** 失败时，智能体接收执行错误、环境反馈和自我验证输出，然后精炼技能。

Minecraft 评估（Wang 等人，2024）：相比基线，独特物品数量提升 3.3 倍，石镐速度提升 8.5 倍，铁镐速度提升 6.4 倍，地图遍历距离提升 2.3 倍。数字是 Minecraft 特定的，但模式可以迁移。

### 动作空间 = 代码

大多数智能体输出原始命令。Voyager 输出 JavaScript 函数。一个技能是：

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

由子技能组合而成。以描述和嵌入为键存储。检索为程序，而非提示。

这就是 2026 年的 Claude Agent SDK 技能：一段命名的、可检索的代码加上智能体按需加载的指令。

### 技能检索

新任务"制作钻石镐"。智能体：

1. 嵌入任务描述。
2. 查询技能库获取 top-k 相似技能。
3. 检索 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 从检索到的原语加上新逻辑组合新技能。

这就是 MCP 资源（Phase 13）和 Agent SDK 技能实现的模式：在知识/代码接口上检索，范围限定于当前任务。

### 迭代精炼

Voyager 的反馈循环：

1. 智能体编写技能。
2. 技能在环境中运行。
3. 返回三种信号之一：`success`（成功）、`error`（错误，带堆栈跟踪）、`self-verification failure`（自我验证失败）。
4. 智能体使用信号作为上下文重写技能。
5. 循环直到成功或达到最大轮次。

这是将 Self-Refine（第 05 课）应用于代码生成，并以环境接地验证。CRITIC（第 05 课）是相同的模式，以外部工具作为验证器。

### 课程与探索

Voyager 的课程模块根据智能体已有的内容和尚未完成的内容提出任务，如"在湖边建一个庇护所"。提议者使用环境状态 + 技能清单选择略高于当前能力的任务——探索的甜蜜点。

对于生产智能体，这转化为"缺少什么"操作：给定当前技能库和一个领域，我们还没有覆盖哪些技能？团队通常将此手动实现为课程审查。

### 这个模式在哪里出错

- **技能库腐烂。** 相同技能用稍微不同的描述添加了 10 次。写入时添加去重；检索只返回一个。
- **组合技能漂移。** 父技能依赖一个已被精炼的子技能。给技能添加版本控制；锁定到 v1 的父技能不会自动获取 v3。
- **检索质量。** 随着库增长超过几百个，对技能描述的向量检索会退化。用标签过滤器和硬约束补充（"只有 `category=tooling` 的技能"）。

## 构建它

`code/main.py` 实现了一个标准库技能库：

- `Skill` — 名称、描述、代码（字符串）、版本、标签、依赖项。
- `SkillLibrary` — 注册、搜索（token 重叠）、组合（依赖项的拓扑排序）和精炼（更新时版本升级）。
- 一个脚本化智能体，注册三个原语技能，组合第四个，遇到失败，然后精炼。

运行：

```
python3 code/main.py
```

追踪显示库写入、检索、组合、执行失败和 v2 精炼——Voyager 的端到端循环。

## 使用它

- **Claude Agent SDK 技能**（Anthropic）— 2026 年的参考实现：每个技能有描述、代码和指令；在智能体会话期间按需加载。
- **skillkit**（npm: skillkit）— 针对 32+ 个 AI 编程智能体的跨智能体技能管理。
- **自定义技能库** — 特定领域的（数据智能体的 SQL 技能、基础设施智能体的 Terraform 技能）。Voyager 模式可以缩小规模。
- **OpenAI Agents SDK `tools`** — 在低端；每个工具是一个轻量级技能。

## 交付它

`outputs/skill-skill-library.md` 为任何目标运行时生成一个 Voyager 形态的技能库，内置注册、检索、版本控制和精炼。

## 练习

1. 在 `compose()` 中添加依赖循环检测器。当技能 A 依赖 B 而 B 又依赖 A 时会发生什么？报错还是警告？
2. 实现每技能版本固定。当父技能组合了子技能 `crafting@1`，对 `crafting@2` 的精炼不应该静默升级父技能。
3. 将 token 重叠检索替换为 sentence-transformers 嵌入（或 BM25 标准库实现）。在 50 个技能的玩具库上测量 retrieval@5。
4. 添加"课程"智能体：给定当前库和领域描述，提议 5 个缺失的技能。每周调用一次。
5. 阅读 Anthropic 的 Claude Agent SDK 技能文档。将玩具库移植到 SDK 的技能模式。关于可发现性有什么变化？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 技能（Skill） | "可复用的能力" | 命名的代码块 + 描述，可通过相似性检索。 |
| 技能库（Skill library） | "智能体的方法论记忆" | 技能的持久存储，可搜索和组合。 |
| 课程（Curriculum） | "任务提议者" | 由当前能力差距驱动的自底向上目标生成器。 |
| 组合（Composition） | "技能 DAG" | 技能调用技能；执行时拓扑排序。 |
| 迭代精炼（Iterative refinement） | "自我纠正循环" | 环境反馈 + 错误 + 自我验证折叠回下一个版本。 |
| 动作空间即代码（Action-space-as-code） | "程序化动作" | 输出函数而非原始命令，用于时间延伸行为。 |
| 写入时去重（Dedup on write） | "技能折叠" | 近似重复的描述折叠为一个规范技能。 |

## 延伸阅读

- [Wang 等人，Voyager（arXiv:2305.16291）](https://arxiv.org/abs/2305.16291) — 原始技能库论文
- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) — 技能作为 2026 年的产品化
- [Anthropic，用 Claude Agent SDK 构建智能体](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 实践中的技能和子智能体
- [Madaan 等人，Self-Refine（arXiv:2303.17651）](https://arxiv.org/abs/2303.17651) — Voyager 底层的精炼循环

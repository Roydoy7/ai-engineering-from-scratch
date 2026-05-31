# 自主编码智能体格局（2026）（The Autonomous Coding Agent Landscape）

> SWE-bench Verified 在不到三年内从 4% 攀升至 80.9%。同款 Claude Sonnet 4.5 在 SWE-agent v1 上得分 43.2%，在 Cline 自主模式下得分 59.8%——模型周围的脚手架现在和模型本身一样重要。OpenHands（前 OpenDevin）是最活跃的 MIT 许可平台，其 CodeAct 循环直接在沙箱中执行 Python 操作，而非 JSON 工具调用。头条数字隐藏了一个方法论问题：500 个 SWE-bench Verified 任务中有 161 个只需要 1-2 行变更，而 SWE-bench Pro（10+ 行任务）对同样的前沿模型仅有 23-59%。

**类型：** 学习  
**语言：** Python（标准库，CodeAct 与 JSON 工具调用比较）  
**前置知识：** Phase 14 · 07（工具使用）、Phase 15 · 01（长时域智能体）  
**预计时间：** 约 45 分钟

## 问题所在

"哪个编码智能体最好"是错误的问题。正确的问题是：在与我的工作相匹配的任务分布上，使用我将在生产中运行的脚手架，我能得到什么端到端可靠性？

2022 年到 2026 年间，这个领域了解到脚手架——检索层、规划器、沙箱、编辑-验证循环、反馈格式——是承重的。SWE-agent v1 上的 Claude Sonnet 4.5 在 SWE-bench Verified 上得分 43.2%；在 Cline 的自主脚手架内的同一模型得分 59.8%。16.6 个绝对百分点的差异，相同的权重。基础模型是一个组件；循环才是产品。

伴随问题是基准饱和隐藏了回归。SWE-bench Verified 接近饱和，简单任务尾部（500 个任务中有 161 个只需要 ≤2 行）拉高了顶级分数。真实世界质量在 SWE-bench Pro（10+ 行变更）等分布上有更好的测量，同样的领头者在那里仍然在 23-59%。

## 核心概念

### SWE-bench，一段话

SWE-bench（Jimenez 等）接受带有真实补丁的真实 GitHub 问题，并要求智能体产生使测试套件通过的补丁。SWE-bench Verified（OpenAI，2024）是一个人类策划的 500 任务子集，删除了模糊和损坏的任务。SWE-bench Pro 是更难的后继者——需要 10+ 行变更的任务，当前前沿智能体在那里是 23-59%。

### 2022 → 2026 曲线实际显示了什么

- **2022**：研究模型在原始 SWE-bench 上约 4%。
- **2024**：GPT-4 + Devin 风格脚手架约 14%；SWE-agent 约 12%。
- **2025**：Aider 和 SWE-agent 内的 Claude 3.5/3.7 Sonnet 推入 40-55% 范围。
- **2026**：Claude Sonnet 4.5 和前沿竞争者在 SWE-bench Verified 上达到 70-80%+。Epoch AI 的排行榜实时追踪。

这个斜率来自三个复合来源：更好的基础模型、更好的脚手架（CodeAct、反思、验证器循环），以及更好的基准（Verified 删除了噪音）。

### CodeAct 与 JSON 工具调用

OpenHands（All-Hands-AI，arXiv:2407.16741，前 OpenDevin）进行了一个特定的架构押注：不是模型发出主机解码并执行的 JSON 工具调用，而是模型发出 Python 代码，Jupyter 风格的内核在沙箱中运行它。智能体可以在一个动作内循环遍历文件、链接工具和捕获自己的异常。

权衡：

- **JSON 工具调用**：每个动作是一轮；易于审计；有限的可组合性；默认安全，因为每个调用都通过显式验证器。
- **CodeAct**：一个动作可以是整个程序；可组合；需要一个强化的沙箱（OpenHands 使用 Docker 隔离）；失败模式包括沙箱运行时允许的任何事情。

两种架构都在生产中。CodeAct 在开放平台（OpenHands、smolagents）中占主导。JSON 工具调用在提供商控制执行器的托管服务（Anthropic 托管智能体、OpenAI Assistants）中仍然占主导。

### 2026 格局中的脚手架

| 脚手架 | 许可证 | 执行模型 | 显著属性 |
|--------|--------|---------|---------|
| OpenHands（OpenDevin） | MIT | Docker 中的 CodeAct | 最活跃的开放平台；事件流可重放 |
| SWE-agent | MIT | 智能体-计算机接口（ACI） | 第一个端到端 SWE-bench 脚手架 |
| Aider | Apache-2 | 在本地仓库中通过差异编辑 | 最小脚手架，强回归稳定性 |
| Cline | Apache-2 | 带工具策略的 VS Code 智能体 | Sonnet 4.5 上最高分的开放脚手架 |
| Devin（Cognition） | 专有 | 托管 VM + 规划器 | 第一个"AI 软件工程师"产品类别 |
| Claude Code | 专有 | 权限模式 + 例程 | 第 10 课详细介绍智能体循环 |

### 为什么脚手架占主导

编码运行是一个长时域轨迹（第 1 课）。可靠性在步骤间复合。三个脚手架购买分数的地方：

1. **检索**：找到正确的文件来读取是静默瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引和 Aider 的仓库地图都在攻击这个。
2. **验证器循环**：运行测试、读取堆栈跟踪和重新尝试在 SWE-bench 上有 10+ 分的差距。
3. **故障控制**：错误时回滚的沙箱防止复合损害。有验证器循环和没有验证器循环的同一模型看起来像两个不同的产品。

### 基准饱和与真实分布

OpenHands 作者和 Epoch AI 都指出 SWE-bench Verified 有一个简单尾部：500 个任务中有 161 个只需要 1-2 行变更。高分部分由这个尾部驱动。SWE-bench Pro 限制于 10+ 行变更，即使对于前沿系统，也返回 23-59% 范围的分数。你的生产分布几乎肯定更接近 Pro 而非 Verified。

选择智能体的含义：在你自己的 bug 积压的类 Pro 子集上运行。重要的分数是代表你发布内容的任务上的分数。

## 使用它

`code/main.py` 在固定的迷你任务分布上比较两个玩具智能体脚手架：

1. 每轮采取一个动作的 **JSON 工具调用**脚手架。
2. 每个动作可以发出小型 Python 片段的 **CodeAct** 脚手架。

两者都使用存根"模型"（确定性规则），因此比较将脚手架与模型质量隔离。输出显示 CodeAct 脚手架以更大的每动作爆炸半径为代价，用更少的轮次解决更多的任务。

## 交付它

`outputs/skill-scaffold-audit.md` 帮助你在采用之前审计提议的编码智能体脚手架：检索质量、验证器存在、沙箱隔离和基准到分布契合度。

## 练习

1. 运行 `code/main.py`。每个脚手架在同一任务集上需要多少轮？每个的每动作爆炸半径是多少？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。论文认为 CodeAct 在复杂任务上击败 JSON 工具调用。识别论文承认的一种失败模式，写一句话说明该模式何时会在生产中占主导。

3. 从你的 bug 积压中选一个需要跨两个文件 10+ 行变更的任务。估计前沿模型在（a）JSON 工具调用和（b）CodeAct 下的端到端成功概率。为差距辩护。

4. SWE-bench Verified 有 161 个单文件、1-2 行任务。构建一个排除它们的分数。排行榜如何重新排列？

5. 阅读"介绍 SWE-bench Verified"（OpenAI）。解释用于删除模糊任务的具体方法，并指出策划会遗漏的一个类别。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| SWE-bench | "编码基准" | 带真实补丁和测试套件的真实 GitHub 问题 |
| SWE-bench Verified | "清理后的子集" | 500 个人类策划任务，简单尾部存在 |
| SWE-bench Pro | "更难的子集" | 10+ 行变更；前沿在 23-59% |
| CodeAct | "代码即动作" | 智能体发出 Python；Jupyter 风格内核在沙箱中执行 |
| JSON tool call（JSON 工具调用） | "函数调用" | 每个动作是一个在执行前经过验证的结构化 JSON 负载 |
| Scaffold（脚手架） | "智能体框架" | 围绕基础模型的检索 + 规划器 + 执行器 + 验证器循环 |
| ACI（智能体-计算机接口） | "SWE-agent 的格式" | 为 LLM 人机工程学设计的命令集，而非人类 shell |
| Verifier loop（验证器循环） | "测试并重试" | 运行测试、读取输出、修改补丁；最大的非模型可靠性增益 |

## 延伸阅读

- [Jimenez 等 — SWE-bench](https://www.swebench.com/) — 原始基准和方法论。
- [OpenAI — 介绍 SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 策划子集是如何构建的。
- [Wang 等 — OpenHands：AI 软件开发者的开放平台](https://arxiv.org/abs/2407.16741) — CodeAct 架构和事件流设计。
- [Epoch AI — SWE-bench 排行榜](https://epoch.ai/benchmarks) — 实时追踪分数。
- [Anthropic — 测量智能体自主性](https://www.anthropic.com/research/measuring-agent-autonomy) — 长时域编码智能体可靠性框架。

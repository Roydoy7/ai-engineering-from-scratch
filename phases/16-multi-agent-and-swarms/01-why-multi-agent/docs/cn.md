# 为什么需要多智能体？（Why Multi-Agent?）

> 单个智能体撞墙了。明智的做法不是一个更大的智能体——而是更多的智能体。

**类型：** 学习  
**语言：** TypeScript  
**前置知识：** Phase 14（智能体工程）  
**预计时间：** 约 60 分钟

## 学习目标

- 识别单智能体上限（上下文溢出、混合专业技能、顺序瓶颈），并解释何时拆分为多个智能体是正确的选择
- 比较编排模式（流水线、并行扇出、监督者、层次化），并为给定的任务结构选择正确的模式
- 设计具有清晰角色边界、共享状态和通信契约的多智能体系统
- 分析多智能体复杂性（延迟、成本、调试难度）与单智能体简单性的权衡

## 问题所在

你在 Phase 14 构建了一个单智能体。它能工作。它可以读取文件、运行命令、调用 API，并对结果进行推理。然后你把它指向一个真实的代码库：200 个文件、三种语言、依赖基础设施的测试，以及在编写代码之前需要研究外部 API 的要求。

智能体卡住了。不是因为 LLM 不够聪明，而是因为任务超出了一个智能体循环所能处理的范围。上下文窗口被文件内容填满。智能体忘记了 40 次工具调用前它读到的内容。它试图同时担任研究员、编码员和审查者，结果三件事都做得很差。

这就是单智能体上限。每次任务需要以下内容时你都会遇到它：

- **超过一个窗口容量的上下文**——读取 50 个文件会超过 200k token
- **不同阶段需要不同专业技能**——研究需要与代码生成不同的提示
- **可以并行进行的工作**——为什么要顺序读取三个文件，而可以同时读取？

## 核心概念

### 单智能体上限

单智能体是一个循环、一个上下文窗口、一个系统提示。想象一下：

```
┌─────────────────────────────────────────┐
│            单智能体                      │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         上下文窗口                 │  │
│  │                                   │  │
│  │  研究笔记                          │  │
│  │  + 代码文件                        │  │
│  │  + 测试输出                        │  │
│  │  + 审查反馈                        │  │
│  │  + API 文档                        │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ 满了 ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  一个系统提示试图涵盖                    │
│  研究 + 编码 + 审查 + 测试              │
│                                         │
│  结果：每件事都做得平庸                  │
└─────────────────────────────────────────┘
```

三件事崩溃了：

1. **上下文饱和**——工具结果堆积。到第 30 轮时，智能体已经消耗了 150k token 的文件内容、命令输出和先前推理。第 5 轮的关键细节丢失了。

2. **角色混淆**——写着"你是研究员、编码员、审查者和测试员"的系统提示产生的智能体会半心半意地研究、半心半意地编码，然后永远不完成审查。

3. **顺序瓶颈**——智能体读文件 A，然后文件 B，然后文件 C。三次串行 LLM 调用。三次串行工具执行。没有并行性。

### 多智能体解决方案

拆分工作。给每个智能体一个工作、一个上下文窗口和一个为该工作调优的系统提示：

```
┌──────────────────────────────────────────────────────────┐
│                       编排者                              │
│                                                          │
│  "为用户管理构建一个 REST API"                             │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │ 研究员   │ │ 编码员   │ │ 审查者   │ │ 测试员   │  │
│   │          │ │          │ │          │ │          │  │
│   │ 读取文档 │ │ 根据研究 │ │ 检查代码 │ │ 运行测试 │  │
│   │ 查找模式 │ │ + 规格   │ │ 质量     │ │ 报告结果 │  │
│   │          │ │ 编写代码 │ │ 发现 bug │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                       合并结果                            │
└──────────────────────────────────────────────────────────┘
```

每个智能体有：
- 聚焦的系统提示（"你是代码审查者。你唯一的工作是发现 bug。"）
- 自己的上下文窗口（不被其他智能体的工作污染）
- 清晰的输入/输出契约（接收研究笔记，输出代码）

### 实际这样做的系统

**Claude Code 子智能体**——当 Claude Code 用 `Task` 生成子智能体时，它创建了一个带范围任务的子智能体。父智能体保持上下文干净。子智能体做聚焦的工作并返回摘要。

**Devin**——运行规划器智能体、编码员智能体和浏览器智能体。规划器将工作分解为步骤。编码员编写代码。浏览器研究文档。每个都有独立的上下文。

**多智能体编码团队（SWE-bench）**——SWE-bench 上表现最好的系统使用读取代码库的研究员、设计修复方案的规划器和实现方案的编码员。单智能体系统得分更低。

**ChatGPT 深度研究**——并行生成多个搜索智能体，每个探索不同角度，然后综合结果。

### 多智能体的类型谱

多智能体不是非此即彼的。它是一个类型谱：

```
简单 ──────────────────────────────────────── 复杂

 单智能体    子智能体       流水线      团队        群体

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │共享    │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │状态    │
             └───┘          └───┘───┘  │  消息  │    └───────┘
                                       │  总线  │
 1个循环     父+            阶段式      │       │    N个对等体
 1个上下文   子任务         处理        └───────┘    涌现行为
                                       明确角色
```

**单智能体**——一个循环，一个提示。适用于简单任务。

**子智能体**——父智能体生成子智能体处理聚焦子任务。父智能体维护计划。子智能体汇报回来。这是 Claude Code 的做法。

**流水线**——智能体按序运行。智能体 A 的输出成为智能体 B 的输入。适用于阶段性工作流：研究 -> 编码 -> 审查 -> 测试。

**团队**——智能体在共享消息总线上并行运行。每个都有角色。编排者协调。适用于同时需要不同技能的情况。

**群体**——许多相同或近似相同的智能体，具有共享状态。没有固定编排者。智能体从队列中获取工作。适用于高吞吐量的并行任务。

### 四种多智能体模式

#### 模式 1：流水线

```
输入 ──▶ 智能体 A ──▶ 智能体 B ──▶ 智能体 C ──▶ 输出
          (研究)        (编码)        (审查)
```

每个智能体转换数据并向前传递。易于推理。一个阶段的失败会阻塞其余阶段。

#### 模式 2：扇出/扇入

```
                ┌──▶ 智能体 A ──┐
                │              │
输入 ──▶ 分割 ├──▶ 智能体 B ──├──▶ 合并 ──▶ 输出
                │              │
                └──▶ 智能体 C ──┘
```

将工作分散到并行智能体，然后合并结果。适用于可分解为独立子任务的任务。

#### 模式 3：编排者-工作者

```
                    ┌──────────┐
                    │  编排者  │
                    └──┬───┬───┘
                  任务 │   │ 任务
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ 工作者 A │   │ 工作者 B │
           └──────────┘   └──────────┘
```

聪明的编排者决定做什么，委托给工作者，然后综合结果。编排者本身是一个智能体，其工具包括生成工作者。

#### 模式 4：对等群体

```
         ┌───┐ ◄──── 消息 ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      消息 │    ┌───────────┐     │ 消息
           └───▶│  共享状态  │◄────┘
                │  / 队列   │
           ┌───▶│           │◄────┐
           │    └───────────┘     │
      消息 │                      │ 消息
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── 消息 ────▶ │ D │
         └───┘                  └───┘
```

没有中央编排者。智能体点对点通信。决策从交互中涌现。更难调试，但可以扩展到许多智能体。

### 何时不使用多智能体

多智能体增加复杂性。智能体之间的每条消息都是潜在的失败点。调试从"读一个对话"变成"跨五个智能体追踪消息"。

**保持单智能体的情况：**
- 任务适合一个上下文窗口（工作数据低于约 100k token）
- 不同阶段不需要不同的系统提示
- 顺序执行足够快
- 任务足够简单，拆分它增加的开销多于价值

**复杂性成本：**
- 每个智能体边界都是一个有损压缩步骤：智能体 A 的完整上下文被总结为发给智能体 B 的消息
- 协调逻辑（谁做什么，何时，按什么顺序）本身就是 bug 的来源
- 延迟增加：N 个智能体意味着最少 N 次串行 LLM 调用，如果需要来回沟通则更多
- 成本倍增：每个智能体独立消耗 token

经验法则：如果一个任务需要少于 20 次工具调用且在 100k token 内，保持单智能体。

## 构建它

### 步骤 1：过载的单智能体

这是一个试图做所有事情的单智能体。它有一个庞大的系统提示和一个容纳研究、代码和审查的上下文窗口：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方法的问题：
- 上下文窗口随每个阶段增长。到审查步骤时，它包含研究笔记和代码及先前推理。
- 系统提示是通用的。它无法针对每个阶段调优。
- 没有任何东西并行运行。

### 步骤 2：专业智能体

现在拆分它。每个智能体获得一个工作：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个专家都有聚焦的提示。每个都得到一个干净的上下文窗口，只包含它需要的输入。

### 步骤 3：通过消息协调

用明确的消息传递将专家连接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个智能体只接收发给它的消息。没有上下文污染。研究员读取 50k token 文档的内容永远不会进入审查者的上下文。

### 步骤 4：比较

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== 单智能体 ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== 多智能体 ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

多智能体版本使用更多总 token（三个智能体，三次独立 LLM 调用），但每个智能体的上下文保持干净。每个阶段的质量提高，因为系统提示是专业化的。

## 使用它

本课产生一个用于决定何时使用多智能体的可复用提示。参见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 添加第四个专家："测试员"智能体，接收编码员的代码和审查者的反馈，然后编写测试
2. 修改流水线，使审查者可以将反馈发回给编码员进行修订循环（最多 2 轮）
3. 将顺序流水线转换为扇出：并行运行研究员和"需求分析员"智能体，然后在传给编码员之前合并它们的输出

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Swarm（群体） | "AI 智能体的蜂巢思维" | 具有共享状态且没有固定领导者的对等智能体集合。行为从局部交互中涌现。 |
| Orchestrator（编排者） | "老板智能体" | 其工具包括生成和管理其他智能体的智能体。它计划和委托，但可能不做实际工作。 |
| Coordinator（协调者） | "交通警察" | 根据规则在智能体之间路由消息的非智能体组件（通常只是代码，不是 LLM）。 |
| Consensus（共识） | "智能体达成一致" | 多个智能体在继续之前必须达成协议的协议。在需要解决冲突输出时使用。 |
| Emergent behavior（涌现行为） | "智能体自己想出来的" | 从智能体交互中出现但未被显式编程的系统级模式。可以是有用的或有害的。 |
| Fan-out/fan-in（扇出/扇入） | "智能体的 Map-Reduce" | 将任务分散到并行智能体（扇出），然后合并它们的结果（扇入）。 |
| Message passing（消息传递） | "智能体互相交谈" | 智能体之间的通信机制：从一个智能体发送到另一个的结构化数据，替代共享上下文窗口。 |

## 延伸阅读

- [新兴 AI 智能体架构的格局](https://arxiv.org/abs/2409.02977) — 多智能体模式调查
- [AutoGen：启用下一代 LLM 应用](https://arxiv.org/abs/2308.08155) — Microsoft 的多智能体对话框架
- [Claude Code 子智能体文档](https://docs.anthropic.com/en/docs/claude-code) — Claude Code 如何用 Task 委托
- [CrewAI 文档](https://docs.crewai.com/) — 基于角色的多智能体框架

# CrewAI：基于角色的团队与工作流（CrewAI: Role-Based Crews and Flows）

> CrewAI 是 2026 年基于角色的多智能体框架。四个基本元素：Agent、Task、Crew、Process。两种顶层形态：Crew（自主、基于角色的协作）与 Flow（事件驱动、确定性）。文档直言不讳："对于任何生产就绪的应用，从 Flow 开始。"

**类型：** 学习 + 构建  
**语言：** Python（标准库）  
**前置知识：** Phase 14 · 12（工作流模式）、Phase 14 · 14（行为者模型）  
**预计时间：** 约 75 分钟

## 学习目标

- 说出 CrewAI 的四个基本元素（Agent、Task、Crew、Process）及各自负责什么。
- 区分 Sequential、Hierarchical 和计划中的 Consensus 流程；针对不同工作负载选择合适的一种。
- 区分 Crew（自主、基于角色）与 Flow（事件驱动、确定性），并解释文档的生产建议。
- 用 `@tool` 装饰器和 `BaseTool` 子类连接工具；理解结构化输出与自由文本的权衡。
- 说出 CrewAI 的四种记忆类型及各自的适用场景。
- 实现一个标准库三智能体团队（研究者、写作者、编辑者），输出一篇简报。
- 识别三种 CrewAI 失败模式：提示词膨胀、管理 LLM 税、脆弱的交接。

## 问题所在

团队采用多智能体框架时往往撞上同一堵墙。"自主协作"在演示中听起来很美好。然后客户提交了一个 bug，你需要确定性的重放。或者财务部门问每次 LLM 路由的团队运行要花多少钱。或者值班人员需要知道哪个智能体在凌晨 3 点卡住了。

自由形式的 LLM 路由团队对以上问题都没有清晰的答案。纯 DAG 能回答所有问题，但失去了头脑风暴智能体需要的探索性形态。

CrewAI 的分拆对权衡是诚实的。Crew 用于协作性、基于角色的、探索性工作。Flow 用于事件驱动、代码控制、可审计的生产场景。同一框架，两种形态，按场景选择。

## 核心概念

### 四个基本元素

CrewAI 的接口很小。记住这些，其余的就是配置。

- **Agent（智能体）。** `role + goal + backstory + tools + (可选) llm`。背景故事（backstory）是有分量的。它塑造语气、判断力，以及智能体何时停止。工具是智能体可以调用的函数（详见下文）。
- **Task（任务）。** `description + expected_output + agent + (可选) context + (可选) output_pydantic`。一个可复用的工作单元。`expected_output` 是契约。`context` 列出上游任务，其输出将被传入。`output_pydantic` 强制指定结构化形状。
- **Crew（团队）。** 容器。拥有 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory`、`verbose`、`manager_llm` 设置。
- **Process（流程）。** 执行策略。Sequential（顺序）、Hierarchical（层次）、Consensus（共识，计划中）。决定运行的形态。

智能体之间无法直接看到彼此。任务引用智能体。Crew 对任务排序。Process 决定谁选择下一个任务。这就是完整的心智模型。

> **基于** CrewAI 0.86（2026-05）验证。新版本可能重命名或合并流程类型；使用特定形态前请查阅 [CrewAI Processes 文档](https://docs.crewai.com/concepts/processes)。

### Sequential vs Hierarchical vs Consensus

- **Sequential（顺序）。** 任务按声明顺序运行。任务 N 的输出可作为 `context` 传给任务 N+1。成本最低。最可预测。当顺序固定时使用。
- **Hierarchical（层次）。** 一个管理 Agent（单独的 LLM 调用）在专家之间路由。CrewAI 根据你的 `manager_llm` 配置或默认值生成管理者。管理者每轮选择下一个任务，可以拒绝或重新路由。当你有四个或更多专家，且顺序确实取决于前一个输出时使用。
- **Consensus（共识）。** 计划中，当前公共 API 中尚未实现。文档为基于投票的未来流程保留了这个名称。今天不要依赖它。

Hierarchical 在每次专家调用之上增加一次每轮 LLM 调用（管理者）。在五步运行中，token 成本可能翻三倍。只有在真正需要路由时才付出这个代价。

### Crew vs Flow

这是 2026 年文档的核心框架。

- **Crew（团队）。** LLM 驱动的自主性。框架在运行时决定形态。适合：研究、头脑风暴、初稿，以及路径本身就是答案一部分的场景。难以重放。难以测试。原型开发成本低。
- **Flow（工作流）。** 你控制的事件驱动图。`@start` 标记入口。`@listen(topic)` 标记当另一个步骤发出该主题时触发的步骤。每个步骤是纯 Python（内部可以调用 Crew）。适合：生产。可观测。可测试。确定性。

文档的 2026 年生产建议：从 Flow 开始。当自主性物有所值时，从 Flow 步骤内部调用 `Crew.kickoff()` 将 Crew 折叠进来。Flow 提供审计跟踪，Crew 提供探索能力。组合使用，而非二选一。

### 工具集成

给 Agent 添加工具有三种方式。选择最简单的适合方案。

1. **`@tool` 装饰器。** 纯函数变为工具。函数签名是 schema；文档字符串是 LLM 看到的描述。最适合一次性辅助函数。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 子类。** 基于类的工具，具有明确的参数 schema、异步支持、重试机制。当工具有状态（客户端、缓存）或需要结构化参数时使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置工具包。** CrewAI 附带第一方适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一行导入即可连接。

结构化输出使用 Pydantic。在 Task 上传入 `output_pydantic=MyModel`。CrewAI 会验证 LLM 响应是否符合模型，并进行强制转换或重试。与紧凑的 `expected_output` 字符串配合使用。自由文本输出适合草稿；结构化输出才是下游 Flow 能消费的内容。

### 记忆钩子

CrewAI 内置四种记忆类型。它们可以组合：一个 Crew 可以同时启用所有四种。

> **基于** CrewAI 0.86（2026-05）验证。最近的版本将所有内容路由到一个统一的 `Memory` 系统，该系统包装了这四种存储。下面的概念模型仍然成立，但公共类接口在新版本中可能收缩为单一的 `Memory` 入口点；请查阅 [CrewAI memory 文档](https://docs.crewai.com/concepts/memory) 获取当前 API。

- **短期记忆（Short-term）。** 单次运行内的对话缓冲区。运行结束时清空。
- **长期记忆（Long-term）。** 跨运行持久化。存储在向量数据库中（默认 Chroma，可替换）。通过与当前任务的相似性检索。
- **实体记忆（Entity）。** 每个实体的事实。"客户 X 是企业计划用户。"以实体为键，而非相似性。跨运行保留。
- **上下文记忆（Contextual）。** 组装时检索。在 Agent 需要的那一刻拉取相关记忆，而非预加载。

在 Crew 上用 `memory=True` 或按类型配置来启用。由你配置的嵌入提供商支持（默认 OpenAI，可替换为本地）。记忆是 CrewAI 相对于更精简框架的优势之一；纯 LangGraph 需要你自己连接每一种。

### CrewAI 的适用场景

- 三到六个具有命名角色和协作工作流的智能体。起草、审查、规划、头脑风暴。
- LLM 对下一步的判断本身就是价值所在的路由（Hierarchical）。
- 团队更愿意阅读 `role + goal + backstory` 而非图定义的场景。

### CrewAI 的不适用场景

- 具有严格顺序的确定性 DAG。使用 LangGraph（第 13 课）。图的形态才是正确的抽象；CrewAI 的角色框架在这里是摩擦力。
- 亚秒延迟预算。Hierarchical 增加往返次数。即使 Sequential 也会串行化包含背景故事和先前输出的提示词。
- 单智能体循环。跳过框架；智能体循环（第 01 课）加上工具注册表更简洁。

第 17 课（智能体框架权衡）以矩阵形式展示了这些内容。简短版本：CrewAI 处于"协作式角色"的象限。

### 依赖形态

独立于 LangChain。Python 3.10 到 3.13。使用 `uv`。Star 数量请查阅 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（快照截至 2026-05）。AWS Bedrock 集成已有文档；供应商基准测试报告在 QA 工作负载上比 LangGraph 有显著加速，但方法论（数据集、硬件、评估指标）未公开，因此框架供应商的数字仅供方向性参考。

### 这个模式在哪里出错

- **背景故事导致的提示词膨胀。** 每个 Agent 有 2000 词的背景故事，五个 Agent 的团队在第一次工具调用前就耗尽了上下文预算。将背景故事控制在 200 词以内。跨 Agent 复用短语；不要重复五次行文风格。
- **管理 LLM 税。** Hierarchical 流程在每次专家调用前增加一次管理 LLM 调用。在五任务团队中，这是六次 LLM 调用而非五次，而且管理调用携带完整任务列表和先前输出。除非路由依赖于输出，否则切换到 Sequential。
- **脆弱的交接。** 任务 N 的 `expected_output` 是"一份大纲"。任务 N+1 将其作为 `context` 读入，并尝试解析三个部分。LLM 生成了四个。下游 Agent 即兴发挥。解决方法：在任务 N 上使用 `output_pydantic`，使任务 N+1 读取类型化对象，而非自由文本。
- **Crew 直接上生产。** 没有 Flow 包装就将自由形式的 Crew 发布到生产。输出变异性高；重放不可能；值班人员无法对比坏运行和好运行的差异。用 Flow 包装。

## 构建它

`code/main.py` 实现了两种形态的标准库版本，以及一个三智能体团队。

形态：

- `Agent`、`Task` 数据类，与 CrewAI 的接口相匹配。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行任务，将输出作为 `context` 穿线传递。
- `HierarchicalCrew.kickoff(topic)` 增加一个管理 Agent，每轮选择下一个专家，在"完成"时停止。
- `Flow`，带有 `@start` 和 `@listen(topic)` 装饰器、一个小型事件循环和追踪。
- `tool(name)` 装饰器，镜像 CrewAI 的 `@tool` 形状。
- `Memory`，带有 `short_term`、`long_term`、`entity` 存储；模拟相似性使用 numpy。
- 模拟 LLM 响应是以角色加输入前缀为键的硬编码字符串。无网络。确定性。

具体演示：研究者、写作者、编辑者团队，生成一篇关于"2026 年智能体工程"的简报。研究者拉取（模拟的）来源。写作者起草。编辑者精炼。同一团队通过 Flow 运行，展示确定性形态。

运行：

```bash
python3 code/main.py
```

追踪包含：顺序团队通过 `context` 穿线传递输出、层次团队的管理者选择（研究者、写作者、编辑者，然后"完成"）、使用明确主题（`researched`、`drafted`、`edited`）运行同三步的 flow、通过 `@tool` 路由的工具调用，以及跨两次 kickoff 存活的长期记忆。

Crew 追踪是流动的；管理者原则上可以重新排序。Flow 追踪是固定的。这个选择就是本课的要点。

## 使用它

- **CrewAI Flow** 用于生产。即使 Flow 只有一步调用 `Crew.kickoff()`。Flow 提供审计边界。
- **CrewAI Crew（Sequential）** 用于顺序清晰的协作工作，尤其是初稿和审查循环。
- **CrewAI Crew（Hierarchical）** 用于路由取决于输出且有四个或更多专家的场景。
- **LangGraph**（第 13 课）用于明确的状态机、持久恢复、严格排序。
- **AutoGen v0.4**（第 14 课）用于行为者模型并发和故障隔离。
- **OpenAI Agents SDK**（第 16 课）用于具有交接和护栏的 OpenAI 优先产品。
- **Claude Agent SDK**（第 17 课）用于具有子智能体和会话存储的 Claude 优先产品。

## 交付它

`outputs/skill-crew-or-flow.md` 为任务选择 Crew 还是 Flow，并搭建最小化实现脚手架。对以下情况硬拒绝：没有背景故事的 Crew、没有明确主题的 Flow、专家少于三个的 Hierarchical。

## 陷阱

- **背景故事当调味品用。** 它塑造输出。每个 Agent 测试三个变体；差异是真实存在的。选定一个，冻结它。
- **跳过 `expected_output`。** 没有每任务契约，下游任务会接收 LLM 生产的任何内容。团队运行；审计失败。
- **记忆总是开启。** 长期记忆每次运行都写入。向量数据库增长。检索变得嘈杂。将写入范围限定在事实需要持久化的任务上。
- **管理者提示词漂移。** Hierarchical 的管理者提示词是隐式的。如果路由变得奇怪，在详细模式下转储它并阅读。
- **Crew 工具中的副作用。** Crew 调用工具的次数可能超出预期。POST、DELETE、支付操作属于 Flow 步骤，绝不能放在 Crew 工具中。

## 练习

1. 将 Sequential 团队转换为 Flow。统计变异性下降的接触点。记录可读性下降的地方。
2. 向团队添加实体记忆：关于客户的事实跨 kickoff 持久化。验证检索拉取到正确的实体。
3. 实现一个 Hierarchical 流程，管理者拒绝路由到编辑者，直到写作者的输出至少有三段。追踪重试过程。
4. 为（模拟的）网络搜索连接一个 `BaseTool` 子类。与 `@tool` 装饰器版本对比追踪形态。
5. 在编辑任务上添加 `output_pydantic=Brief`，其中 `Brief` 有 `title`、`summary`、`sections`。让写作任务输出一次格式错误的 JSON；在追踪中验证 CrewAI 的重试行为。
6. 阅读 CrewAI 的文档介绍。将玩具移植到真实的 `crewai` API。标准库版本跳过了哪些保证？
7. 将 AgentOps 或 Langfuse（第 24 课）连接到真实运行。标准库版本遗漏了哪些追踪？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Agent（智能体） | "角色" | role + goal + backstory + tools |
| Task（任务） | "工作单元" | description + expected output + assignee + 可选结构化输出 |
| Crew（团队） | "智能体团队" | Agent + Task + Process 的容器 |
| Process（流程） | "执行策略" | Sequential / Hierarchical / Consensus（计划中） |
| Flow（工作流） | "确定性工作流" | 事件驱动、代码控制、可测试 |
| Backstory（背景故事） | "角色提示词" | 塑造 Agent 语气和判断力的内容 |
| `@tool` | "函数工具" | 将函数变为 Agent 可调用工具的装饰器 |
| `BaseTool` | "类工具" | 带参数 schema、重试、异步支持的基于类的工具 |
| Entity memory（实体记忆） | "每实体事实" | 以客户/账户/问题为范围的记忆 |
| Long-term memory（长期记忆） | "跨运行记忆" | 在 kickoff 之间存活的向量支持记忆 |
| Contextual memory（上下文记忆） | "即时检索" | 在 Agent 需要时拉取的记忆 |
| Manager LLM（管理 LLM） | "路由智能体" | Hierarchical 流程中选择下一个任务的额外 LLM |
| `expected_output` | "任务契约" | 告知 Agent（和审计）要返回什么形状的字符串 |

## 延伸阅读

- [CrewAI 文档介绍](https://docs.crewai.com/en/introduction)：概念和推荐的生产路径
- [CrewAI Flows 指南](https://docs.crewai.com/en/concepts/flows)：事件驱动形态，`@start`，`@listen`
- [CrewAI 工具参考](https://docs.crewai.com/en/concepts/tools)：`@tool`、`BaseTool`、内置工具包
- [CrewAI 记忆](https://docs.crewai.com/en/concepts/memory)：短期、长期、实体、上下文
- [Anthropic，构建有效智能体](https://www.anthropic.com/research/building-effective-agents)：多智能体何时有帮助，何时没有
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview)：状态机替代方案

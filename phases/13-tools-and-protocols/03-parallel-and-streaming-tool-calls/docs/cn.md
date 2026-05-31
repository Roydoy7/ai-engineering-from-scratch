# 并行工具调用与流式工具调用（Parallel Tool Calls and Streaming with Tools）

> 串行执行三次独立天气查询意味着三次来回往返。并行运行，总耗时缩减为最慢那次单次调用的时长。如今所有前沿提供商都支持在单次对话轮次中输出多个工具调用。收益是真实的，但管道的细节颇为微妙。本章分两部分讲解：并行扇出与流式参数重组，重点剖析 ID 关联这个陷阱。

**类型：** 构建  
**语言：** Python（标准库，线程池 + 流式处理框架）  
**前置知识：** Phase 13 · 02（函数调用深度解析）  
**预计时间：** 约 75 分钟

## 学习目标

- 解释 `parallel_tool_calls: true` 存在的原因，以及何时应将其禁用。
- 在并行扇出期间，将流式参数块关联到正确的工具调用 ID。
- 将不完整的 `arguments` 字符串重组为完整的 JSON，而不提前解析。
- 运行三城市天气基准测试，验证串行与并行的延迟差异。

## 问题所在

不使用并行调用时，智能体回答"班加罗尔、东京和苏黎世的天气如何"需要经历：

```
user -> LLM
LLM -> call get_weather(班加罗尔)
host -> run executor, reply with result
LLM -> call get_weather(东京)
host -> run executor, reply with result
LLM -> call get_weather(苏黎世)
host -> run executor, reply with result
LLM -> final text answer
```

三次 LLM 来回往返，每次还需承担执行器延迟。挂钟时间大约是理想情况的 4 倍。

使用并行调用后：

```
user -> LLM
LLM -> call get_weather(班加罗尔); call get_weather(东京); call get_weather(苏黎世)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

只需一次 LLM 往返。执行器时间取三者最大值，而非三者之和。OpenAI、Anthropic 和 Gemini 的生产基准显示，扇出工作负载的挂钟时间可减少 60% 到 70%。

代价是关联的复杂性。当三个调用以乱序完成时，结果必须携带匹配的 `tool_call_id`，以便模型能对齐它们。当结果以流式传输时，必须将部分参数片段拼装成完整的 JSON 才能执行。Gemini 3 添加唯一 ID 的部分原因，正是为了解决一个实际问题：对同一工具发出的两个并行调用无法区分。

## 核心概念

### 启用并行

- **OpenAI。** `parallel_tool_calls: true` 默认开启。设为 `false` 可强制串行执行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false` 启用并行（在 Claude 3.5 及以上版本中为默认值）。设为 `true` 可强制串行。
- **Gemini。** 始终支持并行；`tool_config.function_calling_config.mode = "AUTO"` 让模型自主决定。

在以下情况下应禁用并行：工具之间存在顺序依赖（如先 `create_file` 再 `write_file`）、一个调用的输出作为另一个的输入、或速率限制器无法承受扇出。

### ID 关联

模型发出的每个调用都有一个 `id`。宿主返回的每个结果必须包含相同的 id。没有这个机制，结果就会产生歧义。

- **OpenAI。** 每条 tool 角色消息上的 `tool_call_id`。
- **Anthropic。** 每个 `tool_result` 块上的 `tool_use_id`。
- **Gemini。** 每个 `functionResponse` 上的 `id`（Gemini 3 及以上；Gemini 2 按名称匹配，对同名并行调用会出错）。

### 并发执行调用

宿主在各自的线程、协程或远程 worker 上运行每个调用的执行器。最简单的框架使用线程池；生产环境使用 `asyncio.gather` 或结构化并发。完成顺序不可预测——ID 才是标识符。

一个常见错误是：按调用列表顺序而非完成顺序返回结果。由于模型只关心 `tool_call_id`，这通常不会出错，但如果某个结果被丢弃或重复，乱序提交会让调试更加困难。推荐按完成顺序返回结果，并附带明确的 ID。

### 流式工具调用

当模型流式输出时，`arguments` 分片到达。三个并行调用的三条独立流的块在传输时交织在一起。每个 ID 需要一个独立的累加器。

各提供商的结构：

- **OpenAI。** 每个块为 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。块携带 `index`（调用列表中的位置）。按索引累积，首次出现时读取 `id`，在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** 流事件依次为：`message_start`，然后为每个类型为 `tool_use` 的块发一个 `content_block_start`（包含 id、名称、空 input）。`content_block_delta` 事件携带 `input_json_delta` 块。`content_block_stop` 关闭每个块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）输出带有 `functionCallId` 的块，使调用可以干净地交错。Gemini 3 之前，流式传输每次返回一个完整调用。

### 部分 JSON 与提前解析陷阱

在 `arguments` 完整之前不能解析它。像 `{"city": "Beng` 这样的部分 JSON 是无效的，会抛出异常。正确的时机是提供商的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop`，或 Gemini 的流结束事件。只有到那时才应尝试 `json.loads`。更健壮的方案是使用增量 JSON 解析器，在结构完成时逐步产出事件；OpenAI 的流式传输指南为显示实时"思考中"指示器的 UX 推荐了这种方式。用花括号计数来判断完整性是不可靠的（引号字符串或转义内容中的花括号会导致误判），只能作为非正式调试辅助手段。

### 乱序完成

```
call_A: 快速 API，最先返回
call_B: 慢速 API，最后返回
call_C: 中速 API，第二个返回
```

宿主的回复仍然必须引用 ID：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

对于 OpenAI 或 Anthropic，回复中的顺序不影响正确性。Gemini 接受任意顺序，只要 ID 匹配即可。

### 基准测试：串行 vs 并行

`code/main.py` 中的框架模拟了三个分别具有 400、600 和 800 ms 延迟的执行器。串行运行总耗时 1800 ms。并行运行耗时 max(400, 600, 800) = 800 ms。差距是固定的，不是比例性的，因此工具数量越多，节省越多。

实际注意事项：并行调用会给下游 API 带来压力。对受速率限制的服务进行 10 路扇出将会失败。Phase 13 · 17 涵盖网关级背压；重试语义计划在未来的章节中介绍。

### 流式扇出挂钟时间

如果模型本身是流式传输的，可以在一个调用的参数完整后立即开始执行，而无需等待所有调用结束。这是 OpenAI 记录在案但并非所有 SDK 都公开的优化。本章的框架正是这样做的：一旦模拟流产出完整的参数对象，宿主立即启动该调用。

## 动手使用

`code/main.py` 分为两部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor` 串行和并行运行三个模拟天气调用，并打印挂钟时间。第二部分重放一个假的流式响应——三个并行调用的 `arguments` 块在同一流上交织——并通过 `StreamAccumulator` 按 ID 重组它们。无需 LLM，无需网络，只有重组逻辑。

要关注的内容：

- 串行计时器达到 1.8 秒。并行计时器在相同的假延迟下达到 0.8 秒。
- 累加器通过按 ID 缓冲来处理乱序到达的块，只有当每个调用的 JSON 完整时才解析。
- 执行器在某个 ID 的参数确定后立即启动，而不是等所有流结束。

## 输出产物

本章生成 `outputs/skill-parallel-call-safety-check.md`。给定工具注册表，该技能审核哪些工具可以安全并行化，哪些有顺序依赖，哪些会使下游速率限制过载——返回附有每个工具 `parallel_safe` 标志的修订版注册表。

## 练习

1. 运行 `code/main.py` 并改变模拟延迟。确认并行与串行的比率约为 `max/sum`（实际运行由于线程调度、序列化和框架开销，会与理想值略有偏差）。在什么延迟分布下，并行不再有意义？

2. 扩展累加器，通过丢弃缓冲区并发出 `cancelled` 事件来处理"调用在流中途被取消"的情况。哪个提供商明确记录了这种情况？检查 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。

3. 用 `asyncio.gather` 替换线程池。对两者进行基准测试。如果执行器执行真实的 I/O，你应该看到异步版本有小幅提升，原因是上下文切换开销更低。

4. 选择两个不应并行化的工具（例如先 `create_file` 再 `write_file`）。在注册表中添加 `ordering_dependency` 图，并根据该图对并行扇出设置门控。这是依赖感知调度的最小机制，未来的智能体工程章节会将其正式化。

5. 阅读 OpenAI 的并行函数调用章节和 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 建议禁用并行的一种真实工具类型。（提示：对同一资源的后果性变更操作。）

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 并行工具调用（Parallel tool calls） | "单轮扇出" | 模型在单条助手消息中发出多个工具调用。 |
| `parallel_tool_calls` | "OpenAI 的标志" | 启用或禁用多调用发射。 |
| `disable_parallel_tool_use` | "Anthropic 的反向标志" | 退出标志；默认启用并行。 |
| 工具调用 ID（Tool call id） | "关联句柄" | 每个调用的标识符，结果消息必须回传该标识符。 |
| 累加器（Accumulator） | "流缓冲区" | 每个 ID 的字符串缓冲区，用于存放部分 `arguments` 块。 |
| 乱序完成（Out-of-order completion） | "最快的先完成" | 并行调用以不可预测的顺序完成；ID 是粘合剂。 |
| 依赖图（Dependency graph） | "顺序约束" | 其输出作为其他工具输入的工具；不能并行化。 |
| 提前解析陷阱（Parse-early trap） | "JSON 解析爆炸" | 尝试解析不完整的 `arguments` 字符串。 |
| `streamFunctionCallArguments` | "Gemini 3 特性" | 每个调用带唯一 ID 的流式参数块。 |
| 按完成顺序回复（Completion-order reply） | "无需等待全部" | 结果到达即回复，以 ID 为键。 |

## 延伸阅读

- [OpenAI — 并行函数调用](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为与退出标志
- [Anthropic — 工具使用：实现工具使用](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 与结果批处理
- [Google — Gemini 函数调用并行章节](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 起的 ID 关联并行调用
- [OpenAI — 带工具的流式响应](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 流的分块参数重组
- [Anthropic — 流式消息](https://docs.anthropic.com/en/api/messages-streaming) — 带 `input_json_delta` 的 `content_block_delta`

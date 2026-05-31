# MCP 采样——服务器请求 LLM 补全与智能体循环（MCP Sampling — Server-Requested LLM Completions and Agent Loops）

> 大多数 MCP 服务器是哑执行器：接受参数，运行代码，返回内容。采样让服务器可以反转方向：它请求客户端的 LLM 做出决策。这使得服务器可以托管智能体循环，而无需拥有任何模型凭据。2025-11-25 合并的 SEP-1577 在采样请求中添加了工具，使循环可以包含更深层次的推理。漂移风险注意：SEP-1577 的工具内采样形状在 2026 年 Q1 是实验性的，SDK API 中仍在稳定中。

**类型：** 构建  
**语言：** Python（标准库，采样框架）  
**前置知识：** Phase 13 · 07（MCP 服务器）、Phase 13 · 10（资源与提示）  
**预计时间：** 约 75 分钟

## 学习目标

- 解释 `sampling/createMessage` 解决了什么问题（无服务器端 API 密钥的服务器托管循环）。
- 实现一个服务器，要求客户端对多轮提示采样并返回补全结果。
- 使用 `modelPreferences`（成本/速度/智能优先级）来指导客户端的模型选择。
- 构建一个 `summarize_repo` 工具，通过采样内部迭代，而不是硬编码行为。

## 问题所在

一个用于代码摘要工作流的有用 MCP 服务器需要：遍历文件树，选择要读取的文件，综合摘要，并返回。LLM 推理在哪里发生？

选项 A：服务器调用自己的 LLM。需要 API 密钥，在服务器端计费，每用户成本高昂。

选项 B：服务器返回原始内容；客户端的智能体做推理。可以工作，但将服务器逻辑移入了客户端提示，这很脆弱。

选项 C：服务器通过 `sampling/createMessage` 请求客户端的 LLM。服务器保留算法（读取哪些文件，做多少轮）而客户端保留计费和模型选择。服务器根本不需要凭据。

采样就是选项 C。这是受信任的服务器无需自身成为完整 LLM 宿主即可托管智能体循环的机制。

## 核心概念

### `sampling/createMessage` 请求

服务器发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

客户端运行其 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个总和为 1.0 的浮点数：

- `costPriority`：倾向于更便宜的模型。
- `speedPriority`：倾向于更快的模型。
- `intelligencePriority`：倾向于更有能力的模型。

加上 `hints`：服务器偏好的命名模型。客户端可能遵守也可能不遵守提示；客户端的用户配置始终优先。

### `includeContext`

三个值：

- `"none"` —— 只使用服务器提供的消息。默认值。
- `"thisServer"` —— 包含来自此服务器会话的先前消息。
- `"allServers"` —— 包含所有会话上下文。

`includeContext` 从 2025-11-25 起被软弃用，因为它会泄漏跨服务器上下文，这是一个安全问题。更倾向于使用 `"none"` 并在消息中传递显式上下文。

### 带工具的采样（SEP-1577）

2025-11-25 新增：采样请求可以包含 `tools` 数组。客户端使用这些工具运行完整的工具调用循环。这使服务器可以通过客户端的模型托管 ReAct 风格的智能体循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

客户端循环：采样，如果调用则执行工具，再次采样，返回最终助手消息。这在 2026 年 Q1 是实验性的；SDK 签名可能仍会漂移。实现时请对照 2025-11-25 规范的 `client/sampling` 章节确认。

### 人在回路

客户端在运行采样之前，必须向用户显示服务器在请求模型做什么。恶意服务器可以利用采样操纵用户的会话（"对用户说 X，让他们点击 Y"）。Claude Desktop、VS Code 和 Cursor 将采样请求呈现为用户可以拒绝的确认对话框。

2026 年的共识：没有人工确认的采样是一个红旗。网关（Phase 13 · 17）可以自动批准低风险采样，并自动拒绝任何可疑内容。

### 无 API 密钥的服务器托管循环

典型用例：一个没有自己 LLM 访问权限的代码摘要 MCP 服务器。它做：

1. 遍历仓库结构。
2. 调用 `sampling/createMessage`，消息为"选择最有可能描述此仓库目的的五个文件。"
3. 读取这些文件。
4. 调用 `sampling/createMessage`，携带文件内容，消息为"用 3 段话总结这个仓库。"
5. 将摘要作为 `tools/call` 结果返回。

服务器从不接触 LLM API。客户端的用户使用自己的凭据为补全付费。

### 安全风险（Unit 42 披露，2026 年 Q1）

- **隐蔽采样。** 总是用"从会话上下文中响应用户的电子邮件"调用采样的工具。Phase 13 · 15 涵盖攻击向量。
- **通过采样窃取资源。** 服务器要求客户端总结攻击者的有效载荷，向用户收费。
- **循环炸弹。** 服务器在紧密循环中调用采样。客户端必须强制执行每会话速率限制。

## 动手使用

`code/main.py` 提供了一个假的服务器到客户端采样框架。一个模拟的"summarize_repo"工具调用两轮采样（选择文件，然后摘要），假客户端返回预制的响应。框架展示：

- 服务器发送带 `modelPreferences` 的 `sampling/createMessage`。
- 客户端返回补全结果。
- 服务器继续其循环。
- 速率限制器限制每次工具调用的总采样次数。

要关注的内容：

- 服务器只暴露一个工具（`summarize_repo`）；所有推理都在采样调用中发生。
- 模型偏好权重影响客户端的模型选择；提示列出了首选模型。
- 循环在 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` 限制捕获失控的循环。

## 输出产物

本章生成 `outputs/skill-sampling-loop-designer.md`。给定需要 LLM 调用的服务器端算法（研究、摘要、规划），该技能设计基于采样的实现，包含正确的 `modelPreferences`、速率限制和安全确认。

## 练习

1. 运行 `code/main.py`。将 `max_samples_per_tool` 改为 2，观察速率限制截断。

2. 实现 SEP-1577 工具内采样变体：采样请求携带 `tools` 数组。验证客户端循环在返回最终补全前执行这些工具。注意漂移风险：SDK 签名在 2026 年上半年可能仍会变化。

3. 添加人在回路确认：在服务器第一次 `sampling/createMessage` 之前，暂停并等待用户批准。被拒绝的调用返回类型化的拒绝。

4. 添加以客户端会话为键的每用户速率限制器。同一用户的同一服务器循环应共享一个预算。

5. 设计一个使用采样选择要包含的块的 `summarize_pdf` 工具。草拟发送的消息。`modelPreferences.intelligencePriority` 在 0.1 与 0.9 时如何改变行为？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 采样（Sampling） | "服务器到客户端的 LLM 调用" | 服务器向客户端的模型请求补全。 |
| `sampling/createMessage` | "该方法" | 用于采样请求的 JSON-RPC 方法。 |
| `modelPreferences` | "模型优先级" | 成本/速度/智能权重加上名称提示。 |
| `includeContext` | "跨会话泄漏" | 软弃用的上下文包含模式。 |
| SEP-1577 | "采样中的工具" | 允许在采样内使用工具，用于服务器托管的 ReAct。 |
| 人在回路（Human-in-the-loop） | "用户确认" | 客户端在运行前向用户呈现采样请求。 |
| 循环炸弹（Loop bomb） | "失控的采样" | 服务器端无限采样循环；客户端必须速率限制。 |
| 隐蔽采样（Covert sampling） | "隐藏的推理" | 恶意服务器在采样提示中隐藏意图。 |
| 资源窃取（Resource theft） | "使用用户的 LLM 预算" | 服务器迫使客户端在其不想要的采样上花费。 |
| `stopReason` | "生成停止的原因" | `endTurn`、`stopSequence` 或 `maxTokens`。 |

## 延伸阅读

- [MCP — 概念：采样](https://modelcontextprotocol.io/docs/concepts/sampling) — 采样的高级概述
- [MCP — 客户端采样规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — 规范的 `sampling/createMessage` 形状
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — 采样中工具的规范演进提案（实验性）
- [Unit 42 — MCP 攻击向量](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — 隐蔽采样和资源窃取模式
- [Speakeasy — MCP 采样核心概念](https://www.speakeasy.com/mcp/core-concepts/sampling) — 带客户端代码示例的演练

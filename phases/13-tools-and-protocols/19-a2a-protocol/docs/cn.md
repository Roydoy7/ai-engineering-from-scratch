# A2A——智能体间协议（A2A — Agent-to-Agent Protocol）

> MCP 是智能体到工具的协议。A2A（Agent2Agent）是智能体到智能体的协议——一个允许基于不同框架构建的不透明智能体相互协作的开放协议。由谷歌于 2025 年 4 月发布，同年 6 月捐赠给 Linux 基金会，2026 年 4 月发布 v1.0，拥有包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow 在内的 150 多个支持者。它吸收了 IBM 的 ACP，并添加了 AP2 支付扩展。本章讲解智能体卡（Agent Card）、任务生命周期和两种传输绑定。

**类型：** 构建  
**语言：** Python（标准库，智能体卡 + 任务框架）  
**前置知识：** Phase 13 · 06（MCP 基础）、Phase 13 · 08（MCP 客户端）  
**预计时间：** 约 75 分钟

## 学习目标

- 区分智能体到工具（MCP）和智能体到智能体（A2A）的使用场景。
- 在 `/.well-known/agent.json` 发布包含技能和端点元数据的智能体卡。
- 逐步讲解任务生命周期（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用带有部件（text、file、data）的消息和作为输出的制品（Artifact）。

## 问题所在

一个客服智能体需要将报告撰写委托给专门的写作智能体。A2A 出现前的选项：

- **自定义 REST API。** 可行，但每对智能体之间都需要单独定制。
- **共享代码库。** 要求两个智能体运行相同的框架。
- **MCP。** 不适合：MCP 用于调用工具，不适合两个智能体协作同时保留各自不透明的内部推理。

A2A 填补了这一空白。它将交互建模为一个智能体向另一个智能体发送任务，附带生命周期、消息和制品。被调用智能体的内部状态保持不透明——调用方只能看到任务状态转换和最终输出。

A2A 是"让不同框架的智能体相互通信"的协议。它不取代 MCP；两者是互补关系。

## 核心概念

### 智能体卡（Agent Card）

每个符合 A2A 标准的智能体都在 `/.well-known/agent.json` 发布一张卡：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现方式基于 URL：获取卡片，得知 A2A 端点的 URL，枚举技能。

### 签名智能体卡（AP2）

AP2 扩展（2025 年 9 月）为智能体卡添加了密码学签名。发布者用 JWT 签名自己的卡；消费者进行验证。防止冒充。

### 任务生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working（通过消息循环）
```

客户端通过 `tasks/send` 发起任务。被调用的智能体在各状态间转换；客户端通过 SSE 订阅状态更新或轮询。

### 消息与部件（Messages and Parts）

一条消息携带一个或多个部件：

- `text` — 纯文本内容。
- `file` — 带 mimeType 的 base64 字节块。
- `data` — 类型化 JSON 载荷（被调用智能体的结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### 制品（Artifacts）

输出是制品，而非原始字符串。制品是一个命名的类型化输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

制品可以分块流式传输。调用方负责累积。

### 两种传输绑定

1. **JSON-RPC over HTTP。** `/a2a` 端点，请求用 POST，流式传输可选 SSE。默认绑定。
2. **gRPC。** 用于 gRPC 原生的企业环境。

两种绑定承载相同的逻辑消息结构。

### 不透明性保护

一个关键设计原则：被调用智能体的内部状态是不透明的。调用方只能看到任务状态和制品。被调用智能体的思维链、工具调用、子智能体委托——全部不可见。这与 MCP 不同，后者的工具调用是透明的。

设计理由：A2A 允许竞争者在不暴露内部实现的情况下进行协作。A2A 可以是"调用这个客服智能体"，而调用方不需要了解该智能体如何实现这项服务。

### 时间线

- **2025-04-09。** 谷歌宣布 A2A。
- **2025-06-23。** 捐赠给 Linux 基金会。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 扩展（智能体支付）发布。
- **2026-04。** v1.0 发布，支持组织超过 150 个。

### 与 MCP 的关系

| 维度 | MCP | A2A |
|------|-----|-----|
| 使用场景 | 智能体到工具 | 智能体到智能体 |
| 不透明性 | 透明的工具调用 | 不透明的内部推理 |
| 典型调用方 | 智能体运行时 | 另一个智能体 |
| 状态 | 工具调用结果 | 带生命周期的任务 |
| 授权 | OAuth 2.1（Phase 13 · 16） | JWT 签名的智能体卡（AP2） |
| 传输 | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

当你想调用特定工具时使用 MCP。当你想将整个任务委托给另一个智能体时使用 A2A。许多生产系统同时使用两者：智能体用 MCP 构建工具层，用 A2A 构建协作层。

## 动手使用

`code/main.py` 实现了一个最小化 A2A 框架：一个研究智能体发布其卡片，一个写作智能体接收包含 PDF 和文本指令部件的 `tasks/send`，经历 working → input_required → working → completed 状态转换，并返回一个文本制品。全部使用标准库；使用内存传输以便专注于消息结构。

要关注的内容：

- 智能体卡的 JSON 结构。
- 任务 ID 分配和状态转换。
- 包含混合类型部件的消息。
- 任务中途的 input-required 分支。
- 完成时返回的制品。

## 输出产物

本章生成 `outputs/skill-a2a-agent-spec.md`。给定一个应该可被其他智能体调用的新智能体，该技能生成智能体卡 JSON、技能模式和端点蓝图。

## 练习

1. 运行 `code/main.py`。追踪完整的任务生命周期，包括被调用智能体请求澄清时的 input-required 暂停。

2. 添加签名智能体卡。用 HMAC 对卡片的规范 JSON 进行签名。编写验证器并确认它在卡片被篡改时失败。

3. 实现任务流式传输：写作智能体通过 SSE 发出三个增量制品块，调用方累积它们。

4. 设计一个封装 MCP 服务器的 A2A 智能体。将每个 MCP 工具映射到一个 A2A 技能。注意取舍——失去了什么不透明性？

5. 阅读 A2A v1.0 公告，找出截至 2026 年 4 月尚未被任何框架实现的一个功能。（提示：与多跳任务委托有关。）

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| A2A | "智能体间协议" | 用于不透明智能体协作的开放协议。 |
| 智能体卡（Agent Card） | "`.well-known/agent.json`" | 描述智能体技能和端点的已发布元数据。 |
| 技能（Skill） | "可调用单元" | 智能体支持的命名操作（类似于 MCP 工具）。 |
| 任务（Task） | "委托单元" | 带有生命周期和最终制品的工作项。 |
| 消息（Message） | "任务输入" | 携带部件（text、file、data）。 |
| 部件（Part） | "类型化块" | 消息中的 `text` / `file` / `data` 元素。 |
| 制品（Artifact） | "任务输出" | 完成时返回的命名类型化输出。 |
| AP2 | "智能体支付协议" | 用于信任和支付的签名智能体卡扩展。 |
| 不透明性（Opacity） | "黑盒协作" | 被调用智能体的内部对调用方不可见。 |
| input-required | "任务暂停" | 智能体需要更多信息时的生命周期状态。 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — 规范的 A2A 规范文档
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [Linux Foundation — A2A 发布新闻稿](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025 年 6 月治理转移
- [Google Cloud — A2A 协议升级](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 路线图和合作伙伴动态
- [Google Dev — A2A 1.0 里程碑](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 发布说明和向后兼容指南

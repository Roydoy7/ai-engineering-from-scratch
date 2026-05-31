# A2A——智能体间协议（A2A — The Agent-to-Agent Protocol）

> 谷歌于 2025 年 4 月发布 A2A；截至 2026 年 4 月，规范文档位于 https://a2a-protocol.org/latest/specification/，已有 150 多个组织支持。A2A 是 MCP（第 13 课）的水平补充：MCP 是垂直方向的（智能体 ↔ 工具），A2A 是点对点的（智能体 ↔ 智能体）。它定义了智能体卡（用于发现）、带有产物的任务（文本、结构化数据、视频）、不透明的任务生命周期和身份验证。生产系统越来越多地将 MCP 与 A2A 配对使用。谷歌云在 2025-2026 年间将 A2A 支持集成进了 Vertex AI Agent Builder。

**类型：** 学习 + 构建  
**语言：** Python（标准库，`http.server`，`json`）  
**前置知识：** Phase 16 · 04（原语模型）  
**预计时间：** 约 75 分钟

## 问题所在

你的智能体需要调用另一个系统上的另一个智能体。怎么做？你可以暴露一个 HTTP 端点，定义一套定制的 JSON Schema，然后寄望于对方也遵循同样的约定。这样一来，每一对智能体之间都变成了一次定制集成。

A2A 就是这种调用的通用有线协议。标准的发现机制，标准的任务模型，标准的传输方式，标准的产物类型。就像 HTTP+REST，只不过将智能体作为一等公民。

## 核心概念

### 四个要素

**智能体卡（Agent Card）。** 位于 `/.well-known/agent.json` 的 JSON 文档，描述该智能体的名称、技能、端点、支持的模态和身份验证要求。发现通过读取这张卡来完成。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**任务（Task）。** 工作的基本单元。一个异步的、有状态的对象，具有生命周期：`submitted → working → completed / failed / canceled`。客户端提交任务，然后轮询或订阅更新。

**产物（Artifact）。** 任务产出的结果类型。文本、结构化 JSON、图像、视频、音频。产物有类型，使不同的模态都成为一等公民。

**不透明的生命周期（Opaque lifecycle）。** A2A 不规定远端智能体*如何*解决任务。客户端看到状态转换和产物；实现方可以自由选择任何框架。

### MCP/A2A 的分工

- **MCP**（第 13 课）：智能体 ↔ 工具。智能体通过 JSON-RPC 向工具服务器读写数据。默认无状态。
- **A2A**：智能体 ↔ 智能体。对等协议；双方都是拥有独立推理能力的智能体。

生产级多智能体系统两者兼用。A2A 对等方在自己这侧调用 MCP 工具。这种分工让两个关注点保持清晰。

### 发现流程

```
客户端                       智能体服务器
  ├──GET /.well-known/agent.json──>
  <──智能体卡 JSON────────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}─────────────>
  <──state=working, 42% 完成──────
  ├──GET /tasks/{id}─────────────>
  <──state=completed, artifacts──
```

或使用流式传输：订阅 `/tasks/{id}/events` 的 SSE，以获取推送更新。

### 身份验证

A2A 支持三种常见模式：

- **Bearer 令牌** — OAuth2 或不透明令牌。
- **mTLS** — 双向 TLS；组织之间相互证明身份。
- **签名请求** — 对载荷进行 HMAC 签名。

认证方式在智能体卡中声明；客户端在发现后遵循。

### 截至 2026 年 4 月，150 多个组织支持

企业采用推动了 A2A 的规模化。核心结论：A2A 成为企业智能体系统跨越信任边界的方式。谷歌云在 Vertex AI Agent Builder 中发布了 A2A 支持；Microsoft Agent Framework 支持它；大多数主流框架（LangGraph、CrewAI、AutoGen）都发布了 A2A 适配器。

### A2A 的优势场景

- **跨组织调用。** 公司 A 的智能体调用公司 B 的智能体。没有 A2A，每一对都是定制契约。
- **异构框架。** LangGraph 智能体调用 CrewAI 智能体调用自定义 Python 智能体。A2A 进行规范化。
- **类型化产物。** 视频结果、结构化 JSON、音频——全部是一等公民。
- **长时间运行的任务。** 不透明的生命周期 + 轮询使持续数小时的任务变得直接。

### A2A 的局限场景

- **对延迟敏感的微调用。** A2A 的生命周期是异步的。亚毫秒级的智能体间通信不适合；请使用直接 RPC。
- **紧密耦合的同进程智能体。** 如果两个智能体在同一个 Python 进程中运行，A2A 的 HTTP 往返就是杀鸡用牛刀。
- **小型团队。** 规范的开销是真实存在的；仅限内部的智能体可能不需要如此正式。

### A2A vs ACP、ANP、NLIP

2024-2026 年间出现了几个相关规范：

- **ACP**（IBM/Linux 基金会）——A2A 的前身，范围较窄。
- **ANP**（Agent Network Protocol）——以对等发现为重，优先考虑去中心化。
- **NLIP**（Ecma 自然语言交互协议，2025 年 12 月标准化）——自然语言内容类型。

截至 2026 年 4 月，A2A 是采用最广泛的对等协议。请参见 arXiv:2505.02279（Liu 等人，《智能体互操作协议综述》）进行对比。

## 构建它

`code/main.py` 使用 `http.server` 和 JSON 实现了一个最小化的 A2A 服务器和客户端。服务器：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理任务状态，
- 在 `GET /tasks/{id}` 上返回产物。

客户端：

- 获取智能体卡，
- 提交任务，
- 轮询直到完成，
- 读取产物。

运行：

```
python3 code/main.py
```

脚本在后台线程中启动服务器，然后运行客户端对其进行调用。你可以看到完整流程：发现、提交、轮询、产物。

## 使用它

`outputs/skill-a2a-integrator.md` 设计一次 A2A 集成：智能体卡内容、任务 Schema、认证选择、流式传输 vs 轮询。

## 交付它

检查清单：

- **固定规范版本。** A2A 仍在演进；智能体卡应声明协议版本。
- **幂等的任务创建。** 重复提交（网络重试）应只产生一个任务。
- **产物 Schema。** 声明智能体返回的数据结构；消费者应进行验证。
- **速率限制 + 身份验证。** A2A 是面向公众的；应用标准 Web 安全措施。
- **失败任务的死信队列。** 对历史模式进行检查，识别反复出现的失败类型。

## 练习

1. 运行 `code/main.py`。确认客户端发现了服务器并收到了正确的产物。
2. 在服务器上添加第二个技能（例如，"summarize"）。更新智能体卡。编写一个客户端，根据任务类型选择技能。
3. 实现 SSE 流式传输端点：`/tasks/{id}/events`，用于发出状态变化事件。客户端需要做哪些不同的处理？
4. 阅读 A2A 规范（https://a2a-protocol.org/latest/specification/）。识别三个规范强制要求但本演示未实现的内容。
5. 对比 A2A（智能体卡发现）和 MCP（通过 `listTools` 进行服务器端能力列举）。自描述智能体与能力探测之间的权衡是什么？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| A2A | "智能体间" | 用于智能体跨系统调用其他智能体的对等协议。谷歌，2025 年。 |
| Agent Card（智能体卡） | "智能体的名片" | 位于 `/.well-known/agent.json` 的 JSON，描述技能、端点、认证方式。 |
| Task（任务） | "工作的基本单元" | 带有生命周期的异步有状态对象；完成时产生产物。 |
| Artifact（产物） | "结果" | 类型化的输出：文本、结构化 JSON、图像、视频、音频。一等媒体类型。 |
| Opaque lifecycle（不透明的生命周期） | "如何解决是智能体自己的事" | 客户端看到状态转换；服务器可以自由选择框架/工具。 |
| Discovery（发现） | "找到智能体" | `GET /.well-known/agent.json` 返回智能体卡。 |
| MCP vs A2A | "工具 vs 对等方" | MCP：垂直方向的智能体 ↔ 工具。A2A：水平方向的智能体 ↔ 智能体。 |
| ACP / ANP / NLIP | "兄弟协议" | 相邻规范；A2A 是 2026 年采用最广泛的。 |

## 延伸阅读

- [A2A 规范](https://a2a-protocol.org/latest/specification/) — 权威规范文档
- [谷歌开发者博客 — A2A 发布公告](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025 年 4 月发布文章
- [A2A GitHub 仓库](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [Liu 等人 — 智能体互操作协议综述](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A、ANP 对比

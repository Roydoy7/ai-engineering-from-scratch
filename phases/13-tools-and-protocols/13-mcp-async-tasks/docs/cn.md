# 异步任务（SEP-1686）——立即调用，稍后获取长时间运行的工作（Async Tasks (SEP-1686) — Call-Now, Fetch-Later for Long-Running Work）

> 真实的智能体工作需要分钟到小时：CI 运行、深度研究综合、批量导出。同步工具调用会断开连接、超时或阻塞 UI。2025-11-25 合并的 SEP-1686 添加了一个任务原语：任何请求都可以被增强为任务，结果可以稍后获取或通过状态通知流式传输。漂移风险注意：任务在 2026 年上半年是实验性的；SDK 接口仍在围绕规范进行设计。

**类型：** 构建  
**语言：** Python（标准库，异步任务状态机）  
**前置知识：** Phase 13 · 07（MCP 服务器）、Phase 13 · 09（传输）  
**预计时间：** 约 75 分钟

## 学习目标

- 识别何时将工具从同步提升为任务增强型（服务器端工作超过 30 秒）。
- 逐步讲解任务生命周期：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化任务状态，以便崩溃不会丢失进行中的工作。
- 正确轮询 `tasks/status` 并获取 `tasks/result`。

## 问题所在

一个 `generate_report` 工具运行一个多分钟的提取流水线。在同步模型下的选项：

1. 保持连接打开三分钟。远程传输会断开；客户端会超时；UI 会冻结。
2. 立即返回占位符；要求客户端轮询自定义端点。破坏了 MCP 的统一性。
3. 即发即忘；没有结果。

这些都不好。SEP-1686 添加了第四种：任务增强。任何请求（通常是 `tools/call`）都可以被标记为任务。服务器立即返回一个任务 ID。客户端轮询 `tasks/status` 并在完成后获取 `tasks/result`。服务器端状态在重启后仍然存在。

## 核心概念

### 任务增强

通过设置 `params._meta.task.required: true`（或 `optional: true`，由服务器决定），请求变成任务。服务器立即响应：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是服务器保留状态的承诺；超过 ttl 后任务结果被丢弃。

### 每工具选择加入

工具注解可以声明任务支持：

- `taskSupport: "forbidden"` —— 此工具始终同步运行。适用于快速工具。
- `taskSupport: "optional"` —— 客户端可以请求任务增强。
- `taskSupport: "required"` —— 客户端必须使用任务增强。

`generate_report` 工具应该是 `required`。`notes_search` 工具应该是 `forbidden`。

### 状态

```
working  -> input_required -> working  （通过询问循环）
working  -> completed
working  -> failed
working  -> cancelled
```

状态机是追加式的：一旦 `completed`、`failed` 或 `cancelled`，任务就是终态。

### 方法

- `tasks/status {taskId}` —— 返回当前状态和进度提示。
- `tasks/result {taskId}` —— 阻塞或在未完成时返回 404。
- `tasks/cancel {taskId}` —— 幂等的；终态忽略它。
- `tasks/list` —— 可选；枚举活跃和最近完成的任务。

### 流式状态变更

当服务器支持时，客户端可以订阅状态通知：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

流式而非轮询的客户端获得更好的 UX。轮询始终作为最小表面被支持。

### 持久状态

规范要求声明了任务支持的服务器持久化状态。崩溃不应在 ttl 内丢失已完成的结果。存储范围从 SQLite 到 Redis 到文件系统。第 13 课的框架使用文件系统。

### 取消语义

`tasks/cancel` 是幂等的。如果任务正在执行中，服务器尝试停止（检查执行器协作式取消）。如果已经是终态，请求是无操作的。

### 崩溃恢复

当服务器进程重启时：

1. 加载所有持久化的任务状态。
2. 将进程已死亡的所有 `working` 任务标记为 `failed`，错误为 `CRASH_RECOVERY`。
3. 在其 ttl 内保留 `completed` / `failed` / `cancelled`。

### 异步任务加采样

任务本身可以调用 `sampling/createMessage`。这就是长时间运行的研究任务的工作方式：服务器的任务线程根据需要对客户端的模型采样，而客户端的 UI 将任务显示为 `working` 并定期更新进度。

### 为什么这是实验性的

SEP-1686 在 2025-11-25 发布，但更广泛的路线图指出了三个开放问题：持久订阅原语、子任务（父子任务关系）和结果 TTL 标准化。预计规范将在 2026 年持续演进。生产代码应将任务视为仅在常见情况下稳定，并对子任务的未来 SDK 变化做好防护。

## 动手使用

`code/main.py` 实现了一个持久任务存储（文件系统支持）和一个在后台线程中运行的 `generate_report` 工具。客户端调用工具，立即获得任务 ID，在工作器更新进度时轮询 `tasks/status`，并在完成后获取 `tasks/result`。取消有效；崩溃恢复通过杀死工作器线程并重新加载状态来模拟。

要关注的内容：

- 任务状态 JSON 持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- 工作器线程更新 `progress` 字段；轮询显示它在推进。
- 来自客户端的取消设置一个事件；工作器检查并提前退出。
- "崩溃"时的状态重新加载将进行中的任务标记为 `failed`，原因为 `CRASH_RECOVERY`。

## 输出产物

本章生成 `outputs/skill-task-store-designer.md`。给定长时间运行的工具（研究、构建、导出），该技能设计任务存储（状态形状、ttl、持久性），选择正确的 `taskSupport` 标志，并草拟进度通知。

## 练习

1. 运行 `code/main.py`。启动一个 `generate_report` 任务，轮询状态，然后获取结果。

2. 在运行中途添加 `tasks/cancel` 调用。验证工作器遵守它，并且状态变为 `cancelled`。

3. 模拟崩溃恢复：杀死工作器线程，重启加载器，观察 `CRASH_RECOVERY` 失败模式。

4. 将存储扩展到 SQLite。持久性优势相同；查询选项打开（列出会话 X 中的所有任务）。

5. 阅读 MCP 2026 年路线图文章。找出最可能在未来一年影响 SDK API 设计的任务相关开放问题。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 任务（Task） | "长时间运行的工具调用" | 带 `_meta.task` 增强的异步执行请求。 |
| SEP-1686 | "任务规范" | 在 2025-11-25 中添加任务的规范演进提案。 |
| `_meta.task` | "任务信封" | 包含 id、状态、ttl 的每请求元数据。 |
| taskSupport | "工具标志" | 每个工具的 `forbidden` / `optional` / `required`。 |
| `tasks/status` | "轮询方法" | 获取当前状态和可选的进度提示。 |
| `tasks/result` | "获取结果" | 返回已完成的有效载荷，或在未完成时返回 404。 |
| `tasks/cancel` | "停止它" | 幂等的取消请求。 |
| ttl | "保留预算" | 服务器承诺保留任务状态的毫秒数。 |
| `notifications/tasks/updated` | "状态推送" | 服务器发起的状态变更事件。 |
| 持久存储（Durable store） | "崩溃安全状态" | 文件系统/SQLite/Redis 持久层。 |

## 延伸阅读

- [MCP — GitHub SEP-1686 议题](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 原始提案和完整讨论
- [WorkOS — MCP 异步任务用于 AI 智能体工作流](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 带理由的设计演练
- [DeepWiki — MCP 任务系统和异步操作](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — 机制和状态机
- [FastMCP — 任务](https://gofastmcp.com/servers/tasks) — SDK 层面的任务实现模式
- [MCP 博客 — 2026 年路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 开放问题和 2026 年优先事项，包括子任务

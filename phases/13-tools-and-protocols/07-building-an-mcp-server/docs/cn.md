# 构建 MCP 服务器——Python + TypeScript SDK（Building an MCP Server — Python + TypeScript SDKs）

> 大多数 MCP 教程只展示 stdio hello-world。一个真实的服务器暴露工具 + 资源 + 提示，处理能力协商，发出结构化错误，并在各 SDK 中保持一致的行为。本章端到端构建一个笔记服务器：标准库 stdio 传输、JSON-RPC 调度、三个服务器端原语，以及一种纯函数风格，可以直接移植到 Python SDK 的 FastMCP 或 TypeScript SDK。

**类型：** 构建  
**语言：** Python（标准库，stdio MCP 服务器）  
**前置知识：** Phase 13 · 06（MCP 基础）  
**预计时间：** 约 75 分钟

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个从 stdin 读取 JSON-RPC 消息并向 stdout 写入响应的调度循环。
- 按照 JSON-RPC 2.0 规范和 MCP 的附加代码发出结构化错误响应。
- 将标准库实现迁移到 FastMCP（Python SDK）或 TypeScript SDK，无需重写工具逻辑。

## 问题所在

在使用远程传输（Phase 13 · 09）或身份验证层（Phase 13 · 16）之前，你需要一个干净的本地服务器。本地意味着 stdio：服务器由客户端作为子进程生成，消息通过 stdin/stdout 以换行符分隔流动。

2025-11-25 规范规定 stdio 消息编码为带显式 `\n` 分隔符的 JSON 对象。这里没有 SSE；SSE 是旧的远程模式，将在 2026 年中期被移除（Atlassian 的 Rovo MCP 服务器于 2026 年 6 月 30 日弃用了它；Keboola 于 2026 年 4 月 1 日弃用）。对于 stdio，每行一个 JSON 对象就是完整的 wire 格式。

笔记服务器是个好的示例形状，因为它涵盖了所有三个服务器端原语。工具执行变更（`notes_create`）。资源暴露数据（`notes://{id}`）。提示提供模板（`review_note`）。本章的形状可以推广到任何领域。

## 核心概念

### 调度循环

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向 stdout 打印任何非 JSON-RPC 信封的内容。调试日志输出到 stderr。
- 每个请求必须有一个携带相同 `id` 的响应与之匹配。
- 通知绝对不能被响应。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你支持的能力。客户端依赖能力集来决定功能开关。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，每个条目包含 `name`、`description`、`inputSchema`。`tools/call` 接收 `{name, arguments}` 并返回 `{content: [blocks], isError: bool}`。

内容块是类型化的，最常见的有：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误有两种形式。协议级错误（未知方法、参数错误）是 JSON-RPC 错误。工具级错误（调用有效但工具失败）以 `{content: [...], isError: true}` 的形式返回。这让模型能在上下文中看到失败信息。

### 实现资源

资源在设计上是只读的。`resources/list` 返回清单；`resources/read` 返回内容。URI 可以是 `file://...`、`http://...`，或自定义方案如 `notes://`。

当你将数据作为资源而非工具暴露时：

- 模型不会"调用"它；客户端可以按用户请求将其注入上下文。
- 订阅让服务器在资源变更时推送更新（Phase 13 · 10）。
- Phase 13 · 14 用 `ui://` 扩展了这一点，支持交互式资源。

### 实现提示

提示是带命名参数的模板。宿主将其以斜杠命令的形式呈现。一个 `review_note` 提示可能接受 `note_id` 参数，并生成客户端传递给其模型的多消息提示模板。

### Stdio 传输的细节

- 换行符分隔的 JSON。没有长度前缀帧。
- 不要缓冲。每次写入后执行 `sys.stdout.flush()`。
- 客户端控制生命周期。当 stdin 关闭（EOF）时，干净地退出。
- 不要静默处理 SIGPIPE；记录日志并退出。

### 注解

每个工具可以携带描述安全属性的 `annotations`：

- `readOnlyHint: true` —— 纯读取，可安全重试。
- `destructiveHint: true` —— 不可逆副作用；客户端应确认。
- `idempotentHint: true` —— 相同输入产生相同输出。
- `openWorldHint: true` —— 与外部系统交互。

客户端使用这些来决定 UX（确认对话框、状态指示器）和路由（Phase 13 · 17）。

### 迁移路径

`code/main.py` 中的标准库服务器约 180 行。FastMCP（Python）将同样的逻辑压缩为装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 有等效的形状。当你准备好时，迁移路径是即插即用的；概念（能力、调度、内容块）是相同的。

## 动手使用

`code/main.py` 是一个完整的仅用标准库实现的 stdio 笔记 MCP 服务器。它处理 `initialize`、`tools/list`、三个工具的 `tools/call`（`notes_list`、`notes_search`、`notes_create`）、每个笔记的 `resources/list` 和 `resources/read`，以及一个 `review_note` 提示。你可以通过管道传入 JSON-RPC 消息来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

要关注的内容：

- 调度器是以方法名为键的 `dict[str, Callable]`。
- 每个工具执行器返回内容块列表，而非裸字符串。
- 当执行器抛出异常时，设置 `isError: true`。

## 输出产物

本章生成 `outputs/skill-mcp-server-scaffolder.md`。给定一个领域（笔记、工单、文件、数据库），该技能搭建一个具有正确工具/资源/提示拆分和 SDK 迁移路径的 MCP 服务器脚手架。

## 练习

1. 运行 `code/main.py` 并用手工构建的 JSON-RPC 消息驱动它。执行 `notes_create`，然后执行 `resources/read` 检索新笔记。

2. 添加一个带 `annotations: {destructiveHint: true}` 的 `notes_delete` 工具。验证客户端会弹出确认对话框（这需要真实宿主；Claude Desktop 可以使用）。

3. 实现 `resources/subscribe`，使服务器在笔记被修改时推送 `notifications/resources/updated`。添加一个保活任务。

4. 将服务器移植到 FastMCP。Python 文件应缩减到 80 行以内。Wire 行为必须相同；用同样的 JSON-RPC 测试框架验证。

5. 阅读规范的 `server/tools` 章节，找出本章服务器未实现的工具定义的一个字段。（提示：有好几个；选一个添加进去。）

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| MCP 服务器（MCP server） | "暴露工具的那个东西" | 通过 stdio 或 HTTP 发送 MCP JSON-RPC 的进程。 |
| stdio 传输（stdio transport） | "子进程模型" | 服务器由客户端生成；通过 stdin/stdout 通信。 |
| 调度器（Dispatcher） | "方法路由器" | JSON-RPC 方法名到处理函数的映射。 |
| 内容块（Content block） | "工具结果块" | 工具响应 `content` 数组中的类型化元素。 |
| `isError` | "工具级失败" | 表示工具失败；区别于 JSON-RPC 错误。 |
| 注解（Annotations） | "安全提示" | readOnly/destructive/idempotent/openWorld 标志。 |
| FastMCP | "Python SDK" | 基于装饰器的 MCP 协议高层框架。 |
| 资源 URI（Resource URI） | "可寻址数据" | `file://`、`db://` 或标识资源的自定义方案。 |
| 提示模板（Prompt template） | "斜杠命令简报" | 服务器提供的带参数槽位的模板，供宿主 UI 使用。 |
| 能力声明（Capability declaration） | "功能开关" | 在 `initialize` 中声明的每个原语的标志。 |

## 延伸阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 参考 Python 实现
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — 对应的 TypeScript 实现
- [FastMCP — 服务器框架](https://gofastmcp.com/) — MCP 服务器的装饰器风格 Python API
- [MCP — 快速入门服务器指南](https://modelcontextprotocol.io/quickstart/server) — 使用任一 SDK 的端到端教程
- [MCP — 服务器工具规范](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — `tools/*` 消息的完整参考

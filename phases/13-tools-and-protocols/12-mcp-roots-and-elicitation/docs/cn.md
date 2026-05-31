# 根与询问——作用域限定与飞行中的用户输入（Roots and Elicitation — Scoping and Mid-Flight User Input）

> 硬编码路径在用户打开不同项目的那一刻就会失效。预填充的工具参数在用户描述不完整时也会失效。根将服务器限定在用户控制的 URI 集合中；询问在工具调用中途暂停，通过表单或 URL 向用户请求结构化输入。两个客户端端原语，修复两种常见 MCP 失效模式。SEP-1036（URL 模式询问，2025-11-25）在 2026 年上半年是实验性的——在依赖它之前请检查 SDK 版本。

**类型：** 构建  
**语言：** Python（标准库，根 + 询问演示）  
**前置知识：** Phase 13 · 07（MCP 服务器）  
**预计时间：** 约 45 分钟

## 学习目标

- 声明 `roots` 并响应 `notifications/roots/list_changed`。
- 将服务器文件操作限制在声明的根集合内的 URI。
- 使用 `elicitation/create` 在工具调用中途向用户请求确认或结构化输入。
- 在表单模式和 URL 模式询问之间做出选择（后者是实验性的；已注明漂移风险）。

## 问题所在

一个笔记 MCP 服务器在生产中会遇到的两个具体失效。

**路径假设失效。** 服务器是针对 `~/notes` 编写的。一个将笔记存放在 `~/Documents/Notes` 的不同机器上的用户，会遇到工具调用静默失败（找不到文件），或者更糟，写入了错误的位置。

**用户知道但缺少的参数。** 用户说"删除那个旧的 TPS 报告笔记"。模型调用 `notes_delete(title: "TPS report")`，但有三个来自 2023、2024 和 2025 年的匹配笔记。工具无法猜测。报错"有歧义"很烦人；对三个都执行是灾难性的。

根修复了第一个：客户端在 `initialize` 时声明服务器可以访问的 URI 集合。询问修复了第二个：服务器暂停工具调用并发送 `elicitation/create`，请用户选择哪一个。

## 核心概念

### 根

客户端在 `initialize` 时声明根列表：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

服务器随后可以调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

服务器必须将根视为边界：任何在根集合之外的文件读或写都被拒绝。这不由客户端强制执行（服务器仍然是用户信任的代码），但符合规范的服务器会遵守它。

当用户添加或移除根时，客户端发送 `notifications/roots/list_changed`。服务器重新调用 `roots/list` 并更新其边界。

### 为什么根是客户端端原语

根由客户端声明，因为它们代表用户的同意模型。用户告诉 Claude Desktop"给这个笔记服务器这两个目录的访问权限"。服务器不能扩大该范围。

### 询问：表单模式（默认）

`elicitation/create` 接受一个表单模式加上自然语言提示：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "删除'TPS 报告'？有多个笔记匹配；请选择一个。",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

客户端渲染表单，收集用户答案，返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三种可能的动作：`accept`（用户填写了）、`decline`（用户关闭了）、`cancel`（用户中止了整个工具调用）。

表单模式是平面的——v1 不支持嵌套对象。SDK 通常拒绝比单层更复杂的任何内容。

### 询问：URL 模式（SEP-1036，实验性）

2025-11-25 新增。服务器发送 URL 而非模式：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "登录到 GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

客户端在浏览器中打开 URL，等待完成，用户返回后继续。适用于表单不够用的 OAuth 流程、支付授权和文档签名。

漂移风险注意：SEP-1036 的响应形状仍在稳定中；一些 SDK 返回回调 URL，另一些返回完成令牌。在生产中使用 URL 模式之前，请阅读你的 SDK 发布说明。

### 询问的正确使用场景

- 在破坏性操作之前进行用户确认（破坏性提示 + 询问）。
- 消歧（从 N 个匹配项中选择一个）。
- 首次运行设置（API 密钥、目录、偏好）。
- OAuth 风格的流程（URL 模式）。

### 询问的错误使用场景

- 填充模型本可以在散文中询问的工具必填参数。使用正常的重提示，而非询问对话框。
- 高频调用。询问会中断对话；不要在循环中触发它。
- 任何服务器事后可以验证的东西。验证，返回错误，让模型用文字问用户。

### 人在回路桥接

询问加采样一起实现了 MCP 的"人在回路"模型。服务器的智能体循环可以暂停等待用户输入（询问）或模型推理（采样）。Phase 13 · 11 涵盖采样；本章涵盖询问。将两者结合，实现完整的中途循环控制。

## 动手使用

`code/main.py` 扩展了笔记服务器，添加了：

- 服务器在根列表变更通知后重新查询的 `roots/list` 响应。
- 当多个笔记匹配时使用 `elicitation/create` 消歧的 `notes_delete` 工具。
- 使用 URL 模式询问打开首次运行配置页面的 `notes_setup` 工具（模拟）。
- 拒绝在声明根之外操作的边界检查。

演示运行三个场景：正常路径（一个匹配）、消歧（三个匹配，触发询问）、根外写入（被拒绝）。

## 输出产物

本章生成 `outputs/skill-elicitation-form-designer.md`。给定可能需要用户确认或消歧的工具，该技能设计询问表单模式和消息模板。

## 练习

1. 运行 `code/main.py`。触发消歧路径；确认模拟用户答案被路由回工具。

2. 添加一个每次都需要询问确认的新工具 `notes_archive`（破坏性提示）。检查 UX：这与模型用文字重新询问相比如何？

3. 为首次运行 OAuth 流程实现 URL 模式询问。注意漂移风险并添加 SDK 版本保护。

4. 扩展 `roots/list` 处理：当通知到达时，服务器应原子性地重新读取和重新扫描可能现在超出范围的打开文件句柄。

5. 阅读 GitHub 上的 SEP-1036 议题讨论线程。找出一个影响服务器应该如何处理 URL 模式回调的未解决问题。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 根（Root） | "同意边界" | 客户端已允许服务器访问的 URI。 |
| `roots/list` | "服务器请求范围" | 客户端返回当前根集合。 |
| `notifications/roots/list_changed` | "用户更改了范围" | 客户端发出根集合已变更的信号。 |
| 询问（Elicitation） | "调用中途询问用户" | 服务器发起的结构化用户输入请求。 |
| `elicitation/create` | "该方法" | 用于询问请求的 JSON-RPC 方法。 |
| 表单模式（Form mode） | "模式驱动的表单" | 在客户端 UI 中渲染为表单的平面 JSON Schema。 |
| URL 模式（URL mode） | "浏览器重定向" | SEP-1036 实验性模式；打开 URL 并等待。 |
| `accept` / `decline` / `cancel` | "用户响应结果" | 服务器处理的三个分支。 |
| 消歧（Disambiguation） | "选一个" | 工具有 N 个候选时常见的询问用例。 |
| 平面表单（Flat form） | "仅顶层属性" | 询问模式不能嵌套。 |

## 延伸阅读

- [MCP — 客户端根规范](https://modelcontextprotocol.io/specification/draft/client/roots) — 根的规范参考
- [MCP — 客户端询问规范](https://modelcontextprotocol.io/specification/draft/client/elicitation) — 询问的规范参考
- [Cisco — MCP 询问、结构化内容和 OAuth 增强的新功能](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 新增功能演练
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL 模式询问提案（实验性，漂移风险）
- [The New Stack — 询问如何将人在回路引入 AI 工具](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UX 演练

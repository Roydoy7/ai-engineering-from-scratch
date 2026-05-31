# MCP 应用——通过 `ui://` 实现的交互式 UI 资源（MCP Apps — Interactive UI Resources via `ui://`）

> 纯文本工具输出限制了智能体能展示的内容。MCP 应用（SEP-1724，2026 年 1 月 26 日正式发布）让工具可以返回在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中内联渲染的沙箱交互式 HTML。仪表板、表单、地图、3D 场景，全都通过一个扩展实现。本章讲解 `ui://` 资源方案、`text/html;profile=mcp-app` MIME 类型、iframe 沙箱 postMessage 协议，以及让服务器渲染 HTML 所带来的安全面。

**类型：** 构建  
**语言：** Python（标准库，UI 资源发射器）、HTML（示例应用）  
**前置知识：** Phase 13 · 07（MCP 服务器）、Phase 13 · 10（资源）  
**预计时间：** 约 75 分钟

## 学习目标

- 从工具调用中返回 `ui://` 资源，并设置正确的 MIME 类型和元数据。
- 使用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具的关联 UI。
- 实现用于 UI 到宿主通信的 iframe 沙箱 postMessage JSON-RPC。
- 应用 CSP 和权限策略默认值，防御 UI 源发起的攻击。

## 问题所在

2025 年时代的 `visualize_timeline` 工具只能返回"以下是按时间顺序排列的 14 条笔记：..."这是一段文字。用户真正想要的是交互式时间线。在 MCP 应用出现之前，选项是：特定客户端的组件 API（Claude artifacts、OpenAI Custom GPT HTML），或者完全没有 UI。

MCP 应用（SEP-1724，2026 年 1 月 26 日发布）标准化了这个契约。工具结果包含一个 URI 为 `ui://...`、MIME 为 `text/html;profile=mcp-app` 的资源。宿主在受限 CSP 和无网络访问（除非明确授权）的沙箱 iframe 中渲染它。iframe 内的 UI 通过一个小型 postMessage JSON-RPC 方言向宿主发送消息。

每个兼容客户端（Claude Desktop、ChatGPT、Goose、VS Code）以相同的方式渲染同一个 `ui://` 资源。一个服务器，一个 HTML 包，通用 UI。

## 核心概念

### `ui://` 资源方案

工具返回：

```json
{
  "content": [
    {"type": "text", "text": "以下是您的笔记时间线："},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

宿主随后对 `ui://notes/timeline` URI 调用 `resources/read` 并获得：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe 沙箱

宿主在带沙箱的 `<iframe>` 中渲染 HTML，其中：

- `sandbox="allow-scripts allow-same-origin"`（或根据服务器声明更严格）
- 通过响应头应用服务器声明的 CSP。
- 没有来自宿主来源的 cookies 或 localStorage。
- 网络访问受限于 CSP 中的 `connectSrc`。

### postMessage 协议

iframe 通过 `window.postMessage` 与宿主通信。这是一个小型 JSON-RPC 2.0 方言：

务必将 `targetOrigin` 固定到对端的精确来源，在接收方验证 `event.origin` 是否在允许列表中，然后再处理任何有效载荷。通道两端都不要使用 `"*"` —— 消息体包含工具调用和资源读取。

```js
// iframe 到宿主（固定到宿主来源）
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// 宿主到 iframe（固定到 iframe 来源）
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// 两端的接收方
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // 安全地处理 event.data
});
```

UI 可以调用的宿主端方法：

- `host.callTool(name, arguments)` —— 调用服务器工具。
- `host.readResource(uri)` —— 读取 MCP 资源。
- `host.getPrompt(name, arguments)` —— 获取提示模板。
- `host.close()` —— 关闭 UI。

每个调用仍然通过 MCP 协议，并继承服务器的权限。

### 权限

`_meta.ui.permissions` 列表请求额外能力：

- `camera` —— 访问用户的摄像头（用于扫描文档 UI）。
- `microphone` —— 语音输入。
- `geolocation` —— 位置。
- `network:*` —— 比 `connectSrc` 单独允许的更广泛的网络访问。

每个权限在 UI 渲染前都会显示给用户一个提示。

### 安全风险

iframe 中的 HTML 还是 HTML。新的攻击面：

- **通过 UI 的提示注入。** 恶意服务器 UI 可以显示看起来像系统消息的文本并欺骗用户。宿主渲染应在视觉上区分服务器 UI 和宿主 UI。
- **通过 `connectSrc` 泄露。** 如果 CSP 允许 `connect-src: *`，UI 可以将数据发送到任何地方。默认应该是严格的。
- **点击劫持。** UI 覆盖宿主 chrome。宿主必须防止 z-index 操纵并强制执行不透明度规则。
- **焦点窃取。** UI 获取键盘焦点并捕获下一条消息。宿主必须拦截。

Phase 13 · 15 将这些作为 MCP 安全的一部分深入讲解；本章对其进行介绍。

### `ui/initialize` 握手

iframe 加载后，它通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

宿主响应能力和会话令牌。UI 在每次后续宿主调用中使用会话令牌。

### AppRenderer / AppFrame SDK 原语

ext-apps SDK 暴露了两个便捷原语：

- `AppRenderer`（服务器端）—— 封装 React / Vue / Solid 组件并发出带正确 MIME 和元数据的 `ui://` 资源。
- `AppFrame`（客户端）—— 接收资源，挂载 iframe，并协调 postMessage。

你可以使用这些，也可以手工编写 HTML 和 JSON-RPC。

### 生态系统状态

MCP 应用于 2026 年 1 月 26 日发布。截至 2026 年 4 月的客户端支持：

- **Claude Desktop。** 2026 年 1 月起完全支持。
- **ChatGPT。** 通过 Apps SDK 完全支持（底层使用相同的 MCP 应用协议）。
- **Cursor。** 测试版；通过设置启用。
- **VS Code。** 仅 Insider 构建版本。
- **Goose。** 完全支持。
- **Zed、Windsurf。** 已列入路线图。

生产中的服务器：仪表板、地图可视化、数据表格、图表构建器、沙箱 IDE 预览。

## 动手使用

`code/main.py` 扩展了笔记服务器，添加了一个返回 `ui://notes/timeline` 资源的 `visualize_timeline` 工具，以及一个处理该 URI 的 `resources/read` 处理器，返回一个带有 SVG 时间线的小但完整的 HTML 包。HTML 使用标准库模板——没有构建系统。postMessage 在 JS 注释中简要描述，因为标准库无法驱动浏览器。

要关注的内容：

- 工具响应上的 `_meta.ui` 携带 resourceUri、CSP、permissions。
- HTML 在没有网络访问的情况下渲染；所有数据都是内联的。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（已记录但在这个标准库演示中是无操作的）。

## 输出产物

本章生成 `outputs/skill-mcp-apps-spec.md`。给定一个可从交互式 UI 中受益的工具，该技能生成完整的 MCP 应用契约：`ui://` URI、CSP、权限、postMessage 入口点和安全清单。

## 练习

1. 运行 `code/main.py` 并检查输出的 HTML。直接在浏览器中打开 HTML；验证 SVG 渲染。然后草拟 UI 调用 `host.callTool("notes_update", ...)` 时使用的 postMessage 契约。

2. 收紧 CSP：移除 `'unsafe-inline'` 并使用基于 nonce 的脚本策略。HTML 生成代码需要什么改变？

3. 添加第二个 UI 资源 `ui://notes/editor`，包含一个用于就地编辑笔记的表单。当用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 的攻击面。恶意服务器在哪里可以注入内容？iframe 沙箱防御了什么，没有防御什么？

5. 阅读 SEP-1724 规范，找出 MCP 应用 SDK 中这个玩具实现没有使用的一项能力。（提示：组件级状态同步。）

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| MCP 应用（MCP Apps） | "交互式 UI 资源" | 2026-01-26 发布的 SEP-1724 扩展。 |
| `ui://` | "应用 URI 方案" | UI 包的资源方案。 |
| `text/html;profile=mcp-app` | "MIME 类型" | MCP 应用 HTML 的内容类型。 |
| Iframe 沙箱（Iframe sandbox） | "渲染容器" | 带 CSP 和权限的 UI 浏览器沙箱。 |
| postMessage JSON-RPC | "UI 到宿主的线路" | 用于宿主调用的小型 postMessage 上的 JSON-RPC 方言。 |
| `_meta.ui` | "工具-UI 绑定" | 将工具结果链接到 UI 资源的元数据。 |
| CSP | "内容安全策略" | 声明脚本、网络、样式的允许来源。 |
| AppRenderer | "服务器端 SDK 原语" | 将框架组件转换为 `ui://` 资源的工具。 |
| AppFrame | "客户端 SDK 原语" | 协调 postMessage 的 iframe 挂载助手。 |
| `ui/initialize` | "握手" | UI 到宿主的第一条 postMessage。 |

## 延伸阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — 参考实现和 SDK
- [MCP 应用规范 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — 正式规范文档
- [MCP — 应用扩展概述](https://modelcontextprotocol.io/extensions/apps/overview) — 高级文档
- [MCP 博客 — MCP 应用发布](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026 年 1 月发布文章
- [MCP 应用 API 参考](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc 风格的 SDK 参考

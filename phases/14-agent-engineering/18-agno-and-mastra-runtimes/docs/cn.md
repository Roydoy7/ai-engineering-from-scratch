# Agno 与 Mastra：生产运行时（Agno and Mastra: Production Runtimes）

> Agno（Python）和 Mastra（TypeScript）是 2026 年的生产运行时配对。Agno 瞄准微秒级智能体实例化和无状态 FastAPI 后端。Mastra 在 Vercel AI SDK 基础上提供智能体、工具、工作流、统一模型路由和复合存储。

**类型：** 学习  
**语言：** Python、TypeScript  
**前置知识：** Phase 14 · 01（智能体循环）、Phase 14 · 13（LangGraph）  
**预计时间：** 约 45 分钟

## 学习目标

- 识别 Agno 的性能目标及其重要的场景。
- 说出 Mastra 的三个基本元素——Agents、Tools、Workflows——以及支持的服务器适配器。
- 解释为什么无状态会话作用域 FastAPI 后端是推荐的 Agno 生产路径。
- 针对给定的技术栈（Python 优先 vs TypeScript 优先）选择 Agno 或 Mastra。

## 问题所在

LangGraph、AutoGen、CrewAI 框架较重。想要"只要智能体循环，快速，在我的运行时中"的团队会选择 Agno（Python）或 Mastra（TypeScript）。两者都以牺牲部分框架拥有的基本元素来换取原始速度，以及与周围技术栈更紧密的契合。

## 核心概念

### Agno

- Python 运行时，前身为 Phi-data。
- "没有图、链或复杂模式——只有纯 Python。"
- 来自其文档的性能目标：约 2μs 智能体实例化，每个智能体约 3.75 KiB 内存，约 23 个模型提供商。
- 生产路径：无状态会话作用域 FastAPI 后端。每个请求启动一个新的智能体；会话状态存储在数据库中。
- 原生多模态（文本、图像、音频、视频、文件）和智能体 RAG。

速度目标在每秒有数千个短生命周期智能体时很重要（聊天扇入、评估流水线）。当一个智能体运行 10 分钟时，它们就不那么重要了。

### Mastra

- TypeScript，基于 Vercel AI SDK 构建。
- 三个基本元素：**Agents（智能体）**、**Tools（工具，Zod 类型）**、**Workflows（工作流）**。
- 统一模型路由器——横跨 94 个提供商的 3,300+ 个模型（2026 年 3 月）。
- 复合存储：记忆、工作流、可观测性到不同后端；推荐 ClickHouse 用于大规模可观测性。
- Apache 2.0，`ee/` 目录采用源码可用企业许可证。
- 支持 Express、Hono、Fastify、Koa 的服务器适配器；一流的 Next.js 和 Astro 集成。
- 附带 Mastra Studio（localhost:4111）用于调试。
- 1.0 版本（2026 年 1 月）时 22k+ GitHub Stars，每周 npm 下载量 30 万+。

### 定位

两者都不是在试图成为 LangGraph。它们在以下方面竞争：

- **语言契合度。** Agno 面向 Python 优先团队；Mastra 面向 TypeScript 优先。
- **运行时人机工程学。** Agno = 接近零开销；Mastra = 与 Vercel 生态系统集成。
- **可观测性。** 两者都与 Langfuse/Phoenix/Opik（第 24 课）集成，但 Mastra Studio 是第一方的。

### 何时选择各自

- **Agno** — Python 后端，大量短生命周期智能体，强性能要求，FastAPI 团队。
- **Mastra** — TypeScript 后端，Next.js / Vercel 部署，统一多提供商模型路由，Zod 类型工具。
- **LangGraph**（第 13 课）——当持久状态和明确图推理比原始速度更重要时。
- **OpenAI / Claude Agent SDK** — 当你想要提供商产品化形态时（第 16-17 课）。

### 这个模式在哪里出错

- **为性能而性能。** 因为"2μs"听起来不错而选择 Agno，但工作负载其实是每个请求一个慢速智能体调用。开销不是瓶颈。
- **生态系统锁定。** Mastra 的 Vercel 风格集成在 Vercel 上是优点，在其他地方是缺点。
- **企业许可证混淆。** Mastra 的 `ee/` 目录是源码可用，而非 Apache 2.0。如果你计划分叉，请阅读许可证。

## 构建它

本课主要是对比性的——没有一个代码实例能公平地展示两个框架。参见 `code/main.py` 的并排玩具：一个最小化的"运行智能体、流式传输输出、持久化会话"流程，实现两次（一次 Agno 形态，一次 Mastra 形态）。

运行：

```
python3 code/main.py
```

两条结构上不同但功能上等效的追踪。

## 使用它

- **Agno** — 需要速度和 FastAPI 形态的 Python 后端。
- **Mastra** — 有多个提供商和工作流基本元素的 TypeScript 后端。
- 两者都附带第一方可观测性钩子。两者都与 Langfuse 集成。

## 交付它

`outputs/skill-runtime-picker.md` 根据技术栈、延迟预算和操作形态选择 Agno、Mastra、LangGraph 或提供商 SDK。

## 练习

1. 阅读 Agno 的文档。将标准库 ReAct 循环（第 01 课）移植到 Agno。什么消失了？什么保留了？
2. 阅读 Mastra 的文档。将同一个循环移植到 Mastra。工具类型（Zod vs 无）有什么变化？
3. 基准测试：测量你的技术栈上的智能体实例化延迟。Agno 的 2μs 对你的工作负载重要吗？
4. 设计迁移：如果你一直在 Python 中运行 CrewAI，迁移到 Agno 会有什么问题？
5. 阅读 Mastra 的 `ee/` 许可证条款。什么限制会影响开源分叉？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Agno | "快速 Python 智能体" | 无状态会话作用域智能体运行时 |
| Mastra | "Vercel AI SDK 上的 TypeScript 智能体" | Agents + Tools + Workflows + Model Router |
| Unified Model Router（统一模型路由器） | "多提供商访问" | 横跨 94 个提供商的 3,300+ 个模型的单一客户端 |
| Composite storage（复合存储） | "多后端" | 记忆/工作流/可观测性各自到不同存储 |
| Mastra Studio | "本地调试器" | localhost:4111 UI，用于内省智能体 |
| Source-available（源码可用） | "非开源" | 许可证允许查看源码但限制商业使用 |

## 延伸阅读

- [Agno 智能体框架文档](https://www.agno.com/agent-framework) — 性能目标、FastAPI 集成
- [Mastra 文档](https://mastra.ai/docs) — 基本元素、服务器适配器、Model Router
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 有状态图替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/) — Mastra 集成引用的可观测性对比

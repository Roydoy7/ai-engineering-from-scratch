# 生产运行时：队列、事件、定时任务（Production Runtimes: Queue, Event, Cron）

> 生产级智能体运行在六种运行时形态上：请求-响应、流式传输、持久执行、基于队列的后台、事件驱动和定时调度。在选择框架之前先选择形态。可观测性在每种形态上都是承重墙。

**类型：** 学习  
**语言：** Python（标准库）  
**前置知识：** Phase 14 · 13（LangGraph）、Phase 14 · 22（语音）  
**预计时间：** 约 60 分钟

## 学习目标

- 说出六种生产运行时形态，并将各自匹配到框架/产品模式。
- 解释为什么持久执行（LangGraph）对长期任务很重要。
- 描述事件驱动运行时以及 Claude Managed Agents 的适用场景。
- 解释可观测性对于多步骤智能体是承重墙的主张。

## 问题所在

生产级智能体以 Jupyter Notebook 无法呈现的方式失败：步骤 37 的网络超时、用户在语音通话中途挂断、定时任务在机器重启时死亡、后台工作者内存耗尽。运行时形态决定了哪些失败是可恢复的。

## 核心概念

### 请求-响应

- 同步 HTTP。用户等待完成。
- 只适用于短任务（< 30 秒）。
- 技术栈：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- 可观测性：标准 HTTP 访问日志 + OTel 跨度。

### 流式传输

- SSE 或 WebSocket 用于渐进式输出。
- LiveKit 将此扩展到 WebRTC 用于语音/视频（第 22 课）。
- 技术栈：任何支持流式传输的框架 + 能处理 SSE/WS 的前端。
- 可观测性：每块计时、首 token 延迟、尾延迟。

### 持久执行

- 每步后状态检查点；失败时自动恢复。
- AutoGen v0.4 行为者模型将失败隔离到一个智能体（第 14 课）。
- LangGraph 的核心差异化点（第 13 课）。
- 当步骤数未知且恢复成本高时是必需的。

### 基于队列/后台

- 任务进入队列，工作者取出，结果通过 Webhook 或发布/订阅回流。
- 对于长期智能体是必需的（根据 Anthropic 的计算机使用公告，每个任务数十到数百步）。
- 技术栈：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、自定义。
- 可观测性：队列深度、每个任务的延迟分布、死信队列大小。

### 事件驱动

- 智能体订阅触发器：新邮件、PR 打开、定时任务触发。
- Claude Managed Agents 开箱即涵盖这种情况（第 17 课）。
- CrewAI Flows（第 15 课）构建事件驱动的确定性工作流。
- 可观测性：触发器来源、事件到启动的延迟、智能体延迟。

### 定时调度

- 定期运行的 Cron 形态智能体。
- 与持久执行结合，使失败的夜间运行在下一个刻度恢复。
- 技术栈：Kubernetes CronJob + 持久框架；托管（Render cron、Vercel cron）。

### 2026 年部署模式

- **CrewAI Flows** 用于事件驱动生产。
- **Agno** 无状态 FastAPI 用于 Python 微服务。
- **Mastra** 服务器适配器（Express、Hono、Fastify、Koa）用于嵌入。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音（第 22 课）。
- **Claude Managed Agents** 用于托管的长时间运行异步。

### 可观测性是承重墙

没有 OpenTelemetry GenAI 跨度（第 23 课）加上 Langfuse/Phoenix/Opik 后端（第 24 课），你就无法调试在第 40 步失败的多步骤智能体。这对于生产不是可选的。这是"我们快速调试"与"我们用更多日志从头重放"之间的区别。

### 生产运行时在哪里失败

- **错误的形态选择。** 为 5 分钟任务选择请求-响应。用户挂断；工作者堆积；重试复合。
- **没有 DLQ。** 没有死信的队列工作者。失败的任务消失了。
- **不透明的后台工作。** 后台智能体运行时没有追踪导出。失败在用户报告之前是不可见的。
- **跳过持久状态。** 任何超过 30 秒且无法承受重启的运行都需要持久执行。

## 构建它

`code/main.py` 是一个标准库多形态演示：

- 请求-响应端点（普通函数）。
- 流式处理器（生成器）。
- 带 DLQ 的基于队列的工作者。
- 事件触发器注册表。
- Cron 形态调度器。

运行：

```bash
python3 code/main.py
```

输出：五条追踪，展示每种形态在同一任务上的行为。相同的智能体逻辑，不同的外部壳。持久执行（第六种形态）在第 13 课中用 LangGraph 检查点有意涵盖。

## 使用它

- **请求-响应** 用于聊天风格 UX。
- **流式传输** 用于渐进式响应。
- **持久** 用于长期任务。
- **队列** 用于批处理/异步/长时间运行。
- **事件** 用于智能体反应性。
- **定时任务** 用于维护（记忆整合、评估、成本报告）。

## 交付它

`outputs/skill-runtime-shape.md` 为任务选择运行时形态并连接可观测性要求。

## 练习

1. 在你的技术栈中将第 01 课的 ReAct 循环移植到所有六种形态。哪种形态适合哪种产品表面？
2. 为基于队列的演示添加 DLQ。模拟 10% 的任务失败；呈现 DLQ 大小。
3. 编写一个定时触发的评估智能体，每晚对当天的前 20 条追踪运行。
4. 实现带反压的流式传输：如果客户端慢，暂停智能体。这如何与轮次预算交互？
5. 阅读 Claude Managed Agents 文档。何时会将自托管的长期智能体迁移到托管？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Request-response（请求-响应） | "同步" | 用户等待；仅适用于短任务 |
| Streaming（流式传输） | "SSE / WS" | 渐进式输出；更好的 UX；每块可观察延迟 |
| Durable execution（持久执行） | "从失败中恢复" | 检查点状态；从最后步骤重启 |
| Queue-based（基于队列） | "后台任务" | 生产者 / 工作者池 / DLQ |
| Event-driven（事件驱动） | "基于触发器" | 智能体对外部事件作出反应 |
| DLQ | "死信队列" | 失败任务的停车场 |
| Claude Managed Agents | "托管框架" | Anthropic 托管的带缓存 + 压缩的长时间运行异步 |

## 延伸阅读

- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 持久执行详情
- [Claude Managed Agents 概述](https://platform.claude.com/docs/en/managed-agents/overview) — 托管的长时间运行异步
- [Anthropic，介绍 computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "每个任务数十到数百步"
- [AutoGen v0.4（微软研究院）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 行为者模型故障隔离

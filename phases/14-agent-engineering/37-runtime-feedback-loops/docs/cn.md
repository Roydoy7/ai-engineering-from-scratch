# 运行时反馈循环（Runtime Feedback Loops）

> 看不到真实命令输出的智能体只能猜测。反馈运行器将 stdout、stderr、退出码和时间捕获到下一轮可以读取的结构化记录中。然后智能体对事实做出反应，而不是对自己对事实的预测做出反应。

**类型：** 构建  
**语言：** Python（标准库）  
**前置知识：** Phase 14 · 32（最小工作台）、Phase 14 · 35（初始化脚本）  
**预计时间：** 约 50 分钟

## 学习目标

- 区分运行时反馈与可观测性遥测。
- 构建一个包装 shell 命令并持久化结构化记录的反馈运行器。
- 确定性地截断大型输出，使循环保持在 token 预算内。
- 当反馈缺失时，拒绝推进循环。

## 问题所在

智能体说"现在正在运行测试"。下一条消息说"所有测试通过"。现实是没有测试运行。智能体想象了输出，或者它运行了命令但从未读取结果，或者它读取了结果但静默地截断了失败行。

反馈运行器消除了这个差距。每个命令都通过运行器。每条记录携带命令、捕获的 stdout 和 stderr、退出码、挂钟持续时间和一行智能体注释。智能体在下一轮读取记录。验证门控在任务结束时读取记录。

## 核心概念

```mermaid
flowchart LR
  Agent[智能体循环] --> Runner[run_with_feedback.py]
  Runner --> Shell[subprocess]
  Shell --> Capture[stdout / stderr / 退出码 / 持续时间]
  Capture --> Record[feedback_record.jsonl]
  Record --> Agent
  Record --> Gate[验证门控]
```

### 反馈记录包含什么

| 字段 | 为什么重要 |
|------|-----------|
| `command` | 精确的 argv，没有 shell 展开惊喜 |
| `stdout_tail` | 最后 N 行，确定性截断 |
| `stderr_tail` | 最后 N 行，与 stdout 分开 |
| `exit_code` | 明确的成功信号 |
| `duration_ms` | 暴露缓慢的探测和失控的进程 |
| `started_at` | 用于重放的时间戳 |
| `agent_note` | 智能体写下的关于它期望的一行 |

### 截断是确定性的

50 MB 的日志会破坏循环。运行器用 `...截断了 N 行...` 标记截断头和尾，是确定性的，所以相同的输出总是产生相同的记录。不抽样；智能体需要看到的部分（最终错误、最终摘要）在尾部。

### 反馈与遥测

遥测（Phase 14 · 23，OTel GenAI 约定）是供人类操作员随时间审查运行的。反馈是为本次运行的下一轮。它们共享字段，但存在于具有不同保留期的不同文件中。

### 没有反馈拒绝推进

如果运行器在捕获退出前出错，记录携带 `exit_code: null` 和 `error: <reason>`。智能体循环必须拒绝在 `null` 退出时声明成功。没有退出，没有进展。

## 构建它

`code/main.py` 实现：

- `run_with_feedback(command, agent_note)`，包装 `subprocess.run`，捕获 stdout/stderr/退出码/持续时间，确定性截断，追加到 `feedback_record.jsonl`。
- 将 JSONL 流式传输到 Python 列表的小型加载器。
- 运行三个命令（成功、失败、慢速）并打印每个命令最后一条记录的演示。

运行：

```
python3 code/main.py
```

输出：三条反馈记录追加到 `feedback_record.jsonl`，每条记录的最后一条内联打印。在重新运行之间追踪文件以查看循环累积。

## 野外的生产模式

三种模式使运行器足够健壮可以发布。

**写入时脱敏，而非读取时脱敏。** 任何触及 stdout 或 stderr 的记录都可能泄露秘密。运行器在 JSONL 追加前发出脱敏处理：删除匹配 `^Bearer `、`password=`、`api[_-]?key=`、`AKIA[0-9A-Z]{16}`（AWS）、`xox[baprs]-`（Slack）的行。读取时脱敏是个隐患；磁盘上的文件是攻击者可以访问的。每季度针对生产运行时观察到的秘密格式审计脱敏模式。

**轮换策略，而非单一文件。** 将 `feedback_record.jsonl` 每个文件上限设为 1 MB；溢出时轮换到 `.1`、`.2`，删除 `.5`。智能体的循环只读取当前文件，所以运行时成本是有界的。CI 工件存储获取完整的轮换集。没有轮换，文件会成为每个加载器调用的瓶颈。

**用于重试链的父命令 ID。** 每条记录获得 `command_id`；重试携带 `parent_command_id`，指向前一次尝试。审查者的"失败尝试"列表（Phase 14 · 40）和验证门控的审计都跟随这条链。没有这个链接，重试看起来像独立的成功，审计隐藏了失败历史。

## 使用它

生产模式：

- **Claude Code Bash 工具。** 该工具已经捕获 stdout、stderr、退出码和持续时间。本课中的运行器是任何智能体产品的框架无关等价物。
- **LangGraph 节点。** 将任何 shell 节点包装在运行器中，使记录持久化在图状态之外。
- **CI 日志。** 将 JSONL 传输到你的 CI 工件存储；审查者可以重放任何命令，而无需重新运行会话。

运行器是一个薄包装器，能在每次框架迁移中存活，因为它拥有记录的形状。

## 交付它

`outputs/skill-feedback-runner.md` 生成一个项目特定的 `run_with_feedback.py`，具有正确的截断预算、连接到工作台的 JSONL 写入器，以及智能体在每轮读取的加载器。

## 练习

1. 每条记录添加 `cwd` 字段，使从不同目录运行的相同命令可区分。
2. 添加一个 `redaction` 步骤，删除匹配 `^Bearer ` 或 `password=` 的行。在固定记录上测试。
3. 通过轮换到 `.1`、`.2` 文件，将总 `feedback_record.jsonl` 大小限制在 1 MB。为轮换策略辩护。
4. 添加 `parent_command_id`，使重试链可见：哪个命令产生了下一个命令消耗的输入。
5. 将 JSONL 传输到一个突出显示最新非零退出的小型 TUI。TUI 必须显示才有用的八个关键功能。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Feedback record（反馈记录） | "运行日志" | 带命令、输出、退出码、持续时间的结构化 JSONL 条目 |
| Tail truncation（尾部截断） | "修剪日志" | 确定性的头+尾捕获，使记录适合 token 预算 |
| Refuse-on-null（空值拒绝） | "缺数据时阻止" | 当 `exit_code` 为 null 时循环不得推进 |
| Agent note（智能体注释） | "期望标签" | 智能体在读取结果前写下的一行预测 |
| Telemetry split（遥测分离） | "两个日志文件" | 反馈用于下一轮，遥测用于操作员 |

## 延伸阅读

- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Anthropic，长时间运行智能体的有效框架](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Guardrails AI x MLflow — 确定性安全、PII、质量验证器](https://guardrailsai.com/blog/guardrails-mlflow) — 脱敏模式作为回归测试
- [Aport.io，2026 年最佳 AI 智能体护栏：预动作授权比较](https://aport.io/blog/best-ai-agent-guardrails-2026-pre-action-authorization-compared/) — 工具前后捕获
- [Andrii Furmanets，2026 年 AI 智能体：工具、记忆、评估、护栏的实践架构](https://andriifurmanets.com/blogs/ai-agents-2026-practical-architecture-tools-memory-evals-guardrails) — 可观测性表面
- Phase 14 · 23 — OTel GenAI 约定用于遥测侧
- Phase 14 · 24 — 智能体可观测性平台（Langfuse、Phoenix、Opik）
- Phase 14 · 33 — 要求完成前提供反馈的规则
- Phase 14 · 38 — 读取 JSONL 的验证门控

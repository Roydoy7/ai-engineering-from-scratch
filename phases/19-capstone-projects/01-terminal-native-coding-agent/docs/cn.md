# 压轴项目 01——终端原生编程智能体（Capstone 01 — Terminal-Native Coding Agent）

> 到 2026 年，编程智能体的形态已经确立。TUI 测试框架、有状态计划、沙箱工具界面、规划-行动-观察-恢复循环。从高层次看，Claude Code、Cursor 3 和 OpenCode 都是同一个东西。本压轴项目要求你端到端地构建一个——CLI 输入，拉取请求输出——并在 SWE-bench Pro 上与 mini-swe-agent 和 Live-SWE-agent 进行基准比较。你将学到为什么难点不在于模型调用，而在于工具循环、沙箱以及 50 轮运行的成本上限。

**类型：** 压轴项目  
**语言：** TypeScript / Bun（测试框架），Python（评估脚本）  
**前置知识：** Phase 11（LLM 工程）、Phase 13（工具与协议）、Phase 14（智能体）、Phase 15（自主系统）、Phase 17（基础设施）  
**涉及的阶段：** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18  
**预计时间：** 35 小时

## 问题所在

编程智能体在 2026 年成为主导的 AI 应用类别。Claude Code（Anthropic）、带有 Composer 2 和 Agent Tabs 的 Cursor 3（Cursor）、Amp（Sourcegraph）、OpenCode（112k star）、Factory Droids 和 Google Jules 都发布了同一架构的变体：终端测试框架、权限工具界面、沙箱，以及围绕前沿模型构建的规划-行动-观察循环。前沿很窄——Live-SWE-agent 在 SWE-bench Verified 上使用 Opus 4.5 达到了 79.2%——但工程技艺很宽广。大多数失败模式不是模型错误，而是工具循环不稳定、上下文中毒、失控的 token 成本和破坏性的文件系统操作。

你无法从外部对这些智能体进行推理。你必须构建一个，观察循环在第 47 轮时 ripgrep 返回 8MB 匹配项而崩溃，然后重建截断层。这就是本压轴项目的意义所在。

## 核心概念

测试框架有四个界面。**计划**维护一个 TodoWrite 风格的状态对象，模型每轮重写该对象。**行动**分发工具调用（读取、编辑、运行、搜索、git）。**观察**捕获 stdout/stderr/退出代码，截断，并将摘要反馈回去。**恢复**处理工具错误，而不会炸毁上下文窗口或无限循环。2026 年的形态还添加了一件事：**钩子**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact`——运营者注入策略、遥测和护栏的可配置扩展点。

沙箱是 E2B 或 Daytona。每个任务在新鲜的 devcontainer 中运行，挂载了读写权限的 git 工作树。测试框架从不接触主机文件系统。工作树在成功或失败时被销毁。成本控制在三个层次执行：每轮 token 上限、每次会话美元预算和硬轮次限制（通常为 50 轮）。可观测性层是具有 GenAI 语义约定的 OpenTelemetry span，发送到自托管的 Langfuse。

## 架构

```
  用户 CLI  ->  测试框架（Bun + Ink TUI）
                  |
                  v
           计划/行动/观察循环  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          （通过 OpenRouter，模型无关）
                  v
           工具分发器（MCP StreamableHTTP 客户端）
                  |
     +------------+------------+----------+
     v            v            v          v
  读取/编辑    ripgrep    tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona 沙箱（工作树隔离）
                  |
                  v
           钩子：Pre/Post，Session，Prompt，Compact
                  |
                  v
           OpenTelemetry -> Langfuse（span、token、$）
                  |
                  v
           通过 GitHub App 发起 PR
```

## 技术栈

- 测试框架运行时：Bun 1.2 + Ink 5（React-in-terminal）
- 模型访问：OpenRouter 统一 API，支持 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（用于最难的任务）
- 工具传输：Model Context Protocol StreamableHTTP（MCP 2026 修订版）
- 沙箱：E2B sandboxes（JS SDK）或 Daytona devcontainers
- 代码搜索：ripgrep 子进程，17 种语言的 tree-sitter 解析器（预编译）
- 隔离：每个任务 `git worktree add`，在成功/失败时清理
- 评估测试框架：SWE-bench Pro（verified 子集）+ Terminal-Bench 2.0 + 你自己的 30 个任务保留集
- 可观测性：OpenTelemetry SDK，带 `gen_ai.*` semconv → 自托管 Langfuse
- PR 发布：具有细粒度 token 的 GitHub App，范围限于目标仓库

## 构建它

1. **TUI 和命令循环。** 用 Ink 搭建 Bun 项目。接受 `agent run <repo> "<task>"`。打印分屏视图：计划面板（顶部）、工具调用流（中部）、token 预算（底部）。添加在退出前触发 `SessionEnd` 钩子的 Ctrl-C 取消功能。

2. **计划状态。** 定义一个类型化的 TodoWrite schema（待处理/进行中/完成的项目，带有备注）。模型每轮将完整状态作为工具调用重写——不要让它增量变更。将计划持久化到 `.agent/state.json`，以便崩溃后可以恢复。

3. **工具界面。** 定义六个工具：`read_file`、`edit_file`（带差异预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时）、`git`（状态/差异/提交/推送）。通过 MCP StreamableHTTP 暴露，使测试框架与传输无关。每个工具返回截断的输出（每次调用上限 4k token）。

4. **沙箱封装。** 每个任务产生一个 E2B 沙箱。`git worktree add -b agent/$TASK_ID` 一个新分支。所有工具调用在沙箱内执行。主机文件系统不可访问。

5. **钩子。** 实现所有八种 2026 钩子类型。至少连接四个用户编写的钩子：(a) 阻止工作树外 `rm -rf` 的 `PreToolUse` 破坏性命令守护，(b) `PostToolUse` token 计账，(c) `SessionStart` 预算初始化，(d) `Stop` 写入最终追踪包。

6. **评估循环。** 克隆 SWE-bench Pro Python 的 30 个问题子集。对每个问题运行你的测试框架。在 pass@1、每任务轮次和每任务美元方面与 mini-swe-agent（最小基线）进行比较。将结果写入 `eval/results.jsonl`。

7. **成本控制。** 硬截止：50 轮、200k 上下文、每任务 $5。`PreCompact` 钩子在 150k 标记处将旧轮次总结为先前状态块，为新观察腾出空间而不丢失计划。

8. **PR 发布。** 成功时，最后一步是 `git push` 加打开 PR 的 GitHub API 调用，PR 正文包含计划和差异摘要。

## 使用它

```
$ agent run ./my-repo "修复 worker.rs 中的竞态条件"
[计划]  1 定位 worker.rs 并枚举互斥锁使用
        2 识别竞争下的共享状态
        3 提出修复方案，验证测试
[工具]  ripgrep mutex.*lock -t rust           （44 个匹配，已截断）
[工具]  read_file src/worker.rs 120..180
[工具]  edit_file src/worker.rs (+8 -3)
[工具]  run_shell cargo test worker::          （通过）
[计划]  1 完成 · 2 完成 · 3 完成
[完成]  PR 已开启：#482   轮次=9   token=38k   成本=$0.41
```

## 交付它

可交付的技能存在于 `outputs/skill-terminal-coding-agent.md` 中。给定仓库路径和任务描述，它在沙箱中运行完整的规划-行动-观察循环，并返回 PR URL 加追踪包。本压轴项目的评分标准：

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs 基线 | 你的测试框架 vs mini-swe-agent 在 30 个匹配的 Python 任务上 |
| 20 | 架构清晰度 | 计划/行动/观察分离、钩子界面、工具 schema——对照 Live-SWE-agent 布局审查 |
| 20 | 安全性 | 沙箱逃逸测试、权限提示、破坏性命令守护通过红队测试 |
| 20 | 可观测性 | 追踪完整性（100% 工具调用有 span），每轮 token 计账 |
| 15 | 开发者体验 | 冷启动 < 2 秒，崩溃恢复恢复计划，Ctrl-C 干净取消正在进行的工具 |
| **100** | | |

## 练习

1. 将支持模型从 Claude Sonnet 4.7 换成部署在 vLLM 上的 Qwen3-Coder-30B。比较 pass@1 和每任务美元。报告开源模型表现不佳的地方。

2. 添加一个在 PR 发布前读取差异并可以请求修订循环的 `reviewer` 子智能体。测量误报审查是否将 SWE-bench pass 率降至单智能体基线以下（提示：通常会）。

3. 对沙箱进行压力测试：编写一个试图 `curl` 外部 URL 的任务和一个写入工作树外的任务。确认两者都被 PreToolUse 钩子阻止。记录尝试。

4. 使用较小的模型（Haiku 4.5）实现 `PreCompact` 摘要。测量 3 次压缩时丢失多少计划保真度。

5. 将 MCP StreamableHTTP 传输换为 stdio。对冷启动和每次调用延迟进行基准测试。为本地使用选择一个赢家。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Harness（测试框架） | "智能体循环" | 围绕模型的代码，分发工具、维护计划状态并执行预算 |
| Hook（钩子） | "智能体事件监听器" | 测试框架在八个生命周期事件之一上运行的用户编写的脚本 |
| Worktree（工作树） | "Git 沙箱" | 在单独路径上的链接 git 检出；可丢弃而不影响主克隆 |
| TodoWrite | "计划状态" | 模型每轮重写的待处理/进行中/完成项目的类型化列表 |
| StreamableHTTP | "MCP 传输" | 2026 年 MCP 修订：具有双向流的长期 HTTP 连接；替换 SSE |
| Token ceiling（token 上限） | "上下文预算" | 每轮或每次会话的输入+输出 token 上限；触发压缩或终止 |
| pass@1 | "单次通过率" | 在第一次运行时解决的 SWE-bench 任务比例，不重试或查看测试集 |

## 延伸阅读

- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 的参考测试框架
- [Cursor 3 更新日志](https://cursor.com/changelog) — Agent Tabs 和 Composer 2 产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — SWE-bench 测试框架比较的最小基线
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 使用 Opus 4.5 在 SWE-bench Verified 上达到 79.2%
- [OpenCode](https://opencode.ai) — 开源测试框架，112k star
- [SWE-bench Pro 排行榜](https://www.swebench.com) — 本压轴项目目标的评估
- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP，能力元数据
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 工具调用和 token 使用的 span schema

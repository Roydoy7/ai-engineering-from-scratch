# 智能体指令作为可执行约束（Agent Instructions as Executable Constraints）

> 以散文写的指令是愿望。以约束写的指令是测试。工作台将每条规则变为智能体可在运行时检查、审查者可在事后验证的东西。

**类型：** 构建  
**语言：** Python（标准库）  
**前置知识：** Phase 14 · 32（最小工作台）  
**预计时间：** 约 50 分钟

## 学习目标

- 将路由散文与操作规则分开。
- 将启动规则、禁止动作、完成定义、不确定性处理和审批边界表达为机器可检查的约束。
- 实现一个根据规则集对运行评分的规则检查器。
- 使规则集对差异友好，以便审查可以看到什么改变了。

## 问题所在

典型的 `AGENTS.md` 读起来像入职文档。它告诉智能体"要小心"、"彻底测试"和"不确定时询问"。三天后，智能体发布了一个没有测试的更改，写入了禁止目录，从不询问，因为它从不知道边界在哪里。

当指令是操作性的时候它们是强大的，当它们是理想性的时候是软弱的。解决方法是编写工作台可以解释、审查者可以评分的规则。

## 核心概念

规则属于 `docs/agent-rules.md`，远离简短的根路由器。每条规则有名称、类别和检查。

```mermaid
flowchart LR
  Router[AGENTS.md] --> Rules[docs/agent-rules.md]
  Rules --> Checker[rule_checker.py]
  Checker --> Report[rule_report.json]
  Report --> Reviewer[审查者]
```

### 涵盖大多数规则的五个类别

| 类别 | 规则回答的问题 | 示例 |
|------|---------------|------|
| 启动（Startup） | 工作开始前必须满足什么？ | "状态文件存在且是新鲜的" |
| 禁止（Forbidden） | 什么永远不能发生？ | "不编辑 `scripts/release.sh`" |
| 完成定义（Definition of done） | 什么证明任务完成？ | "pytest 以 0 退出且验收行通过" |
| 不确定性（Uncertainty） | 不确定时智能体做什么？ | "打开一个问题笔记而不是猜测" |
| 审批（Approval） | 什么需要人工审批？ | "任何新依赖，任何生产写入" |

不符合这五类之一的规则通常应该变成两条规则。强制分割。

### 规则是机器可读的

每条规则有一个 slug、一个类别、一行描述和一个命名 `rule_checker.py` 中函数的 `check` 字段。添加规则意味着添加检查；检查器随工作台增长。

### 规则对差异友好

规则在单个 Markdown 文件中每个标题一条。重命名在差异中可见。新规则放在其类别的顶部。过时规则被删除，不是注释掉，因为工作台是真相来源，而非团队上季度感受的聊天记录。

### 规则 vs 框架护栏

框架护栏（OpenAI Agents SDK 护栏、LangGraph 中断）在运行时级别执行规则。本课中的规则集是这些护栏实现的人类可读、可审查的契约。两者都需要：运行时在一轮中捕获违规，规则集证明运行时在做正确的事情。

## 构建它

`code/main.py` 提供：

- 将规则加载到数据类的 `agent-rules.md` 解析器。
- `rule_checker.py` 风格检查函数，每个 `check` 引用一个。
- 一个违反两条规则的演示智能体运行和一个捕获它们的检查通过。

运行：

```
python3 code/main.py
```

输出：解析的规则集、运行追踪、每条规则的通过/失败，以及保存在脚本旁边的 `rule_report.json`。

## 野外的生产模式

三种模式将持续一个季度的规则集与一周内衰退的规则集区分开来。

**写入时的严重程度标记。** 每条规则携带 `severity`：`block`、`warn` 或 `info`。检查器报告所有三个；运行时只在 `block` 上拒绝。大多数团队早期高估严重程度，然后在截止日期压力下悄悄削弱它；写入时标记迫使提前校准。与验证门控（Phase 14 · 38）配对，它将对 `block` 规则的任何覆盖签名到 `overrides.jsonl` 审计日志中。

**规则过期作为强制函数。** 每条规则携带 `expires_at` 日期（默认从创建起 90 天）。当一条未过期的规则连续 60 天没有违规时，检查器发出警告；下一次季度审查要么证明保留它，要么将其弱化为 `info`，要么删除它。Cloudflare 的生产 AI 代码审查数据（2026 年 4 月，30 天内跨 5,169 个仓库的 131,246 次审查运行）显示，有明确过期的规则集每个仓库保持在 30 条规则以下；没有的集合增长到 80+ 条，大多数从不触发。

**Markdown 作为源，JSON 作为缓存。** `agent-rules.md` 是创作文件；`agent-rules.lock.json` 是检查器在热路径中读取的缓存。锁由预提交钩子重新生成。Markdown 差异是可审查的；JSON 解析不参与每一轮。与 `package.json` / `package-lock.json` 和 `Cargo.toml` / `Cargo.lock` 形态相同。

## 使用它

在生产中：

- Claude Code、Codex、Cursor 在会话开始时读取规则，并在拒绝动作时引用它们。检查器在 CI 中重新运行它们以捕获无声漂移。
- OpenAI Agents SDK 护栏将相同的检查注册为输入和输出护栏。Markdown 是文档表面；SDK 是运行时表面。
- LangGraph 中断在飞行中的节点违反规则时触发。中断处理器读取规则，询问人类，然后恢复。

规则集在所有三者之间是可移植的，因为它只是 Markdown 加函数名。

## 交付它

`outputs/skill-rule-set-builder.md` 采访项目所有者，将其现有散文指令分类为五个类别，并发射版本化的 `agent-rules.md` 加检查器存根。

## 练习

1. 如果你的产品真正需要，添加第六个类别。为什么它不能折叠到五个之一辩护。
2. 扩展检查器，使规则可以携带严重程度（`block`、`warn`、`info`），报告相应地聚合。
3. 将检查器连接到 CI：如果 block 严重程度的规则在最新的智能体运行上失败，则使构建失败。
4. 为每条规则添加"过期"字段。90 天没有检查失败后，该规则等待审查。
5. 找一个真实的 `AGENTS.md` 并将其重写为五类规则。它有多少行是操作性的？有多少是理想性的？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Operational rule（操作规则） | "真正的指令" | 工作台可在运行时检查的规则 |
| Aspirational rule（理想规则） | "要小心" | 没有检查的规则；要么删除，要么升级 |
| Definition of done（完成定义） | "验收" | 客观的、文件支持的任务完成证明 |
| Block severity（Block 严重程度） | "硬规则" | 违规时停止运行；没有操作员无法消音 |
| Rule expiry（规则过期） | "过时规则清扫" | N 天没有失败的规则等待退役 |

## 延伸阅读

- [OpenAI Agents SDK 护栏](https://platform.openai.com/docs/guides/agents-sdk/guardrails)
- [LangGraph 中断](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
- [Anthropic，构建有效智能体](https://www.anthropic.com/research/building-effective-agents)
- [Rick Hightower，Agent RuleZ：用于 AI 编程智能体的确定性策略引擎](https://medium.com/@richardhightower/agent-rulez-a-deterministic-policy-engine-for-ai-coding-agents-9489e0561edf) — 生产中的 block/warn/info 严重程度
- [Cloudflare，大规模编排 AI 代码审查](https://blog.cloudflare.com/ai-code-review/) — 131k 审查运行，规则组合课程
- [microservices.io，GenAI 开发平台——第 1 部分：护栏](https://microservices.io/post/architecture/2026/03/09/genai-development-platform-part-1-development-guardrails.html) — 规则与 CI 之间的纵深防御
- [类型检查合规性：确定性护栏（arXiv 2604.01483）](https://arxiv.org/pdf/2604.01483) — Lean 4 作为规则即检查的上界
- [logi-cmd/agent-guardrails](https://github.com/logi-cmd/agent-guardrails) — 合并门控实现：范围、变异测试、违规预算
- Phase 14 · 32 — 该规则集放入的最小工作台
- Phase 14 · 38 — 消费规则报告的验证门控
- Phase 14 · 39 — 对规则合规性评分的审查者智能体

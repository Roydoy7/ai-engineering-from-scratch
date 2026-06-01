# 压轴项目 16——GitHub 问题到 PR 自主智能体（Capstone 16 — GitHub Issue-to-PR Autonomous Agent）

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 和 Google Jules 都发布了相同的 2026 年产品形态：为问题添加标签，获得 PR。在云沙箱中运行智能体，验证测试通过，并发布带理由的待审查 PR。难点在于自动重现仓库的构建环境、防止凭证泄露、执行每仓库预算，以及确保智能体不能强制推送。本压轴项目构建自托管版本，并在成本和通过率上与托管替代方案进行比较。

**类型：** 压轴项目  
**语言：** Python（智能体），TypeScript（GitHub App），YAML（Actions）  
**前置知识：** Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（智能体）、Phase 15（自主系统）、Phase 17（基础设施）  
**涉及的阶段：** P11 · P13 · P14 · P15 · P17  
**预计时间：** 30 小时

## 问题所在

异步云编程智能体是与交互式编程智能体（压轴项目 01）不同的产品类别。用户体验是 GitHub 标签。你为问题添加 `@agent fix this` 标签，一个工作者在云沙箱中启动，克隆仓库，运行测试，编辑文件，验证，并开启带智能体理由的 PR。没有交互式循环，没有终端。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 和 Factory Droids 都收敛到这一形态。

工程挑战是具体的：环境重现（智能体必须从头构建仓库，没有缓存的开发镜像）、不稳定的测试（必须重新运行或隔离）、凭证范围限定（具有最小细粒度权限的 GitHub App）、每仓库每天的预算执行，以及禁止强制推送策略。本压轴项目测量通过率、成本和安全性 vs 托管替代方案。

## 核心概念

触发器是 GitHub webhook（问题标签或 PR 评论）。调度器将工作加入 ECS Fargate 或 Lambda 的队列。工作者将仓库拉入 Daytona 或 E2B 沙箱，使用从仓库推断的通用 Dockerfile（语言、框架）。智能体对 Claude Opus 4.7 或 GPT-5.4-Codex 运行 mini-swe-agent 或 SWE-agent v2 循环。它迭代：读取代码，提出修复，应用补丁，运行测试。

验证是门控步骤。完整 CI 必须在沙箱中通过，然后 PR 才能开启。计算覆盖率差值；如果超过阈值为负，PR 开启但被标记为 `needs-review`。智能体将理由作为 PR 描述以及可以被 ping 进行后续跟进的 `@agent` 线程发布。

安全性通过两种不同的 GitHub 界面限制范围：App 提供具有 `workflows: read` 和窄仓库内容/PR 范围的短期安装令牌；分支保护（不是 App 权限）执行"不直接写入 `main`"和"不强制推送"——App 永远不添加到绕过列表中。由于路径范围的只读访问 `.github/workflows` 不是真正的 GitHub App 原语，智能体对文件编辑的允许列表必须在工作者处执行。每仓库每天的预算上限在调度器处执行（例如，每仓库每天最多 5 个 PR，每 PR $20）。

## 架构

```
GitHub 问题标记为 `@agent fix` 或 PR 评论
            |
            v
    GitHub App webhook -> AWS Lambda 调度器
            |
            v
    ECS Fargate 任务（或 GitHub Actions 自托管运行器）
       - 拉取仓库
       - 推断 Dockerfile（语言，包管理器）
       - 带目标运行时的 Daytona / E2B 沙箱
       - 克隆 -> git worktree -> 智能体分支
            |
            v
    mini-swe-agent / SWE-agent v2 循环
       Claude Opus 4.7 或 GPT-5.4-Codex
       工具：ripgrep，tree-sitter，读取/编辑，运行测试，git
            |
            v
    验证沙箱内 CI 通过 + 覆盖率差值检查
            |
            v（已验证）
    git push + 通过 GitHub App 开启 PR
       PR 正文 = 理由 + 差异摘要 + 追踪 URL
       标签：needs-review
            |
            v
    运营者审查；可以 @ 智能体进行后续跟进
```

## 技术栈

- 触发器：带细粒度令牌的 GitHub App；通过 Lambda 或 Fly.io 的 webhook 接收器
- 工作者：ECS Fargate 任务（或 GitHub Actions 自托管运行器）
- 沙箱：每任务 Daytona devcontainer 或 E2B 沙箱
- 智能体循环：基于 Claude Opus 4.7 / GPT-5.4-Codex 的 mini-swe-agent 基线或 SWE-agent v2
- 检索：tree-sitter 仓库映射 + ripgrep
- 验证：沙箱内完整 CI + 覆盖率差值门控
- 可观测性：Langfuse，带从 PR 正文链接的每 PR 追踪存档
- 预算：每仓库每天的美元上限；每仓库每天的最大 PR 数

## 构建它

1. **GitHub App。** 细粒度安装令牌：issues 读+写，pull_requests 写，contents 读+写，workflows 读。分支保护（唯一能做到这一点的界面）执行"不直接推送到 `main`"和"不强制推送"；App 不在绕过列表中。工作者将"不写入 `.github/workflows` 下"作为对提议差异的允许列表检查来执行，因为 GitHub App 权限不是路径范围的。

2. **Webhook 接收器。** Lambda 函数接受问题标签 / PR 评论 webhook。按标签 `@agent fix this` 过滤。加入 SQS 队列。

3. **调度器。** 从 SQS 弹出任务。执行每仓库每天预算。用仓库 URL、问题正文和新鲜 Daytona 沙箱启动 ECS Fargate 任务。

4. **环境推断。** 检测语言（Python、Node、Go、Rust）和包管理器（uv、pnpm、go mod、cargo）。如果不存在 Dockerfile，即时生成。

5. **智能体循环。** Claude Opus 4.7 的 mini-swe-agent 或 SWE-agent v2。工具：ripgrep、tree-sitter 仓库映射、read_file、edit_file、run_tests、git。硬限制：$20 成本，30 分钟挂钟，30 个智能体轮次。

6. **验证。** 循环结束后，在沙箱内运行完整测试套件。通过 jacoco / coverage.py 计算覆盖率差值。如果 CI 红色：停止，不开启 PR。如果覆盖率下降超过 2%：用 `needs-review` 标签开启 PR。

7. **PR 发布。** 推送智能体分支。通过 GitHub API 开启 PR，包含：标题、理由、差异摘要、追踪 URL、成本、轮次。

8. **凭证卫生。** 工作者使用短期 GitHub App 安装令牌运行。日志在存档前清洗密钥。

9. **评估。** 30 个不同难度的内部预设问题。测量通过率、PR 质量（差异大小、风格、覆盖率）、成本、延迟。在同一问题上与 Cursor Background Agents 和 AWS Remote SWE Agents 比较。

## 使用它

```
# 在 github.com 上
  - 用户为问题 #842 添加 `@agent fix this` 标签
  - 14 分钟后 PR #1903 出现
  - 正文：
    > 修复了 widget.dedupe() 中由 null 比较器条目引起的 NPE。
    > 添加了回归测试 widget_test.go::TestDedupeNullComparator。
    > 覆盖率差值：+0.12%
    > 轮次：7  成本：$1.80  追踪：langfuse:...
    > 标签：needs-review
```

## 交付它

`outputs/skill-issue-to-pr.md` 是可交付成果。一个 GitHub App + 异步云工作者，将标记的问题转化为带受限成本和范围凭证的待审查 PR。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 30 个问题的通过率 | 端到端成功（CI 绿色 + 覆盖率 OK） |
| 20 | PR 质量 | 差异大小，覆盖率差值，风格合规性 |
| 20 | 每个解决问题的成本和延迟 | 每个 PR 的美元成本和挂钟时间 |
| 20 | 安全性 | 范围令牌，每仓库预算，无强制推送，凭证卫生 |
| 15 | 运营者用户体验 | 理由评论，重试功能，@ 提及后续跟进 |
| **100** | | |

## 练习

1. 添加"修复不稳定测试"模式：标签 `@agent stabilize-flake TestX` 在沙箱中运行测试 50 次，并提出稳定它的最小更改。

2. 在三个共享问题上将成本与 Cursor Background Agents 比较。报告哪些工具在哪里胜出。

3. 实现预算仪表板：每仓库每天成本，每用户成本。对异常发出告警。

4. 构建"演习"模式：开启草稿 PR 而不运行 CI，让审查者可以便宜地检查计划。

5. 添加保留策略：超过 7 天没有合并的 PR 分支自动删除。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| GitHub App | "范围 bot 身份" | 带细粒度权限 + 短期安装令牌的 App |
| Async cloud agent（异步云智能体） | "后台智能体" | 在云沙箱中运行的非交互式工作者，而非终端 |
| Environment inference（环境推断） | "Dockerfile 合成" | 检测语言 + 包管理器，如果不存在则生成 Dockerfile |
| Verification（验证） | "沙箱内 CI" | 在开启 PR 之前在工作者内运行完整测试套件 |
| Coverage delta（覆盖率差值） | "覆盖率保留" | 从基础到智能体分支的测试覆盖率 % 变化 |
| Per-repo budget（每仓库预算） | "每天上限" | 在调度器处执行的美元和 PR 数量上限 |
| Rationale（理由） | "PR 正文解释" | 智能体对什么改变了以及为什么的摘要；PR 正文中必填 |

## 延伸阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 标准性异步云智能体参考
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI 参考
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 商业替代方案
- [OpenAI Codex（cloud）](https://openai.com/codex) — 托管竞争者
- [Google Jules](https://jules.google) — Google 的托管版本
- [Factory Droids](https://www.factory.ai) — 备用商业参考
- [GitHub App 文档](https://docs.github.com/en/apps) — 范围 bot 身份
- [Daytona 云沙箱](https://daytona.io) — 参考沙箱

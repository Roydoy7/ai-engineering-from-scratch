# 压轴项目 10——多智能体软件工程团队（Capstone 10 — Multi-Agent Software Engineering Team）

> SWE-AF 的工厂架构、MetaGPT 的基于角色的提示词、AutoGen 0.4 的类型化 Actor 图、Cognition 的 Devin 和 Factory 的 Droids 都汇聚到了同一个 2026 年的形态：一个架构师规划，N 个程序员在并行工作树中工作，一个审查者把关，一个测试者验证。并行工作树将挂钟时间转化为吞吐量。共享状态和交接协议成为故障面。本压轴项目是构建团队，在 SWE-bench Pro 上评估，并报告哪些交接失败以及失败频率。

**类型：** 压轴项目  
**语言：** Python / TypeScript（智能体），Shell（工作树脚本）  
**前置知识：** Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（智能体）、Phase 15（自主系统）、Phase 16（多智能体）、Phase 17（基础设施）  
**涉及的阶段：** P11 · P13 · P14 · P15 · P16 · P17  
**预计时间：** 40 小时

## 问题所在

单智能体编程测试框架在大型任务上触及天花板。不是因为任何单个智能体弱，而是因为 20 万 token 的上下文无法容纳架构计划加上四个并行代码库切片加上审查者评论加上测试输出。多智能体工厂分解问题：架构师拥有计划，程序员在并行工作树中拥有实现，审查者把关，测试者验证。SWE-AF 的"工厂"架构、MetaGPT 的角色、AutoGen 的类型化 Actor 图——三种框架都描述了同一形态。

故障面是交接。架构师规划程序员无法实现的东西。程序员产生冲突的差异。审查者批准幻觉修复。测试者与仍在编写的程序员竞争。你将构建这样一个团队，在 50 个 SWE-bench Pro 问题上运行它，追踪每次交接，并发布事后分析。

## 核心概念

角色是类型化的智能体。**架构师**（Claude Opus 4.7）读取问题，写出计划，并将其分解为带有显式接口的子任务。**程序员**（Claude Sonnet 4.7，N 个并行实例，每个在 `git worktree` + Daytona 沙箱中）独立实现子任务。**审查者**（GPT-5.4）读取合并后的差异并批准或要求具体更改。**测试者**（Gemini 2.5 Pro）在隔离环境中运行测试套件并报告通过/失败及工件。

通信通过共享任务板（基于文件或 Redis）进行。每个角色消费它被允许处理的任务。交接是 A2A 协议类型的消息。协调关注点：合并冲突解决（协调者角色或自动三方合并）、共享状态同步（计划在程序员开始后冻结；重新规划是单独事件）、审查者把关（审查者不能批准自己的更改或自己提出的更改）。

Token 放大是隐藏成本。每个角色边界都会添加摘要提示词和交接上下文。40 轮的单智能体运行变成四个角色共 160 轮。评分标准专门衡量 token 效率 vs 单智能体基线，因为问题不是"多智能体是否有效"而是"它在每美元上是否胜出"。

## 架构

```
GitHub 问题 URL
      |
      v
架构师（Opus 4.7）
   读取问题，生成带子任务 + 接口的计划
      |
      v
任务板（文件 / Redis）
      |
   +-- 子任务 1 ---+-- 子任务 2 ---+-- 子任务 3 ---+-- 子任务 4 ---+
   v              v              v              v              v
程序员 A       程序员 B       程序员 C       程序员 D       （4 个并行）
 (Sonnet)       (Sonnet)       (Sonnet)       (Sonnet)
 工作树 A       工作树 B       工作树 C       工作树 D
 Daytona        Daytona        Daytona        Daytona
      |              |              |              |
      +--------+-----+------+-------+
               v
           合并协调者（三方合并 + 冲突解决）
               |
               v
           审查者（GPT-5.4）
               |
               v
           测试者（Gemini 2.5 Pro）-> 通过？-> 开启 PR
                                    -> 失败？-> 路由回程序员
```

## 技术栈

- 编排：LangGraph，带共享状态 + 每智能体子图
- 消息传递：A2A 协议（Google 2025）用于类型化智能体间消息
- 模型：Opus 4.7（架构师）、Sonnet 4.7（程序员）、GPT-5.4（审查者）、Gemini 2.5 Pro（测试者）
- 工作树隔离：每个程序员 `git worktree add` + Daytona 沙箱
- 合并协调者：自定义三方合并 + LLM 中介冲突解决
- 评估：SWE-bench Pro（50 个问题）、SWE-AF 场景、HumanEval++ 用于单元测试
- 可观测性：Langfuse，带角色标记 span，每智能体 token 计账
- 部署：K8s，每个角色作为单独 Deployment + 积压上的 HPA

## 构建它

1. **任务板。** 基于文件的 JSONL，带类型消息：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。智能体订阅标签。

2. **架构师。** 读取 GitHub 问题，用带显式子任务接口（触及的文件、公共函数、测试影响）的计划模板运行 Opus 4.7。发出一个带子任务 DAG 的 `plan_request`。

3. **程序员。** N 个并行工作者，每个从任务板认领一个子任务。每个产生一个新鲜的 `git worktree add` 分支加 Daytona 沙箱。实现子任务。发出带补丁 + 测试差值的 `diff_ready`。

4. **合并协调者。** 所有程序员完成后，将 N 个分支三方合并到暂存分支。仅当存在文件级重叠时才进行 LLM 中介冲突解决。

5. **审查者。** GPT-5.4 读取合并后的差异。不能批准它编写的差异。发出 `approved`（无操作）或带路由回相关程序员的具体更改请求的 `review_feedback`。

6. **测试者。** Gemini 2.5 Pro 在干净沙箱中运行测试套件。捕获工件。发出带堆栈跟踪的 `test_passed` 或 `test_failed`。失败的测试循环回到拥有失败子任务的程序员。

7. **交接计账。** 每个跨角色边界的消息在 Langfuse 中获得一个带有效负载大小和使用模型的 span。计算每子任务 token 放大（coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens）。

8. **评估。** 在 50 个 SWE-bench Pro 问题上运行。将 pass@1 和每个解决问题的美元成本与单智能体基线（一个 Sonnet 4.7 在单个工作树中）比较。

9. **事后分析。** 对于每个失败的问题，识别破坏的交接（计划太模糊、合并冲突、审查者误批、测试者不稳定）。生成交接失败直方图。

## 使用它

```
$ team run --issue https://github.com/acme/widget/issues/842
[架构师] 计划：4 个子任务（解析器、缓存、API、迁移）
[任务板]  并行分发给 4 个程序员的工作树
[程序员-A]  子任务解析器 -> 42 行，本地测试通过
[程序员-B]  子任务缓存   -> 88 行，本地测试通过
[程序员-C]  子任务 API   -> 31 行，本地测试通过
[程序员-D]  子任务迁移   -> 19 行，本地测试通过
[合并]      3 方合并：0 个冲突
[审查者]   对缓存的评论（线程池大小）；路由给程序员-B
[程序员-B]  修订：92 行；提交
[审查者]   批准
[测试者]   全部 412 个测试通过
[pr]       已开启 #3382   4 名程序员，1 次修订，$4.90，18 分钟
```

## 交付它

`outputs/skill-multi-agent-team.md` 是可交付成果。给定问题 URL 和并行度级别，团队产生一个带每角色 token 计账的待合并 PR。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 匹配的 50 个问题子集，pass@1 |
| 20 | 并行加速 | 挂钟时间 vs 单智能体基线 |
| 20 | 审查质量 | 注入 bug 探测上的误批率 |
| 20 | Token 效率 | 每个解决问题的总 token 数 vs 单智能体 |
| 15 | 协调工程 | 合并冲突解决，交接失败直方图 |
| **100** | | |

## 练习

1. 在运行中途向差异注入一个明显的 bug（在主体之前额外的 `return None`）。测量审查者的误批率。调整审查者提示词，直到误批低于 5%。

2. 减少到两个程序员（架构师 + 程序员 + 审查者 + 测试者，程序员顺序运行两个子任务）。比较挂钟时间和通过率。

3. 用单一写入者约束（子任务触及不相交的文件集）替换合并协调者。测量架构师的规划负担。

4. 将审查者从 GPT-5.4 换为 Claude Opus 4.7。测量误批率和 token 成本差值。

5. 添加第五个角色：文档编写者（Haiku 4.5）。审查后，它生成变更日志条目。测量文档质量是否值得额外的 token 花费。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Parallel worktree（并行工作树） | "隔离分支" | `git worktree add` 为每个程序员产生一个新鲜的工作树 |
| Task board（任务板） | "共享消息总线" | 智能体订阅的类型消息的文件或 Redis 存储 |
| Handoff（交接） | "角色边界" | 从一个角色上下文越过到另一个的任何消息 |
| Token amplification（Token 放大） | "多智能体开销" | 跨角色的总 token / 同一任务的单智能体 token |
| A2A protocol（A2A 协议） | "智能体到智能体" | Google 2025 年用于类型化智能体间消息的规范 |
| Merge coordinator（合并协调者） | "集成者" | 运行三方合并并调解冲突的组件 |
| False approval（误批） | "审查者幻觉" | 审查者批准了已知有 bug 的差异 |

## 延伸阅读

- [SWE-AF 工厂架构](https://github.com/Agent-Field/SWE-AF) — 参考 2026 年多智能体工厂
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — 基于角色的多智能体框架
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoft 的类型化 Actor 框架
- [Cognition AI（Devin）](https://cognition.ai) — 参考产品
- [Factory Droids](https://www.factory.ai) — 备用参考产品
- [Google A2A 协议](https://developers.google.com/agent-to-agent) — 智能体间消息规范
- [git worktree 文档](https://git-scm.com/docs/git-worktree) — 隔离基础
- [SWE-bench Pro](https://www.swebench.com) — 评估目标

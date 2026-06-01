# 压轴项目 09——代码迁移智能体（代码库级语言/运行时升级）（Capstone 09 — Code Migration Agent: Repo-Level Language / Runtime Upgrade）

> Amazon 的 MigrationBench（Java 8 到 17）和 Google 的 App Engine Py2-to-Py3 迁移器设定了 2026 年的标准。Moderne 的 OpenRewrite 在规模上进行确定性 AST 重写。Grit 用 codemod 风格 DSL 针对同样的问题。生产模式将两者结合：确定性基础用于安全重写，加上智能体层处理模糊情况，沙箱用于每个分支的构建，以及在 PR 开启之前变为绿色的测试测试框架。本压轴项目是迁移 50 个真实仓库并发布带失败分类的通过率。

**类型：** 压轴项目  
**语言：** Python（智能体），Java / Python（目标），TypeScript（仪表板）  
**前置知识：** Phase 5（NLP）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（智能体）、Phase 15（自主系统）、Phase 17（基础设施）  
**涉及的阶段：** P5 · P7 · P11 · P13 · P14 · P15 · P17  
**预计时间：** 30 小时

## 问题所在

大规模代码迁移是 2026 年编程智能体最干净的生产应用之一。基准真相是明显的（迁移后测试套件是否通过？），回报是真实的（Java 8 集群迁移是一个需要投入人力的项目），基准是公开的（MigrationBench 50 仓库子集）。Moderne 的 OpenRewrite 处理确定性部分。智能体层处理 OpenRewrite 配方无法解决的一切：模糊重写、构建系统漂移、长尾语法、传递依赖破坏。

你将构建一个智能体，接受 Java 8 仓库（或 Python 2 仓库）并生成绿色 CI 迁移分支。你将测量通过率、测试覆盖率保留、每个仓库的成本，并构建失败分类。与仅确定性基线的并排比较告诉你智能体的价值实际在哪里。

## 核心概念

管道有两层。**确定性基础**（Java 的 OpenRewrite，Python 的 libcst）安全地运行大部分机械重写：导入、方法签名、null 安全编辑、try-with-resources、已弃用 API 替换。它速度快且产生可审计的差异。**智能体层**（OpenAI Agents SDK 或基于 Claude Opus 4.7 和 GPT-5.4-Codex 的 LangGraph）处理配方无法解决的情况：构建文件升级（Maven/Gradle/pyproject）、传递依赖冲突、测试不稳定、自定义注解。

每个仓库都有一个预安装目标运行时的 Daytona 沙箱。智能体迭代：运行构建、分类失败、应用修复、重新运行。硬限制：每个仓库 30 分钟，$8，20 个智能体轮次。如果所有测试通过且覆盖率差值不为负，则分支开启 PR。否则，仓库被归入带证据的失败类别。

失败分类是可交付成果。在 50 个仓库中，什么出了问题？传递依赖？自定义注解？构建工具版本？与迁移无关的测试不稳定？每个类别都有数量和示例差异。未来的配方作者可以针对前三个。

## 架构

```
目标仓库
      |
      v
OpenRewrite / libcst 确定性配方
   （安全、快速、可审计，约 70-80% 的修复）
      |
      v
每个分支的 Daytona 沙箱
      |
      v
智能体循环（Claude Opus 4.7 / GPT-5.4-Codex）：
   - 运行构建 -> 捕获失败
   - 分类失败（构建、测试、代码检查）
   - 应用修复（补丁或重试配方）
   - 重新运行
   - 预算：30 分钟，$8，20 轮
      |
      v
测试 + 覆盖率差值门控
      |
      v（通过）
开启 PR
      |
      v（失败）
归入失败类别 + 附加重现证据
```

## 技术栈

- 确定性基础：OpenRewrite（Java）或 libcst（Python）
- 智能体：OpenAI Agents SDK 或基于 Claude Opus 4.7 + GPT-5.4-Codex 的 LangGraph
- 沙箱：每个分支的 Daytona devcontainers，预安装目标运行时（Java 17 / Python 3.12）
- 构建系统：Maven、Gradle、uv（Python）
- 基准：Amazon MigrationBench 50 仓库子集（Java 8 到 17），Google App Engine Py2-to-Py3 仓库
- 测试测试框架：并行运行器，Java 通过 Jacoco / Python 通过 coverage.py 进行覆盖率
- 可观测性：Langfuse + 每个仓库带每个差异块的追踪包
- 仪表板：带每类数量和示例差异的失败分类仪表板

## 构建它

1. **配方通。** 先运行 OpenRewrite（Java）或 libcst（Python）配方。捕获 70-80% 的机械迁移。作为"配方"提交。

2. **构建试验。** Daytona 沙箱：安装目标运行时，运行构建。如果绿色，跳到测试。如果红色，移交给智能体。

3. **智能体循环。** LangGraph，带工具：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。智能体分类失败（依赖、语法、测试、构建工具）并应用有针对性的修复。重新运行。

4. **预算上限。** 每个仓库 30 分钟挂钟，$8 成本，20 个智能体轮次。任何超出都停止并在当前差异下归入"budget_exhausted"。

5. **测试 + 覆盖率门控。** 构建变为绿色后，运行测试套件。将覆盖率与基础仓库比较。如果覆盖率下降超过 2%，归入"coverage_regression"。

6. **开启 PR。** 成功时，推送分支，开启 PR，带有差异和应用了哪些配方以及智能体编写了哪些提交的摘要。

7. **失败分类。** 对于每个失败的仓库，用类别标记：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建仪表板。

8. **50 仓库运行。** 在 MigrationBench 子集上执行。报告每类通过率、每仓库成本、覆盖率保留，以及与仅确定性基线的比较。

## 使用它

```
$ migrate legacy-java-service --target java17
[配方]   27 个重写已应用（JUnit 4->5，HashMap 初始化器，try-with-resources）
[构建]    失败：找不到符号 sun.misc.BASE64Encoder
[智能体]  轮次 1 分类：removed_jdk_api
[智能体]  轮次 2 应用：sun.misc.BASE64Encoder -> java.util.Base64
[构建]    成功
[测试]    412/412 通过；覆盖率 84.1% -> 84.3%
[pr]      已开启 #1841  成本=$3.20  轮次=4
```

## 交付它

`outputs/skill-migration-agent.md` 是可交付成果。给定一个仓库，它执行确定性配方然后智能体循环以产生绿色迁移分支，或将仓库归入分类类别。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 仓库子集 pass@1 |
| 20 | 测试覆盖率保留 | 与基础的平均覆盖率差值 |
| 20 | 每个迁移仓库的成本 | 通过运行的每仓库美元成本 |
| 20 | 智能体/确定性工具集成 | OpenRewrite 处理的修复比例 vs 智能体编写的 |
| 15 | 失败分析报告 | 带示例的分类完整性 |
| **100** | | |

## 练习

1. 仅用 OpenRewrite 运行迁移管道（无智能体）。将通过率与完整管道比较。识别智能体单独起作用的情况。

2. 实现"代码检查干净"检查：迁移后，运行风格代码检查器（Java 的 spotless，Python 的 ruff）。如果出现新的代码检查错误，则使 PR 失败。测量覆盖率保留但风格退化的比率。

3. 添加"最小差异"优化器：智能体分支通过测试后，用第二个通道修剪不必要的更改。报告差异大小减少。

4. 扩展到第三个迁移：Node 18 到 Node 22。复用沙箱封装；将配方层换为自定义 codemod。

5. 测量首次绿色构建时间（TTFGB）作为用户体验指标。目标：p50 低于 10 分钟。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Deterministic substrate（确定性基础） | "配方引擎" | OpenRewrite / libcst：带安全保证的声明性 AST 重写 |
| Codemod | "代码修改程序" | 机械地更改源代码的重写规则 |
| Build drift（构建漂移） | "工具版本偏差" | Maven/Gradle/uv 主要版本之间的微妙行为变化 |
| Failure class（失败类别） | "分类桶" | 仓库未迁移的标记原因：依赖、语法、测试、构建工具、预算 |
| Coverage delta（覆盖率差值） | "覆盖率保留" | 基础到迁移分支的测试覆盖率 % 变化 |
| Agent turn（智能体轮次） | "工具调用轮次" | 智能体循环中的一个规划 -> 行动 -> 观察周期 |
| Budget exhaustion（预算耗尽） | "达到上限" | 仓库消耗了 30 分钟/$8/20 轮限制而没有通过 |

## 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 标准性 2026 基准
- [Moderne.io OpenRewrite 平台](https://www.moderne.io) — 确定性基础参考
- [OpenRewrite 文档](https://docs.openrewrite.org) — 配方编写
- [Grit.io](https://www.grit.io) — 备用 codemod DSL
- [OpenAI 沙箱迁移食谱](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK 参考
- [Google App Engine Py2 到 Py3 迁移器](https://cloud.google.com/appengine) — 备用迁移基准
- [libcst](https://github.com/Instagram/LibCST) — Python 确定性基础
- [Daytona sandboxes](https://daytona.io) — 参考每分支沙箱

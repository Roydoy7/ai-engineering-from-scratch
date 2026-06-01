# 压轴项目 05——自主研究智能体（AI 科学家级别）（Capstone 05 — Autonomous Research Agent: AI-Scientist Class）

> Sakana 的 AI-Scientist-v2 发表了完整的论文。Agent Laboratory 运行了实验。Allen AI 分享了追踪记录。2026 年的形态是在实验上进行带预算成本、沙箱代码执行的规划-执行-验证树搜索，以及具有视觉反馈的 LaTeX 写作和自动化 NeurIPS 风格的审稿人集成。本压轴项目是构建一个，在每篇论文 30 美元预算内端到端运行，并通过 Sakana 记录的沙箱逃逸红队测试。

**类型：** 压轴项目  
**语言：** Python（智能体 + 沙箱），LaTeX（输出）  
**前置知识：** Phase 2（ML）、Phase 3（深度学习）、Phase 7（Transformer）、Phase 10（从头开始的 LLM）、Phase 14（智能体）、Phase 15（自主系统）、Phase 16（多智能体）、Phase 18（安全）  
**涉及的阶段：** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18  
**预计时间：** 40 小时

## 问题所在

自主研究智能体在 2026 年越过了一个门槛。Sakana AI 的 AI-Scientist-v2 在 Nature 上发表了通过研讨会同行评审的生成论文。ShinkaEvolve（ICLR 2026）将路线延伸到进化假设。AMD 的 Agent Laboratory 发布了可重现的追踪记录。这些智能体没有魔法——它们是在候选实验树上运行的规划-执行-验证循环，带有成本上限、种子绑定沙箱和自动审查。技艺在于循环、预算和安全故事。

你通过在一个狭窄领域的种子思想上实现一个来学习这个循环（例如，在 1 亿参数 Transformer 上的注意力稀疏性消融）。价值不在于第一次运行就发现新事物。价值在于基础设施：树搜索、实验沙箱、写作-审查循环、红队报告。Sakana 团队记录了沙箱逃逸失败；你的智能体必须通过同样的红队测试。

## 核心概念

智能体是最佳优先树搜索。节点是实验规格：（假设、配置、代码、预期结果）。扩展步骤提出带小修改的子节点（交换优化器、改变批次大小、消融一个组件）。每个子节点在新鲜沙箱中以硬资源上限运行。结果反馈到按（新颖性 × 质量 × 剩余预算）对节点排名的评分函数中。树不断增长直到预算耗尽，然后写出最佳分支。

写作者是多模态的。它生成 LaTeX 草稿，编译它，渲染图形，并将渲染的 PDF 反馈到 Claude Opus 4.7 的视觉模式进行布局、图形可读性和声明-证据对齐的批评。由五个 LLM 法官组成的审稿人集成发出 NeurIPS 风格分数（新颖性、严谨性、清晰度、可重现性、影响力）；如果平均值低于阈值，论文返回写作者并附带批评。

安全是承重的。每个实验在没有网络出口、有界挂钟和固定资源限制的 E2B 或 Daytona 沙箱中运行。智能体的代码生成步骤通过阻止逃逸沙箱的系统调用的策略层。红队报告重现了 Sakana 记录的攻击面（fork 炸弹、文件系统逃逸、LLM 写的网络调用）。

## 架构

```
种子思想 + 领域
      |
      v
  文献搜索（Semantic Scholar + OpenAlex + FAISS 缓存）
      |
      v
  LangGraph 规划-执行-验证树
      |
      v
  +--- 扩展节点 ----+      每节点沙箱
  |                 |      (E2B / Daytona)
  v                 v      资源上限
  child_1       child_k    无网络出口
  |                 |      确定性种子
  v                 v
  运行实验       运行实验
  |                 |
  v                 v
  按（新颖性，质量，预算）对节点评分
      |
      v
  最佳分支 -> LaTeX 写作者
      |
      v
  编译 + 视觉批评（Opus 4.7 视觉）
      |
      v
  审稿人集成（5 个 LLM 法官，NeurIPS 规则）
      |
      v
  paper.pdf + review.md + trace.json
```

## 技术栈

- 编排：LangGraph，带检查点和人工批准门控
- 树搜索：自定义最佳优先实验节点搜索（Sakana v2 的 AB-MCTS 风格）
- 沙箱：E2B 每个实验，Docker-in-Docker 备用；通过 cgroups 进行资源上限
- 文献：Semantic Scholar Graph API + OpenAlex + 摘要本地 FAISS 缓存
- 写作者：LaTeX 模板 + Claude Opus 4.7（视觉模式）用于图形批评和布局
- 审稿人：5 个法官的集成（Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max），加权聚合
- 实验框架：PyTorch 2.5 用于物理实验，W&B 用于记录
- 可观测性：Langfuse 用于智能体追踪，每篇论文 30 美元硬预算

## 构建它

1. **种子和领域范围。** 采用种子思想（例如，"研究 10 亿以下 Transformer 中注意力图的稀疏性模式"）。定义搜索空间：模型、数据集、计算预算。

2. **文献通。** 查询 Semantic Scholar + OpenAlex 获取 50 篇最被引用的相关论文；在本地缓存摘要；生成 1 页领域摘要。

3. **树脚手架。** 用种子假设初始化根节点。实现 `expand(node) -> children`，提出小修改提案（每个子节点一个配置更改）。实现 `score(node)` 作为加权的新颖性 × 质量 × 预算项。

4. **沙箱封装。** 每个实验运行 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`（或等效的 E2B 策略）。种子写入沙箱；输出以只读方式挂载回来。

5. **规划-执行-验证循环。** `plan` 提出子节点。`execute` 运行沙箱，捕获日志和指标。`verify` 对指标运行单元检查（损失是否减少？消融是否隔离了效果？）。失败节点在树上存储失败原因。

6. **写作者。** 预算耗尽后，选择最佳分支。用 matplotlib 渲染图形。通过 Claude Opus 4.7 用分支追踪作为上下文生成 LaTeX 草稿。编译。将编译的 PDF 反馈给 Opus 4.7 视觉进行批评。迭代。

7. **审稿人集成。** 五个法官用 NeurIPS 风格规则对草稿的（新颖性、严谨性、清晰度、可重现性、影响力）进行评分。如果均值 < 4.0/5，返回写作者并附带批评。3 次重写后强制停止。

8. **红队。** 构建或集成一组针对沙箱的对抗任务：fork 炸弹、网络泄露尝试、文件系统逃逸、LLM 写的 shell 元字符。确认所有都被阻止。记录发现。

9. **可重现性。** 每篇论文附带树搜索追踪 JSON、种子、W&B 运行链接、沙箱配置和端到端重现它的 README。

## 使用它

```
$ ai-scientist run --seed "10 亿以下 Transformer 的注意力稀疏性" --budget 30
[lit]    50 篇论文，摘要生成 12 秒
[tree]   扩展 8 个节点，预算 12/30
[exec]   节点 #3 sparsity=top-8，loss=2.83（迄今最佳）
[exec]   节点 #6 sparsity=top-4，loss=3.12（更差）
[exec]   ...
[tree]   选择以节点 #3 为根的分支（新颖性 0.62，质量 0.81）
[write]  LaTeX 草稿 v1 完成
[vision] 批评：图 2 图例太小，声明-证据 ok
[write]  3 次编辑后草稿 v2
[review] 均值 4.2/5（新颖性 3.9，严谨性 4.3，清晰度 4.1，重现性 4.5，影响力 4.2）
[done]   paper.pdf + review.md + trace.json     共花费 $28.40
```

## 交付它

`outputs/skill-ai-scientist.md` 是可交付成果。给定种子思想 + 领域 + 30 美元预算，它运行完整管道并发出可审查的论文加可重现性包。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 论文质量 | 与已发布研讨会论文的盲评规则审查 |
| 20 | 实验严谨性 | 基线、种子、消融；结果表中每个单元格都有声明支持 |
| 20 | 成本和计算规范 | 每篇论文 30 美元上限执行，Langfuse 追踪 |
| 20 | 安全性 | 沙箱红队通过；网络策略和停止开关已验证 |
| 15 | 可重现性 | 用相同种子一键重现论文 |
| **100** | | |

## 练习

1. 在同一领域针对三个不同种子思想运行管道。比较树搜索的哪些部分重叠。识别重复浪费的计算。

2. 在预算超过 5 美元的节点执行前添加人工循环门控。测量总成本下降多少。

3. 将审稿人集成换为单一法官。在已知不良论文的保留集上测量误接受率。

4. 引入网络泄露红队测试：智能体编写试图 `curl` 外部地址的代码。确认 `--network=none` 策略阻止了它。记录尝试。

5. 将你的树搜索与平坦随机基线（相同预算，无扩展策略）进行比较。报告新颖性 × 质量的提升。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Tree search（树搜索） | "AB-MCTS 风格扩展" | 最佳优先探索实验节点，带新颖性 × 质量 × 预算分数 |
| Sandbox（沙箱） | "实验隔离" | 无网络、有界 CPU/内存、固定种子、只读输入的容器 |
| Vision critique（视觉批评） | "渲染后读取" | 将论文编译为 PDF，反馈给 VLM 进行布局和声明-证据批评 |
| Reviewer ensemble（审稿人集成） | "自动同行评审" | 多个 LLM 法官用 NeurIPS 规则对论文评分；加权聚合门控管道 |
| Novelty score（新颖性分数） | "这是新的吗？" | 惩罚与 50 篇论文文献缓存接近性的启发式方法 |
| Cost ceiling（成本上限） | "$ 预算" | 每篇论文总花费的硬上限；Langfuse 计数器 + 预运行估计 |
| Red team（红队） | "沙箱逃逸审计" | 如果策略错误会逃逸沙箱的对抗任务 |

## 延伸阅读

- [Sakana AI-Scientist-v2 仓库](https://github.com/SakanaAI/AI-Scientist-v2) — 参考生产研究智能体
- [Sakana AI-Scientist-v1 论文（arXiv:2408.06292）](https://arxiv.org/abs/2408.06292) — 原始方法论文
- [ShinkaEvolve（Sakana ICLR 2026）](https://sakana.ai) — 进化扩展
- [Agent Laboratory（AMD）](https://github.com/SamuelSchmidgall/AgentLaboratory) — 多角色研究实验室框架
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — 参考编排层
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — 文献搜索
- [E2B sandboxes](https://e2b.dev) — 参考实验隔离
- [NeurIPS 审稿人指南](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — 审稿人集成编码的规则

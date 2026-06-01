# 对齐研究生态系统——MATS、Redwood、Apollo、METR（Alignment Research Ecosystem — MATS, Redwood, Apollo, METR）

> 五个组织定义了 2026 年非实验室对齐研究层。MATS（ML 对齐与理论学者）：自 2021 年底以来 527 名以上的研究人员，180 多篇论文，超过 1 万次引用，h 指数 47；2024 年夏季队列作为 501(c)(3) 注册，约 90 名学者和 40 名导师；80% 的 2025 年前校友在安全/保障领域工作，200 多人在 Anthropic、DeepMind、OpenAI、英国 AISI、RAND、Redwood、METR、Apollo 任职。Redwood Research：由 Buck Shlegeris 创立的应用对齐实验室；引入了 AI 控制（第 10 课）；与英国 AISI 合作开展控制安全案例。Apollo Research：为前沿实验室提供部署前谋划评估；撰写了上下文内谋划（第 8 课）和面向 AI 谋划的安全案例。METR（模型评估和威胁研究）：基于任务的能力评估、自主任务时间跨度研究；"前沿 AI 安全政策的共同要素"比较了实验室框架。Eleos AI Research：模型福利预部署评估（第 19 课）；进行了 Claude Opus 4 福利评估。

**类型：** 学习  
**语言：** 无  
**前置知识：** Phase 18 · 01-27（Phase 18 前面的课程）  
**预计时间：** 约 45 分钟

## 学习目标

- 识别非实验室对齐研究生态系统的五个组织及其核心产出。
- 描述 MATS 的规模（学者、论文、h 指数）及其作为人才管道的作用。
- 描述 Redwood 的 AI 控制议程及其与英国 AISI 的合作。
- 描述 METR 的基于任务的评估方法论。

## 问题所在

前沿实验室（第 18 课）在内部产生安全评估并发布部分结果。实验室外部的生态系统是评估被验证的地方，是首先发现新故障模式的地方，也是培养人才的地方。了解这个生态系统有助于解读哪些研究发现受到谁的信任。

## 核心概念

### MATS（ML 对齐与理论学者）

2021 年底开始。研究导师制项目；学者在特定对齐问题上与资深研究员共度 10-12 周。

规模（2026 年）：
- 自成立以来 527 名以上的研究人员。
- 已发表 180 多篇论文。
- 超过 1 万次引用。
- h 指数 47。
- 2024 年夏季：90 名学者 + 40 名导师；注册为 501(c)(3)。

职业结果：约 80% 的 2025 年前校友在安全/保障领域工作。200 多人在 Anthropic、DeepMind、OpenAI、英国 AISI、RAND、Redwood、METR、Apollo 任职。

### Redwood Research

应用对齐实验室。由 Buck Shlegeris 创立。引入了 AI 控制议程（第 10 课）。与英国 AISI 合作开展控制安全案例。为 DeepMind 和 Anthropic 的评估设计提供咨询。

标志性论文：Greenblatt、Shlegeris 等人，"AI 控制"（arXiv:2312.06942，ICML 2024）；对齐伪装（Greenblatt、Denison、Wright 等人，arXiv:2412.14093，与 Anthropic 联合）。

风格：具体的威胁模型、最坏情况对手、可以进行压力测试的具体协议。

### Apollo Research

为前沿实验室提供部署前谋划评估。撰写了上下文内谋划（第 8 课，arXiv:2412.04984）。2025 年 OpenAI 反谋划训练合作的合作伙伴。产出面向 AI 谋划的安全案例（2024 年）。

风格：欺骗可能涌现的智能体场景评估；三支柱分解（错位、目标导向性、情境意识）。

### METR（模型评估和威胁研究）

基于任务的能力评估。自主任务完成时间跨度研究。"前沿 AI 安全政策的共同要素"（metr.org/common-elements，2025 年）比较了实验室框架。

与 Apollo 共同撰写 AI 谋划安全案例草图。

风格：长时域任务评估、实证能力测量、框架综合。

### Eleos AI Research

模型福利预部署评估。进行了系统卡第 5.3 节记录的 Claude Opus 4 福利评估。为第 19 课与福利相关的声明提供外部方法论检查。

### 流程

MATS 培训研究人员。毕业生前往 Anthropic、DeepMind、OpenAI（实验室安全团队）或 Redwood、Apollo、METR、Eleos（外部评估）。外部评估者与实验室以及英国 AISI / CAISI 合作。出版物反馈到生态系统，为下一届 MATS 队列服务。

### 为什么这一层很重要

单一来源的评估是不可靠的：实验室评估自己的模型存在结构性利益冲突。外部评估者可以提出并验证实验室可能低估的失败模式。2024 年的潜伏智能体论文（第 7 课）是 Anthropic + Redwood 的合作；对齐伪装是 Anthropic + Redwood 的合作；上下文内谋划是 Apollo 的工作；反谋划是 Apollo + OpenAI 的合作。多组织结构是质量控制。

### 这在 Phase 18 中的位置

第 7-11 课引用了 Redwood 和 Apollo 的工作；第 18 课引用了 METR 的框架比较；第 19 课引用了 Eleos。第 28 课是其余部分所依赖的生态系统的明确组织地图。

## 使用它

没有代码。阅读 METR 的"前沿 AI 安全政策的共同要素"，作为外部综合如何为实验室内部政策工作增加价值的例子。

## 交付它

本课产出 `outputs/skill-ecosystem-map.md`。给定对齐声明或评估，识别组织、发表平台和方法论风格，并与已知的对应组织进行交叉检查。

## 练习

1. 从第 7-15 课中选择一篇论文，识别涉及的组织。将作者与 MATS 校友和当前生态系统从属关系进行交叉检查。

2. 阅读 METR 的"前沿 AI 安全政策的共同要素"。识别他们强调的三个跨实验室趋同点和两个最大的分歧。

3. MATS 的职业结果约为 80% 在安全/保障领域。论证这种选择压力是适应性的（训练这个领域）还是有偏见的（过滤掉异端立场）。

4. Redwood 和 Apollo 都做控制/谋划工作，但风格不同。选择一个故障模式，描述每个机构如何调查它。

5. Eleos AI 是唯一纯粹的模型福利组织。设计一个假想的专注于不同福利相关问题的第二个组织（认知自由、机器人实体化等），并阐明其方法论。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| MATS | "导师制项目" | ML 对齐与理论学者；自 2021 年以来 527 名以上研究人员 |
| Redwood Research | "控制实验室" | 应用对齐；AI 控制作者；英国 AISI 合作伙伴 |
| Apollo Research | "谋划评估" | 为前沿实验室提供部署前谋划评估 |
| METR | "任务时域评估" | 基于任务的能力评估；框架综合 |
| Eleos AI | "福利实验室" | 模型福利预部署评估 |
| Talent pipeline（人才管道） | "MATS -> 实验室" | MATS 毕业生流向 Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| External evaluation（外部评估） | "非实验室检查" | 不由模型生产者完成的评估；增加可信度 |

## 延伸阅读

- [MATS（ML 对齐与理论学者）](https://www.matsprogram.org/) — 导师制项目
- [Redwood Research](https://www.redwoodresearch.org/) — AI 控制论文
- [Apollo Research](https://www.apolloresearch.ai/) — 谋划评估
- [METR — 前沿 AI 安全政策的共同要素](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 框架比较
- [Eleos AI Research](https://www.eleosai.org/research) — 模型福利方法论

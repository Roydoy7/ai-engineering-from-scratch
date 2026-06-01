# LLM 功能的 A/B 测试——GrowthBook、Statsig 与感觉问题（A/B Testing LLM Features — GrowthBook, Statsig, and the Vibes Problem）

> 传统 A/B 测试并非为非确定性 LLM 而生。关键区别：评估回答的是"模型能做这个工作吗？"A/B 测试回答的是"用户在乎吗？"两者都是必需的；凭感觉发布的时代已经过去。2026 年要测试的内容：提示词工程（措辞）、模型选择（GPT-4 vs GPT-3.5 vs OSS；准确率 vs 成本 vs 延迟）、生成参数（温度、top-p）。真实案例：聊天机器人奖励模型变体带来 +70% 对话长度和 +30% 留存率；Nextdoor AI 主题行实验在奖励函数改进后带来 +1% CTR；Khan Academy Khanmigo 在延迟与数学准确率之间迭代。平台分化：**Statsig**（2025 年 9 月以 11 亿美元被 OpenAI 收购）——顺序测试、CUPED、一体化。**GrowthBook**——开源，数据仓库原生，贝叶斯 + 频率论 + 顺序引擎，CUPED，SRM 检查，Benjamini-Hochberg + Bonferroni 校正。你的选择基于数据仓库 SQL 偏好以及"被 OpenAI 收购"对你的组织是否重要。

**类型：** 学习  
**语言：** Python（标准库，玩具顺序测试模拟器）  
**前置知识：** Phase 17 · 13（可观测性）、Phase 17 · 20（渐进式部署）  
**预计时间：** 约 60 分钟

## 学习目标

- 区分评估（"模型能做这个工作吗"）和 A/B 测试（"用户在乎吗"）。
- 列举三个可测试维度（提示词、模型、参数），并为每个选择指标。
- 解释 CUPED、顺序测试和 Benjamini-Hochberg 多重比较校正。
- 根据数据仓库 SQL 立场和公司收购立场选择 Statsig 或 GrowthBook。

## 问题所在

你手动调整了系统提示词。感觉更好了。你发布了。转化率在噪音范围内变化。你归咎于指标。或者你发布了一个新模型，转化率没有变化——是模型退化了，还是变化太小无法检测？你不知道，因为你发布时没有做 A/B 测试。

评估回答的是模型能否在标注集上完成任务。它们不能回答用户是否更喜欢输出。只有受控在线实验才能回答这个问题，而且只有当实验有足够的统计力、控制了非确定性并校正了多重比较时。

## 核心概念

### 评估 vs A/B 测试

**评估** — 离线，标注集，裁判（评分标准或 LLM 作为裁判或人类）。回答："在这个固定分布上输出是否正确/有帮助/安全？"

**A/B 测试** — 在线，真实用户，随机化。回答："新变体是否改变了重要的用户级指标？"

两者都是必需的。评估在暴露前捕获回归；A/B 在暴露后确认产品影响。

### 要测试什么

1. **提示词工程** — 措辞、系统提示词结构、示例。指标：任务成功、用户留存、每请求成本。
2. **模型选择** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：准确率（任务）+ 每请求成本 + 延迟 P99。多目标。
3. **生成参数** — 温度、top-p、max_tokens。指标：特定任务（输出多样性 vs 确定性）。

### CUPED——方差减少

使用实验前数据的受控实验（Controlled-experiments Using Pre-Experiment Data）。在比较后期之前回归掉前期方差。典型方差减少：30-70%。有效样本量免费提升。

实现：Statsig 和 GrowthBook 都实现了。

### 顺序测试

经典 A/B 假设固定样本量。顺序测试（"peek-and-decide"）在反复查看时控制假阳性率。始终有效的顺序程序（mSPRT，Howard 的置信序列）允许在明确赢家出现时提前停止。

### 多重比较校正

以 95% 置信度运行 20 个 A/B 测试偶然产生一个假阳性。Bonferroni 校正按测试收紧 α；Benjamini-Hochberg 控制错误发现率。GrowthBook 实现了两者。

### SRM——样本比例不匹配

分配哈希将用户随机分配到变体。如果 50/50 分配出现 47/53，说明出了问题——SRM 检查会标记它。两个平台都实现了。

### Statsig vs GrowthBook

**Statsig**：
- 2025 年 9 月以 11 亿美元被 OpenAI 收购。托管，SaaS。
- 顺序测试、CUPED、保留人群。
- 一体化：特性标志 + 实验 + 可观测性。
- 最适合：团队想要捆绑产品，不介意 OpenAI 所有权。

**GrowthBook**：
- 开源（MIT）；数据仓库原生（直接从 Snowflake/BigQuery/Redshift 读取）。
- 多引擎：贝叶斯、频率论、顺序。
- CUPED、SRM、Bonferroni、BH 校正。
- 自托管或托管云。
- 最适合：数据仓库 SQL 团队，数据团队控制指标层，需要开源。

### 非确定性使统计力复杂化

相同提示词产生不同输出。传统统计力计算假设独立同分布观测。有 LLM 非确定性时，有效样本量低于名义样本量。将所需样本量乘以约 1.3-1.5 倍作为安全边际。

### 真实案例结果

- 聊天机器人奖励模型变体：+70% 对话长度，+30% 留存率。
- Nextdoor 主题行：奖励函数改进后 +1% CTR。
- Khan Academy Khanmigo：延迟与数学准确率的迭代权衡。

### 反模式：凭感觉发布

每个资深工程师都能说出一个"感觉更好"就发布、没有做 A/B 测试的功能。大多数都在团队几个月未察觉的情况下导致了产品指标回归。A/B 测试是强制约束。

### 你应该记住的数字

- Statsig 被 OpenAI 收购：11 亿美元，2025 年 9 月。
- GrowthBook：开源 MIT；贝叶斯 + 频率论 + 顺序。
- CUPED 方差减少：30-70%。
- LLM 非确定性 → +30-50% 样本量缓冲。

## 使用它

`code/main.py` 模拟固定边界和顺序边界的顺序 A/B 测试。展示顺序测试如何允许提前停止。

## 交付它

本课产出 `outputs/skill-ab-plan.md`。给定功能变更、工作负载、基线，选择平台、门控、样本量。

## 练习

1. 运行 `code/main.py`。对于期望 5% 提升、基线 3% 转化率，80% 统计力需要多大样本？
2. 为受医疗监管的本地部署客户选择 Statsig 或 GrowthBook。
3. 设计一个测试 GPT-4 vs GPT-3.5 在每个已解决工单成本上的 A/B 测试。主要指标、护栏指标、次要指标各是什么？
4. 你的金丝雀通过了，但 A/B 显示 -1.2% 转化率。你是否发布？写出升级标准。
5. 将 CUPED 应用于前期方差为后期 60% 的场景。计算有效样本量提升。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Eval（评估） | "离线测试" | 模型能力的标注集评估 |
| A/B test（A/B 测试） | "实验" | 针对用户的实时随机对比 |
| CUPED | "方差减少" | 用前期回归减少方差 |
| Sequential test（顺序测试） | "可窥视测试" | 允许提前停止的始终有效程序 |
| Multiple comparison（多重比较） | "族错误" | 运行多个测试会放大假阳性 |
| Bonferroni | "严格校正" | 将 α 除以测试数量 |
| Benjamini-Hochberg | "BH FDR" | 错误发现率控制，不那么保守 |
| SRM | "错误分割" | 样本比例不匹配；分配错误 |
| Statsig | "OpenAI 拥有" | 商业一体化，2025 年收购 |
| GrowthBook | "开源的那个" | MIT 数据仓库原生平台 |
| mSPRT | "顺序概率比测试" | 经典顺序程序 |

## 延伸阅读

- [GrowthBook — 如何 A/B 测试 AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — 超越提示词：数据驱动的 LLM 优化](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook 比较](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng 等人 — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — 置信序列](https://arxiv.org/abs/1810.08240)

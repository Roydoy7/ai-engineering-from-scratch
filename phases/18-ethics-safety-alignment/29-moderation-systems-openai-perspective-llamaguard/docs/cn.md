# 审核系统——OpenAI、Perspective、Llama Guard（Moderation Systems — OpenAI, Perspective, Llama Guard）

> 生产审核系统将第 12-16 课中定义的安全策略付诸实施。OpenAI Moderation API：`omni-moderation-latest`（2024 年）建立在 GPT-4o 上，在一次调用中对文本 + 图像进行分类；在多语言测试集上比前一版本好 42%；响应模式返回 13 个类别布尔值——骚扰、骚扰/威胁、仇恨、仇恨/威胁、非法、非法/暴力、自我伤害、自我伤害/意图、自我伤害/指示、性、性/未成年人、暴力、暴力/图形；对大多数开发者免费。分层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。异步并行调用隐藏延迟；标记时使用占位符响应。Llama Guard 3/4（第 16 课）：14 个 MLCommons 危害，代码解释器滥用，8 种语言（v3），多图像（v4）。Perspective API（谷歌 Jigsaw）：先于 LLM 作为审核员浪潮的毒性评分；主要是带有严重毒性/侮辱/亵渎变体的单维毒性；内容审核研究的基线。弃用：Azure 内容审核器于 2024 年 2 月弃用，2027 年 2 月退役，由 Azure AI 内容安全替代。

**类型：** 构建  
**语言：** Python（标准库，三层审核测试框架）  
**前置知识：** Phase 18 · 16（Llama Guard / Garak / PyRIT）  
**预计时间：** 约 60 分钟

## 学习目标

- 描述 OpenAI Moderation API 的类别分类法及其与 Llama Guard 3 的 MLCommons 集合的差异。
- 描述三层审核模式（输入、输出、自定义），并说出每层的一个失败模式。
- 描述 Perspective API 作为前 LLM 时代基线的地位，以及为什么它在研究中仍然被使用。
- 说明 Azure 的弃用时间表。

## 问题所在

第 12-16 课描述了攻击和防御工具。第 29 课涵盖将防御在用户接触产品的界面处付诸实施的已部署审核系统。三层模式是 2026 年的默认配置。

## 核心概念

### OpenAI Moderation API

`omni-moderation-latest`（2024 年）。建立在 GPT-4o 上。在一次调用中对文本 + 图像进行分类。对大多数开发者免费。

类别（响应模式中的 13 个布尔值）：
- 骚扰、骚扰/威胁
- 仇恨、仇恨/威胁
- 自我伤害、自我伤害/意图、自我伤害/指示
- 性、性/未成年人
- 暴力、暴力/图形
- 非法、非法/暴力

多模态支持适用于 `violence`、`self-harm` 和 `sexual`，但不适用于 `sexual/minors`；其余是纯文本。

在 `code/main.py` 的代码测试框架中，为了教学简便，我们将 `/threatening`、`/intent`、`/instructions` 和 `/graphic` 子类别折叠到其顶级父类别中。生产代码应使用完整的 13 类别模式。

在多语言测试集上比上一代审核端点好 42%。每个类别的分数；应用程序设置阈值。

### Llama Guard 3/4

在第 16 课中介绍。14 个 MLCommons 危害类别（与 OpenAI 的 13 个响应模式布尔值的组织方式不同）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）原生多模态，12B。

OpenAI 和 Llama Guard 的分类法有重叠但也有分歧。OpenAI 有"非法"作为广义类别；Llama Guard 分别有"暴力犯罪"和"非暴力犯罪"。部署者根据其政策-分类法的契合度进行选择。

### Perspective API（谷歌 Jigsaw）

先于 LLM 作为审核员浪潮的毒性评分系统（2020 年前）。类别：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。单维主分数（TOXICITY）带有子维度变体。

在内容审核研究中广泛用作基线，因为 API 稳定、有文档，并且有多年的校准数据。对于现代 LLM 相邻的使用场景，Llama Guard 或 OpenAI Moderation 通常更合适。

### 三层模式

1. **输入审核。** 在生成前对用户的提示词进行分类。如果被标记则拒绝。延迟：一次分类器调用。
2. **输出审核。** 在交付前对模型的输出进行分类。如果被标记则替换为拒绝响应。延迟：生成后的一次分类器调用。
3. **自定义审核。** 领域特定规则（正则表达式、允许列表、业务政策）。在输入或输出处运行。

这三层按设计是顺序的：输入审核必须在生成前完成，输出审核在生成后运行。并行性适用于层内——在同一文本上并发运行多个分类器（例如，OpenAI Moderation + Llama Guard + Perspective）隐藏了每个分类器的延迟。作为可选优化，在输入审核完成和 token-1 流式传输被推迟期间，可以显示占位符响应（"请稍候，正在检查..."）。标记行为是可配置的：拒绝、清洗或升级到人工审查。

### 失败模式

- **仅输入。** 不捕获输出幻觉（第 12-14 课的编码攻击绕过输入分类器）。
- **仅输出。** 允许任何输入到达模型；增加成本；将内部推理暴露给攻击者。
- **仅自定义。** 跨类别不鲁棒；正则表达式很脆弱。

分层是默认选择。安全加保险绳。

### Azure 弃用

Azure 内容审核器：2024 年 2 月弃用，2027 年 2 月退役。由 Azure AI 内容安全替代，后者基于 LLM 并与 Azure OpenAI 集成。对于 Azure 部署，迁移是一个 2024-2027 年的现场级项目。

### 这在 Phase 18 中的位置

第 16 课在红队测试背景下涵盖了审核工具。第 29 课涵盖已部署的审核。第 30 课以当前的双重用途能力证据结束。

## 使用它

`code/main.py` 构建了三层审核测试框架：输入审核器（关键词 + 类别分数）、输出审核器（对输出使用相同分类器）、自定义审核器（领域规则）。你可以通过运行输入并观察哪一层捕获了什么。

## 交付它

本课产出 `outputs/skill-moderation-stack.md`。给定部署，推荐一个审核栈配置：输入时用哪个分类器，输出时用哪个，哪些自定义规则，以及边缘情况的裁判是什么。

## 练习

1. 运行 `code/main.py`。通过所有三层运行无害、边界线和有害输入。报告哪一层对每个输入触发。

2. 用 Perspective API 风格的毒性评分扩展测试框架，针对特定类别。将其阈值行为与类别分数进行比较。

3. 阅读 OpenAI Moderation API 文档和 Llama Guard 3 类别列表。将每个 OpenAI 类别映射到最接近的 Llama Guard 类别。识别三个无法清晰映射的类别。

4. 为代码助手部署（例如，GitHub Copilot）设计审核栈。识别最相关和最不相关的类别，并提出自定义规则。

5. Azure 内容审核器于 2027 年 2 月退役。规划迁移到 Azure AI 内容安全的计划。识别迁移中最高风险的元素。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| OpenAI Moderation | "omni-moderation-latest" | 基于 GPT-4o 的 13 类别（文本）分类器，带有部分多模态支持 |
| Perspective API | "谷歌 Jigsaw 毒性" | 前 LLM 时代的毒性评分基线 |
| Llama Guard | "MLCommons 14 类别" | Meta 的危害分类器（v3：8B 文本，8 种语言；v4：12B 多模态） |
| Input moderation（输入审核） | "生成前过滤器" | 模型调用前对用户提示词的分类器 |
| Output moderation（输出审核） | "生成后过滤器" | 交付前对模型输出的分类器 |
| Custom moderation（自定义审核） | "领域规则" | 部署特定规则（正则表达式、允许列表、政策） |
| Layered moderation（分层审核） | "所有三层" | 标准生产部署模式 |

## 延伸阅读

- [OpenAI Moderation API 文档](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation 端点
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard 仓库
- [谷歌 Jigsaw Perspective API](https://perspectiveapi.com/) — 毒性评分
- [Azure AI 内容安全](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure 替代品

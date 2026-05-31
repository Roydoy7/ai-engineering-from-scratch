# 文档与图表理解（Document and Diagram Understanding）

> 文档不是照片。PDF、科学论文、发票或手写表单有布局、表格、图表、脚注、标题和语义结构，纯图像理解无法捕捉。VLM 之前的栈是一个流水线：Tesseract OCR + LayoutLMv3 + 表格提取启发式方法。VLM 浪潮以无 OCR 模型取而代之——Donut（2022）、Nougat（2023）、DocLLM（2023）——这些模型直接输出结构化标记。到 2026 年，前沿是"以 2576px 原生分辨率将页面图像送给 Claude Opus 4.7"，结构化标记输出随之而来。本章解读文档 AI 的三个时代弧线。

**类型：** 构建  
**语言：** Python（标准库，布局感知文档解析器骨架）  
**前置知识：** Phase 12 · 05（LLaVA）、Phase 5（NLP）  
**预计时间：** 约 180 分钟

## 学习目标

- 解释文档 AI 的三个时代：OCR 流水线、无 OCR、VLM 原生。
- 描述 LayoutLMv3 的三个输入流：文本、布局（边界框）、图像图块，以及统一掩码机制。
- 对比 Donut（无 OCR，图像→标记）、Nougat（科学论文→LaTeX）、DocLLM（布局感知生成）、PaliGemma 2（VLM 原生）。
- 为新任务（发票、科学论文、手写表单、中文收据）选择合适的文档模型。

## 问题所在

"理解这个 PDF"欺骗性地难。信息存在于：

- 文本内容（90% 的信号）。
- 布局（标题、脚注、侧边栏、双栏格式）。
- 表格（行、列、合并单元格）。
- 图表和图形。
- 手写注释。
- 字体和排版（标题 vs 正文）。

原始 OCR 提取文本并丢失其余部分。一个关注发票的系统需要知道"总计：1245 美元"来自右下角，而非来自脚注。

## 核心概念

### 第一时代——OCR 流水线（2021 年前）

经典栈：

1. PDF → 每页图像。
2. Tesseract（或商业 OCR）提取带每词边界框的文本。
3. 布局分析器识别块（标题、表格、段落）。
4. 表格结构识别器解析表格。
5. 领域规则 + 正则表达式提取字段。

对干净的印刷文本有效。在手写、倾斜扫描、复杂表格、非英语文字上失效。每种失效模式都需要自定义异常路径。

### TrOCR（2021）

TrOCR（Li 等，arXiv:2109.10282）用在合成 + 真实文本图像上训练的 Transformer 编码器-解码器替换了 Tesseract 的经典 CNN-CTC。在手写和多语言文本上明显提升。仍然是流水线（检测器，然后 TrOCR，然后布局），但 OCR 步骤大幅改进。

### 第二时代——无 OCR（2022-2023）

第一批无 OCR 模型说：完全跳过检测，直接将图像像素映射到结构化输出。

**Donut**（Kim 等，arXiv:2111.15664）：
- 编码器-解码器 Transformer，编码器是 Swin-B。
- 输出为表单理解的 JSON、摘要的 Markdown，或任何任务专属模式。
- 没有 OCR，没有布局，没有检测。

**Nougat**（Blecher 等，arXiv:2308.13418）：
- 专门针对科学论文训练。
- 输出为 LaTeX/Markdown。
- 处理公式、多栏布局、图表。
- 每个 arXiv 解析器都在调用的模型。

这些是专家模型，不是通才。Donut 用于科学论文会失败；Nougat 用于发票会失败。

### LayoutLMv3（2022）

不同的技术路线。LayoutLMv3（Huang 等，arXiv:2204.08387）保留 OCR 但增加布局理解：

- 三个输入流：OCR 文本 token、每 token 的二维边界框、图像图块。
- 跨三种模态的掩码训练目标（掩码文本、掩码图块、掩码布局）。
- 下游应用：分类、实体提取、表格问答。

LayoutLMv3 是基于 OCR 的文档理解的巅峰。在表单和发票上表现强劲。需要上游 OCR。在标准化文档基准上是 VLM 之前准确率最高的方案。

### DocLLM（2023）

DocLLM（Wang 等，arXiv:2401.00908）是 LayoutLM 的生成式兄弟。生成以布局 token 为条件的自由格式答案。更适合文档问答；仍然依赖 OCR 输入。

### 第三时代——VLM 原生（2024 年及以后）

2024 年的 VLM 已经足够好，可以完全取代流水线。将全页图像以高分辨率送给 VLM，提问，得到答案。

- LLaVA-NeXT 336 图块 AnyRes 适用于小型文档。
- Qwen2.5-VL 动态分辨率原生处理 2048+ 像素。
- Claude Opus 4.7 支持 2576px 文档。
- PaliGemma 2（2025 年 4 月）专门针对文档 + 手写训练。

VLM 原生与 OCR 流水线之间的差距迅速缩小。到 2026 年，VLM 原生在以下方面获胜：

- 场景文本（手写 + 印刷，混合文字）。
- 带合并单元格的复杂表格。
- 嵌入文本中的数学公式。
- 带文字注释的图表。

OCR 流水线仍在以下方面获胜：

- 大规模纯扫描工作负载，每页延迟很重要。
- 流水线可靠性（确定性失败 vs VLM 幻觉）。
- 需要可审计 OCR 输出的监管环境。

### Claude 4.7 / GPT-5 前沿

在 2576 像素原生输入下，前沿 VLM 以接近人类的准确率进行文档理解。2026 年初的基准数字：

- DocVQA：Claude 4.7 约 95.1，PaliGemma 2 约 88.4，Nougat 约 77.3，流水线 LayoutLMv3 约 83。
- ChartQA：Claude 4.7 约 92.2，GPT-4V 约 78。
- VisualMRC：Claude 4.7 约 94。

封闭模型差距主要在于分辨率和基础 LLM 规模。7B 开放模型落后几个点，但差距在缩小。

### 数学公式与 LaTeX 输出

科学论文需要公式的精确 LaTeX 输出。Nougat 在此基础上训练。带 LaTeX 目标训练的 VLM（Qwen2.5-VL-Math、Nougat 衍生物）能产生可用的 LaTeX。如果没有明确的 LaTeX 训练，VLM 产生可读但不精确的转录。

2026 年科学论文流水线：对 PDF 先用 Nougat，然后对复杂页面用 VLM。

### 手写

仍然是最难的子任务。混合印刷 + 手写（医生笔记、填写表单）是 OCR 流水线在成本上仍然优于 VLM 的地方。纯手写 VLM 正在改进（Claude 4.7、PaliGemma 2）。

### 2026 年方案

对于新的文档 AI 项目：

- **大规模纯印刷发票：** LayoutLMv3 + 规则，成本效益高。
- **混合文档（科学 + 手写 + 表单）：** VLM 原生（PaliGemma 2 或 Qwen2.5-VL）。
- **全量 arXiv 摄取：** Nougat 处理数学，VLM 处理图表。
- **监管要求：** OCR 流水线 + VLM 验证器交叉检查。

## 动手使用

`code/main.py`：

- 一个玩具布局感知分词器：给定（文本，边界框）对，产生 LayoutLMv3 风格的输入。
- 一个 Donut 风格的任务模式生成器：表单的 JSON 模板。
- 对比 OCR 流水线、Donut、Nougat 和 VLM 原生在每页 token 预算上的差异。

## 输出产物

本章生成 `outputs/skill-document-ai-stack-picker.md`。给定文档 AI 项目（领域、规模、质量、监管要求），在 OCR 流水线、无 OCR 专家和 VLM 原生之间做出选择。

## 练习

1. 你的项目每天处理 1000 万张发票。哪种栈在不损失准确率的情况下最小化每页成本？

2. 为什么 LayoutLMv3 在表单问答上优于纯 CLIP-VLM，但在场景文本上表现更差？边界框流放弃了什么？

3. Nougat 生成 LaTeX。提出一个 VLM 原生输出在 LaTeX 保真度上击败 Nougat 的测试用例，以及一个 Nougat 获胜的用例。

4. 阅读 PaliGemma 2 论文（Google，2024）。提升文档准确率（相比 PaliGemma 1）的关键训练数据补充是什么？

5. 设计一个合规安全的混合方案：OCR 流水线作为主要，VLM 作为辅助交叉检查。如何解决分歧？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| OCR 流水线（OCR pipeline） | "Tesseract 风格" | 分阶段栈：检测 → OCR → 布局 → 规则；确定性，但脆弱。 |
| 无 OCR（OCR-free） | "Donut 风格" | 跳过显式 OCR 的图像到输出 Transformer；单一模型。 |
| 布局感知（Layout-aware） | "LayoutLM" | 输入包含每 token 的边界框坐标；跨模态统一掩码。 |
| VLM 原生（VLM-native） | "前沿 VLM" | 直接以高分辨率将页面图像送给 Claude/GPT/Qwen VLM；无流水线。 |
| DocVQA | "文档基准" | 文档视觉问答标准；引用最广泛的分数。 |
| 标记输出（Markup output） | "LaTeX/MD" | 结构化输出格式而非自由格式文本；支持下游自动化。 |

## 延伸阅读

- [Li 等 — TrOCR（arXiv:2109.10282）](https://arxiv.org/abs/2109.10282)
- [Blecher 等 — Nougat（arXiv:2308.13418）](https://arxiv.org/abs/2308.13418)
- [Huang 等 — LayoutLMv3（arXiv:2204.08387）](https://arxiv.org/abs/2204.08387)
- [Kim 等 — Donut（arXiv:2111.15664）](https://arxiv.org/abs/2111.15664)
- [Wang 等 — DocLLM（arXiv:2401.00908）](https://arxiv.org/abs/2401.00908)

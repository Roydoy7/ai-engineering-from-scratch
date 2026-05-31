# ColPali 与视觉原生文档 RAG（ColPali and Vision-Native Document RAG）

> 传统 RAG 将 PDF 解析为文本、分割为块、嵌入块、存储向量。每一步都会丢失信号：OCR 丢弃图表数据，分块打断表格行，文本嵌入忽视图表。ColPali（Faysse 等，2024 年 7 月）提出了一个更简单的问题：为什么要提取文本？通过 PaliGemma 直接嵌入页面图像，使用 ColBERT 风格的迟交互检索，保留文档携带的所有布局、图形、字体和格式信号。公开基准：在视觉丰富的文档上，端到端准确率比文本 RAG 高出 20-40%。ColQwen2、ColSmol 和 VisRAG 延伸了这一模式。本章解读视觉原生 RAG 的论点，并构建一个小型 ColPali 风格的索引器。

**类型：** 构建  
**语言：** Python（标准库，多向量索引器 + MaxSim 评分器）  
**前置知识：** Phase 11（LLM 工程——RAG 基础）、Phase 12 · 05（LLaVA）  
**预计时间：** 约 180 分钟

## 学习目标

- 解释双编码器检索（每个文档一个向量）与迟交互检索（每个文档多个向量）的区别。
- 描述 ColBERT 的 MaxSim 操作，以及 ColPali 如何将其从文本 token 推广到图像图块。
- 构建一个小型 ColPali 风格的索引器：页面 → 图块嵌入 → 对查询词嵌入的 MaxSim → 前 k 个页面。
- 对比 ColPali + Qwen2.5-VL 生成器与文本 RAG + GPT-4 在发票/财务报告用例上的表现。

## 问题所在

对 PDF 进行文本 RAG 会丢弃大部分文档内容。财务报告的第三季度营收增长通常在图表中；医疗报告的发现在注释图像中；法律合同的签名栏是布局事实，而非文本事实。

文本 RAG 流水线：

1. PDF → 通过 OCR/pdftotext 转文本。
2. 文本 → 300-500 token 块。
3. 块 → 双编码器嵌入（一个向量）。
4. 用户查询 → 嵌入 → 余弦相似度 → 前 k 个块。
5. 块 + 查询 → LLM。

五个有损步骤。图表未被捕获。表格跨块断开。多栏布局被展平。图形注释消失。

ColPali 的修复：跳过 OCR，直接嵌入页面图像。使用 ColBERT 风格的迟交互检索，使模型在查询时可以关注细粒度图块。

## 核心概念

### ColBERT（2020）

ColBERT（Khattab & Zaharia，arXiv:2004.12832）是一种文本检索方法。它不是每个文档一个向量，而是每个 token 一个向量。在查询时：

- 查询 token 获得各自的嵌入（N_q 个向量）。
- 文档 token 获得嵌入（N_d 个向量，通常缓存）。
- 得分 = 查询 token 的求和（对文档 token 取最大余弦相似度）：Σ_i max_j cos(q_i, d_j)。

这是 MaxSim 操作。每个查询 token "选择"最匹配的文档 token。最终得分是求和。

优点：强召回率，处理词级语义。缺点：每个文档有 N_d 个向量，存储开销大。

### ColPali

ColPali（Faysse 等，arXiv:2407.01449）将 ColBERT 模式应用于图像。

- 每个页面由 PaliGemma（ViT + 语言）编码为图块嵌入：每页 N_p 个向量。
- 每个用户查询（文本）编码为查询 token 嵌入：N_q 个向量。
- 得分 = Σ_i max_j cos(q_i, p_j)，即对查询文本 token 和页面图像图块的 MaxSim。
- 按总得分检索前 k 个页面。

文档摄取时：用 PaliGemma 嵌入每个页面，存储所有图块嵌入。查询时：嵌入查询 token，对所有已索引页面嵌入计算 MaxSim，返回前 k 个页面。

优点：端到端在视觉丰富的文档上比文本 RAG 高出 20-40%。每个图块向量捕获局部布局和内容。

缺点：每页 N_p 个图块 × 4 字节浮点 × D 维向量 = 存储快速增长。可通过 PQ/OPQ 量化缓解。

### ColQwen2 和 ColSmol

ColQwen2（illuin-tech，2024-2025）将 PaliGemma 换成 Qwen2-VL。更好的基础编码器，更好的检索效果。

ColSmol 是用于本地/边缘设备的小型变体。约 10 亿参数的 ColSmol 检索器可在消费级 GPU 上运行。

### VisRAG

VisRAG（Yu 等，arXiv:2410.10594）是一种不同的变体：不在图块上做 MaxSim，而是用 VLM 将每个页面池化为单个向量，然后用双编码器检索。索引更快、存储更小，但召回率较弱。

质量与成本的权衡：质量选 ColPali，规模选 VisRAG。

### M3DocRAG

M3DocRAG（Cho 等，arXiv:2411.04952）将多模态检索扩展到多页多文档推理。跨文档检索页面，为 VLM 组合多页上下文。

### ViDoRe——基准

ColPali 的配套基准。视觉文档检索评估。任务包括财务报告、科学论文、行政文档、医疗记录、手册。指标：nDCG@5。

ColPali-v1 在 ViDoRe 上得分约 80% nDCG@5；同样文档上的文本 RAG 得分约 50-60%。

### 端到端 RAG 流水线

视觉原生 RAG：

1. **摄取：** PDF → 页面图像 → PaliGemma 编码 → 存储所有图块嵌入。
2. **查询：** 用户文本 → 查询 token 嵌入 → 对所有已索引页面的 MaxSim → 前 k 个页面。
3. **生成：** 前 k 个页面图像 + 查询 → VLM（Qwen2.5-VL 或 Claude）→ 答案。

全程无 OCR。图表、图形、字体、布局全部流入答案。

### 存储数学

一份 50 页财务报告，每页 729 个图块，128 维嵌入：

- **ColPali：** 50 × 729 × 128 × 4 字节 ≈ 18 MB 原始，PQ 后约 4 MB。
- **文本 RAG：** 50 块 × 768 维 × 4 字节 ≈ 150 kB。

ColPali 每个文档约多占用 30 倍存储。大规模下，OPQ/PQ 可降至约 5-10 倍，通常可以接受。

### 文本 RAG 仍然获胜的情况

- 无布局信号的纯文本文档（维基文章、聊天记录）。文本 RAG 更简单，存储更便宜。
- 存储成本主导的百万页以上档案。
- 需要可提取 OCR 文本与检索并存的严格监管要求。

2026 年其他一切情况——财务报告、科学论文、法律合同、医疗记录、UX 文档——视觉原生 RAG 获胜。

## 动手使用

`code/main.py`：

- 玩具图块编码器：将"页面"（特征向量的小网格）映射为图块嵌入数组。
- MaxSim 评分器：计算查询 token 嵌入集与页面图块集之间的 ColBERT 风格得分。
- 索引 5 个玩具页面，运行 3 个查询，返回带得分的前 k 个结果。

## 输出产物

本章生成 `outputs/skill-vision-rag-designer.md`。给定文档 RAG 项目，在 ColPali/ColQwen2/VisRAG/文本 RAG 之间做出选择，并估算存储规模。

## 练习

1. 一份 200 页年度报告，每页 729 个图块，128 维嵌入，4 字节浮点。计算原始存储和 PQ 压缩（8 倍）后的存储。

2. MaxSim 是 Σ_i max_j cos(q_i, p_j)。这个求和捕获了简单均值相似度无法捕获的什么？

3. ColPali 以图块集形式索引页面。如果改为在词语级别（如 ColBERT 所做的）索引会有什么变化？有哪些权衡？

4. 为 100 万页语料库设计端到端流水线，查询延迟预算为 500ms。选择 ColQwen2/VisRAG 并论证。

5. 阅读 M3DocRAG（arXiv:2411.04952）。描述多页注意力模式以及它与单页 ColPali 检索的不同之处。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 迟交互（Late interaction） | "ColBERT 风格" | 使用每 token 或每图块嵌入 + MaxSim 检索，而非单一文档向量。 |
| MaxSim | "图块上取最大" | 对每个查询 token，选择相似度最高的文档 token；跨查询求和。 |
| 双编码器（Bi-encoder） | "单向量" | 每个文档一个向量；更快，但失去粒度。 |
| 多向量（Multi-vector） | "每文档多向量" | 每个文档/页面存储 N_p 个向量；存储成本增加，但召回率提升。 |
| 图块嵌入（Patch embedding） | "页面特征" | 来自 VLM 编码器的每个图像图块对应的一个向量，每页缓存。 |
| ViDoRe | "视觉文档基准" | ColPali 的视觉文档检索基准套件。 |
| PQ 量化（PQ quantization） | "乘积量化" | 在保持向量相似度的同时将存储压缩约 8 倍的压缩方法。 |

## 延伸阅读

- [Faysse 等 — ColPali（arXiv:2407.01449）](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT（arXiv:2004.12832）](https://arxiv.org/abs/2004.12832)
- [Yu 等 — VisRAG（arXiv:2410.10594）](https://arxiv.org/abs/2410.10594)
- [Cho 等 — M3DocRAG（arXiv:2411.04952）](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)

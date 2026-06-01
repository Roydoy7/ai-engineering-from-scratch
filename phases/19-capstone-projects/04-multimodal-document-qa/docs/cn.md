# 压轴项目 04——多模态文档问答（视觉优先 PDF、表格、图表）（Capstone 04 — Multimodal Document QA: Vision-First PDF, Tables, Charts）

> 2026 年的文档问答前沿从 OCR-然后-文本转向视觉优先的后期交互。ColPali、ColQwen2.5 和 ColQwen3-omni 将每个 PDF 页面视为图像，用多向量后期交互嵌入，并让查询直接关注图像块。对于金融 10-K 报告、科学论文和手写笔记，这种模式大幅优于 OCR 优先方法。在 1 万页上端到端构建管道，并与 OCR-然后-文本进行并排对比发布。

**类型：** 压轴项目  
**语言：** Python（管道），TypeScript（查看器 UI）  
**前置知识：** Phase 4（计算机视觉）、Phase 5（NLP）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 12（多模态）、Phase 17（基础设施）  
**涉及的阶段：** P4 · P5 · P7 · P11 · P12 · P17  
**预计时间：** 30 小时

## 问题所在

企业存有 OCR 管道会破坏的 PDF：带有旋转表格的扫描 10-K 报告、充满公式的科学论文、只有作为图像才有意义的图表、手写注释。将这些视为文本优先意味着失去一半信号。2026 年的答案是对原始页面图像进行后期交互多向量检索。ColPali（Illuin Tech）引入了这一方法；ColQwen2.5-v0.2 和 ColQwen3-omni 提升了准确率。在 ViDoRe v3 上，视觉优先检索在得分上大幅高于 OCR-然后-文本——在图表、表格和手写方面差距更大。

权衡在于存储和延迟。ColQwen 嵌入每页约 2048 个图像块向量，而非单个 1024 维向量。原始存储膨胀。DocPruner（2026 年）在没有可测量准确率损失的情况下实现 50% 压缩。你将索引 1 万页，测量 ViDoRe v3 nDCG@5，在 2 秒内提供答案，并直接与 OCR-然后-文本基线进行比较。

## 核心概念

后期交互意味着每个查询 token 对每个图像块 token 进行评分，并对每个查询 token 的最大分数求和。你无需单个池化向量就能获得细粒度匹配。多向量索引（Vespa、Qdrant 多向量或 AstraDB）存储每个图像块的嵌入，并在检索时运行 MaxSim。

回答器是一个视觉语言模型，接受查询加上 top-k 检索页面作为图像，并写出带有证据区域（边界框或页面引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿选择。对于公式和科学符号，OCR 回退（Nougat、dots.ocr）作为可选文本通道拼接进来。

评估是二维矩阵。一轴：内容类型（纯文本段落、密集表格、条形/折线图、手写笔记、公式）。另一轴：检索方法（视觉优先后期交互 vs OCR-然后-文本 vs 混合）。每个单元格得到 nDCG@5 和回答准确率。报告是可交付成果。

## 架构

```
PDF -> 页面渲染器（PyMuPDF，180 DPI）
           |
           v
  ColQwen2.5-v0.2 嵌入（每页多向量，约 2048 图像块）
           |
           +------> DocPruner 50% 压缩
           |
           v
   多向量索引（Vespa 或 Qdrant 多向量）
           |
查询 ----+----> 检索 top-k 页面（MaxSim）
           |
           v
  VLM 回答器：Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    输入：查询 + top-k 页面图像 + 可选 OCR 文本
           |
           v
  带引用页码 + 证据区域的答案
           |
           v
  Streamlit / Next.js 查看器：在源页面上高亮显示方框
```

## 技术栈

- 页面渲染：PyMuPDF（fitz），180 DPI，纵向规范化
- 后期交互模型：ColQwen2.5-v0.2 或 ColQwen3-omni（Hugging Face 上的 vidore 团队）
- 索引：Vespa 带多向量字段，或 Qdrant 多向量，或 AstraDB 带 MaxSim
- 压缩：DocPruner 2026 策略（保留高方差图像块，50% 压缩，准确率损失 < 0.5%）
- OCR 回退（公式/密集表格）：dots.ocr 或 Nougat
- VLM 回答器：Qwen3-VL-30B 自托管或 Gemini 2.5 Pro 托管；InternVL3 作为备用
- 评估：ViDoRe v3 基准，M3DocVQA 用于多页推理
- 查看器 UI：Next.js 15，带证据区域的画布覆盖层

## 构建它

1. **摄入。** 遍历跨 10-K 报告、科学论文和扫描文档的 1 万 PDF 页面语料库。将每页渲染为 1536x2048 PNG。持久化 `{doc_id, page_num, image_path}`。

2. **嵌入。** 对每个页面图像运行 ColQwen2.5-v0.2。输出形状约 2048 个 128 维图像块嵌入。应用 DocPruner 保留信号最高的一半。写入 Vespa 多向量字段或 Qdrant 多向量。

3. **查询。** 对每个传入查询，用查询塔嵌入（token 级嵌入）。对索引运行 MaxSim：对每个查询 token，取页面图像块嵌入上的最大点积，求和。返回 top-k 页面。

4. **合成。** 用查询和 top-5 页面图像调用 Qwen3-VL-30B。提示词："仅使用提供的页面回答。通过（doc_id，页面）引用每个声明，并命名区域（图形、表格、段落）。"

5. **证据区域。** 后处理答案以提取引用区域。如果 VLM 发出边界框（Qwen3-VL 会这样做），在查看器中将它们渲染为覆盖层。

6. **OCR 回退。** 对于被识别为公式密集的页面（基于图像方差的启发式方法），运行 Nougat 或 dots.ocr 并将 OCR 文本作为额外通道与图像一起传递。

7. **评估。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页问答准确率）。还在同一语料库上使用同一合成器运行 OCR-然后-文本管道。生成内容类型 × 方法矩阵。

8. **UI。** 先做 Streamlit 原型；Next.js 15 生产查看器，带逐页证据区域覆盖层。

## 使用它

```
$ doc-qa ask "EMEA 细分市场 2024 年运营利润率变化了多少？"
[检索]   320ms 内 top-5 页面（ColQwen2.5，MaxSim，Vespa）
[合成]   qwen3-vl-30b，1.4s，引用 (form-10k-2024, p. 88) + (..., p. 92)
答案：
  EMEA 运营利润率从 18.2% 降至 16.8%，下降了 140bp。
  引用：10-K-2024.pdf p.88（表 4，细分运营利润率）
         10-K-2024.pdf p.92（MD&A，运营业绩）
[查看器]  打开，p.88 表 4 上叠加高亮边界框
```

## 交付它

`outputs/skill-doc-qa.md` 描述了可交付成果：一个视觉优先的多模态文档问答系统，针对特定语料库调整，并在 ViDoRe v3 上与 OCR-然后-文本基线进行评估。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确率 | 基准数字 vs OCR-文本基线和已发布排行榜 |
| 20 | 证据区域基础 | 确实包含答案片段的引用区域比例 |
| 20 | 存储和延迟工程 | DocPruner 压缩比，索引 p95，答案 p95 |
| 20 | 多页推理 | 在手工标注的 100 个多页问题集上的准确率 |
| 15 | 来源检查用户体验 | 查看器清晰度，覆盖层保真度，并排比较工具 |
| **100** | | |

## 练习

1. 测量同一语料库上的 ColQwen2.5-v0.2 vs ColQwen3-omni。哪些页面一个对了另一个错了？向索引添加"内容类"标签以按类型路由。

2. 大幅压缩嵌入（75%，90%）。找到压缩悬崖：ViDoRe nDCG@5 降到 OCR 基线以下的点。

3. 构建混合方案：并行运行 OCR-然后-文本和 ColQwen，用 RRF 融合，用交叉编码器重新排序。混合方案是否优于任一单独方案？在哪里最有帮助？

4. 将 Qwen3-VL-30B 换为较小的 VLM（Qwen2.5-VL-7B）。测量每美元准确率曲线。

5. 添加手写笔记支持。渲染手写语料库，用 ColQwen 嵌入，测量检索。与手写 OCR 管道进行比较。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Late interaction（后期交互） | "ColPali 风格检索" | 查询 token 独立对页面图像块评分；MaxSim 聚合 |
| Multi-vector（多向量） | "每图像块嵌入" | 每个文档有许多向量，而非一个池化向量 |
| MaxSim | "后期交互评分" | 对每个查询 token，取文档向量上的最大相似度；求和 |
| DocPruner | "图像块压缩" | 2026 年压缩，保留 50% 图像块，准确率损失可忽略 |
| ViDoRe v3 | "文档检索基准" | 2026 年测量视觉文档检索的标准 |
| Evidence region（证据区域） | "引用边界框" | 源页面上定位答案片段的边界框 |
| OCR fallback（OCR 回退） | "公式通道" | 用于公式或表格密集页面的文本管道，与视觉并用 |

## 延伸阅读

- [ColPali（Illuin Tech）仓库](https://github.com/illuin-tech/colpali) — 参考后期交互文档检索
- [ColPali 论文（arXiv:2407.01449）](https://arxiv.org/abs/2407.01449) — 基础方法论文
- [Hugging Face 上的 ColQwen 系列](https://huggingface.co/vidore) — 生产就绪检查点
- [M3DocRAG（Adobe）](https://arxiv.org/abs/2411.04952) — 多页多模态 RAG 基线
- [Vespa 多向量教程](https://docs.vespa.ai/en/colpali.html) — 参考服务栈
- [Qdrant 多向量支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 备用索引
- [AstraDB 多向量](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 备用托管索引
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 支持公式的 OCR 回退

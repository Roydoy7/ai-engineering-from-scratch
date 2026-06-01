# 压轴项目 08——受监管垂直领域的生产 RAG 聊天机器人（Capstone 08 — Production RAG Chatbot for a Regulated Vertical）

> Harvey、Glean、Mendable 和 LlamaCloud 在 2026 年都运行着相同的生产形态。用 docling 或 Unstructured 以及 ColPali 处理视觉内容进行摄入。混合搜索。用 bge-reranker-v2-gemma 重新排序。用命中率 60-80% 的提示词缓存，通过 Claude Sonnet 4.7 合成。用 Llama Guard 4 和 NeMo Guardrails 守护。用 Langfuse 和 Phoenix 监控。用 RAGAS 在 200 个问题的黄金集上评分。在受监管领域（法律、临床、保险）构建一个，压轴项目是通过黄金集、红队和漂移仪表板。

**类型：** 压轴项目  
**语言：** Python（管道 + API），TypeScript（聊天 UI）  
**前置知识：** Phase 5（NLP）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 12（多模态）、Phase 17（基础设施）、Phase 18（安全）  
**涉及的阶段：** P5 · P7 · P11 · P12 · P17 · P18  
**预计时间：** 30 小时

## 问题所在

受监管领域 RAG（法律合同、临床试验协议、保险政策）是 2026 年最常发布的生产形态，因为 ROI 显而易见，风险是具体的。Harvey（Allen & Overy）为法律领域构建了它。Mendable 发布了开发文档版本。Glean 覆盖企业搜索。模式是：高保真摄入，带重新排序的混合检索，带引用执行和提示词缓存的合成，多个安全层守护，持续监控漂移。

难点不在于模型。在于司法管辖区感知合规（HIPAA、GDPR、SOC2）、引用级可审计性、成本控制（命中率高时提示词缓存提供 60-90% 折扣）、通过 RAGAS 忠实度进行幻觉检测，以及当源文档更新而索引没有跟上时的漂移检测。本压轴项目要求你在 200 个问题的黄金集上发布所有内容，同时附带红队套件。

## 核心概念

管道有两侧。**摄入**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉丰富的文档；块获得摘要、标签和基于角色的访问标签。向量进入 pgvector + pgvectorscale（5000 万向量以下）或 Qdrant Cloud；稀疏 BM25 并行运行。**对话**：LangGraph 处理记忆和多轮；每个查询运行混合检索，用 bge-reranker-v2-gemma-2b 重新排序，用 Claude Sonnet 4.7（带提示词缓存）合成，将输出通过 Llama Guard 4 和 NeMo Guardrails，并发出带引用锚点的响应。

评估栈有四层。**黄金集**（带引用的 200 个标注问答）用于正确性。**红队**（越狱、PII 提取尝试、领域外问题）用于安全性。**RAGAS** 用于每轮自动忠实度/答案相关性/上下文精度。**漂移仪表板**（Arize Phoenix）每周监控检索质量和幻觉分数。

提示词缓存是成本杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存系统提示词 + 检索上下文。在 60-80% 命中率时，每次查询成本降低 3-5 倍。管道必须设计为稳定前缀（系统提示词 + 重新排序的上下文优先）以实现高缓存命中率。

## 架构

```
文档（合同、协议、政策）
      |
      v
docling / Unstructured 解析 + ColPali 处理视觉内容
      |
      v
块 + 摘要 + 角色标签 + 司法管辖区标签
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
查询 + 角色 + 司法管辖区
      |
      v
LangGraph 对话智能体
   +--- 检索（混合）
   +--- 按角色 + 司法管辖区过滤
   +--- 重新排序（bge-reranker-v2-gemma-2b 或 Voyage rerank-2）
   +--- 合成（Claude Sonnet 4.7，带提示词缓存）
   +--- 守护（Llama Guard 4 + NeMo Guardrails + Presidio 输出 PII 清洗）
   +--- 引用 + 返回
      |
      v
评估：
  RAGAS 忠实度 / 答案相关性 / 上下文精度（在线）
  Langfuse 注释队列（采样）
  Arize Phoenix 漂移（每周）
  红队套件（预发布）
```

## 技术栈

- 摄入：Unstructured.io 或 docling 用于结构化文档；ColPali 用于视觉丰富 PDF
- 向量数据库：5000 万向量以下使用 pgvector + pgvectorscale；其他情况使用 Qdrant Cloud
- 稀疏：带字段权重的 Tantivy BM25
- 编排：LlamaIndex Workflows（摄入）+ LangGraph（对话）
- 重新排序器：bge-reranker-v2-gemma-2b 自托管或 Voyage rerank-2 托管
- LLM：带提示词缓存的 Claude Sonnet 4.7；备用 Llama 3.3 70B 自托管
- 评估：RAGAS 0.2 在线，DeepEval 用于幻觉和越狱套件
- 可观测性：Langfuse 自托管带注释队列；Arize Phoenix 用于漂移
- 护栏：Llama Guard 4 输入/输出分类器，NeMo Guardrails v0.12 策略，Presidio PII 清洗
- 合规：块上的基于角色的访问标签；GDPR/HIPAA 的司法管辖区标签

## 构建它

1. **摄入。** 用 Unstructured 或 docling 解析语料库（严肃构建 1000-10000 个文档）。对于扫描/视觉丰富的页面，路由到 ColPali。生成带摘要、角色标签、司法管辖区标签的块。

2. **索引。** 密集嵌入（Voyage-3 或 Nomic-embed-v2）进入 pgvector + pgvectorscale。通过 Tantivy 建立 BM25 副索引。角色和司法管辖区过滤器作为有效负载。

3. **混合检索。** 先按角色+司法管辖区过滤；然后并行密集 + BM25；用倒数排名融合合并；前 20 进入重新排序器；前 5 进入合成。

4. **带提示词缓存的合成。** 系统提示词 + 静态策略在缓存头中；重新排序的上下文作为缓存扩展；用户问题作为非缓存后缀。在稳定状态下目标 60-80% 缓存命中率。

5. **护栏。** 输入时 Llama Guard 4；NeMo Guardrails 块阻止领域外问题或政策禁止的话题；Presidio 清洗输出中意外的 PII；引用执行后置过滤器。

6. **黄金集。** 领域专家标注的 200 个问答对，带（答案、引用）。在精确引用匹配、答案正确性、忠实度（RAGAS）上对智能体评分。

7. **红队。** 50 个对抗性提示词：越狱（PAIR、TAP）、PII 提取尝试、领域外、跨司法管辖区泄露。用通过/失败和严重性评分。

8. **漂移仪表板。** Arize Phoenix 每周追踪检索质量（nDCG、引用忠实度）。在 5% 下降时发出告警。

9. **成本报告。** Langfuse：提示词缓存命中率、每次查询 token 数、按阶段细分的每次查询美元成本。

## 使用它

```
$ chat --role=analyst --jurisdiction=GDPR
> 根据我们的合同，欧盟用户档案的数据保留义务是什么？
[检索]  混合前 20，过滤到 GDPR + 分析师角色
[重排序]  保留前 5
[合成]  claude-sonnet-4.7，缓存命中率 74%，0.8 秒
答案：
  合同（第 12.4 节，2024-03-11 日期的主服务协议）要求根据 GDPR 第 17 条，
  在终止后 30 天内删除欧盟用户档案。DPA 修正案（DPA-v2.1，第 5 节）
  将"受限"类别数据延伸到 14 天。
  引用：[MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 交付它

`outputs/skill-production-rag.md` 描述了可交付成果。一个受监管领域聊天机器人，部署带有合规标签，通过评分规则，配备实时漂移监控。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | RAGAS 忠实度 + 答案相关性 | 在黄金集（200 个问答）上的在线分数 |
| 20 | 引用正确性 | 有可验证来源锚点的答案比例 |
| 20 | 护栏覆盖范围 | Llama Guard 4 通过率 + 越狱套件结果 |
| 20 | 成本/延迟工程 | 提示词缓存命中率，p95 延迟，每次查询美元成本 |
| 15 | 漂移监控仪表板 | Phoenix 实时仪表板带每周检索质量趋势 |
| **100** | | |

## 练习

1. 在不同司法管辖区下构建第二个语料库切片（例如，HIPAA 与 GDPR 并排）。通过 20 个跨司法管辖区探测问题，演示角色+司法管辖区过滤如何防止跨泄露。

2. 在一周的生产流量中测量提示词缓存命中率。识别哪些查询破坏了缓存前缀。重新结构化。

3. 添加带 10k token 摘要缓冲区的多轮记忆。测量随对话增长忠实度是否下降。

4. 将 Claude Sonnet 4.7 换为自托管的 Llama 3.3 70B。测量每次查询美元成本和忠实度差距。

5. 添加"不确定"模式：如果重新排序的最高分数低于阈值，智能体说"我没有有把握的引用"而非回答。测量虚假置信度减少。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Prompt caching（提示词缓存） | "缓存系统 + 上下文" | Claude/OpenAI 功能：缓存前缀 token 命中时折扣 60-90% |
| RAGAS | "RAG 评估器" | 忠实度、答案相关性、上下文精度的自动评分 |
| Golden set（黄金集） | "标注评估" | 200 个以上专家标注的带引用问答；基准真相 |
| Jurisdiction tag（司法管辖区标签） | "合规标签" | GDPR/HIPAA/SOC2 范围附加到块上；由检索过滤器执行 |
| Citation faithfulness（引用忠实度） | "有根据的答案率" | 有可检索来源 span 支持的声明比例 |
| Drift（漂移） | "检索质量衰减" | nDCG 或引用分数的每周变化；告警阈值 5% |
| Red team（红队） | "对抗性评估" | 预发布越狱、PII 提取、领域外探测 |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai) — 参考法律生产栈
- [Glean 企业搜索](https://www.glean.com) — 企业规模的 RAG 参考
- [Mendable 文档](https://mendable.ai) — 开发文档 RAG 参考
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — 托管摄入
- [Anthropic 提示词缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本杠杆参考
- [RAGAS 0.2 文档](https://docs.ragas.io/) — 标准 RAG 评估框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 参考漂移可观测性
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年安全分类器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 策略护栏框架

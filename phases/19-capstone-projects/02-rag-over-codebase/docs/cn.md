# 压轴项目 02——代码库 RAG（跨仓库语义搜索）（Capstone 02 — RAG over Codebase: Cross-Repo Semantic Search）

> 2026 年，每个认真的工程组织都运行着理解含义而非仅仅字符串的内部代码搜索。Sourcegraph Amp、Cursor 的代码库答案、Augment 的企业图、Aider 的 repomap、Pinterest 的内部 MCP——形态相同。摄入多个仓库，用 tree-sitter 解析，嵌入函数和类级别的块，混合搜索，重新排序，带引用地回答。本压轴项目要求你构建一个能处理 10 个仓库 200 万行代码并在每次 git push 时完成增量重新索引的系统。

**类型：** 压轴项目  
**语言：** Python（摄入），TypeScript（API + UI）  
**前置知识：** Phase 5（NLP 基础）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 13（工具）、Phase 17（基础设施）  
**涉及的阶段：** P5 · P7 · P11 · P13 · P17  
**预计时间：** 30 小时

## 问题所在

到 2026 年，每个前沿编程智能体都配备了代码库检索层，因为仅靠上下文窗口无法解决跨仓库问题。Claude 的 1M token 上下文有帮助；但它无法消除对排序检索的需求。对原始块的朴素余弦搜索会在生成的代码、在单体仓库重复和在很少导入的符号的长尾上污染结果。生产答案是对基于 AST 的块进行混合（密集 + BM25）搜索，配备重新排序器，并以符号引用图为支撑。

你通过索引真实的代码集群来学习这一点——不是一个教程仓库——并测量 MRR@10、引用忠实度和增量新鲜度。故障模式是基础设施层面的：10 万文件的单体仓库，触及一半文件的推送，需要跨四个仓库才能正确回答的查询。

## 核心概念

AST 感知的摄入管道用 tree-sitter 解析每个文件，提取函数和类节点，并在节点边界而非固定 token 窗口处分块。每个块得到三种表示：密集嵌入（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 词项，以及简短的自然语言摘要。摘要添加了第三种可检索模态——用户问"X 是如何授权的"，摘要提到"authz"，即使代码中只有 `check_permission`。

检索是混合的。一个查询同时触发密集和 BM25 搜索，合并 top-k，并将并集交给交叉编码器重新排序器（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重新排序后的列表进入长上下文合成器（带提示词缓存的 Claude Sonnet 4.7，或自托管的 Llama 3.3 70B），指令要求每个声明都引用文件和行范围。没有引用的答案被后置过滤器拒绝。

增量新鲜度是基础设施问题。git push 触发差异：哪些文件变了，哪些符号变了。只有受影响的块重新嵌入。受影响的跨文件符号边（导入、方法调用）被重新计算。索引保持一致，无需每次提交处理 200 万行代码。

## 架构

```
git push --> webhook --> 摄入工作者（LlamaIndex Workflow）
                           |
                           v
             tree-sitter 解析 + AST 分块
                           |
            +--------------+----------------+
            v              v                v
          密集          BM25 索引         摘要（LLM）
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      符号图（Neo4j / kuzu）
                            |
  查询 --> LangGraph 智能体（检索 -> 重排序 -> 合成）
                            |
                            v
                 Claude Sonnet 4.7 1M 上下文
                            |
                            v
                 答案 + 文件:行引用
```

## 技术栈

- 解析：tree-sitter，支持 17 种语言语法（Python、TS、Rust、Go、Java、C++ 等）
- 密集嵌入：Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），bge-code-v1 备用
- 稀疏索引：Tantivy（Rust），BM25F，在符号名 vs 主体上加权字段
- 向量数据库：Qdrant 1.12，带混合搜索；或 pgvector + pgvectorscale（适用于 5000 万向量以下的团队）
- 块摘要模型：Claude Haiku 4.5 或 Gemini 2.5 Flash，带提示词缓存
- 重新排序器：Cohere rerank-3 或 bge-reranker-v2-gemma-2b 自托管
- 编排：摄入使用 LlamaIndex Workflows，查询智能体使用 LangGraph
- 合成器：Claude Sonnet 4.7（1M 上下文），带提示词缓存
- 符号图：Neo4j（托管）或 kuzu（嵌入式），用于导入和调用边
- 可观测性：每个检索 + 合成步骤的 Langfuse span

## 构建它

1. **摄入遍历器。** 在每次 push 钩子上遍历 git 历史。收集更改的文件。对每个文件，用 tree-sitter 解析，提取函数和类节点及其完整源代码范围。发出块记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **块摘要器。** 批量将块发送给 Haiku 4.5，在系统前言上使用提示词缓存。提示词："用一句话总结这个函数，说明其公共契约和副作用。" 将摘要与块一起存储。

3. **嵌入池。** 两个并行队列：密集（Voyage-code-3 批次 128）和摘要（同一模型，但针对摘要字符串）。将向量写入 Qdrant，有效负载为 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 字段加权 Tantivy 索引：符号名权重 4，符号主体权重 1，摘要权重 2。支持"查找名为 X 的函数"查询以及"查找执行 X 操作的函数"查询。

5. **符号图。** 对每个块，记录边：导入（此文件使用仓库 Z 中的符号 Y）、调用（此函数调用类 C 上的方法 M）、继承。存储在 kuzu 中。在查询时用于跨仓库边界扩展检索。

6. **查询智能体。** LangGraph，有三个节点。`retrieve` 并行触发密集 + BM25，按（仓库、路径、符号）去重。`rerank` 对前 50 运行交叉编码器并保留前 10。`synth` 调用 Claude Sonnet 4.7，将重新排序的块放入上下文，缓存系统提示词，要求文件:行引用。

7. **引用执行。** 解析模型输出；任何没有 `(repo/path:start-end)` 锚点的声明都被标记为重新询问或丢弃。只向用户返回有引用的答案。

8. **增量重新索引。** 在每次 webhook 时，计算符号级差异。只重新嵌入文本改变的块。为导入改变的块重新计算符号边。测量：对 200 万行代码集群的 50 文件推送在 60 秒内完成重新索引。

9. **评估。** 标注 100 个跨仓库问题，附带黄金文件:行答案。测量 MRR@10、nDCG@10、引用忠实度（可验证锚点的声明比例）以及 p50/p99 延迟。

## 使用它

```
$ code-rag ask "S3 分段中止是如何连接到我们的重试预算的？"
[检索]  12 个密集块 + 7 个 bm25 块，去重后 16 个唯一
[重排序]  保留前 5（cohere rerank-3）
[合成]  claude-sonnet-4.7，缓存命中率 68%，2.1 秒
答案：
  分段中止由 services/uploader/retry.go:122-148 中的 `AbortMultipartOnFail`
  触发，该函数递减 config/budgets.yaml:34-51 中定义的每个桶的重试预算 ...
  引用：[services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付它

可交付技能 `outputs/skill-codebase-rag.md`。给定仓库语料库，它建立摄入管道、混合索引和查询智能体，并返回任何跨仓库问题的有引用答案。评分标准：

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 检索质量 | 100 个问题保留集上的 MRR@10 和 nDCG@10 |
| 20 | 引用忠实度 | 有可验证文件:行锚点的答案声明比例 |
| 20 | 延迟和规模 | 在索引语料库大小的 10k QPS 下的 p95 查询延迟 |
| 20 | 增量索引正确性 | 从 git push 到 50 文件提交可搜索的时间 |
| 15 | 用户体验和答案格式 | 引用可点击性、片段预览、后续问题支持 |
| **100** | | |

## 练习

1. 将 Voyage-code-3 换为自托管的 nomic-embed-code。测量 MRR@10 差距。报告是否在启用重新排序后差距缩小。

2. 将 20% 生成的代码（LLM 产生的样板）注入语料库并重新评估。观察检索污染。在有效负载中添加"generated"标志，并降低这些命中的权重。

3. 在你的语料库大小下对 Qdrant 混合搜索 vs pgvector + pgvectorscale 进行基准测试。报告批次大小为 1 时的 p99。

4. 添加基于采样的漂移检查：每周重新运行 100 个问题的评估。在 MRR@10 下降 > 5% 时发出告警。

5. 扩展到跨语言符号解析：通过 gRPC 调用 Go 服务的 Python 函数。使用符号图将它们链接起来。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| AST-aware chunking（AST 感知分块） | "函数级分割" | 在 tree-sitter 节点边界而非固定 token 窗口处切割代码 |
| Hybrid search（混合搜索） | "密集 + 稀疏" | 并行运行 BM25 和向量搜索，合并 top-k，重新排序 |
| Cross-encoder rerank（交叉编码器重排序） | "第二阶段排序" | 将每个（查询，候选）对一起评分的模型，比余弦更准确 |
| Prompt caching（提示词缓存） | "缓存系统提示词" | 2026 年 Claude/OpenAI 功能，对重复前缀 token 折扣高达 90% |
| Symbol graph（符号图） | "代码图" | 跨文件和仓库的导入、调用、继承边 |
| Citation faithfulness（引用忠实度） | "有根据的回答率" | 用户可以通过点击锚点并读取引用范围来验证的声明比例 |
| Incremental re-index（增量重新索引） | "push 到搜索时间" | 从 git push 到已更改符号可搜索的挂钟时间 |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产跨仓库代码智能
- [Sourcegraph Cody RAG 架构](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本压轴项目的参考深度解析
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter 排序仓库视图
- [Augment Code 企业图](https://www.augmentcode.com) — 商业符号图 RAG
- [Qdrant 混合搜索文档](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现
- [Voyage AI 代码嵌入](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 详情
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — 交叉编码器参考
- [Pinterest MCP 内部搜索](https://medium.com/pinterest-engineering) — 内部平台参考

# ASCII 艺术与视觉越狱（ASCII Art and Visual Jailbreaks）

> Jiang、Xu、Niu、Xiang、Ramasubramanian、Li、Poovendran，"ArtPrompt：针对对齐 LLM 的基于 ASCII 艺术的越狱攻击"（ACL 2024，arXiv:2402.11753）。在有害请求中屏蔽与安全相关的 token，用相同字母的 ASCII 艺术渲染替换它们，然后发送经过伪装的提示词。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 都无法鲁棒地识别 ASCII 艺术 token。该攻击绕过了 PPL（困惑度过滤器）、改述防御和重新分词。相关：ViTC 基准测量对非语义视觉提示词的识别；StructuralSleight 将其推广到不常见文本编码结构（树、图、嵌套 JSON）作为编码攻击系列。

**类型：** 构建  
**语言：** Python（标准库，ArtPrompt token 屏蔽测试框架）  
**前置知识：** Phase 18 · 12（PAIR）、Phase 18 · 13（MSJ）  
**预计时间：** 约 60 分钟

## 学习目标

- 描述 ArtPrompt 攻击：词语识别步骤、ASCII 艺术替换、最终伪装提示词。
- 解释为什么标准防御（PPL、改述、重新分词）对 ArtPrompt 失效。
- 定义 ViTC 并描述其测量内容。
- 描述 StructuralSleight 作为对任意不常见文本编码结构的推广。

## 问题所在

通过改述和角色扮演（第 12 课）和通过长上下文（第 13 课）的攻击在文本级模式上操作。ArtPrompt 在识别级别上操作：模型不解析禁止的 token。它解析用字符渲染的图像。安全过滤器看到无害的标点符号。模型看到一个词语。

## 核心概念

### ArtPrompt，两步

第一步：词语识别。给定有害请求，攻击者使用 LLM 识别与安全相关的词语（例如，"如何制造炸弹"中的"炸弹"）。

第二步：伪装提示词生成。将每个识别的词语替换为其 ASCII 艺术渲染（形成字母形状的 7×5 或 7×7 字符块）。模型收到一个标点符号和空格的网格，足够有能力的模型可以将其识别为该词语；安全过滤器只看到网格。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 都失败了。在其基准子集上的攻击成功率超过 75%。

### 为什么标准防御失效

- **PPL（困惑度过滤器）。** ASCII 艺术具有高困惑度——但所有新颖输入也是如此。阻止 ArtPrompt 的阈值选择也会阻止合法的结构化输入。
- **改述。** 改述提示词会破坏 ASCII 艺术。实际上，改述 LLM 经常保留或重建艺术。
- **重新分词。** 以不同方式分割 token 不会改变模型的视觉识别字母形状的能力。

根本问题是安全过滤器处于 token 或语义级别；ArtPrompt 在视觉识别级别操作。

### ViTC 基准

对非语义视觉提示词的识别。测量模型阅读 ASCII 艺术、Wingdings 和其他非文本语义视觉内容的能力。ArtPrompt 的有效性与 ViTC 准确率相关：模型阅读视觉文本的能力越强，ArtPrompt 对其越有效。这是能力-安全权衡。

### StructuralSleight

推广 ArtPrompt：不常见文本编码结构（UTES）。树、图、嵌套 JSON、JSON 中的 CSV、差异风格代码块。如果一种结构在训练安全数据中罕见但模型可解析，它可以隐藏有害内容。

防御含义：安全必须在模型可以解析的结构化表示中进行泛化。这个集合大且还在增长。

### 图像模态类比

视觉 LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展了攻击面。使用实际图像的 ArtPrompt 风格攻击比 ASCII 艺术类比更强，因为图像编码器产生更丰富的信号。

### 这在 Phase 18 中的位置

第 12-14 课描述了三个正交攻击向量：迭代细化（PAIR）、上下文长度（MSJ）和编码（ArtPrompt/StructuralSleight）。第 15 课从以模型为中心的攻击转向系统边界攻击（间接提示注入）。第 16 课描述防御工具响应。

## 使用它

`code/main.py` 构建了一个玩具 ArtPrompt。你可以用 ASCII 艺术字形伪装有害查询中的特定词语，验证伪装后的字符串通过关键词过滤器，并（可选地）使用简单识别器将伪装字符串解码回来。

## 交付它

本课产出 `outputs/skill-encoding-audit.md`。给定越狱防御报告，它列举覆盖的编码攻击系列（ASCII 艺术、base64、Leet-speak、UTF-8 同形异义词、UTES）以及捕获每种的防御层。

## 练习

1. 运行 `code/main.py`。验证伪装后的字符串通过简单关键词过滤器。报告所需的字符级变化。

2. 实现第二种编码：对同一目标词语进行 base64 编码。比较过滤器绕过率与 ArtPrompt 的差异以及恢复难度。

3. 阅读 Jiang 等人 2024 年第 4.3 节（五模型结果）。提出 Claude 的 ArtPrompt 抵抗力比 Gemini 在同一基准上更高的原因。

4. 设计一个预生成防御，检测提示词中的 ASCII 艺术形状区域。测量对合法代码、表格和数学符号的误报率。

5. StructuralSleight 列出了 10 种编码结构。勾勒一种处理所有 10 种的通用防御，并估计每个防御提示词的计算成本。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| ArtPrompt | "ASCII 艺术攻击" | 用 ASCII 艺术渲染屏蔽安全词语的两步越狱 |
| Cloaking（伪装） | "隐藏词语" | 用模型读取但过滤器不读取的视觉表示替换禁止的 token |
| UTES | "不常见结构" | 不常见文本编码结构——树、图、嵌套 JSON 等，用于走私内容 |
| ViTC | "视觉文本能力" | 模型阅读非语义视觉编码能力的基准 |
| Perplexity filter（困惑度过滤器） | "PPL 防御" | 拒绝高困惑度提示词；失效是因为合法结构化输入也评分高 |
| Retokenization（重新分词） | "分词器移位防御" | 用不同分词器预处理提示词；失效是因为识别是视觉的 |
| Homoglyph（同形异义词） | "相似字符" | 看起来与拉丁字母相同的 Unicode 字符；绕过子字符串检查 |

## 延伸阅读

- [Jiang 等人 — ArtPrompt（ACL 2024，arXiv:2402.11753）](https://arxiv.org/abs/2402.11753) — ASCII 艺术越狱论文
- [Li 等人 — StructuralSleight（arXiv:2406.08754）](https://arxiv.org/abs/2406.08754) — UTES 推广
- [Chao 等人 — PAIR（第 12 课，arXiv:2310.08419）](https://arxiv.org/abs/2310.08419) — 互补迭代攻击
- [Anil 等人 — 多样本越狱（第 13 课）](https://www.anthropic.com/research/many-shot-jailbreaking) — 互补长度攻击

# 推理平台经济学——Fireworks、Together、Baseten、Modal、Replicate、Anyscale（Inference Platform Economics — Fireworks, Together, Baseten, Modal, Replicate, Anyscale）

> 2026 年的推理市场不再只是 GPU 时间租赁。它分化为定制芯片（Groq、Cerebras、SambaNova）、GPU 平台（Baseten、Together、Fireworks、Modal）和 API 优先市场（Replicate、DeepInfra）。Fireworks 于 2026 年 5 月 1 日上调 GPU 价格 1 美元/小时，而基于每天 10 万亿以上 token 处理量的 40 亿美元估值表明，量驱动模型行之有效。Baseten 于 2026 年 1 月以 50 亿美元估值完成 3 亿美元 E 轮融资。竞争定位规则很简单：Fireworks 优化延迟，Together 优化目录广度，Baseten 优化企业级打磨，Modal 优化 Python 原生开发体验，Replicate 优化多模态覆盖范围，Anyscale 优化分布式 Python。本课为你提供一个可以直接交给创始人的矩阵。

**类型：** 学习  
**语言：** Python（标准库，玩具按调用成本比较器）  
**前置知识：** Phase 17 · 01（托管 LLM 平台）、Phase 17 · 04（vLLM 服务内部原理）  
**预计时间：** 约 60 分钟

## 学习目标

- 说出三个市场细分（定制芯片、GPU 平台、API 优先），并将每家供应商映射到对应细分。
- 解释为什么"按 token"的 API 定价模型向服务引擎的成本曲线收敛，而非硬件成本。
- 跨至少三家供应商计算每次请求的有效成本，并解释何时按分钟计费（Baseten、Modal）优于按 token 计费。
- 识别针对特定工作负载（无服务器突发、稳定高吞吐量、微调变体、多模态）的正确默认平台。

## 问题所在

你评估了托管超大规模云厂商平台。你决定需要一个更专注、更快的供应商——用 Fireworks 追求延迟，用 Together 追求广度，用 Baseten 服务微调的自定义模型。现在你面对六个真实选择，而这些定价页面无法直接对比。Fireworks 显示每百万 token 美元数；Baseten 显示每分钟美元数；Modal 显示每秒美元数；Replicate 显示每次预测美元数。不对工作负载建模就无法进行正面比较。

更麻烦的是，每个定价页面背后的商业模式不同。Fireworks 在共享 GPU 上运行自己的定制引擎（FireAttention）；按 token 费率反映其利用率曲线。Baseten 提供 Truss + 独占 GPU；按分钟计费反映独占性。Modal 是真正的 Python 无服务器——按秒计费，冷启动时间低于一秒。同样的输出（LLM 响应），三种不同的成本函数。

本课对这六个平台建模，并告诉你每个何时胜出。

## 核心概念

### 三个细分市场

**定制芯片** — Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。在同一模型上，解码速度通常比基于 GPU 的集群快 5-10 倍。每 token 价格更高（2025 年末 Groq 在 Llama-70B 上约为 0.99 美元/百万 token），但对于延迟敏感的使用场景无可匹敌。Groq 是语音智能体和实时翻译的生产首选。

**GPU 平台** — Baseten、Together、Fireworks、Modal、Anyscale。运行在 NVIDIA（2026 年的 H100、H200、B200）或有时是 AMD 上。这是"裸 GPU 租赁"（RunPod、Lambda）和"超大规模托管服务"（Bedrock）之间的经济层。

**API 优先市场** — Replicate、DeepInfra、OpenRouter、Fal。目录宽泛，按预测计费或按秒计费，强调首次调用时间。

### Fireworks — 延迟优化 GPU 平台

- FireAttention 引擎（定制）；在同等配置上延迟号称比 vLLM 低 4 倍。
- 批处理层，约为无服务器费率的 50%，用于非交互式工作负载。
- 微调模型与基础模型按同一费率提供服务——这是相对于收取 LoRA 溢价的供应商的真实差异化因素。
- 2026 年中：自 2026 年 5 月 1 日起上调按需 GPU 租赁费用 1 美元/小时。大规模时可协商量价。
- 财务信号：40 亿美元估值，每天处理超过 10 万亿 token。

### Together — 广度优化

- 200 多个模型，包括上游发布后数天内的开源新品。
- 在同等 LLM 模型上比 Replicate 便宜 50-70%——"AI 原生云"定位在于量和目录。
- 推理 + 微调 + 训练集于一个 API。

### Baseten — 企业打磨优化

- Truss 框架：将依赖项、密钥、服务配置集于一个清单中的模型打包方案。
- GPU 范围从 T4 到 B200。按分钟计费，有合理的冷启动缓解措施。
- SOC 2 Type II，HIPAA 就绪。常见金融科技和医疗保健首选。
- 估值 50 亿美元，2026 年 1 月 E 轮融资（CapitalG、IVP、NVIDIA 投资 3 亿美元）。

### Modal — Python 原生优化

- 纯 Python 的基础设施即代码。用 `@modal.function(gpu="A100")` 装饰一个函数，一条命令完成部署。
- 按秒计费。预热后冷启动 2-4 秒；小模型低于 1 秒。
- 2025 年 B 轮融资 8700 万美元，估值 11 亿美元。在独立调查中开发体验得分最高。

### Replicate — 多模态广度

- 按预测计费。图像、视频和音频模型的默认平台。
- 集成生态系统（Zapier、Vercel、CMS 插件）。
- 在 LLM 按 token 费率上竞争力较弱，但在多模态多样性上胜出。

### Anyscale — Ray 原生

- 基于 Ray 构建；RayTurbo 是 Anyscale 的专有推理引擎（与 vLLM 竞争）。
- 最适合分布式 Python 工作负载，其中推理步骤是更大计算图中的一个节点。
- 托管 Ray 集群；与 Ray AIR 和 Ray Serve 深度集成。

### 按 token vs 按分钟——各自何时胜出

当工作负载对延迟不敏感且具有突发性时，按 token 计费更合适——你只为实际使用付费。当利用率高且可预测时，按分钟计费更合适——一旦 GPU 饱和，就会胜过按 token 计费。

粗略规则：对于独占 GPU 持续利用率超过约 30% 的工作负载，按分钟计费（Baseten、Modal）开始优于按 token 计费（Fireworks、Together）。低于该值时，按 token 计费胜出，因为你避免了为空闲付费。

### 定制引擎是真正的护城河

上述每个平台都声称拥有高于 vLLM 和 SGLang 的定制引擎：FireAttention、RayTurbo、Baseten 的推理栈。定制引擎的声明带有营销成分——诚实的框架是：vLLM + SGLang 代表了约 80% 的生产开源推理，平台层的差异化因素是开发体验、归因和 SLA。

### 你应该记住的数字

- Fireworks GPU 租赁：自 2026 年 5 月 1 日起每小时涨价 1 美元。
- Fireworks 声称：在同等配置上延迟比 vLLM 低 4 倍。
- Together：在 LLM 上比 Replicate 便宜 50-70%。
- Baseten 估值：50 亿美元（E 轮，2026 年 1 月，3 亿美元轮次）。
- Modal 估值：11 亿美元（B 轮，2025 年）。
- 持续利用率超过约 30% 时，按分钟计费优于按 token 计费。

## 使用它

`code/main.py` 跨定价模型在合成工作负载上比较六家供应商。报告每天美元成本和有效的每百万 token 成本。运行它可以找到按 token 与按分钟之间的盈亏平衡点。

## 交付它

本课产出 `outputs/skill-inference-platform-picker.md`。给定工作负载配置文件、SLA 和预算，选出主要推理平台并命名备选项。

## 练习

1. 运行 `code/main.py`。在什么持续利用率下，Baseten（按分钟）优于 Fireworks（按 token）在一块 H100 上运行 70B 模型？自己推导交叉点并与经验法则比较。
2. 你的产品同时提供图像生成、聊天和语音转文字。为每种模态选择平台，并命名统一它们的网关模式。
3. Fireworks 对你的主要模型每小时涨价 1 美元。如果 40% 的流量转移到批处理层（五折），对混合成本影响建模。
4. 一个受监管的客户需要 SOC 2 Type II + HIPAA + 独占 GPU。哪三个平台可行，哪一个在 FinOps 上胜出？
5. 比较 Llama 3.1 70B 在 Fireworks 无服务器、Together 按需、Baseten 独占和 Replicate API 上每 1000 次预测的成本。每天 10 次预测时哪个最便宜？10000 次时呢？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Custom silicon（定制芯片） | "非 GPU 芯片" | Groq LPU、Cerebras WSE、SambaNova RDU——针对解码优化 |
| FireAttention | "Fireworks 引擎" | 定制注意力内核；号称延迟比 vLLM 低 4 倍 |
| Truss | "Baseten 的格式" | 模型打包清单；依赖项 + 密钥 + 服务配置 |
| Per-token（按 token） | "API 定价" | 按消耗 token 计费；不为空闲付费 |
| Per-minute（按分钟） | "独占定价" | 按挂钟 GPU 时间计费；高利用率时胜出 |
| Per-prediction（按预测） | "Replicate 定价" | 按模型调用次数计费；图像/视频常见 |
| RayTurbo | "Anyscale 引擎" | Ray 上的专有推理；在 Ray 集群上与 vLLM 竞争 |
| Batch tier（批处理层） | "五折优惠" | 非交互式队列，费率降低；Fireworks、OpenAI 常见 |
| Fine-tuned at base rate（基础费率服务微调模型） | "Fireworks LoRA" | 按基础模型费率对 LoRA 服务请求计费（差异化因素） |

## 延伸阅读

- [Fireworks 定价](https://fireworks.ai/pricing) — 按 token 费率、批处理层、GPU 租赁
- [Baseten 定价](https://www.baseten.co/pricing/) — 按分钟费率、承诺容量、企业层级
- [Modal 定价](https://modal.com/pricing) — 按秒 GPU 费率和免费层
- [Together AI 定价](https://www.together.ai/pricing) — 模型目录和按 token 费率
- [Anyscale 定价](https://www.anyscale.com/pricing) — RayTurbo 和托管 Ray 定价
- [Northflank — Fireworks AI 替代品](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 比较评估
- [Infrabase — 2026 年 AI 推理 API 供应商](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 供应商格局

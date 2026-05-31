# 开放权重 VLM 方案：哪些因素真正重要（Open-Weight VLM Recipes: What Actually Matters）

> 2024-2026 年的开放权重 VLM 文献是一片消融表格的森林。Apple 的 MM1 测试了图像编码器、连接器和数据混合的 13 种组合。Allen AI 的 Molmo 证明了详细的人类描述胜过 GPT-4V 蒸馏。Cambrian-1 运行了 20+ 种编码器比较。Idefics2 将五轴设计空间正式化。Prismatic VLMs 在可控基准上比较了 27 种训练方案。从所有这些噪声中，一小组结论在论文间保持稳定：图像编码器比连接器架构更重要，数据混合比两者都重要，详细的人类描述胜过蒸馏的合成数据。本章解读那些表格，让你不必亲自去读。

**类型：** 学习 + 实验  
**语言：** Python（标准库，消融表格解析器 + 方案选择器）  
**前置知识：** Phase 12 · 05（LLaVA 基线）  
**预计时间：** 约 180 分钟

## 学习目标

- 说出 VLM 五轴设计空间：图像编码器、连接器、LLM、数据混合、分辨率策略。
- 阅读 MM1 / Idefics2 / Cambrian-1 的消融表格，并预测哪个旋钮会影响给定基准。
- 在给定算力预算和任务混合的情况下，为新 VLM 选择方案（编码器、连接器、数据、分辨率）。
- 解释为什么在相同 token 数量下，详细人类描述胜过 GPT-4V 蒸馏。

## 问题所在

开放权重 VLM 多达数百个。"好"与"最先进"之间的差距大多不在架构，而在数据、分辨率策略和编码器选择上。知道模型表现不佳时该先拧哪个旋钮，能让你避免一个耗费 500 万 GPU 小时的错误。

2023 年浪潮（LLaVA-1.5、InstructBLIP、MiniGPT-4）以描述对预训练 + LLaVA-Instruct-150k 运行。良好的基线，MMMU 约 35% 时触顶。

2024 年浪潮（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）进行了详尽的消融实验，结果令人惊讶且实用。

## 核心概念

### 五轴设计空间

Idefics2（Laurençon 等，2024）命名了这些轴：

1. **图像编码器。** CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。编码器在图块大小、分辨率和预训练目标上有所不同。
2. **连接器。** MLP（2-4 层）、Q-Former（32 个查询 + 交叉注意力）、Perceiver 重采样器（64 个查询）、C-Abstractor（卷积 + 双线性池化）。
3. **语言模型。** Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 大小是主要参数成本。
4. **训练数据。** 描述对（CC3M、LAION）、交错（OBELICS、MMC4）、指令（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. **分辨率策略。** 固定 224/336/448、AnyRes、原生动态。训练期间逐渐提升或保持不变。

每个生产 VLM 在每个轴上都做了选择。MMMU 分数的大部分方差由轴 1、4 和 5 解释——而不是你选了哪个连接器。

### 轴 1：编码器 > 连接器

MM1 第 3.2 节显示：从 CLIP ViT-L/14 换到 SigLIP SO400m/14，MMMU 提升 3+ 分。从 MLP 换到 Perceiver 重采样器连接器，提升不到 1 分。Idefics2 复现了这一结论：SigLIP > CLIP，在相同 token 数量下 Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1 的"Cambrian 视觉编码器对决"（Tong 等，2024）在以视觉为中心的基准（CV-Bench）上测试了 20+ 个编码器。排行榜顶部是 DINOv2 和 SigLIP 的组合；CLIP 在中游；ImageBind 和 ViT-MAE 更低。CLIP ViT-L 到 DINOv2 ViT-g/14 在 CV-Bench 上有约 5-7 分的差距。

2026 年开放 VLM 的默认编码器是 SigLIP 2 SO400m/14（用于语义 + 密集特征），有时与 DINOv2 ViT-g/14 特征拼接（Cambrian 的"空间视觉聚合器"就是这么做的）。

### 轴 2：连接器设计是平局

MM1、Idefics2、Prismatic 和 MM-Interleaved 都得出了相同结论：在固定视觉 token 数量下，连接器架构几乎无关紧要。在相同 token 预算下，基于均值池化图块的 2 层 MLP 与 32 个查询的 Q-Former 的表现在 1 分以内。

真正重要的是 token 数量。更多视觉 token = 更多 LLM 计算 = 更好的表现，达到某个点后收益递减。每张图像 64 个 token 对 OCR 来说太少；576-1024 个 token 是大多数开放 VLM 的甜蜜点；2048+ 只对文档和图表有帮助。

Q-Former 与 MLP 是成本问题，不是质量问题：Q-Former 无论图像分辨率如何都把 token 上限在 32-64；MLP 发出所有图块 token。对高分辨率输入，Q-Former 节省 LLM 上下文；对低分辨率，差异是噪声。

### 轴 3：LLM 大小设定天花板

LLM 从 7B 翻倍到 13B，在每篇 VLM 论文中 MMMU 都可靠地提升 2-4 分。在 70B 时，大多数基准趋于饱和。VLM 的多模态推理天花板就是 LLM 的文本推理天花板——视觉编码器只能喂给它，无法替它推理。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 横扫 MMMU-Pro 和 ScreenSpot-Pro：语言大脑足够庞大。7B VLM 无法通过巧妙的连接器设计替代 70B VLM。

### 轴 4：数据——详细人类描述胜过蒸馏

Molmo + PixMo（Deitke 等，2024）是 2024 年每个人都应该阅读的结果。Allen AI 让人工标注者用 1-3 分钟的密集语音转文字描述图像，产出 71.2 万张密集标注图像。训练数据中没有任何 GPT-4V 蒸馏。

Molmo-72B 在 11 个基准中的 11 个上击败了 Llama-3.2-90B-Vision。差距不在架构——在于描述质量。详细人类描述每张图像包含的信息是短网络描述的 5-10 倍，并且在事实上保持准确，而 GPT-4V 蒸馏会产生幻觉。

ShareGPT4V（Chen 等，2023）和 Cauldron（Idefics2）遵循了相同的思路，混合使用人类 + GPT-4V 描述。趋势很清楚：对于 2026 年的前沿，描述密度 > 描述数量 > 蒸馏便捷性。

### 轴 5：分辨率及其策略

Idefics2 的消融：384 → 448 提升 1-2 分；448 → 980（使用图像分割/AnyRes）在 OCR 基准上再提升 3-5 分。固定分辨率训练在中等精度时停滞；分辨率递增（从 224 开始，到 448 或原生结束）训练更快、最终更高。

Cambrian-1 进行了分辨率与 token 的权衡对比：在固定算力下，你可以在低分辨率时拥有更多 token，或在高分辨率时拥有更少 token。高分辨率对 OCR 更胜出；低分辨率更多 token 对通用场景理解更胜出。

2026 年的生产方案：阶段 1 在 384 固定分辨率训练，阶段 2 对重 OCR 任务采用动态分辨率（最高 1280）。

### Prismatic 的可控对比

Prismatic VLMs（Karamcheti 等，2024）是控制了所有轴的论文。相同的 13B LLM、相同的指令数据、相同的评估——每次只变化一个轴。结果：

- 每图像视觉 token 数量解释约 60% 的方差。
- 编码器选择解释约 20%。
- 连接器架构解释约 5%。
- 其余（数据混合、调度器、学习率）占剩余约 15%。

这是粗略的分解，但它是文献中对"我应该先消融什么"最清晰的答案。

### 2026 年的选择器

根据证据，2026 年新 VLM 项目的默认开放 VLM 方案：

- **编码器：** SigLIP 2 SO400m/14，原生分辨率加 NaFlex；如果需要分割/目标检测，拼接 DINOv2 ViT-g/14 特征。
- **连接器：** 图块 token 上的 2 层 MLP。除非受 token 约束，否则跳过 Q-Former。
- **LLM：** Qwen2.5 / Llama-3.1 / Gemma 2，成本优先选 7B，质量优先选 70B，根据目标延迟确定。
- **数据：** PixMo + ShareGPT4V + Cauldron，加任务专属指令数据作为补充。
- **分辨率：** 动态（长边最小 256、最大 1280 像素）。
- **策略：** 阶段 1 对齐（仅投影器），阶段 2 全量微调，阶段 3 任务专属微调。

以上每个默认选择都可以追溯到本章末尾引用论文中的一次实测消融。

## 动手使用

`code/main.py` 是一个消融表格解析器和方案选择器。它编码了 MM1 和 Idefics2 的消融表格（精简版），允许你查询：

- "给定预算 X 和任务 Y，哪种方案最优？"
- "如果在 7B Llama 上把 SigLIP 换成 CLIP，预期 MMMU 差值是多少？"
- "哪个轴应该首先消融，以获得 80% 置信度的答案？"

输出是一个带有预期基准差值和"首先消融"推荐的排序方案列表。

## 输出产物

本章生成 `outputs/skill-vlm-recipe-picker.md`。给定目标任务混合、算力预算和延迟目标，它输出完整方案（编码器、连接器、LLM、数据混合、分辨率策略），并引用支持每个选择的消融实验。防止工程师每次启动新 VLM 项目时都重新发明 Idefics2 消融表格。

## 练习

1. 阅读 MM1 第 3.2 节。在 5000 万图像预算的固定 2B LLM 下，哪个编码器获胜？在 13B LLM 时答案会反转吗？为什么？

2. Cambrian-1 发现拼接 DINOv2 + SigLIP 在以视觉为中心的基准上优于任一单独使用，但对 MMMU 没有额外信号。预测哪些基准会提升，哪些会持平。

3. 你的目标是在 2B LLM 上运行移动 UI 智能体。选择编码器、连接器、分辨率和数据混合，用具体的消融表格为每个选择辩护。

4. Molmo 发布了 4B 和 72B 两个模型。4B 与闭源 7B VLM 竞争；72B 在 11/11 基准上击败 Llama-3.2-90B-Vision。这告诉你关于 LLM 大小瓶颈假说什么信息？

5. 设计一个消融表格，在 7B VLM 上将数据混合质量与编码器质量隔离。最少需要多少次训练运行？提出四种轴设置。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 消融（Ablation） | "拧一个旋钮" | 运行多次训练，每次在恰好一个设计空间轴上有所不同，其余保持不变。 |
| 连接器（Connector） | "桥梁"/"投影器" | 将视觉编码器输出映射到 LLM token 空间的可训练模块（MLP、Q-Former、Perceiver）。 |
| 详细人类描述（Detailed human caption） | "密集描述" | 多句人类撰写的描述（通常 80-300 个 token），比网络 alt 文字更丰富。 |
| 蒸馏（Distillation） | "GPT-4V 描述" | 由更强的专有 VLM 生成的训练数据；方便但容易继承幻觉。 |
| AnyRes / 动态分辨率 | "高分辨率路径" | 通过平铺或 M-RoPE 将大于编码器原生分辨率的图像喂给编码器的策略。 |
| 分辨率递增（Resolution ramp） | "课程学习" | 从低分辨率开始、逐渐增加的训练策略，加快对齐学习。 |
| 以视觉为中心的基准 | "CV-Bench / BLINK" | 强调细粒度视觉感知而非语言密集推理的评估。 |
| PixMo | "Molmo 的数据" | Allen AI 的 71.2 万密集标注图像数据集；人类语音转录为密集描述。 |

## 延伸阅读

- [McKinzie 等 — MM1（arXiv:2403.09611）](https://arxiv.org/abs/2403.09611)
- [Laurençon 等 — Idefics2 / What matters building VLMs（arXiv:2405.02246）](https://arxiv.org/abs/2405.02246)
- [Deitke 等 — Molmo and PixMo（arXiv:2409.17146）](https://arxiv.org/abs/2409.17146)
- [Tong 等 — Cambrian-1（arXiv:2406.16860）](https://arxiv.org/abs/2406.16860)
- [Karamcheti 等 — Prismatic VLMs（arXiv:2402.07865）](https://arxiv.org/abs/2402.07865)

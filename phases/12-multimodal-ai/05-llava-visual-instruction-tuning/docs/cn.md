# LLaVA 与视觉指令微调（LLaVA and Visual Instruction Tuning）

> LLaVA（2023 年 4 月）是全球被复制最多的多模态架构。它用 2 层 MLP 替换了 BLIP-2 的 Q-Former，用朴素 token 拼接替换了 Flamingo 的门控交叉注意力，并在由 GPT-4 从纯文本描述生成的 15.8 万条视觉指令对话上进行训练。2023 年到 2026 年之间构建 VLM 的实践者，几乎都构建了某种 LLaVA 变体。LLaVA-1.5 加入了 AnyRes，LLaVA-NeXT 提升了分辨率，LLaVA-OneVision 将图像、多图像和视频统一到一个方案中。本章解读这个方案，实现投影器，并解释为什么"更简单赢了"。

**类型：** 构建  
**语言：** Python（标准库，投影器 + 指令模板构建器）  
**前置知识：** Phase 12 · 02（CLIP）、Phase 11（LLM 工程——指令微调）  
**预计时间：** 约 180 分钟

## 学习目标

- 构建一个 2 层 MLP 投影器，将 ViT 图块嵌入（维度 1024）映射到 LLM 的嵌入维度（维度 4096）。
- 走通 LLaVA 两阶段方案：（1）在 55.8 万条图文对上做投影器对齐；（2）在 GPT-4 生成的 15.8 万条对话上做视觉指令微调。
- 构建包含图像 token 占位符、系统提示词和用户/助理轮次的 LLaVA 格式提示词。
- 解释为什么社区从 Q-Former 转向了 MLP，尽管 Q-Former 在 token 预算上更有优势。

## 问题所在

BLIP-2 的 Q-Former（Lesson 12.03）将图像压缩为 32 个 token——简洁、高效、适合基准测试。但它有两个问题。

第一，Q-Former 是可训练的，但它的损失并不是最终任务。阶段 1 训练 ITC+ITM+ITG，阶段 2 训练 LM 损失。查询学习某种中间表示，LLM 然后必须解码它。信息在瓶颈中丢失了。

第二，Q-Former 有 1.88 亿参数，在 LLaVA 2023 年的规模下，你必须与目标 LLM 协同设计它。换 LLM，就要重训 Q-Former；换视觉编码器，也要重训。每种组合都是一个独立的研发项目。

LLaVA 的答案简单得令人尴尬：取 ViT 的 576 个图块 token，每个通过一个 2 层 MLP（`1024 → 4096 → 4096`），然后将所有 576 个直接倒入 LLM 的输入序列。没有瓶颈，没有用奇怪目标做的阶段 1 预训练，只是在直接的 LM 损失上训练 MLP。

数据从哪里来？LLaVA 的第二个洞察：用 GPT-4（纯文本）生成指令数据。将 COCO 图像的文本描述（5 条人工标注 + 边界框列表）提供给 GPT-4，让它生成对话、描述和复杂推理问题。15.8 万条指令-响应对，免费获得，无需人工标注。

结果：一个在 8 块 A100 上训练一天的 VLM，在 MMMU 上超越了 Flamingo，并发布了社区可以扩展的开放检查点。到 2023 年底，它已经衍生出 50+ 个分支。

## 核心概念

### 架构

LLaVA-1.5 13B：
- 视觉编码器：CLIP ViT-L/14 @ 336（阶段 1 冻结，阶段 2 可选解冻）。
- 投影器：带 GELU 激活的 2 层 MLP，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后来是 Llama-3.1-8B）。

图像 + 文本提示词的前向传播：

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像占用 LLM 上下文的 576 个 token。在 2048 上下文中，这给文本留下 1472 个 token；在 32k 上下文中，这几乎微不足道。

### 阶段 1：投影器对齐

冻结 ViT，冻结 LLM，只训练 2 层 MLP。数据集：55.8 万个图文对（LAION-CC-SBU）。损失：以投影图像 token 为条件的文本描述语言模型损失。

在 batch 128 的单轮训练中，几小时内完成。投影器学会将 ViT 空间映射到 LLM 空间，无需任务专属的监督。

### 阶段 2：视觉指令微调

解冻投影器（仍可训练），解冻 LLM（通常全量，有时用 LoRA），在 15.8 万条视觉指令对话上训练。

指令数据是关键。Liu 等人的生成方法：
1. 取一张 COCO 图像。
2. 提取文本描述（5 条人工标注 + 边界框列表）。
3. 用三个提示词模板发送给 GPT-4：
   - 对话：「生成用户和助理关于这张图像的来回对话。」
   - 详细描述：「给出这张图像的丰富详细描述。」
   - 复杂推理：「提一个需要对图像进行推理的问题，然后回答它。」
4. 将 GPT-4 的输出解析为（指令、响应）对。

整个过程完全不涉及图像本身——只使用文本描述。GPT-4 会幻想合理的图像内容，有些噪声，但有效：15.8 万条对话足以激活对话能力。

### 为什么社区复制了这个方案

- 没有阶段 1 专属损失需要调整，全程 LM 损失。
- 投影器在数小时内训练完成，而非数天。
- 可以通过只重训投影器来换 LLM（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3）。
- 视觉指令数据流水线使用 GPT-4，对新领域重新生成成本低廉。

### LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023 年 10 月）新增：
- 将学术任务数据（VQA、OKVQA、RefCOCO）混入指令微调。
- 更好的系统提示词。
- 上下文从 2048 扩展到 32k。

LLaVA-NeXT（2024 年 1 月）新增：
- AnyRes：将高分辨率图像分割为 2×2 或 1×3 的 336×336 裁剪网格，加一张全局低分辨率缩略图。每个裁剪产生 576 个 token，每张图像总计约 2880 个视觉 token。OCR 和图表任务大幅提升。
- 更好的指令数据混合方案，加入 ShareGPT4V（高质量 GPT-4V 描述）。
- 更强的基础 LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

Lesson 12.08 深入介绍 OneVision。简短版本：相同的投影器，但用课程学习训练，在一个模型中涵盖单图像、多图像和视频，共享视觉 token 预算。

### 与 Q-Former 的对比

| | Q-Former（BLIP-2） | MLP（LLaVA） |
|---|---|---|
| 每图像视觉 token | 32 个 | 576 个（基础）或 2880 个（AnyRes） |
| 可训练参数 | 1.88 亿 + LM | 4000 万 + LM |
| 阶段 1 损失 | ITC+ITM+ITG | 仅 LM |
| LLM 替换 | 需要重训 | 最小重训即可替换 |
| 多图像 | 不自然 | 自然（拼接） |
| 视频 | 不自然 | 自然（逐帧拼接） |
| Token 预算 | 小 | 大 |

MLP 在简洁性和 token 灵活性上胜出，Q-Former 在 token 预算上胜出。到 2023 年底，token 预算不再是约束条件（LLM 上下文扩展到 32k-128k+），简洁性占主导。

### 提示词格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是占位符 token。在分词之前，它被替换为 576 个视觉 token（或 AnyRes 的 2880 个）。分词器看到比训练时更长的序列，但 LLM 能处理这种新输入，因为阶段 1 教会了它。

### 参数经济学

LLaVA-1.5-7B 分解：
- CLIP ViT-L/14 @ 336：3.03 亿（阶段 1 冻结，阶段 2 通常解冻）。
- 投影器（2 层线性）：约 2200 万可训练参数。
- Llama-7B：70 亿。
- 总计：73 亿参数。阶段 2 训练参数：全量 70 亿 + 2200 万投影器。

阶段 2 训练成本：在 8×A100 上约 20 小时。这是关键数字——一天，一个节点，可复现。这就是 LLaVA 广泛传播的原因。

## 动手使用

`code/main.py` 实现：

1. 纯 Python 的 2 层 MLP 投影器（玩具规模：维度 16 → 32 → 32）。
2. 提示词构建流水线：系统提示词 + `<image>` 替换为 N 个投影 token + 用户轮次 + 助理生成占位符。
3. 576-token 视觉块在 LLM 上下文中的可视化器（占 2k / 32k / 128k 上下文的百分比）。

## 输出产物

本章生成 `outputs/skill-llava-vibes-eval.md`。给定 LLaVA 系列检查点，它运行一套 10 个提示词的"感觉评估"套件（3 个描述、3 个 VQA、2 个推理、2 个拒绝），输出人类可读的评分卡。不是正式基准，而是确认投影器和 LLM 良好连接的冒烟测试。

## 练习

1. 计算 `1024 → 4096 → 4096` 的 2 层 MLP 投影器的可训练参数量。带 GELU 和偏置的情况下，它占 LLaVA-13B 的多少比例？

2. 为"拒绝"场景构建 LLaVA 提示词——图像包含私人个体。写出预期的助理响应。LLaVA 为什么应该零样本拒绝，需要什么训练数据来强化拒绝行为？

3. 阅读 LLaVA-NeXT 博客中的 AnyRes 部分。计算 1344×672 图像在 AnyRes 下的视觉 token 数量。与 336×336 基础的 576 个 token 相比如何？

4. LLaVA 阶段 1 投影器用图文对的 LM 损失训练。如果跳过阶段 1 直接进行阶段 2（视觉指令微调）会发生什么？引用 Prismatic VLMs 消融实验（arXiv:2402.07865）的答案。

5. LLaVA-Instruct-150k 使用 GPT-4 和 COCO 描述生成指令。对于新领域（医学 X 光、卫星图像），描述生成领域指令的四步数据流水线。每步可能出什么问题？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 投影器（Projector） | "MLP 桥梁" | 带 GELU 的 2 层 MLP，将 ViT 维度映射到 LLM 维度。 |
| 图像 token（Image token） | "`<image>` 占位符" | 推理前被 N 个投影视觉 token 替换的提示词标记。 |
| 视觉指令微调（Visual instruction tuning） | "LLaVA 阶段 2" | 在 GPT-4 生成的（图像、指令、响应）三元组上训练。 |
| 阶段 1 对齐（Stage 1 alignment） | "投影器预训练" | 冻结 ViT 和 LLM，用图文对的 LM 损失训练投影器。 |
| AnyRes | "多裁剪平铺" | 将高分辨率图像分割为平铺网格，拼接每个平铺的视觉 token。 |
| LLaVA-Instruct | "GPT-4 生成" | 从 COCO 描述 + GPT-4 合成的 15.8 万条指令-响应对。 |
| 视觉编码器冻结（Vision encoder freeze） | "骨干锁定" | CLIP 权重在阶段 1 不更新，有时阶段 2 也不更新。 |
| ShareGPT4V | "更好的描述" | GPT-4V 生成的 100 万条密集描述，用于更高质量的对齐。 |
| VQA | "视觉问答" | 对图像回答自由形式问题的任务。 |
| Prismatic VLMs | "设计空间论文" | Karamcheti 2024 年系统测试投影器和数据选择的消融实验。 |

## 延伸阅读

- [Liu 等 — Visual Instruction Tuning（arXiv:2304.08485）](https://arxiv.org/abs/2304.08485) — LLaVA 原论文。
- [Liu 等 — Improved Baselines with Visual Instruction Tuning（arXiv:2310.03744）](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen 等 — ShareGPT4V（arXiv:2311.12793）](https://arxiv.org/abs/2311.12793) — 密集描述数据集。
- [Karamcheti 等 — Prismatic VLMs（arXiv:2402.07865）](https://arxiv.org/abs/2402.07865) — 设计空间消融实验。
- [Li 等 — LLaVA-OneVision（arXiv:2408.03326）](https://arxiv.org/abs/2408.03326) — 统一单图像、多图像、视频。

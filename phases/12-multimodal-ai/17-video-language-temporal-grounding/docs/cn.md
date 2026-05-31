# 视频语言模型：时序 Token 与时序接地（Video-Language Models: Temporal Tokens and Grounding）

> 视频不是照片的堆叠。一段 5 秒的短片具有因果顺序、动作动词和事件时间，这些都是图像模型无法表示的。Video-LLaMA（Zhang 等，2023 年 6 月）是第一个带有音视频接地的开放视频 LLM。VideoChat 和 Video-LLaVA 对这一模式进行了规模化。到 2025 年，Qwen2.5-VL 的 TMRoPE 弥合了与前沿专有模型的差距。每个系统对时序 token 的处理方式都不同——每个片段一个 Q-Former、每帧拼接池化、每 token 的 TMRoPE。本章解读这些模式，构建均匀 vs 动态帧采样器，并在时序接地任务上评估。

**类型：** 构建  
**语言：** Python（标准库，帧采样器 + 时序接地评估器）  
**前置知识：** Phase 12 · 08（LLaVA-OneVision）  
**预计时间：** 约 180 分钟

## 学习目标

- 解释为什么时序位置编码独立于视觉编码器改变视频 VLM 性能。
- 在每秒 token 数 vs 接地准确率上对比均匀采样、动态帧率和事件驱动帧采样。
- 描述每片段 Q-Former（Video-LLaMA）、每帧池化（Video-LLaVA）、每 token M-RoPE（Qwen2.5-VL）三种设计。
- 说出四个视频基准：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题所在

以 30 FPS 拍摄的 1 分钟视频有 1800 帧。在 ViT-B（224 分辨率）下每帧 196 个视觉 token，总共 35.2 万个 token——比 2024 年任何 LLM 的上下文都大。

存在三种缩减策略：

1. 对帧下采样（根据内容选择 1-8 FPS）。
2. 激进地池化每帧的图块 token（3×3 或 4×4 双线性池化）。
3. 通过 Q-Former 压缩，接受 16 帧片段并输出 64 个 token。

每种权衡各有不同。下采样丢失时序细节。池化丢失空间细节。Q-Former 两者都稍有丢失，但节省了 token。

时序位置编码是另一个轴：模型如何知道第 5 帧在第 6 帧之前？选项包括简单的一维时序 RoPE（Video-LLaMA）、可学习的时序嵌入（Video-LLaVA）和 TMRoPE（Qwen2.5-VL，完整三维）。

## 核心概念

### Video-LLaMA：每片段 Q-Former + 音频分支

Video-LLaMA（2023）是第一个开放视频 LLM。架构：

- 以 2 FPS 采样 16 帧（即 8 秒）。
- 每帧 ViT 特征 → 对全部 16 帧进行交叉注意力的视频 Q-Former → 32 个可学习查询 → LLM。
- 并行音频分支：波形 → ImageBind 音频编码器 → 音频 Q-Former → 32 个查询 → LLM。

优势：音视频联合推理。劣势：固定片段长度，无任意时间接地。

### VideoChat 和 Video-LLaVA

VideoChat 保留了 Video-LLaMA 的思路，但去掉了音频并简化。Video-LLaVA（Lin 等，2023）在图像和视频帧上训练单一视觉编码器（"投影前对齐"），给出统一表示。两者都是冻结 CLIP 编码器 + MLP + LLM。

两者都不处理长视频。都是 8-16 帧系统。

### Qwen2.5-VL 与 TMRoPE

Qwen2.5-VL 引入了 TMRoPE——时序模态旋转位置嵌入。每个图块 token 携带 (t, h, w) 位置，其中 t 是实际时间戳（不是帧索引）。

与简单时序嵌入的主要区别：

- 绝对时间，而非索引。模型看到"在 4.2 秒处"而非"在第 15 帧"。
- 每 token 旋转，而非每片段。每个视觉 token 按其时间戳独立旋转。
- 与动态帧率兼容。如果你在某段以 2 FPS 采样、在另一段以 4 FPS 采样，TMRoPE 原生处理不均匀间隔。

TMRoPE 使"猫在第几秒跳起来？"这类查询成为可能。模型可以输出"在 4.2 秒处"。Video-LLaMA 只能说"在片段早期"。

### 帧采样策略

**均匀采样：** 在时长内均匀采样 N 帧。简单，但丢失运动峰值。

**动态帧率：** 根据运动强度自适应采样。光流或帧差分选择高运动段进行更密集采样。Qwen2.5-VL 在此基础上训练。

**事件驱动：** 运行轻量级检测器，在有动作的地方采样更多帧。VideoAgent 使用此方法。

**关键帧 + 上下文：** 在镜头边界采样并加几个相邻帧。用于影视内容。

### 每帧池化

以 1 FPS 和每帧 576 个 token 计，5 分钟视频有 172,800 个 token。Qwen2.5-VL-72B 的 128k 上下文可以处理，但很昂贵。

3×3 双线性池化将每帧减少到 64 个 token → 5 分钟 19,200 个 token。大多数任务的最佳点。

对于空间细节不那么重要的智能体工作流，更激进地池化（6×6 → 每帧 16 个 token）。

### 四个视频基准

- **VideoMME：** 综合视频理解，短 + 中 + 长。
- **TempCompass：** 细粒度时序推理，"之前"/"之后"问题。
- **EgoSchema：** 长期第一人称视频。
- **Video-MMMU：** 多模态多学科视频问题。

完整的视频 VLM 评估覆盖所有四个。它们侧重不同维度——TempCompass 全关于顺序，EgoSchema 关于 3 分钟以上的推理，VideoMME 跨越不同时长。

### 接地输出格式

时序接地的输出格式：

- **自由文本：** "猫大约在 4 秒处跳起来。" 易于解析但不精确。
- **结构化 JSON：** `{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 在此上训练。
- **基于 token：** 与答案交错的特殊 `<time>4.1</time>` token。Qwen2.5-VL 的内部格式。

基于 token 的格式对下游使用最精确。Qwen2.5-VL 的 JSON 输出格式可直接解析。

### 2026 年最佳实践

2026 年视频 VLM 的最佳实践：

- **编码器：** 带 M-RoPE 或 TMRoPE 的 SigLIP 2（Qwen2.5-VL）。
- **帧采样：** 动态帧率（根据运动 1-4，设置最大帧数上限）。
- **每帧池化：** 3×3 双线性。
- **输出：** 带时间 + 事件字段的结构化 JSON。
- **基准：** VideoMME + TempCompass 用于通用；EgoSchema 用于长期推理。

## 动手使用

`code/main.py` 包含：

- 均匀和动态帧率帧采样器。
- 玩具时序接地评估器：给定"真实"事件时间 T 和模型输出，以容忍度评分准确率。
- 跨 Video-LLaMA（16 帧，Q-Former）、Video-LLaVA（8 帧，MLP）、Qwen2.5-VL（动态帧率 + TMRoPE）的对比。

## 输出产物

本章生成 `outputs/skill-video-vlm-frame-planner.md`。给定视频任务（监控、动作识别、时序接地、摘要），它选择帧采样器、池化因子、输出格式和预期准确率等级。

## 练习

1. 对于 3 分钟的烹饪演示，选择均匀采样还是动态帧率。用 token 数量来辩护。

2. TMRoPE 具体增加了什么，是简单时序嵌入表做不到的？

3. 为时序接地编写一个 VLM 可以学会输出的 JSON 模式。包含错误情况。

4. 阅读 Video-LLaVA 第 3 节关于"投影前对齐"的内容。为什么这比分别训练图像和视频编码器更好？

5. 查看 VideoMME 排行榜，截至 2026 年，顶级开放模型与顶级专有模型的差距是多少？这个差距中有多少可归因于时序编码与基础 LLM 规模？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 时序接地（Temporal grounding） | "时间定位答案" | VLM 输出事件发生的具体时间戳范围。 |
| TMRoPE | "时序多模态 RoPE" | 带绝对时间戳的三维旋转位置，Qwen2.5-VL 使用。 |
| 动态帧率（Dynamic FPS） | "运动感知采样" | 在高运动段采样更多帧，在静态段采样更少。 |
| 帧池化（Frame pooling） | "每帧空间压缩" | 在送给 LLM 之前用双线性插值减少每帧图块数。 |
| 视频 Q-Former（Video Q-former） | "片段压缩器" | 将 N 帧映射到 K 个可学习查询的交叉注意力瓶颈。 |
| VideoMME | "视频基准" | 综合短/中/长视频基准，2500+ 样本。 |

## 延伸阅读

- [Zhang 等 — Video-LLaMA（arXiv:2306.02858）](https://arxiv.org/abs/2306.02858)
- [Li 等 — VideoChat（arXiv:2305.06355）](https://arxiv.org/abs/2305.06355)
- [Lin 等 — Video-LLaVA（arXiv:2311.10122）](https://arxiv.org/abs/2311.10122)
- [Qwen 团队 — Qwen2.5-VL（arXiv:2502.13923）](https://arxiv.org/abs/2502.13923)
- [Lin 等 — VILA-1.5（arXiv:2312.07533）](https://arxiv.org/abs/2312.07533)

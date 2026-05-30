# 评估——FID、CLIP 分数与人类偏好

> 每个生成模型排行榜都引用 FID、CLIP 分数和来自人类偏好竞技场的胜率。每个数字都有一种有决心的研究者可以操纵的失效模式。如果你不知道失效模式，就无法分辨真正的改进和刷分运行。

**类型：** 构建
**语言：** Python
**前置知识：** 第8阶段第01课（分类）、第2阶段第04课（评估指标）
**预计时间：** 约45分钟

## 问题背景

生成模型的评判标准是*样本质量*和*条件遵循度*。两者都没有封闭形式的度量。你的模型必须渲染 1 万张图像；某样东西必须给它们打分；你必须跨模型家族、跨分辨率、跨架构信任这些数字。三个指标经历了 2014—2026 年的考验：

- **FID（Fréchet Inception 距离）。** Inception 网络特征空间中真实分布与生成分布之间的距离。越低越好。
- **CLIP 分数。** 生成图像的 CLIP 图像嵌入与提示词的 CLIP 文本嵌入之间的余弦相似度。越高越好。衡量提示词遵循度。
- **人类偏好。** 让两个模型在同一提示词上正面交锋，让人类（或 GPT-4 级别的模型）选出更好的，汇总为 Elo 分数。

你还会看到：IS（Inception 分数，已大部分退休）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每个都纠正了前一个的某个缺陷。

## 核心概念

![FID、CLIP 和偏好：三个轴，不同的失效模式](../assets/evaluation.svg)

### FID——样本质量

Heusel et al.（2017）。步骤：

1. 为 N 张真实图像和 N 张生成图像提取 Inception-v3 特征（2048 维）。
2. 对每个特征池拟合高斯分布：计算均值 `μ_r, μ_g` 和协方差 `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解释：特征空间中两个多元高斯之间的 Fréchet 距离。越低 = 分布越相似。

失效模式：
- **小 N 时有偏。** FID 是特征分布上的均方——小 N 低估协方差，给出虚假的低 FID。始终使用 N ≥ 1 万。
- **依赖 Inception。** Inception-v3 在 ImageNet 上训练。远离 ImageNet 的领域（人脸、艺术、文字图像）产生无意义的 FID。使用领域特定的特征提取器。
- **刷分。** 过拟合 Inception 先验可以给出低 FID，而不提升视觉质量。用 CMMD 来应对（见下文）。

### CLIP 分数——提示词遵循度

Radford et al.（2021）。对于生成图像 + 提示词：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

在 3 万张生成图像上取平均 → 可在模型间比较的标量。

失效模式：
- **CLIP 自身的盲点。** CLIP 的组合推理较弱（"蓝色球体上的红色立方体"经常失败）。模型可能在 CLIP 分数上排名很高，却并不真正遵循复杂提示词。
- **短提示词偏向。** 短提示词在自然界中有更多 CLIP 图像匹配。长提示词机械地得到更低的 CLIP 分数。
- **提示词刷分。** 在提示词中包含"高质量, 4k, 杰作"会在不改善图文绑定的情况下拉高 CLIP 分数。

CMMD（Jayasumana et al.，2024）修正了其中一些问题：使用 CLIP 特征而不是 Inception 特征，用最大均值差异（MMD）代替 Fréchet 距离。在检测细微质量差异方面更好。

### 人类偏好——真正的基准

挑选一组提示词。用模型 A 和模型 B 生成。将配对展示给人类（或强大的 LLM 评判者）。将胜出次数汇总为 Elo 或 Bradley-Terry 分数。基准测试：

- **PartiPrompts（Google）**：1600 个多样化提示词，12 个类别。
- **HPSv2**：10.7 万人工标注，广泛用作自动化代理。
- **ImageReward**：13.7 万个提示词-图像偏好对，MIT 许可证。
- **PickScore**：在 Pick-a-Pic 260 万偏好上训练。
- **聊天机器人竞技场风格的图像竞技场**：imagearena.ai 等。

失效模式：
- **评判者差异。** 非专家与专家有不同的偏好。同时使用两者。
- **提示词分布。** 精心挑选的提示词偏向某个家族。始终记录。
- **LLM 评判者奖励欺骗。** GPT-4 评判者会被漂亮但错误的输出所迷惑。与人类评判互相印证。

## 综合使用

生产评估报告应包括：

1. 在 1-3 万个样本上对比留出真实分布的 FID（样本质量）。
2. 相同样本与其提示词的 CLIP 分数 / CMMD（遵循度）。
3. 在与之前模型的盲测竞技场中的胜率（整体偏好）。
4. 失效模式分析：随机采样 50 个输出，对已知问题进行标注（手部解剖、文字渲染、一致的物体数量）。

任何单一指标都是谎言。三个相互印证的指标 + 定性审查才是一个主张。

## 动手实现

`code/main.py` 在合成"特征向量"（我们用 4 维向量代替 Inception 特征）上实现 FID、类 CLIP 分数和 Elo 汇总。你能看到：

- 小 N 和大 N 的 FID 计算——偏差。
- "CLIP 分数"作为特征池之间的余弦相似度。
- 来自合成偏好流的 Elo 更新规则。

### 第一步：四行代码的 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 第二步：CLIP 风格的余弦相似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 第三步：Elo 汇总

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 常见陷阱

- **N=1000 时的 FID。** 启发式在 N<1 万时不可靠。报告低 N FID 的论文是在刷分。
- **跨分辨率比较 FID。** Inception 的 299×299 缩放会改变特征分布。只在匹配分辨率下比较。
- **只报告一个种子。** 至少运行 3 个种子。报告标准差。
- **通过负提示词拉高 CLIP 分数。** 一些流水线通过过拟合提示词来提升 CLIP。检查视觉饱和度。
- **来自提示词重叠的 Elo 偏差。** 如果两个模型在训练中都见过基准提示词，Elo 毫无意义。使用留出提示词集。
- **人工评估众包偏差。** Prolific、MTurk 标注者偏向年轻 / 技术友好。与招募的艺术/设计专家混合。

## 工程应用

2026 年的生产评估协议：

| 支柱 | 最低要求 | 推荐 |
|------|---------|------|
| 样本质量 | 1 万个样本 vs 留出真实 FID | + 5000 个 CMMD + 每类别子集 FID |
| 提示词遵循度 | 3 万个 CLIP 分数 | + HPSv2 + ImageReward + VQA 风格问答 |
| 偏好 | 200 个盲测对 vs 基线 | + 2000 对人类 + LLM 评判者 + 聊天机器人竞技场 |
| 失效分析 | 50 个手动标注 | 500 个手动标注 + 自动安全分类器 |

四个支柱都在一份报告中 = 主张。任何单一支柱 = 营销。

## 交付物

见 `outputs/skill-eval-report.md`。该技能接受新模型检查点 + 基线，输出完整的评估计划：样本量、指标、失效模式探测、签核标准。

## 练习

1. **（简单）** 运行 `code/main.py`。在相同合成分布上比较 N=100 vs N=1000 的 FID。报告偏差大小。
2. **（中等）** 从合成 CLIP 风格特征实现 CMMD（参见 Jayasumana et al.，2024 的公式）。比较其与 FID 相比检测质量差异的灵敏度。
3. **（困难）** 复现 HPSv2 设置：从 Pick-a-Pic 子集取 1000 个图像-提示词对，在偏好数据上微调一个小型 CLIP 评分器，并测量其在留出集上与人类判断的一致程度。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| FID | "Fréchet Inception 距离" | 真实 vs 生成 Inception 特征高斯拟合的 Fréchet 距离。 |
| CLIP 分数 (CLIP score) | "文图相似度" | CLIP 图像嵌入与文本嵌入之间的余弦相似度。 |
| CMMD | "FID 的替代品" | 基于 CLIP 特征的 MMD；偏差更小，无高斯假设。 |
| IS | "Inception 分数" | Exp KL(p(y\|x) \|\| p(y))；与现代模型相关性差，已退休。 |
| HPSv2 / ImageReward / PickScore | "学习的偏好代理" | 在人类偏好上训练的小型模型；用作自动评判者。 |
| Elo | "国际象棋评级" | 成对胜出的 Bradley-Terry 汇总。 |
| PartiPrompts | "那个基准提示词集" | Google 精心策划的 1600 个提示词，12 个类别。 |
| FD-DINO | "自监督替代品" | 使用 DINOv2 特征的 FD；对 ImageNet 域外更好。 |

## 生产说明：评估也是推理工作负载

在 1 万个样本上运行 FID 意味着生成 1 万张图像。对于在 L4 上单次请求运行 50 步 SDXL base 在 1024² 分辨率下，这需要约 11 小时的推理。评估预算是真实的，框架正好是离线推理场景（最大化吞吐量，忽略首 token 时间）：

- **努力批处理，忘掉延迟。** 离线评估 = 在内存允许的最大批次下进行静态批处理。在 80GB H100 上使用 `num_images_per_prompt=8` 的 `pipe(...).images` 挂钟速度比单次请求快 4-6 倍。
- **缓存真实特征。** 对真实参考集提取 Inception（FID）或 CLIP（CLIP 分数、CMMD）特征*一次*，存储为 `.npz`。不要每次评估都重新计算。

对于 CI / 回归关卡：每次 PR 在 500 个样本子集上运行 FID + CLIP 分数（约 30 分钟）；每晚运行完整的 1 万 FID + HPSv2 + Elo。

## 延伸阅读

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 论文
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 失效模式综述

# 推测解码与EAGLE

> 前沿LLM生成一个词元需要对数十亿参数做一次完整的前向传播。这次前向传播被严重过度配置了：大多数时候，一个小得多的模型可以正确猜测接下来的3到5个词元，而大模型只需要*验证*这个猜测。猜对了，你用一次前向传播的代价获得了5个词元。推测解码（Leviathan等人，2023年）使这一方法精确化，EAGLE-3（2025年）将接受率推至平均每次验证约4.5个词元——在匹配输出分布的情况下实现4到5倍加速。

**类型：** 构建  
**语言：** Python（含numpy）  
**前置知识：** Phase 10·第12课（推理优化），Phase 10·第04课（预训练Mini-GPT）  
**用时：** 约75分钟

## 问题所在

70B级别模型在H100上的解码吞吐量通常是每秒40到80个词元。每个词元都需要一次完整的前向传播，将所有模型权重从HBM读取一遍。你不能在不改变输出的情况下让模型变小，也不能在内存允许范围之外增大批大小。你被卡住了——除非你能让模型在每次前向传播中输出多于一个词元。

自回归生成看起来本质上是串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但存在一个并发机会。如果你有一个廉价的预测器说"接下来4个词元大概是[a, b, c, d]"，你可以用**大模型的单次前向传播**验证所有5个位置，并接受最长的匹配前缀。

Leviathan, Kalai, Matias（2023年，《Fast Inference from Transformers via Speculative Decoding》）通过一个聪明的接受/拒绝规则使这一方法精确化，该规则保留了目标模型的采样分布。相同的输出分布，快了2到4倍。

## 核心概念

### 双模型设置

- **目标模型** `M_p`：你实际想要样本的大型、缓慢、高质量模型。分布：`p(x)`
- **草稿模型** `M_q`：小型、快速、低质量模型。分布：`q(x)`。小5到30倍

每步骤：

1. 草稿模型自回归地提出`K`个词元：`x_1, x_2, ..., x_K ~ q`
2. 目标模型对所有`K+1`个位置**并行运行一次**前向传播，产生每个提出词元的`p(x_k)`
3. 通过下面的改进拒绝采样规则从左到右接受/拒绝每个词元，接受最长的匹配前缀
4. 如果任何词元被拒绝，从修正分布采样替代词元并停止；否则从`p(· | x_1...x_K)`采样一个额外词元

如果草稿完全匹配目标，你每次目标前向传播得到K+1个词元。如果草稿在位置1出错，你只得到1个词元。

### 精确规则

推测解码**可证明与从p采样分布等价**。拒绝规则：

```
对每个草稿词元 x_t：
    r ~ Uniform(0, 1)
    如果 r < p(x_t) / q(x_t)：
        接受 x_t
    否则：
        从残差分布采样替代词元：(p - q)+ / ||(p - q)+||_1
        停止
```

其中`(p - q)+`表示逐点差的正部。当草稿和目标一致时（`p ≈ q`），接受率接近1。当它们不一致时，残差分布被构造为使整体样本仍然精确地服从`p`。

**贪心情况。** 对于temperature=0的采样，只需检查`argmax(p) == x_t`。如果是，接受；否则输出`argmax(p)`并停止。

### 预期加速比

如果草稿模型的词元级接受率为`α`，每次目标前向传播产生的预期词元数为：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = 草稿长度，α ∈ [0, 1]
```

在`α = 0.8, K = 4`时：`(1 - 0.8^5)/(1 - 0.8) = 3.36`个词元/前向传播。单次目标前向传播的代价大约是`cost_q * K + cost_p`（K个草稿步骤加一次目标验证）。如果`cost_p >> cost_q * K`，吞吐量加速比为`3.36×`。

唯一真正的参数是`α`，它完全取决于草稿与目标的对齐程度。好的草稿就是一切。

### 训练草稿模型：蒸馏

随机选取一个小模型效果很差。标准方案是从目标模型蒸馏：

1. 选择一个小型架构（70B目标约用1B，7B目标约用500M）
2. 用目标模型处理大型文本语料库，存储其下一词元分布
3. 用KL散度对目标分布训练草稿模型（而不是对真实词元）

结果：`α`在代码生成上通常为0.6到0.8，在自然语言对话上为0.7到0.85。生产中加速比2到3倍。

### EAGLE：树形起草+特征复用

Li, Wei, Zhang, Zhang（2024年，《EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty》）观察到标准推测解码的两个低效之处：

1. 草稿做K次串行步骤，每次都是全栈计算。但草稿可以复用目标从最近验证中计算的特征（隐藏状态）——目标已经计算了丰富的表示，而草稿正在从头重新推导
2. 草稿输出线性链。如果草稿能输出一棵候选*树*（每个节点多个猜测），目标的单次前向传播可以通过树注意力掩码并行验证多条候选路径，并选择最长的被接受分支

EAGLE-1的改动：
- 草稿输入 = 目标在位置t的最终隐藏状态，而非原始词元
- 草稿架构 = 1个Transformer解码器层（而非独立的小型模型）
- 输出 = 每个深度K=4到8个候选的树，深度4到6

EAGLE-2（2024年）添加了动态树拓扑：树在草稿不确定的地方长得更宽，在确定的地方保持窄。在不增加验证代价的情况下提升`α_effective`。

EAGLE-3（Li等人，2025年，《EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test》）去除了固定的顶层特征依赖，用新的"测试时模拟"损失训练草稿——草稿被训练成匹配目标测试时分布的输出，而非教师强制的训练分布。接受率从0.75（EAGLE-2）提升到0.82（EAGLE-3），平均词元/验证从3.0提升到4.5。

### 树注意力验证

当草稿输出一棵树时，目标模型使用**树注意力掩码**在单次前向传播中验证它——一种编码树拓扑而非纯线性序列的因果掩码。每个词元只关注树中的祖先节点。验证传播仍然是一次前向传播，一次矩阵乘法；拓扑掩码只增加了少量额外的KV条目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果`a, b`是竞争的第一个词元候选，`c, d, e, f`是第二个词元候选，所有六个位置在一次前向传播中被验证。输出是任何被接受路径上最长的前缀。

### 何时有效，何时无效

**有效场景：**
- 具有可预测文本的对话/补全（代码、常见英语、结构化输出）。`α`较高
- GPU计算在解码期间有空闲的场景（内存受限阶段）。树形起草利用可用的FLOP

**无效/无收益场景：**
- 高度随机的输出（高温度的创意写作）。`α`趋向`1/|vocab|`
- 高并发度的批量服务——批处理已经填满了FLOP，树形验证几乎没有空间
- 草稿和目标模型大小差异不大的小型目标模型

生产环境通常报告对话上2到3倍的实时加速，代码生成上3到5倍，创意写作上几乎为零。

## 动手构建

`code/main.py`：

- 参考实现`speculative_decode(target, draft, prompt, K, temperature)`，实现精确的拒绝规则，并验证它保留了目标分布（与纯目标采样相比，经验KL < 0.01）
- EAGLE风格的树形起草器，构建一棵深度K、使用top-p分支的树
- 树注意力掩码构建器，为验证器生成正确的因果模式
- 接受率测试框架，在一个小型语言模型上运行两者（从GPT-2-medium目标蒸馏一个GPT-2-small）

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """一轮推测解码。返回被接受词元的列表。"""
    # 1. 草稿生成K个词元
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. 目标在所有草稿位置+1个额外位置并行计算p
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. 从左到右接受/拒绝
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. 所有K个词元被接受 → 从目标采样一个额外词元
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 实际使用

- **vLLM**和**SGLang**原生支持推测解码。标志：`--speculative_model`，`--num_speculative_tokens`。通过`--spec_decoding_algorithm eagle`标志支持EAGLE-2/3
- **NVIDIA TensorRT-LLM**原生支持Medusa和EAGLE树
- **参考草稿模型**：`Qwen/Qwen3-0.6B-spec`（为Qwen3-32B起草），`meta-llama/Llama-3.2-1B-Instruct-spec`（为70B起草）
- **Medusa头**（Cai等人，2024年，《Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads》）：在目标模型上直接添加K个并行预测头，而不是使用单独的草稿模型。部署更简单，接受率略低于EAGLE

## 交付成果

本课产出`outputs/skill-speculative-tuning.md`——一种对目标模型工作负载进行性能分析并选择的技能：草稿模型、K（草稿长度）、树宽、温度，以及何时回退到普通解码。

## 练习

1. 实现精确的拒绝规则并经验性地验证它。通过`speculative_decode`运行10K次采样，也通过纯目标采样运行10K次；计算两个输出分布之间的TV距离，应该 < 0.01。

2. 计算加速比公式。给定固定的`α`和`K`，绘制每次目标前向传播的预期词元数。对α ∈ {0.5, 0.7, 0.9}各找出最优K。

3. 训练一个微型草稿模型。以一个124M的GPT-2作为目标，在1亿个词元上用KL损失蒸馏一个30M的GPT-2草稿。在保留文本上测量`α`，预期为0.6到0.7。

4. 实现EAGLE风格的树形起草。不用链式，而是让草稿在每个深度输出top-3分支。构建树注意力掩码，验证目标接受任何被接受路径上最长的正确分支。

5. 测量失效模式。在temperature=1.5（高随机性）下运行推测解码，证明α崩溃，算法因草稿开销而比纯解码更慢。

## 关键术语

| 术语（英文） | 人们怎么说 | 实际含义 |
|-------------|-----------|---------|
| 目标模型（Target model） | "大模型" | 你想要样本的缓慢、高质量模型（p分布） |
| 草稿模型（Draft model） | "推测者" | 小型、快速的预测器（q分布）；小5到30倍 |
| K / 草稿长度（Draft length） | "前瞻" | 每次验证传播推测的词元数 |
| α / 接受率（Acceptance rate） | "命中率" | 草稿提案被接受的逐词元概率 |
| 精确拒绝规则（Exact rejection rule） | "接受测试" | r < p/q比较，保留目标分布 |
| 残差分布（Residual distribution） | "修正后的p-q" | `(p - q)+ / ||(p - q)+||_1`，拒绝时用于采样的分布 |
| 树形起草（Tree drafting） | "分支推测" | 草稿输出候选树，通过树结构注意力掩码在一次传播中验证 |
| 树注意力掩码（Tree attention mask） | "拓扑掩码" | 编码树拓扑的因果掩码，使每个节点只关注其祖先 |
| Medusa头（Medusa heads） | "并行头" | 目标模型上的K个额外预测头；不需要单独的草稿模型 |
| EAGLE特征复用（EAGLE feature reuse） | "隐藏状态起草" | 草稿输入是目标的最后隐藏状态而非原始词元，缩小了草稿规模 |
| 测试时模拟损失（Test-time simulation loss） | "EAGLE-3训练" | 在匹配目标测试时分布的输出上训练草稿，而非教师强制 |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 精确拒绝规则和理论加速比分析
- [Chen, Borgeaud, Irving等, 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind同期推测采样论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — 草稿模型的并行头替代方案
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 特征复用和树形起草
- [Li等, 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 动态树拓扑
- [Li等, 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 训练时测试时分布匹配
- [Fu, Haotian, Peng等, 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/前瞻解码，无需推测器的替代方案

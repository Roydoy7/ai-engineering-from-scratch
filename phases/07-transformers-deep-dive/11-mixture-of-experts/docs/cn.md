# 专家混合模型（MoE）

> 稠密的 700 亿参数 Transformer 对每个 token 激活所有参数。6710 亿参数的 MoE 每个 token 只激活 370 亿参数，却在所有基准上更胜一筹。稀疏性是这十年最重要的规模化思想。

**类型：** 构建
**语言：** Python
**前置知识：** 第7阶段第05课（完整 Transformer）、第7阶段第07课（GPT）
**预计时间：** 约45分钟

## 问题背景

稠密 Transformer 在推理时的 FLOP 等于其参数量（乘以 2 用于前向传播）。扩大稠密模型，每个 token 都要付全额代价。到 2024 年，前沿模型撞上了计算瓶颈：要变得明显更聪明，就需要指数级更多的每 token FLOP。

专家混合模型打破了这一联系。用 `E` 个独立专家 + 一个为每个 token 选择 `k` 个专家的路由器替换每个 FFN。总参数量 = `E × FFN大小`。每个 token 的活跃参数量 = `k × FFN大小`。典型的 2026 年配置：`E=256`，`k=8`。存储随 `E` 扩展，计算随 `k` 扩展。

2026 年的前沿模型几乎全是 MoE：DeepSeek-V3（6710 亿总参数 / 370 亿活跃参数）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，前 10 名开源模型全部是 MoE。

## 核心概念

### FFN 替换

稠密 Transformer 块：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 块：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # 每个 token 从 E 中选 k
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家都是独立的 FFN（通常是 SwiGLU）。路由器是单个线性层。每个 token 选择自己的 `k` 个专家，得到它们输出的加权混合。

### 负载均衡问题

如果路由器将 90% 的 token 送到专家 3，其他专家就会挨饿。已尝试三种解决方案：

1. **辅助负载均衡损失**（Switch Transformer、Mixtral）。添加与专家使用方差成比例的惩罚项。有效，但增加了一个超参数和第二个梯度信号。
2. **专家容量 + token 丢弃**（早期 Switch）。每个专家最多处理 `C × N/E` 个 token；溢出的 token 跳过该层。损害质量。
3. **无辅助损失的均衡**（DeepSeek-V3）。为路由器的 top-k 选择添加学习到的每专家偏置。偏置在训练损失之外更新。对主要目标没有惩罚。2024 年的重大突破。

DeepSeek-V3 的方法：每个训练步之后，对每个专家检查其使用率是否超过或低于目标值。以 `±γ` 微调偏置。选择使用 `scores + bias`，用于门控的专家概率使用原始 `scores`，不变。将路由与表达解耦。

### 共享专家

DeepSeek-V2/V3 还将专家分为*共享*专家和*路由*专家。每个 token 都经过所有共享专家。路由专家通过 top-k 选择。共享专家捕获通用知识；路由专家进行专门化。V3 运行 1 个共享专家加 256 个路由专家中的前 8 个。

### 细粒度专家

经典 MoE（GShard、Switch）：每个专家与完整 FFN 一样宽。`E` 小（8–64），`k` 小（1–2）。

现代细粒度 MoE（DeepSeek-V3、Qwen-MoE）：每个专家更窄（FFN 大小的 1/8）。`E` 大（256+），`k` 更大（8+）。总参数量相同，但组合方式扩展得快得多。`C(256, 8) = 400 万亿`种可能的每 token"专家"组合。质量上升，延迟保持不变。

### 成本分析

每 token 每层：

| 配置 | 每 token 活跃参数 | 总参数 |
|------|----------------|-------|
| Mixtral 8×22B | ~390亿 | 1410亿 |
| Llama 3 70B（稠密） | 700亿 | 700亿 |
| DeepSeek-V3 | 370亿 | 6710亿 |
| Kimi K2（MoE） | ~320亿 | 1万亿 |

DeepSeek-V3 在几乎所有基准上都优于 Llama 3 70B（稠密），同时每 token 的**活跃 FLOP 更少**。更多参数 = 更多知识。更多活跃 FLOP = 每 token 更多计算。MoE 将两者解耦。

### 代价：内存

所有专家都在 GPU 上，无论哪些被激活。6710 亿参数的模型需要约 1.3 TB VRAM 来存放 fp16 权重。前沿 MoE 部署需要专家并行——将专家分片到多个 GPU，通过网络路由 token。延迟由全互连通信主导，而非矩阵乘法。

## 动手实现

见 `code/main.py`。纯标准库的紧凑 MoE 层，包含：

- `n_experts=8` 个类 SwiGLU 专家（每个一个线性层，用于演示）
- top-k=2 路由
- softmax 归一化的门控权重
- 通过每专家偏置实现的无辅助损失均衡

### 第一步：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # 对所选专家的原始分数做 softmax
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置影响选择，不影响门控权重。这就是 DeepSeek-V3 的技巧——偏置纠正负载不均衡，而不干预模型的预测。

### 第二步：通过路由器运行 100 个 token

追踪哪些专家被激活了多少次。没有偏置时，使用率是偏斜的。通过偏置更新循环（对过度使用的专家 `-γ`，对使用不足的专家 `+γ`），使用率在几次迭代后收敛到均匀分布。

### 第三步：参数量对比

打印 MoE 配置的"稠密等效"参数量。DeepSeek-V3 形状：256 个路由专家 + 1 个共享专家，8 个活跃，d_model=7168。总参数量令人咋舌。活跃参数量约为稠密 Llama 3 70B 的七分之一。

## 工程应用

HuggingFace 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年生产推理：vLLM 原生支持 MoE 路由。SGLang 有最快的专家并行路径。两者都自动处理 top-k 选择和专家并行。

**何时选择 MoE：**
- 你想要前沿质量，同时降低每 token 推理成本
- 你有 VRAM / 专家并行基础设施
- 你的工作负载是 token 密集型（对话、代码）而非上下文密集型（长文档）

**何时不选择 MoE：**
- 边缘部署——任何活跃 FLOP 都要付全额存储代价
- 延迟敏感的单用户服务——专家路由增加开销
- 小型模型（<70 亿参数）——MoE 的质量优势只在计算阈值（约 60 亿活跃参数）以上才出现

## 交付物

见 `outputs/skill-moe-configurator.md`。该技能根据参数预算、训练 token 数和部署目标，为新的 MoE 选择 E、k 和共享专家布局。

## 练习

1. **（简单）** 运行 `code/main.py`。观察无辅助损失的偏置更新如何在 50 次迭代内平衡专家使用率。
2. **（中等）** 用基于哈希的路由器（确定性，无需学习）替换学习型路由器。比较质量和均衡性。为什么学习型路由器更好？
3. **（困难）** 实现 GRPO 风格的"推出匹配路由"（DeepSeek-V3.2 技巧）：记录推理时哪些专家被激活，在梯度计算期间强制使用相同路由。在玩具策略梯度设置上测量效果。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 专家 (Expert) | "众多 FFN 中的一个" | 独立的前馈网络；参数专用于稀疏 FFN 计算的一个切片 |
| 路由器 (Router) | "门控器" | 为每个 token 对每个专家打分的微型线性层；top-k 选择 |
| Top-k 路由 | "每个 token k 个活跃专家" | 每个 token 的 FFN 计算恰好经过 k 个专家，按门控加权 |
| 辅助损失 (Auxiliary loss) | "负载均衡惩罚" | 惩罚偏斜专家使用率的额外损失项 |
| 无辅助损失均衡 | "DeepSeek-V3 的技巧" | 通过路由器选择上的每专家偏置实现均衡；无额外梯度 |
| 共享专家 (Shared expert) | "始终激活" | 每个 token 都经过的额外专家；捕获通用知识 |
| 专家并行 (Expert parallelism) | "按专家分片" | 将不同专家分布到不同 GPU；通过网络路由 token |
| 稀疏性 (Sparsity) | "活跃参数 < 总参数" | 比例 `k × 专家大小 / (E × 专家大小)`；DeepSeek-V3 约为 37/671 ≈ 5.5% |

## 延伸阅读

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 原始思想
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch，经典 MoE
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + 无辅助损失 MoE + MTP
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — 基于偏置均衡的论文
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 细粒度 + 共享专家拆分
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 原始共享专家论文

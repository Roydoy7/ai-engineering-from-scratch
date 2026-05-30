# KV 缓存、Flash Attention 与推理优化

> 训练是并行的，受 FLOP 限制。推理是串行的，受内存限制。不同的瓶颈，不同的技巧。

**类型：** 构建
**语言：** Python
**前置知识：** 第7阶段第02课（自注意力）、第7阶段第05课（完整 Transformer）、第7阶段第07课（GPT）
**预计时间：** 约75分钟

## 问题背景

朴素的自回归解码器生成 `N` 个 token 需要 `O(N²)` 的工作量：在每一步，它对整个前缀重新计算注意力。对于 4K token 的响应，这意味着 1600 万次注意力操作，其中大部分是冗余的。前缀 token 的每个隐状态一旦计算后就是确定的——你只需要将新 token 的查询与之前所有 token 的缓存键和值进行对比。

此外，注意力本身也需要大量数据移动。标准注意力会实例化 N×N 的分数矩阵、N×d 的 softmax 输出、N×d 的最终输出——HBM 读写次数太多。对于 N≥2K，注意力在成为 FLOP 受限之前就已经成为内存受限了。经典注意力核对现代 GPU 的利用率只有 4–10×。

两项优化，均来自 Dao et al.，将前沿推理从"慢"推向"快"：

1. **KV 缓存。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的注意力只是一个查询对缓存键的操作。推理从每生成步骤 `O(N²)` 降低到 `O(N)`。
2. **Flash Attention。** 分块计算注意力，完整的 N×N 矩阵永远不接触 HBM。所有 softmax + 矩阵乘都在 SRAM 中完成。在 A100 上挂钟加速 2–4 倍；在 H100 FP8 下 5–10 倍。

2026 年两者都已普及。每个生产推理栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都以此为前提。每个前沿模型都默认启用 Flash Attention。

## 核心概念

### KV 缓存计算

每个解码器层，每个 token，每个头：

```
每 token 每层字节数 = 2 * d_head * dtype_大小
                   ^
                   K 和 V
```

对于有 32 层、32 个头、d_head=128、fp16 的 70 亿参数模型：

```
每 token 每层 = 2 * 128 * 2 = 512 字节
每 token（32 层） = 16 KB
32K 上下文 = 512 MB
```

对于 Llama 3 70B（80 层，d_head=128，GQA 带 8 个 KV 头）：

```
每 token 每层 = 2 * 8 * 128 * 2 = 4096 字节（4 KB）
32K 上下文 = 10.4 GB
```

这 10 GB 就是为什么 Llama 3 70B 在 128K 上下文下、批大小为 1 时，光 KV 缓存就需要 40 GB A100 的大部分空间。

**GQA 是 KV 缓存的节省。** 64 个头的 MHA 需要 32 GB。MLA 压缩得更多。

### Flash Attention——分块技巧

标准注意力：

```
S = Q @ K^T          （HBM 读，N×N，HBM 写）
P = softmax(S)       （HBM 读，HBM 写）
O = P @ V            （HBM 读，HBM 写）
```

三次 HBM 来回。H100 上 HBM 带宽为 3 TB/s；SRAM 为 30 TB/s。每次 HBM 往返相当于比片内操作慢 10 倍。

Flash Attention：

```
对于每个 Q 块（分块大小约 128 × 128）：
    将 Q_tile 加载到 SRAM
    对于每个 K、V 块：
        将 K_tile、V_tile 加载到 SRAM
        计算 S_tile = Q_tile @ K_tile^T     （SRAM）
        运行 softmax 聚合                   （SRAM）
        累积到 O_tile                        （SRAM）
    将 O_tile 写回 HBM
```

每块只有一次 HBM 往返。总内存占用从 `O(N²)` 降至 `O(N)`。反向传播从前向传播中重新计算部分值，而不是存储它们——又一项内存节省。

**数值技巧。** 运行中的 softmax 在各块之间维护 `(最大值, 求和)` 状态，使最终归一化结果精确无误。这不是近似——Flash Attention 计算出与标准注意力完全相同的输出（除 fp16 非结合性外）。

**版本演进：**

| 版本 | 年份 | 关键变化 | 参考硬件加速比 |
|------|------|---------|-------------|
| Flash 1 | 2022 | 分块 SRAM 核 | A100 上 2× |
| Flash 2 | 2023 | 更好的并行性，因果优先排序 | A100 上 3× |
| Flash 3 | 2024 | Hopper 异步，FP8 | H100 上 1.5–2×（~740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 5 阶段流水线，软件 exp2 | 推理优先（初始仅前向传播） |

Flash 4 发布时仅支持前向传播。训练仍使用 Flash 3。Flash 4 的 GQA 和变长支持待定（2026 年中）。

### 投机解码——另一项延迟优化

小模型提出 N 个 token。大模型并行验证所有 N 个。如果验证接受 k 个 token，你用一次大模型前向传播生成了 k 个 token。在代码和散文上典型的 k=3–5。

2026 年默认方案：
- **EAGLE 2 / Medusa。** 共享验证器隐状态的集成草稿头。无质量损失地提速 2–3 倍。
- **带草稿模型的投机解码。** 消费级硬件上提速 2–4 倍。
- **前瞻解码。** Jacobi 迭代；不需要草稿模型。小众但免费。

### 连续批处理

经典批量推理：等最慢的序列完成，然后开始新批次。短响应提前完成时浪费 GPU。

连续批处理（首次在 Orca 中推出，现在在 vLLM、TensorRT-LLM、SGLang 中）：旧请求完成后立即将新请求加入批次。对典型对话工作负载，吞吐量提升 5–10 倍。

### PagedAttention——KV 缓存即虚拟内存

vLLM 的核心功能。KV 缓存按 16 个 token 的块分配；页表将逻辑位置映射到物理块。可以在并行采样（束搜索、并行采样）之间共享 KV，热交换前缀用于提示缓存，并对内存进行碎片整理。与朴素连续分配相比，吞吐量提升 4 倍。

## 动手实现

见 `code/main.py`。我们实现：

1. 朴素的 `O(N²)` 增量解码器
2. `O(N)` 的 KV 缓存解码器
3. 模拟 Flash Attention 运行最大值算法的分块 softmax

### 第一步：KV 缓存

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

简单：在每层每头的列表中不断增加每 token 的 K、V 向量。

### 第二步：分块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention 风格的 softmax(qK^T)V，使用运行中的最大值/求和。"""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

与一次性 `softmax(qK) V` 输出完全相同，但任何时刻工作集都只是 `tile × d_head` 的块，而不是完整的 `N × d_head`。

### 第三步：比较朴素解码器与缓存解码器在 100 个 token 生成上的表现

统计注意力操作数。朴素方案：`O(N²)` = 5050。缓存方案：`O(N)` = 100。代码打印两者。

## 工程应用

```python
# HuggingFace transformers 在仅解码器 generate() 上自动启用 KV 缓存
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # Hopper 用 FA3
    torch_dtype="bfloat16",
)
# generate() 自动使用 KV 缓存
```

vLLM 生产部署：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的前缀缓存是 2026 年的重要优化——相同的系统提示、少样本示例或长上下文文档在多个调用间复用 KV。对于带有重复工具提示的 Agent 工作负载，前缀缓存通常可以带来 5 倍的吞吐量提升。

## 交付物

见 `outputs/skill-inference-optimizer.md`。该技能为新的推理部署选择注意力实现、KV 缓存策略、量化方案和投机解码。

## 练习

1. **（简单）** 运行 `code/main.py`。确认朴素解码器和缓存解码器产生相同输出；注意操作数的差异。
2. **（中等）** 实现前缀缓存：给定提示 P 和多个补全，对 P 做一次前向传播填充 KV 缓存，然后每个补全分支。测量与为每个补全重新编码 P 相比的加速比。
3. **（困难）** 实现玩具版 PagedAttention：KV 缓存按固定 16 个 token 的块存放，带空闲列表。序列完成时，将其块归还到池中。模拟 1000 次不同长度的对话补全。比较与连续分配的内存碎片率。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| KV 缓存 (KV cache) | "让解码变快的技巧" | 存储每个前缀 token 的 K 和 V；新查询对其进行关注而不重新计算 |
| HBM | "GPU 主内存" | 高带宽内存；H100 80 GB，B200 192 GB。带宽约 3 TB/s |
| SRAM | "片内内存" | H100 上每个 SM 约 256 KB 的快速内存。带宽约 30 TB/s |
| Flash Attention | "分块注意力核" | 不在 HBM 中实例化 N×N 的情况下计算注意力 |
| 连续批处理 (Continuous batching) | "无等待批处理" | 完成的序列换出，新序列换入，无需排空批次 |
| PagedAttention | "vLLM 的核心" | KV 缓存按固定块分配，带页表；消除碎片 |
| 前缀缓存 (Prefix caching) | "复用长提示" | 跨请求缓存共享前缀的 KV；Agent 的重要成本削减 |
| 投机解码 (Speculative decoding) | "草稿 + 验证" | 廉价草稿模型提出 token；大模型一次性并行验证 k 个 |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5 阶段流水线和软件 exp2 技巧
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 论文
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 投机解码
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1/2 集成草稿方法
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 方法
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 16 token 块和页表设计的权威深度解析

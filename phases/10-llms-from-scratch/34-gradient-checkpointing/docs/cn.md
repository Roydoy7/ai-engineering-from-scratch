# 梯度检查点与激活重计算

> 反向传播保存每一个中间激活值。在700亿参数和12.8万词元的上下文下，每个rank的激活量高达3 TB。检查点用FLOP换内存：重新计算而不是保存。问题在于丢弃哪些片段，答案不是"全部丢弃"。

**类型：** 构建  
**语言：** Python（含numpy，可选torch）  
**前置知识：** Phase 10·第04课（预训练Mini-GPT），Phase 10·第05课（扩展与分布式）  
**用时：** 约70分钟

## 问题所在

训练Transformer时，每一层都要为反向传播中被微分的每个操作保存输入：注意力输入、Q/K/V投影、softmax输出、FFN输入、归一化输出和残差流。对于隐藏维度`d`、序列长度`L`、批大小`B`的层，每层的激活量约为`12 * B * L * d`个浮点数。

取`d=8192, L=8192, B=1`，BF16下每层约800 MB。64层模型需要51 GB的激活内存——还没乘以微批大小，没加注意力softmax中间值（每头`L^2`），也没计入张量并行的部分副本。

两面的账单：BF16权重加优化器状态可能装进80 GB，但激活内存会把你推过去。梯度检查点（又称激活重计算）是标准修复方案：丢弃大部分激活，在反向传播时重新运行前向传播来取回它们。代价：额外FLOP。收益：内存减少的比例等于检查点片段数与总层数之比。

朴素地做，检查点每步约增加33%的前向传播FLOP。做好了——按照Korthikanti等人"智能选择"实现选择性检查点——你以不到5%的FLOP开销节省5倍内存。在FP8矩阵乘法、FSDP卸载和专家并行MoE的背景下，这真的很重要：你既负担不起内存，也承受不起浪费的计算。

## 核心概念

### 反向传播实际需要什么

`output = layer(input)`。反向传播需要`grad_input`和`grad_params`。要计算它们需要：

- `input`（用于计算`grad_params = input.T @ grad_output`，针对线性层）
- 一些激活导数中间值（ReLU/GELU/softmax的导数取决于激活值）

前向传播将这些自动存储在自动微分图中。每个`tensor.retain_grad()`和每个需要其输入的操作都会保留引用。

### 朴素完全检查点

将网络分成`N`个片段。前向传播期间，只保存每个片段的*输入*。当反向传播需要中间值时，重新运行该片段的前向传播来具体化它们，然后再微分。

示例：32层Transformer分成每层1个片段的32个片段。

- 内存：32个层输入（小）vs 32 × （每层激活量）（巨大）
- 额外计算：每个片段1次额外前向传播，即约多33%的前向FLOP（因为反向传播是前向的2倍，完整步骤变成1+1+2=4个单位而非1+2=3个单位）

这是原始的Chen等人2016年方案：每`sqrt(L)`层设一个检查点，以平衡内存和计算。对于L=64，就是8个检查点。

### 选择性检查点（Korthikanti，2022年）

并非所有激活的代价都相同。注意力softmax输出是`B*L*L*heads`，随序列长度**二次方增长**。FFN隐藏激活是`B*L*4d`，线性增长。对于长序列，softmax占主导。

选择性检查点保留廉价的激活（线性投影、残差），只重计算昂贵的激活（注意力）。你付出最小的FLOP来重计算，但节省了O(L^2)的内存。

Megatron-Core将其实现为"选择性"激活重计算，被2024年以后大多数前沿训练运行采用。

### 卸载

重计算的替代方案：在前向和反向传播之间将激活卸载到CPU RAM。需要PCIe带宽；当空闲带宽超过重计算代价时有益。混合策略很常见：某些层检查点，其他层卸载。

FSDP2将卸载作为一等选项提供。当GPU受内存限制但CPU-GPU传输有余量时，卸载效果最好。

### 重计算代价模型

L层中每k层做一次朴素检查点时，每步的FLOP：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # 片段中每层额外一次前向传播
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

选择性检查点只重计算注意力核心，而非整个层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活量：`A`。对于`L`层，总激活内存：`L * A`。

完全检查点（片段大小为1）：只保存`L * input_volume`（标准Transformer约为`L * 1/10 A`）。节省约`9 * L * A * 1/10`。

每k层检查点：保存`L/k * A`加上活跃片段内k-1层的激活。

在`k = sqrt(L)`时，内存和重计算代价都随`sqrt(L)`缩放——均匀代价层的最优权衡。

### 何时不使用检查点

- 流水线阶段中已在运行的最内层，它们无论如何都要完成
- 主导阶段计算的第一层和最后一层（在Transformer中罕见）
- 已使用FlashAttention的注意力核心——Flash已经快速重计算softmax，因此额外的层级检查点几乎不增加收益

### 实现模式

1. **函数包装器：** 将片段包装在`torch.utils.checkpoint.checkpoint(fn, input)`中。PyTorch只保存`input`，在反向传播时重计算其他所有内容

2. **基于装饰器：** 将层标记为可检查点；训练器在配置时决定哪些片段被包装

3. **手动显式重计算：** 自己编写反向传播，调用自定义的`recompute_forward`来复制带有存储输入的前向传播

三种方式都给出相同的功能结果。包装器是标准习惯用法。

### 与TP/PP/FP8的交互

- **张量并行：** 检查点输入在重计算时必须被聚集或重新分散；处理通信代价
- **流水线并行：** 典型模式是对每个流水线阶段的前向传播做检查点，使逆序微批次可以复用激活内存
- **FP8重计算：** 重计算期间更新的amax历史必须与原始前向传播的匹配，否则FP8缩放会漂移。大多数框架会快照缩放值

## 动手构建

### 第一步：带片段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 第二步：需要所有激活的朴素反向传播

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 第三步：每k层检查点的内存

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 第四步：代价模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 第五步：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 第六步：最优片段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 第七步：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 实际使用

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint`——PyTorch中的标准包装器。包装一个函数；只保存输入，在反向传播时重计算其他内容
- **Megatron-Core激活重计算**：支持`selective`、`full`和`block`模式，2024年以后的前沿训练的标准配置
- **FSDP2卸载**：FSDP2中带`offload_policy`的`module.to_empty(device="cpu")`将激活分片到CPU而非重计算
- **DeepSpeed ZeRO-Offload**：优化器状态和激活的CPU卸载，作为检查点的补充

## 交付成果

本课产出`outputs/prompt-activation-recompute-policy.md`——一个提示词，接收你的模型配置（层数、hidden大小、序列长度、批大小）和可用GPU内存，输出逐层重计算策略（无/选择性/完全/卸载）。

## 练习

1. 验证正确性。运行`model_forward` + `model_backward`（完整激活）与`model_forward_checkpointed` + `model_backward_checkpointed`（片段）。参数梯度必须精确到机器精度保持一致。

2. 将片段大小`k`从1扫描到`L`，绘制FLOP开销和内存。找出曲线的拐点。

3. 实现选择性检查点：保存注意力模块输入但不保存其中间值。对seq=8192的32层模型，测量相对于全层检查点的FLOP开销。

4. 添加卸载。将片段输入保存到模拟的"CPU缓冲区"（单独的列表），将"PCIe带宽"建模为字节/时间，找出卸载与重计算的盈亏平衡点。

5. 对真实的PyTorch Transformer分别测试带和不带`torch.utils.checkpoint`的情况。测量内存（通过`torch.cuda.max_memory_allocated`）和步骤时间。

## 关键术语

| 术语（英文） | 人们怎么说 | 实际含义 |
|-------------|-----------|---------|
| 梯度检查点（Gradient checkpointing） | "通过重做前向传播节省内存" | 只保存片段输入；在反向传播时重计算中间值以获取梯度支持张量 |
| 激活重计算（Activation recomputation） | "与检查点相同" | 相同技术的HPC领域术语 |
| 片段大小k（Segment size k） | "每个检查点多少层" | 其中间值被丢弃并一起重计算的层数 |
| 选择性检查点（Selective checkpointing） | "Korthikanti的技巧" | 只重计算昂贵存储的激活（注意力softmax）；保留廉价的 |
| 完全检查点（Full checkpointing） | "朴素版本" | 重计算每个片段中每层的中间值 |
| 块检查点（Block checkpointing） | "粗粒度" | 对整个Transformer块做检查点；最大粒度 |
| FLOP开销（FLOP overhead） | "计算税" | 每步额外FLOP = （重计算FLOP）/（前向 + 反向FLOP）；朴素33%，选择性5% |
| 激活卸载（Activation offload） | "发送到CPU" | 在前向→反向之间将激活移到CPU RAM；重计算的替代方案 |
| sqrt-L规则（sqrt-L rule） | "经典最优" | 对于均匀代价层，最优检查点间距是sqrt(L)层 |
| 注意力softmax体量（Attention-softmax volume） | "O(L^2)问题" | L^2 × heads × batch个浮点数；在长上下文时主导激活内存 |

## 延伸阅读

- [Chen等, 2016 — "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) — 形式化梯度检查点的原始论文
- [Korthikanti等, 2022 — "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) — 选择性激活重计算和形式化代价分析
- [Pudipeddi等, 2020 — "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) — 通过逆向模式重计算实现恒定内存的替代方法
- [Ren等, 2021 — "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) — 大规模激活卸载
- [PyTorch torch.utils.checkpoint文档](https://pytorch.org/docs/stable/checkpoint.html) — 标准API
- [Megatron-Core激活重计算文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) — 选择性、完全和块模式

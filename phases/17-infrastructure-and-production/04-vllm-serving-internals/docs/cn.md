# vLLM 服务内部原理：PagedAttention、连续批处理、分块预填充（vLLM Serving Internals: PagedAttention, Continuous Batching, Chunked Prefill）

> vLLM 在 2026 年的主导地位来自三个叠加的默认特性，而非单一技巧。PagedAttention 始终开启。连续批处理在解码迭代之间将新请求注入活跃批次。分块预填充将长提示词切片，让解码 token 不会饥饿。三者全部开启后，在一块 H100 SXM5 上运行 Llama 3.3 70B FP8，128 并发时可推到 2200-2400 tok/s——大约比 vLLM 自身的默认值高 25%，是朴素 PyTorch 循环的 3-4 倍。本课以可供你画图的层次阅读调度器和注意力内核，并以 `code/main.py` 中一个玩具连续批处理器结尾，其调度预填充和解码的方式与 vLLM 相同。

**类型：** 学习  
**语言：** Python（标准库，玩具连续批处理调度器）  
**前置知识：** Phase 17 · 01（模型服务）、Phase 11（LLM 工程）  
**预计时间：** 约 75 分钟

## 学习目标

- 将 PagedAttention 解释为 KV 缓存分配器：块、块表，以及为什么在生产负载下碎片率低于 4%。
- 在迭代级别画出连续批处理图：已完成的序列如何离开批次，新序列如何加入而无需排空。
- 用一句话描述分块预填充，并说出它保护的是哪个延迟指标（提示：是 TTFT 尾部，而非平均吞吐量）。
- 说出 2026 年 vLLM v0.18.0 让同时开启所有优化的团队踩坑的问题。

## 问题所在

朴素的 PyTorch 服务循环一次处理一个请求：分词、预填充、解码直到 EOS，返回。只有一个用户时这没问题。有一百个用户时，就是一队耐心等待的人。显而易见的修复——静态批处理——将每个请求填充到窗口中最长提示词的长度，将每次解码填充到预期最长输出的长度，整个批次因最慢的序列而阻塞。你为从未使用的填充付费，而快速请求等待慢请求。

vLLM 同时解决了三个问题。PagedAttention 阻止了 KV 缓存碎片，而经典的连续分配方式会吃掉 60-80% 的 GPU 内存。连续批处理让请求在每次解码迭代之间加入和离开批次，使批次始终充满真实工作。分块预填充将 32k token 的提示词分成约 512 token 的切片与解码交错，使长提示词不会冻结 GPU 上所有其他序列的解码 token。

2026 年的生产默认是三者全部开启。你需要理解每个特性的作用，因为故障模式都在调度器上，而非模型上。

## 核心概念

### PagedAttention 作为虚拟内存系统

每个序列的 KV 缓存大小为 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。对于 8192 个 token 的 Llama 3.3 70B，BF16 下每个序列约为 1.25 GB。如果你为每个请求预留 8192 个槽位，但平均请求只使用 1500 个 token，你浪费了约 82% 预留的 HBM。经典批处理要付这笔代价。

PagedAttention 借鉴了操作系统虚拟内存的思路。每个序列的 KV 缓存不再是连续的，而是以固定大小的块（默认 16 个 token）分配。每个序列有一个块表，将其逻辑 token 位置映射到物理块 ID。当序列超出已分配的块时，再添加一个块。完成时，其块返回池中。

碎片率从经典方式的 60-80% 降至 PagedAttention 的 4% 以下。你不需要用标志启用 PagedAttention——它是 vLLM 唯一内置的分配器。可调的参数是 `--gpu-memory-utilization`（默认 0.9），它告诉 vLLM 在加载权重和激活后为 KV 块保留多少 HBM。

### 迭代级别的连续批处理

旧的"动态批处理"等待一个时间窗（比如 10 毫秒）来填充批次，然后运行预填充 + 解码 + 解码 + 解码，直到每个序列完成。快速序列提前结束，而 GPU 完成慢序列时它们处于闲置状态。

连续批处理在每个解码步骤之间操作。将正在运行的序列集合称为 `RUNNING` 列表。每次迭代：

1. `RUNNING` 中刚刚到达 EOS 或 max_tokens 的任何序列被移除。
2. 调度器查看等待队列。如果有空闲 KV 块，它接受新序列（预填充或恢复）。
3. 对 `RUNNING` 中的所有序列运行前向传递，每个序列产生一个新 token。

批次大小永远不会填充到固定数量。处于不同输出位置的序列共享一次融合前向传递。在 2026 年的 vLLM 中，这称为 `V1 调度器`。关键不变量：调度器每次解码迭代运行一次，而非每个请求运行一次。

### 分块预填充保护 TTFT 尾部

预填充是计算密集型的。在一块 H100 上，Llama 3.3 70B 处理 32k token 的提示词需要约 800 毫秒的纯预填充时间。预填充运行期间，批次中所有其他序列的解码 token 都在等待。在服务循环中，一个长提示词的首 token 延迟（TTFT）变成了数十个其他用户的 token 间隔延迟（ITL）峰值。

分块预填充将预填充分割成固定大小的块（默认 512 个 token），并将每个块作为一个单元调度。在块与块之间，调度器可以将解码序列推进一个 token。你用少量的绝对预填充延迟增加（每块几毫秒）换来了更低的解码时间抖动。在已发布的基准中，混合负载下的 P99 ITL 从约 50 毫秒降至约 15 毫秒。

### 三个默认特性相互依存

三个特性都假定对方存在。PagedAttention 为调度器提供了可交换的细粒度 KV 资源。连续批处理需要这种细粒度资源，这样接受新序列就不会强制进行全局重排。分块预填充是调度器在同一 `RUNNING` 列表上做出的决策——它是另一个调度器策略，而非独立的系统。

你不需要了解每一个标志。你需要知道调度器优化的目标：在 KV 块预算约束下的吞吐量，以及分块预填充切片。

### 2026 年 v0.18.0 的坑

在 vLLM v0.18.0 中，你不能将 `--enable-chunked-prefill` 与草稿模型推测解码（`--speculative-model`）组合使用。文档中记录的例外是 V1 调度器中的 N-gram GPU 推测解码。不读发行说明就翻开所有开关的团队会在启动时遇到运行时错误，而非软性退化。如果你的推测收益值得开启分块预填充，重新考虑这个选择——2026 年通常正确的答案是不带分块预填充的 EAGLE-3，而非无法编译的草稿模型加分块预填充组合。

### 你应该记住的数字

- Llama 3.3 70B FP8，H100 SXM5，128 并发，三者全开：2200-2400 tok/s。
- 同一模型，默认 vLLM（无分块预填充）：约 1800 tok/s。
- 同一模型，朴素 PyTorch 前向循环：约 600 tok/s。
- 生产负载下 PagedAttention 的 KV 碎片浪费：< 4%。
- 混合负载下的 P99 ITL：分块预填充约 15 毫秒，无分块预填充约 50 毫秒。

### 调度器的样子

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # 在一个批次中调度预填充块 + 解码
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # 例如 512 个 token
        else:
            batch.append(decode_one_token(s))     # 1 个 token

    run_forward(batch)                            # 一次融合的 GPU 调用
```

`code/main.py` 正是这个循环，用标准库 Python 实现，使用虚假的 token 计数和虚假的前向延迟。运行它可以看到分块预填充如何在长时间预填充期间保持解码序列活跃。

## 使用它

`code/main.py` 模拟带有可切换特性的 vLLM 风格调度器。运行它可以看到：

- `NAIVE` 模式：一次一个请求，无批处理。
- `STATIC` 模式：填充并等待，经典批处理。
- `CONTINUOUS` 模式：迭代级别的接受和释放。
- `CONTINUOUS + CHUNKED` 模式：预填充切片与解码交错。

输出显示总吞吐量（每虚拟秒 token 数）、TTFT 均值和 P99 ITL。`CONTINUOUS + CHUNKED` 行在混合流量上应该占主导地位。

## 交付它

本课产出 `outputs/skill-vllm-scheduler-reader.md`。给定服务配置（批次大小、KV 内存利用率、分块预填充大小、推测配置），它产出一个调度器诊断，指出三个默认特性中哪个是瓶颈以及应调整什么。

## 练习

1. 运行 `code/main.py`。在混合短长请求的工作负载上比较 `STATIC` 与 `CONTINUOUS`。吞吐量差距来自哪里——预填充效率、解码效率还是尾部延迟？
2. 修改玩具调度器以添加 `--max-num-batched-tokens`。对于在 H100 上运行 Llama 3.3 70B FP8 的正确值是多少？（提示：它是 KV 块大小和空闲块数量的函数，而非原始 HBM。）
3. 重新阅读 vLLM v0.18.0 发行说明。哪些标志组合是互斥的？列出它们。
4. 对 1000 个请求的轨迹计算 KV 缓存碎片浪费，均值 1500 个输出 token，标准差 600 个 token，分别在（a）每请求连续分配最多 8192 个槽位和（b）16 token 块的 PagedAttention 下。
5. 用一段话解释分块预填充为什么单独来看帮助 P99 ITL 但不帮助吞吐量。实践中吞吐量的提升来自哪里？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| PagedAttention | "KV 技巧" | KV 缓存的固定大小块分配器；碎片 < 4% |
| Block table（块表） | "页表" | 每序列从逻辑 token 位置到物理 KV 块的映射 |
| Continuous batching（连续批处理） | "动态批处理，但正确" | 每次解码迭代都做接受/释放决策 |
| Chunked prefill（分块预填充） | "预填充切片" | 将长预填充分成 512 token 切片与解码交错 |
| TTFT | "首 token 时间" | 预填充 + 队列 + 网络；长提示词时主要由预填充决定 |
| ITL | "token 间隔延迟" | 连续解码 token 之间的时间；主要由批次大小决定 |
| Goodput（吞吐量） | "满足 SLO 的吞吐量" | 每秒所有请求仍满足 TTFT 和 ITL 目标的 token 数 |
| V1 scheduler（V1 调度器） | "新调度器" | vLLM 的 2026 调度器；N-gram 推测解码是分块预填充兼容路径 |
| `--gpu-memory-utilization` | "内存旋钮" | 权重和激活后为 KV 块保留的 HBM 比例 |

## 延伸阅读

- [vLLM 文档 — 推测解码](https://docs.vllm.ai/en/latest/features/spec_decode/) — 分块预填充和推测解码兼容性的官方资料
- [vLLM 发行说明（NVIDIA）](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 年发布节奏和版本特定行为
- [vLLM 博客 — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 定义分配器思考方式的原始文章
- [PagedAttention 论文（arXiv:2309.06180）](https://arxiv.org/abs/2309.06180) — 碎片分析和调度器设计
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) — 带火焰图的 V1 调度器详细解读

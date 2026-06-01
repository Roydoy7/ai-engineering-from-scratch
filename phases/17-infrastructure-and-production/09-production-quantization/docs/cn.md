# 生产量化——AWQ、GPTQ、GGUF K-quants、FP8、MXFP4/NVFP4（Production Quantization — AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4）

> 量化格式不是通用选择——它是硬件、服务引擎和工作负载的函数。GGUF Q4_K_M 或 Q5_K_M 主导 CPU 和边缘场景，通过 llama.cpp 和 Ollama 交付。当你需要在同一基础模型上运行多个 LoRA 时，GPTQ 在 vLLM 中胜出。AWQ 配合 Marlin-AWQ 内核在 7B 级模型上实现约 741 tok/s，在 INT4 格式中具有最佳 Pass@1——这是 2026 年数据中心生产的默认选择。FP8 在 Hopper、Ada 和 Blackwell 上仍然是中间地带——几乎无损且被广泛支持。NVFP4 和 MXFP4（Blackwell 微缩放）更激进，需要逐块验证。两个陷阱会让团队踩坑：校准数据集必须与部署领域匹配；KV 缓存与权重量化是分开的——AWQ 的"我的模型现在只有 4 GB"忘记了生产批次大小下 10-30 GB 的 KV 缓存。

**类型：** 学习  
**语言：** Python（标准库，跨格式的玩具内存和吞吐量比较器）  
**前置知识：** Phase 10 · 13（量化基础）、Phase 17 · 04（vLLM 服务内部原理）  
**预计时间：** 约 75 分钟

## 学习目标

- 说出 2026 年六种生产量化格式及其最适合的场景。
- 给定硬件（CPU vs GPU，Hopper vs Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）和工作负载（日常对话、推理、多 LoRA），选择一种格式。
- 计算所选格式节省的权重内存以及未触动的 KV 缓存。
- 说出导致量化模型在领域流量上退化的校准数据集陷阱。

## 问题所在

量化减少了内存和 HBM 带宽，而这正是解码所需要的。FP16 70B 模型的权重是 140 GB。将权重量化到 INT4（AWQ 或 GPTQ），模型变成 35 GB——可以放入一块 H100 并留有 KV 缓存的空间，这很重要，因为在 128 并发序列、2k 上下文的情况下，KV 缓存本身就是 20-30 GB。

但量化不是免费的。激进的量化会降低质量，特别是在推理密集型任务上。不同格式与不同引擎配合。不同硬件原生支持不同精度。2026 年的格式动物园是真实存在的，你无法复制别人的选择——你必须根据你的栈来决定。

## 核心概念

### 六种格式

| 格式 | 位数 | 最适合场景 | 支持的引擎 |
|------|------|-----------|-----------|
| GGUF Q4_K_M / Q5_K_M | 4-5 位 | CPU、边缘、笔记本 | llama.cpp、Ollama |
| GPTQ | 4-8 位 | vLLM 上的多 LoRA | vLLM、TGI |
| AWQ | 4 位 | 数据中心 GPU 生产 | vLLM（Marlin-AWQ）、TGI |
| FP8 | 8 位 | Hopper/Ada/Blackwell 数据中心 | vLLM、TRT-LLM、SGLang |
| MXFP4 | 4 位 | Blackwell 多用户 | TRT-LLM |
| NVFP4 | 4 位 | Blackwell 多用户 | TRT-LLM |

### GGUF——CPU/边缘默认

GGUF 是一种文件格式，而非量化方案本身——它将 K-quant 变体（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）捆绑在一个容器中。Q4_K_M 和 Q5_K_M 是生产默认值——4-5 位下接近 BF16 质量。CPU 或边缘服务的最佳选择，因为 llama.cpp 是目前最快的 CPU 推理引擎。

vLLM 中的吞吐量惩罚：7B 模型约 93 tok/s——该格式未针对 GPU 内核优化。当部署目标是 CPU/边缘时使用 GGUF，否则不用。

### GPTQ——vLLM 中的多 LoRA

GPTQ 是带有校准过程的训练后量化算法。Marlin 内核使其在 GPU 上快速运行（比非 Marlin GPTQ 加速 2.6 倍）。7B 模型约 712 tok/s。

独特优势：GPTQ-Int4 在 vLLM 中支持 LoRA 适配器。如果你在服务一个基础模型加 10-50 个微调变体（每个作为 LoRA），GPTQ 是你的路径。截至 2026 年初，NVFP4 还不支持 LoRA。

### AWQ——数据中心 GPU 默认

激活感知权重量化（Activation-aware Weight Quantization）。在量化期间保护约 1% 最显著的权重。Marlin-AWQ 内核：相比朴素方式 10.9 倍加速。7B 模型约 741 tok/s，INT4 格式中最佳 Pass@1。

选择 AWQ 用于新的 GPU 服务，除非你需要多 LoRA（GPTQ）或激进的 Blackwell FP4（NVFP4）。

### FP8——可靠的中间地带

8 位浮点。几乎无损。被广泛支持。Hopper Tensor Core 原生加速 FP8。Blackwell 继承。当质量不可妥协时（推理、医疗、代码生成），FP8 是 2026 年的安全默认选择。内存节省是 INT4 的一半，但质量风险远低于 INT4。

### MXFP4 / NVFP4——Blackwell 激进方案

微缩放 FP4。每个权重块都有自己的缩放因子。激进但在 Blackwell Tensor Core 上硬件加速。相比 FP8 每 token 字节减半——Phase 17 · 07 中的经济优势。

注意事项：
- 截至 2026 年初尚不支持 LoRA。
- 在推理密集型工作负载上质量下降明显。
- 按模型逐一在你的评估集上验证。

### 校准陷阱

AWQ 和 GPTQ 需要校准数据集——通常是 C4 或 WikiText。对于领域模型（代码、医疗、法律），用通用网页文本校准会让算法对哪些权重需要保护做出错误判断。HumanEval 上的 Pass@1 可能下降几个百分点。

修复方法：用领域内数据校准。通常数百个领域样本就足够了。在发布之前在评估集上测试。

### KV 缓存陷阱

AWQ 将权重压缩到 4 位。KV 缓存是独立的，保持在 FP16/FP8。对于使用 AWQ 的 70B 模型：

- 权重：约 35 GB（从 140 GB 压缩到 INT4）。
- 128 并发 × 2k 上下文的 KV 缓存：约 20 GB。
- 激活：约 5 GB。
- 总计：约 60 GB——适合 H100 80GB。

天真地认为"我把模型量化到 4 GB"忘了另外 30-50 GB。从整体上规划 HBM 预算。

另外，KV 缓存量化（FP8 KV 或 INT8 KV）是一个独立的选择，有其自身的权衡——它直接影响注意力精度，不是免费的收益。

### AWQ INT4 对推理任务有风险

思维链、数学、长上下文代码生成——这些在激进量化下会有可见的退化。AWQ INT4 在 MATH 基准上损失约 3-5 个百分点。对于推理密集型工作负载，使用 FP8 或 BF16；接受内存成本。

### 2026 年选择指南

- CPU/边缘服务：GGUF Q4_K_M。完事。
- GPU 服务，日常对话，无 LoRA：AWQ。
- GPU 服务，多 LoRA：带 Marlin 的 GPTQ。
- 推理工作负载：FP8。
- Blackwell 数据中心，已验证质量：NVFP4 + FP8 KV。
- 不确定：在每个候选格式上运行 1000 样本评估。

## 使用它

`code/main.py` 计算跨六种格式、不同模型大小的内存占用（权重 + KV + 激活）和相对吞吐量。展示 KV 缓存在哪里占主导、权重压缩在哪里有收益，以及 FP8 是安全选择的场景。

## 交付它

本课产出 `outputs/skill-quantization-picker.md`。给定硬件、模型大小、工作负载类型和质量容忍度，选择一种格式并产出校准/验证计划。

## 练习

1. 运行 `code/main.py`。对于 128 并发、2k 上下文的 70B 模型，计算每种格式的总 HBM。哪种格式能放入一块 H100 80GB？
2. 你有一个 7B 编程模型。选择一种格式并论证。如果你对质量容忍度判断有误，恢复路径是什么？
3. 计算为医疗领域模型校准 AWQ 所需的校准数据集大小。为什么更多数据不总是更好？
4. 阅读 Marlin-AWQ 内核论文或发行说明。用三句话解释为什么 AWQ 在 7B 模型上能达到 741 tok/s，而原始 GPTQ 约为 712 tok/s。
5. 何时将 AWQ 权重与 FP8 KV 缓存组合使用有意义，而非将 KV 保持在 BF16？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| GGUF | "llama.cpp 格式" | 捆绑 K-quant 变体的文件格式；CPU/边缘默认 |
| Q4_K_M | "Q4 K M" | 4 位 K-quant 中档；GGUF 生产默认 |
| GPTQ | "GPTQ" | 带校准的训练后 INT4；在 vLLM 中支持 LoRA |
| AWQ | "AWQ" | 激活感知 INT4；Marlin 内核；INT4 中最佳 Pass@1 |
| Marlin kernels（Marlin 内核） | "快速 INT4 内核" | Hopper 上 INT4 的自定义 CUDA 内核；10 倍加速 |
| FP8 | "8 位浮点" | Hopper/Ada/Blackwell 上安全的精度默认值 |
| MXFP4 / NVFP4 | "微缩放四位" | 带每块缩放因子的 Blackwell 4 位 FP |
| Calibration dataset（校准数据集） | "校准数据" | 用于选择量化参数的输入文本；必须与领域匹配 |
| KV cache quantization（KV 缓存量化） | "KV INT8" | 与权重量化分开的独立选择；影响注意力精度 |

## 延伸阅读

- [VRLA Tech — 2026 年 LLM 量化](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 各格式比较基准
- [Jarvis Labs — vLLM 量化完整指南](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 按格式的吞吐量数字
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 按格式的选择指南
- [vLLM 文档 — 量化](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 支持的格式和标志
- [AWQ 论文（arXiv:2306.00978）](https://arxiv.org/abs/2306.00978) — 原始 AWQ 表述
- [GPTQ 论文（arXiv:2210.17323）](https://arxiv.org/abs/2210.17323) — 原始 GPTQ 表述

# 自托管服务选型——llama.cpp、Ollama、TGI、vLLM、SGLang（Self-Hosted Serving Selection — llama.cpp, Ollama, TGI, vLLM, SGLang）

> 2026 年，四个引擎主导自托管推理。根据硬件、规模和生态系统进行选择。**llama.cpp** 在 CPU 上最快——最广泛的模型支持，完全控制量化和线程。**Ollama** 是开发笔记本的一键安装方案，比 llama.cpp 慢约 15-30%（Go + CGo + HTTP 序列化），在生产级负载下有 3 倍吞吐量差距。**TGI 于 2025 年 12 月 11 日进入维护模式**——只进行 bug 修复，原始吞吐量历史上比 vLLM 慢约 10%，但历史上可观测性顶级且与 HF 生态系统集成最佳。维护状态使其成为长期风险赌注——对于新项目，SGLang 或 vLLM 是更安全的默认选择。**vLLM** 是通用生产默认选择——v0.15.1（2026 年 2 月）添加了 PyTorch 2.10、RTX Blackwell SM120、H200 优化。**SGLang** 是智能体多轮/前缀密集型专家——生产中部署于 40 万以上 GPU（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件约束：仅 CPU → 只能用 llama.cpp。AMD/非 NVIDIA → 只能用 vLLM（TRT-LLM 锁定 NVIDIA）。2026 年管道模式：开发 = Ollama，暂存 = llama.cpp，生产 = vLLM 或 SGLang。全程使用相同的 GGUF/HF 权重。

**类型：** 学习  
**语言：** Python（标准库，引擎决策树遍历器）  
**前置知识：** Phase 17 中所有涵盖引擎的课程（04、06、07、09、18）  
**预计时间：** 约 45 分钟

## 学习目标

- 给定硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1 用户 / 100 / 10000）和工作负载（通用对话 / 智能体 / 长上下文），选择一个引擎。
- 说出 2026 年 TGI 维护模式状态（2025 年 12 月 11 日），以及为什么它使新项目倾向于 vLLM 或 SGLang。
- 描述全程使用相同 GGUF 或 HF 权重的开发/暂存/生产管道。
- 解释为什么"仅 CPU"强制使用 llama.cpp，以及"AMD"为何排除 TRT-LLM。

## 问题所在

你的团队开始一个新的自托管 LLM 项目。一个工程师说 Ollama，另一个说 vLLM，第三个说"TGI 不是开箱即用吗？"三者对于不同场景都是正确的。没有一个对所有场景都正确。

2026 年，选择树很重要：硬件优先，规模其次，工作负载第三。还有一个特定的 2025 年事件——TGI 在 12 月 11 日进入维护模式——改变了新项目的默认选择。

## 核心概念

### 五个引擎

| 引擎 | 最适合 | 备注 |
|------|--------|------|
| **llama.cpp** | CPU / 边缘 / 最少依赖 / 最广泛模型支持 | CPU 上最快，完全控制 |
| **Ollama** | 开发笔记本，单用户，一键安装 | 比 llama.cpp 慢 15-30%；生产吞吐量差距 3 倍 |
| **TGI** | HF 生态，受监管行业 | **2025 年 12 月 11 日进入维护模式** |
| **vLLM** | 通用生产，100+ 用户 | 广泛生产默认；v0.15.1 2026 年 2 月 |
| **SGLang** | 智能体多轮，前缀密集型工作负载 | 生产中 40 万以上 GPU |

### 硬件优先决策

**仅 CPU** → llama.cpp。Ollama 也可以但更慢。在 CPU 上没有其他引擎有竞争力。

**AMD GPU** → vLLM（AMD ROCm 支持）。SGLang 也可以。TRT-LLM 锁定 NVIDIA，所以排除。

**NVIDIA Hopper（H100 / H200）** → vLLM 或 SGLang 或 TRT-LLM。三者都是顶级。

**NVIDIA Blackwell（B200 / GB200）** → TRT-LLM 是吞吐量领先者（Phase 17 · 07）。vLLM 和 SGLang 紧随其后。

**Apple Silicon（M 系列）** → llama.cpp（Metal）。Ollama 对其进行了封装。

### 规模次要决策

**1 用户 / 本地开发** → Ollama。一个命令，首 token 在几秒内。

**10-100 用户 / 小团队** → vLLM 单 GPU。

**100-10000 用户 / 生产** → vLLM production-stack（Phase 17 · 18）或 SGLang。

**10000 以上用户 / 企业** → vLLM production-stack + 分离式（Phase 17 · 17）+ LMCache（Phase 17 · 18）。

### 工作负载第三决策

**通用对话 / 问答** → vLLM 在广泛默认场景下胜出。

**智能体多轮（工具、规划、记忆）** → SGLang 的 RadixAttention（Phase 17 · 06）占主导。

**前缀高度复用的 RAG** → SGLang。

**代码生成** → vLLM 可以；SGLang 在缓存上略好。

**长上下文（128K+）** → vLLM + 分块预填充；SGLang + 分层 KV。

### TGI 维护陷阱

Hugging Face TGI 于 2025 年 12 月 11 日进入维护模式——此后只进行 bug 修复。历史上：顶级可观测性，最佳 HF 生态系统集成（模型卡、安全工具），原始吞吐量略低于 vLLM。

对于 2026 年的新项目：默认远离 TGI。现有 TGI 部署可以继续，但最终应迁移。SGLang 和 vLLM 是更安全的默认选择。

### 管道模式

开发（Ollama）→ 暂存（llama.cpp）→ 生产（vLLM）。全程使用相同的 GGUF 或 HF 权重。工程师在笔记本上快速迭代；暂存镜像生产量化；生产是服务目标。

### Ollama 注意事项

Ollama 非常适合开发。它不适合共享生产：Go HTTP 序列化增加了开销，并发管理比 vLLM 简单，OpenTelemetry 支持滞后。在它擅长的地方使用 Ollama——单用户，一个命令——对于共享场景切换到 vLLM。

### 自托管 vs 托管是单独的决策

Phase 17 · 01（托管超大规模）、· 02（推理平台）涵盖了托管方案。本课假设你已经决定自托管。自托管的理由：数据驻留、自定义微调、大规模总拥有成本、托管平台上没有的领域模型。

### 你应该记住的数字

- TGI 维护模式：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；Blackwell SM120 支持。
- SGLang 生产部署规模：40 万以上 GPU。
- Ollama vs llama.cpp 吞吐量差距：慢 15-30%；生产负载下慢 3 倍。

## 使用它

`code/main.py` 是一个决策树遍历器：给定硬件 + 规模 + 工作负载，选择一个引擎并解释原因。

## 交付它

本课产出 `outputs/skill-engine-picker.md`。给定约束，选择引擎并写出迁移计划。

## 练习

1. 用你的硬件/规模/工作负载运行 `code/main.py`。输出是否与你的直觉一致？
2. 你的基础设施有 12 块 H100 和 8 块 MI300X AMD。选什么引擎？为什么 TRT-LLM 被排除？
3. 一个团队想在 2026 年使用 TGI，因为"这是我们熟悉的"。论证迁移理由。
4. 从 Ollama 开发到 vLLM 生产：量化、配置和可观测性上有什么变化？
5. P99 前缀长度 8K 且租户间高度复用的 RAG 产品。选择一个引擎并与 Phase 17 · 11 + 18 叠加。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| llama.cpp | "CPU 那个" | 最广泛模型支持，CPU 上最快 |
| Ollama | "笔记本那个" | 一键安装，开发级吞吐量 |
| TGI | "HF 的服务" | 自 2025 年 12 月起进入维护模式 |
| vLLM | "默认的那个" | 2026 年广泛生产基线 |
| SGLang | "智能体那个" | 前缀密集型，RadixAttention |
| TRT-LLM | "NVIDIA 锁定" | Blackwell 吞吐量领先者，仅 NVIDIA |
| GGUF | "llama.cpp 格式" | 捆绑 K-quant 变体 |
| Production-stack | "vLLM K8s" | Phase 17 · 18 参考部署 |
| Pipeline pattern（管道模式） | "开发→暂存→生产" | Ollama → llama.cpp → vLLM，使用相同权重 |

## 延伸阅读

- [AI Made Tools — 2026 年 vLLM vs Ollama vs llama.cpp vs TGI](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — 2026 年 llama.cpp vs Ollama](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — LLM 推理引擎综合比较](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 2026 年 10 个最佳 vLLM 替代方案](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI 维护公告](https://github.com/huggingface/text-generation-inference) — 发行说明
- [vLLM v0.15.1 发行说明](https://github.com/vllm-project/vllm/releases)

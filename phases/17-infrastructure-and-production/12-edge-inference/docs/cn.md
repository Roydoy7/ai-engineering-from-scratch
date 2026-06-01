# 边缘推理——Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson（Edge Inference — Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson）

> 边缘推理的核心约束是内存带宽，而非计算能力。移动 DRAM 速度在 50-90 GB/s；数据中心 HBM3 超过 2-3 TB/s——30-50 倍的差距。解码是内存带宽受限的，因此这个差距是决定性的。2026 年，这个领域分成四个方向。Apple M4/A18 神经引擎峰值 38 TOPS，使用统一内存（无需 CPU↔NPU 拷贝）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon 达到 45 TOPS。WebGPU + WebLLM 在 M3 Max 上以约 41 tok/s 运行 Llama 3.1 8B（Q4），约为原生性能的 70-80%；GitHub 17.6k star，兼容 OpenAI API，移动端覆盖约 70-75%。NVIDIA Jetson Orin Nano Super（8GB）可以运行 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 以约 40 tok/s 运行 gpt-oss-20b；Jetson T4000（JetPack 7.1）是 AGX Orin 的 2 倍。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、分块预填充——在 2026 年 CES 上由博世、雷石、联发科展示。

**类型：** 学习  
**语言：** Python（标准库，玩具带宽受限解码模拟器）  
**前置知识：** Phase 17 · 04（vLLM 服务内部原理）、Phase 17 · 09（生产量化）  
**预计时间：** 约 60 分钟

## 学习目标

- 解释为什么移动 LLM 推理是内存带宽受限的，计算能力是次要的。
- 列举四个边缘目标（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson），并将每个与使用场景匹配。
- 说出 2026 年 WebGPU 覆盖缺口（Firefox Android 正在追赶）以及 Safari iOS 26 落地情况。
- 为每个目标选择量化格式（ANE 的 Core ML INT4 + FP16、Hexagon 的 QNN INT8/INT4、浏览器的 WebGPU Q4、Jetson Thor 的 NVFP4）。

## 问题所在

一个客户想要设备端聊天机器人：语音优先、默认私密、离线可用。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 以约 55 tok/s 运行——没问题。在 iPhone 16 Pro 上，相同模型以 3 tok/s 运行——不行。在搭载 Snapdragon 8 Gen 3 的中端 Android 上，7 tok/s。在 Chrome Android v121+ 通过 WebGPU 运行时，根据设备不同为 4-8 tok/s。

吞吐量差异不是移植问题。它是带宽差距乘以量化格式乘以 NPU 是否可从用户空间访问的结果。2026 年的边缘推理是四个不同的问题，有四种不同的解决方案。

## 核心概念

### 带宽才是真正的上限

解码对每个 token 都要读取完整的权重集合。Q4 格式的 7B 模型是 3.5 GB。以 50 GB/s 读取 3.5 GB 需要 70 毫秒——理论上限约为 14 tok/s。在 90 GB/s（高端移动 DRAM）时，上限移到约 25 tok/s。在这个数字以下，再多计算能力也没用。

数据中心 HBM3 在 3 TB/s 下用 1.2 毫秒清除相同的 3.5 GB——上限是 830 tok/s。同一模型，同样的权重。不同的内存子系统。

### Apple Neural Engine（M4 / A18）

- 最高 38 TOPS。统一内存（CPU 和 ANE 共享同一内存池）——无拷贝开销。
- 通过 Core ML + `.mlmodel` 编译模型访问，或通过 PyTorch 的 Metal Performance Shaders（MPS）访问。
- Llama.cpp Metal 后端使用 MPS，而非直接使用 ANE；原生 ANE 需要 Core ML 转换。
- 2026 年 iOS 应用的最佳实践路径：Core ML + INT4 权重 + FP16 激活。

### Qualcomm Hexagon（Snapdragon X Elite / 8 Gen 4）

- 最高 45 TOPS。与 SoC 中的 CPU 和 GPU 集成，但内存域独立。
- QNN（Qualcomm 神经网络）SDK 和 AI Hub 提供从 PyTorch/ONNX 的转换。
- 聊天模板、Llama 3.2、Phi-3 都在 AI Hub 上作为一等工件提供。

### Intel / AMD NPU（Lunar Lake、Ryzen AI 300）

- 40-50 TOPS。软件落后于 Apple/Qualcomm；OpenVINO 在改进但仍属小众。
- 最适合 Windows ARM Copilot 应用；在 AMD/Intel 台式机上原生用于本地优先。

### WebGPU + WebLLM

- 通过 WebGPU 计算着色器在浏览器中运行模型；无需安装。
- M3 Max 上 Llama 3.1 8B Q4 约 41 tok/s——通过同一后端约为原生的 70-80%。
- WebLLM 有 17.6k GitHub star；兼容 OpenAI 的 JS API；Apache 2.0 许可。
- 2026 年覆盖：Chrome Android v121+，Safari iOS 26 GA，Firefox Android 仍在追赶。整体移动端覆盖约 70-75%。

### NVIDIA Jetson 系列

- Orin Nano Super（8GB）：以良好的 tok/s 运行 Llama 3.2 3B、Phi-3。
- AGX Orin：通过 vLLM 以约 40 tok/s 运行 gpt-oss-20b。
- Thor / T4000（JetPack 7.1）：AGX Orin 性能的 2 倍，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM（2026）支持 EAGLE-3 推测解码、NVFP4 权重、分块预填充——数据中心优化移植到边缘。

### 各目标的量化格式选择

| 目标 | 格式 | 备注 |
|------|------|------|
| Apple ANE | INT4 权重 + FP16 激活 | Core ML 转换路径 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC（q4f16_1） | 使用 `mlc_llm convert_weight` + 编译的 `.wasm`；不支持 GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 内存带宽受限 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路径 |

### 边缘上的长上下文陷阱

Llama 3.1 的 128K 上下文是数据中心特性。在有 8 GB RAM 的手机上，4 GB 模型 + 32K token 的 2 GB KV 缓存 + 操作系统开销 = OOM。除非接受激进的 KV 量化（Q4 KV），否则边缘部署将上下文保持在 4K-8K。

### 语音是杀手应用

语音智能体对延迟敏感（首 token < 500 毫秒）。本地推理完全消除了网络延迟。结合语音转文字（Whisper Turbo 变体可在边缘运行），边缘推理成为生产质量的语音循环。

### 你应该记住的数字

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4 约 41 tok/s。
- AGX Orin：通过 vLLM 在 gpt-oss-20b 上约 40 tok/s。
- 数据中心-边缘带宽差距：30-50 倍。
- WebGPU 移动端覆盖：约 70-75%（Firefox Android 落后）。

## 使用它

`code/main.py` 从带宽受限数学计算各边缘目标的理论解码吞吐量上限。与观测到的基准进行比较，并突出带宽（而非计算）成为瓶颈的场景。

## 交付它

本课产出 `outputs/skill-edge-target-picker.md`。给定平台（iOS/Android/浏览器/Jetson）、模型和延迟/内存预算，选择量化格式和转换管道。

## 练习

1. 运行 `code/main.py`。对于 Snapdragon 8 Gen 3（约 77 GB/s 带宽）上的 Q4 格式 7B 模型，计算解码上限。与观测到的 6-8 tok/s 比较——运行时效率如何？
2. Android 上的 WebGPU 需要 Chrome v121+。为旧版浏览器设计备用方案——通过同一兼容 OpenAI API 的服务端。
3. 你的 iOS 应用需要 4K 上下文流式输出。哪种模型/格式组合让你在 iPhone 16 上的活跃内存低于 4 GB？
4. Jetson AGX Orin 以 40 tok/s 运行 gpt-oss-20b，Jetson Nano 只能容纳 3B。如果你的产品同时面向两者，如何统一推理栈？
5. 就"WebLLM 在 2026 年是否生产就绪"进行论证。引用覆盖率、性能和 Firefox Android 缺口。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| ANE | "Apple 神经引擎" | M 系列和 A 系列中的设备端 NPU；统一内存 |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU；通过 QNN SDK 访问 |
| WebGPU | "浏览器 GPU" | W3C 标准化的浏览器 GPU API；Chrome/Safari 2026 |
| WebLLM | "浏览器 LLM 运行时" | MLC-LLM 项目；Apache 2.0；兼容 OpenAI 的 JS 接口 |
| Jetson | "NVIDIA 边缘" | Orin Nano / AGX / Thor / T4000 系列 |
| TRT Edge-LLM | "边缘 TensorRT" | 2026 年 TensorRT-LLM 的边缘移植；EAGLE-3 + NVFP4 |
| Unified memory（统一内存） | "共享内存池" | CPU 和 NPU 共享同一 RAM；无拷贝开销 |
| Bandwidth-bound（带宽受限） | "内存受限" | 解码受读取权重的字节/秒限制 |
| Core ML | "Apple 转换" | 用于 ANE 原生模型的 Apple 框架 |
| QNN | "Qualcomm 栈" | Qualcomm 神经网络 SDK |

## 延伸阅读

- [设备端 LLM 现状 2026](https://v-chandra.github.io/on-device-llms/) — 格局和基准
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 年边缘移植公告
- [WebLLM（arXiv:2412.15803）](https://arxiv.org/html/2412.15803v2) — 设计和基准
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE 原生转换
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — 为 Hexagon 预转换的模型

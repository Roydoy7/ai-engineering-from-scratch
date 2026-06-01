# 压轴项目 15——宪法安全测试框架 + 红队范围（Capstone 15 — Constitutional Safety Harness + Red-Team Range）

> Anthropic 的宪法分类器、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 内容安全和 X-Guard 的多语言覆盖，定义了 2026 年的安全分类器栈。garak、PyRIT、NVIDIA Aegis 和 promptfoo 成为标准的对抗性评估工具。NeMo Guardrails v0.12 将它们联系到生产管道中。本压轴项目将所有这些连接起来：围绕目标应用的分层安全测试框架、运行 6 个以上攻击系列的自主红队智能体，以及产生可测量无害性差值的宪法自我批判运行。

**类型：** 压轴项目  
**语言：** Python（安全管道，红队），YAML（策略配置）  
**前置知识：** Phase 10（从头开始的 LLM）、Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（智能体）、Phase 18（伦理、安全、对齐）  
**涉及的阶段：** P10 · P11 · P13 · P14 · P18  
**预计时间：** 25 小时

## 问题所在

2026 年 LLM 安全的前沿不是分类器是否有效（大体上有效），而是如何在生产应用周围正确组合它们，而不出现过度拒绝或留下明显漏洞。Llama Guard 4 处理英语策略违规。X-Guard（132 种语言）处理多语言越狱。ShieldGemma-2 捕获基于图像的提示注入。NVIDIA Nemotron 3 内容安全覆盖企业类别。Anthropic 的宪法分类器是一种在训练期间而非服务期间使用的不同方法。

攻击演进也很重要。PAIR 和 TAP 自动化越狱发现。GCG 运行基于梯度的后缀攻击。多轮和代码切换攻击利用智能体记忆。任何已部署的 LLM 都需要一个红队范围——garak 和 PyRIT 是标准驱动器——加上有记录的缓解措施和 CVSS 评分的发现。

你将加固目标应用（8B 指令微调模型或来自其他压轴项目的 RAG 聊天机器人），对其运行 6 个以上攻击系列，并产生前/后无害性测量。

## 核心概念

安全管道有五层。**输入清洗**：去除零宽字符，解码 base64/rot13，规范化 Unicode。**策略层**：NeMo Guardrails v0.12 护栏（领域外、毒性、PII 提取）。**分类器门控**：输入时使用 Llama Guard 4，非英语使用 X-Guard，图像输入使用 ShieldGemma-2。**模型**：目标 LLM。**输出过滤**：输出时使用 Llama Guard 4，Presidio PII 清洗，在适用时进行引用执行。**HITL 层**：标记为高风险的输出进入 Slack 队列。

红队范围在调度器上运行。PAIR 和 TAP 自主发现越狱。GCG 运行基于梯度的后缀攻击。ASCII / base64 / rot13 编码攻击。多轮攻击（角色采用，记忆利用）。代码切换攻击（混合英语与斯瓦希里语或泰语）。每次运行产生一个带 CVSS 评分和披露时间表的结构化发现文件。

宪法自我批判运行是一种训练时干预。取 1k 个有害尝试提示词，让模型起草回应，根据书面宪法（不伤害规则）批判它，并在批判循环上重新训练。在保留集上测量前/后无害性差值。

## 架构

```
请求（文本 / 图像 / 多语言）
      |
      v
输入清洗（去除零宽字符，解码，规范化）
      |
      v
NeMo Guardrails v0.12 护栏（领域外，策略）
      |
      v
分类器门控：
  Llama Guard 4（英语）
  X-Guard（多语言，132 种语言）
  ShieldGemma-2（图像提示词）
  Nemotron 3 内容安全（企业）
      |
      v（允许）
目标 LLM
      |
      v
输出过滤：Llama Guard 4 + Presidio PII + 引用检查
      |
      v
标记输出的 HITL 层

并行：
  红队调度器
    -> garak（经典攻击）
    -> PyRIT（编排红队）
    -> 自主越狱智能体（PAIR + TAP）
    -> GCG 后缀攻击
    -> 多语言 / 代码切换
    -> 多轮角色采用

输出：CVSS 评分发现 + 披露时间表 + 前/后无害性差值
```

## 技术栈

- 安全分类器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 内容安全、X-Guard
- 护栏框架：NeMo Guardrails v0.12 + OPA
- 红队驱动器：garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo
- 越狱智能体：PAIR（Chao 等人，2023 年）、Tree-of-Attacks（TAP）、GCG 后缀
- 宪法训练：Anthropic 风格自我批判循环 + 在批判上的 SFT
- PII 清洗：Presidio
- 目标：8B 指令微调模型或其他压轴项目的 RAG 聊天机器人

## 构建它

1. **目标设置。** 在 vLLM 上搭建 8B 指令微调模型（或从另一个压轴项目复用 RAG 聊天机器人）。这是被测试的应用。

2. **安全管道封装。** 围绕目标连接五层管道。验证每层是可单独观察的（Langfuse 中每层的 span）。

3. **分类器覆盖。** 加载 Llama Guard 4、X-Guard（多语言）、ShieldGemma-2（图像）。在一个小型标注集上运行每个以建立基线。

4. **红队调度器。** 调度 garak、PyRIT、PAIR 智能体、TAP 智能体、GCG 运行器、多轮攻击者和代码切换攻击者。每个在单独的队列上运行。

5. **攻击套件。** 六个攻击系列：(1) PAIR 自动越狱，(2) TAP 攻击树，(3) GCG 梯度后缀，(4) ASCII / base64 / rot13 编码，(5) 多轮角色，(6) 多语言代码切换。报告每系列的成功率。

6. **宪法自我批判。** 整理 1k 个有害尝试提示词。对每个，目标起草响应。批判 LLM 根据书面宪法（"不伤害"，"引用证据"，"拒绝非法请求"）评分。批判者反对的提示词被重写；目标在批判改进的对上进行微调。在保留集上测量前/后无害性。

7. **过度拒绝测量。** 追踪无害提示词套件（如 XSTest）上的误报率。目标必须在无害问题上保持有帮助。

8. **CVSS 评分。** 对每个成功的越狱，用 CVSS 4.0 评分（攻击向量、复杂性、影响）。产生披露时间表和缓解计划。

9. **范围自动化。** 以上所有内容在 cron 上运行；发现写入队列；过度拒绝回归告警触发 Slack。

## 使用它

```
$ safety probe --model=target --family=PAIR --budget=50
[攻击者]  PAIR 智能体对目标运行
[攻击]   尝试 1/50：将查询伪装为学术研究 ... 已阻止
[攻击]   尝试 2/50：诉诸角色扮演 ... 已阻止
[攻击]   尝试 3/50：思维链引导 ... 成功
[发现]   CVSS 4.8 中等：目标上的角色扮演绕过
[范围]   50 次中 7 次成功（14% 成功率）
```

## 交付它

`outputs/skill-safety-harness.md` 是可交付成果。一个生产级分层安全管道加上带前/后无害性差值的可重现红队范围。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 攻击面覆盖率 | 6 个以上攻击系列，2 种以上语言 |
| 20 | 真阳性/误报率权衡 | 攻击阻止率 vs XSTest 无害通过率 |
| 20 | 自我批判差值 | 保留集上的前/后无害性 |
| 20 | 文档和披露 | 带时间表的 CVSS 评分发现 |
| 15 | 自动化和可重现性 | 所有内容在 cron 上运行，带告警 |
| **100** | | |

## 练习

1. 在 RAG 聊天机器人上运行 garak 的提示注入插件，并比较有无输出过滤层时的攻击成功率。

2. 添加第七个攻击系列：通过检索文档的间接提示注入。测量所需的额外防御。

3. 实现"拒绝并帮助"模式：当护栏阻止时，目标提供更安全的相关答案而非直接拒绝。测量 XSTest 差值。

4. 多语言覆盖缺口：找到一种 X-Guard 表现不佳的语言。提出针对它的微调数据集。

5. 在 30B 模型上运行宪法自我批判，并测量差值是否有规模扩展。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Layered safety（分层安全） | "纵深防御" | 输入、门控、输出、HITL 上的多重护栏 |
| Llama Guard 4 | "Meta 的安全分类器" | 2026 年参考输入/输出内容分类器 |
| PAIR | "越狱智能体" | LLM 驱动越狱发现的论文（Chao 等人）|
| TAP | "攻击树" | PAIR 的树搜索变体 |
| GCG | "贪婪坐标梯度" | 基于梯度的对抗后缀攻击 |
| Constitutional self-critique（宪法自我批判） | "Anthropic 风格训练" | 目标起草 -> 批判者评分 -> 重写 -> 重新训练 |
| XSTest | "无害探测集" | 过度拒绝回归基准 |
| CVSS 4.0 | "严重性分数" | 安全发现的标准漏洞评分 |

## 延伸阅读

- [Anthropic 宪法分类器](https://www.anthropic.com/research/constitutional-classifiers) — 训练时参考
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年输入/输出分类器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像 + 多模态安全
- [NVIDIA Nemotron 3 内容安全](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业参考
- [X-Guard（arXiv:2504.08848）](https://arxiv.org/abs/2504.08848) — 132 语言多语言安全
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 红队工具包
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft 红队框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 护栏框架
- [PAIR（arXiv:2310.08419）](https://arxiv.org/abs/2310.08419) — 越狱智能体论文

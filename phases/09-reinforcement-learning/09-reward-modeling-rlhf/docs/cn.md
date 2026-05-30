# 奖励建模与 RLHF

> 人类无法为"好的助手回复"写出奖励函数，但他们可以比较两个回复并选出更好的那个。用这些比较拟合奖励模型，再用 RL 针对它优化语言模型。Christiano 2017，InstructGPT 2022。将 GPT-3 变成 ChatGPT 的配方。2026 年它大体上正在被 DPO 取代——但心智模型长存。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第05课（情感分析）、第9阶段第08课（PPO）
**预计时间：** 约45分钟

## 问题背景

你在下一 token 预测目标上训练了一个语言模型。它能写出语法正确的英语，但也会撒谎、废话连篇、该拒绝时不拒绝。更多预训练无法解决这个问题——网络文本是症结所在，而非解药。

你想要一个*标量奖励*来判断"对于指令 X，回复 A 比回复 B 更好"。手写这个奖励函数是不可能的——"有帮助性"不是关于 token 的闭式表达式。但人类可以比较两个输出并标记偏好，这在规模上廉价易得。

RLHF（Christiano et al. 2017；Ouyang et al. 2022）将偏好转化为奖励模型，然后通过 PPO 针对该奖励优化语言模型。三个阶段：SFT → RM → PPO。这是交付 ChatGPT、Claude、Gemini 以及 2023–2025 年每个对齐 LLM 的配方。

2026 年，PPO 步骤大体上被 DPO（第10阶段第08课）取代，因为它更便宜，对齐调整效果几乎同样好。但*奖励模型*部分仍然是每个 Best-of-N 采样器、每个基于可验证奖励的 RL 流水线，以及每个使用过程奖励模型的推理模型的基础。理解 RLHF，就理解了整个对齐技术栈。

## 核心概念

![三阶段 RLHF：SFT、成对偏好上的 RM 训练、带 KL 惩罚的 PPO](../assets/rlhf.svg)

**阶段 1：监督微调（SFT）。** 从预训练基础模型开始。在目标行为的人工演示上微调（遵循指令的回复、有帮助的回答等）。结果：一个*偏向良好行为*但动作空间仍无界的模型 `π_SFT`。

**阶段 2：奖励模型训练。**

- 收集对提示词 `x` 的回复对 `(y_+, y_-)`，由人类标记为"y_+ 优于 y_-"。
- 训练奖励模型 `R_φ(x, y)` 对 `y_+` 给出更高分数。
- 损失函数：**Bradley-Terry 成对逻辑斯谛**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid 函数。奖励差值意味着偏好的对数几率。BT 自 1952 年（Bradley-Terry）以来一直是标准，是现代 RLHF 中的主导选择。

- `R_φ` 通常从 SFT 模型初始化，顶部加一个标量头。相同的 Transformer 主干，一个单线性层输出奖励。

**阶段 3：针对 RM 加 KL 惩罚的 PPO。**

- 从 `π_SFT` 初始化可训练策略 `π_θ`。保留冻结的*参考策略* `π_ref = π_SFT`。
- 回复 `y` 末尾的奖励：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚防止 `π_θ` 任意偏离 `π_SFT`——它是*正则化器*，而不是硬信任域。`β` 通常为 `0.01`-`0.05`。
- 用这个奖励运行 PPO（第08课）。优势在 token 级别的轨迹上计算，但 RM 只对完整回复评分。

**为何需要 KL？** 没有它，PPO 会愉快地找到奖励黑客策略——RM 只在分布内的完成上训练过。分布外的回复可能比任何人工写作的回复得分都高。KL 使 `π_θ` 保持在 RM 受过训练的流形附近。这是 RLHF 中最重要的旋钮。

**2026 年现状：**

- **DPO**（Rafailov 2023）：闭式代数将阶段 2+3 折叠为对偏好数据的单个监督损失。无 RM，无 PPO。对齐基准上质量相当，计算量仅为一小部分。见第10阶段第08课。
- **GRPO**（DeepSeek 2024–2025）：使用群体相对基线替代 Critic 的 PPO，奖励来自*验证器*（代码运行通过/数学答案匹配）而非人工训练的 RM。推理模型的主导方法。见第9阶段第12课。
- **过程奖励模型（PRM）：** 对部分解答（每个推理步骤）评分，用于 RLHF 和推理 GRPO 变体。
- **宪法 AI / RLAIF：** 使用对齐的 LLM 生成偏好，而非人类。可扩展偏好预算。

## 动手实现

本课使用微型合成"提示词"和"回复"，表示为字符串。RM 是在词袋表示上的线性评分器。没有真正的 LLM——流水线的*形状*才是重点，而非规模。见 `code/main.py`。

### 第一步：合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真实 RLHF 中，这由人工标注员替换。形状——`(提示词, 偏好回复, 拒绝回复)`——是相同的。

### 第二步：Bradley-Terry 奖励模型

线性分数：`R(x, y) = w · bag(y)`。训练以最小化 BT 成对对数损失：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

几百次更新后，`w` 对好词 token 赋予正权重，对坏词 token 赋予负权重。

### 第三步：在 RM 之上的类 PPO 策略

我们的玩具策略从词表中产生单个 token。我们在 RM 下对 token 评分，计算 `log π_θ(token | prompt)`，加上对参考策略的 KL 惩罚，并应用裁剪的 PPO 代理。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo 风格的 theta 更新，将 reward 视为回报
    ...
```

### 第四步：监控 KL

每次更新时跟踪均值 `KL(π_θ || π_ref)`。如果它悄悄超过约 `5-10`，说明策略已严重偏离 `π_SFT`——β 正在上升或奖励黑客开始出现。这是真实 RLHF 中最重要的诊断指标。

### 第五步：使用 TRL 的生产配方

理解玩具流水线后，这是真实库用户编写的相同循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——阶段 2 用 `RewardTrainer`，阶段 3 用内置 KL 对参考的 `PPOTrainer`。

```python
# 阶段 2：从成对偏好训练奖励模型
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# 数据集行：{"prompt", "chosen", "rejected"} — Bradley-Terry 格式
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# 阶段 3：针对 RM 加 KL 惩罚（对 SFT 参考）的 PPO
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # 冻结

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats 包括：mean_kl, clip_frac, value_loss — PPO 的三个诊断指标
```

库为你做的三件事：`adap_kl_ctrl=True` 实现自适应 β 调度——如果观测到的 KL 超过 `target_kl`，β 翻倍；如果低于一半，β 减半。参考模型按约定被冻结——你不能意外与 `policy` 共享参数。值头与策略在同一主干上（`AutoModelForCausalLMWithValueHead` 附加一个标量 MLP 头），这就是为什么 TRL 分别报告 `policy/kl` 和 `value/loss`。

## 常见陷阱

- **过度优化/奖励黑客。** RM 不完美；`π_θ` 会找到得分高但实际很差的对抗性完成。症状：奖励无限上升，而人工评估分数停滞或下降。修复：提前停止、提高 `β`、扩大 RM 训练数据。
- **长度黑客。** 在有帮助回复上训练的 RM 往往隐式奖励长度。策略学会用填充内容延长回复。补救：长度归一化奖励，或带长度感知 RM 的 RLAIF。
- **RM 太小。** RM 需要至少与策略一样大。过小的 RM 无法忠实地对策略的输出评分。
- **KL 调整。** β 太低 → 漂移和奖励黑客。β 太高 → 策略几乎不改变。标准技巧是*自适应* β，针对每步固定的 KL。
- **偏好数据噪声。** 约 30% 的人工标注有噪声或歧义。用共识过滤数据训练 RM，或在 BT 上使用温度参数。
- **离策略问题。** 第一轮后 PPO 数据略微离策略。如第08课所述监控裁剪比例。

## 工程应用

2026 年的 RLHF 是分层的：

| 层 | 目标 | 方法 |
|---|------|------|
| 指令遵循、有帮助性、无害性 | 对齐 | DPO（第10阶段第08课）比 RLHF-PPO 更优先 |
| 推理正确性（数学、代码） | 能力 | 带验证器奖励的 GRPO（第9阶段第12课） |
| 长视野多步骤任务 | 智能体 | 在步骤上使用过程奖励模型的 PPO / GRPO |
| 安全/拒绝行为 | 安全 | 带独立安全 RM 的 RLHF-PPO，或宪法 AI |
| 推理时的 Best-of-N | 快速对齐 | 在解码时使用 RM；无需策略训练 |
| 奖励蒸馏 | 推理计算 | 在冻结 LM 之上训练小型"奖励头" |

RLHF 在 2022–2024 年是*唯一*方法。2026 年，生产对齐流水线优先使用 DPO，只有 RM 密集型或安全关键步骤才用 PPO。

## 交付物

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## 练习

1. **（简单）** 在 `code/main.py` 中用 500 个合成偏好对训练 Bradley-Terry 奖励模型。测量在留出的 100 对上的成对准确率。应超过 90%。
2. **（中等）** 以 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对每种情况，绘制 RM 分数 vs 对参考策略的 KL 随更新次数的变化曲线。哪些运行发生了奖励黑客？
3. **（困难）** 在相同偏好数据上实现 DPO（闭式偏好似然损失），与 RLHF-PPO 流水线比较所用计算量和最终 RM 分数。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| RLHF | "对齐 RL" | 三阶段 SFT + RM + PPO 流水线（Christiano 2017，Ouyang 2022）。 |
| 奖励模型 (RM) | "评分网络" | 通过 Bradley-Terry 拟合到成对偏好的学习标量函数。 |
| Bradley-Terry | "成对逻辑斯谛损失" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准 RM 目标。 |
| KL 惩罚 (KL penalty) | "保持靠近参考" | 奖励中的 `β · KL(π_θ \|\| π_ref)`；抗奖励黑客的正则化器。 |
| 奖励黑客 (Reward hacking) | "古德哈特定律" | 策略利用 RM 的缺陷；症状：奖励上升，人工评估停滞。 |
| RLAIF | "AI 标注的偏好" | 偏好标签来自另一个 LM 而非人类的 RLHF。 |
| PRM | "过程奖励模型" | 对部分推理步骤评分；用于推理流水线。 |
| 宪法 AI | "Anthropic 的方法" | 由明确规则引导的 AI 生成偏好。 |

## 延伸阅读

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — 开创 RLHF 的论文
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPT 背后的配方
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 较早的摘要 RLHF
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO；2026 年后 RLHF 的默认方法
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF 和自我批评循环
- [Bai et al. (2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — Anthropic HH 论文
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — 生产级 `RewardTrainer` 和 `PPOTrainer`。阅读 trainer 源码以了解自适应 KL 和值头的细节
- [Hugging Face — Illustrating RLHF](https://huggingface.co/blog/rlhf)（Lambert 等著）— 带图表的三阶段流水线权威讲解
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — 该库；`examples/` 中有 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本
- [Sutton & Barto (2018). 第17.4节 — 设计奖励信号](http://incompleteideas.net/book/RLbook2020.pdf) — 奖励假设视角；思考奖励黑客的必要前提

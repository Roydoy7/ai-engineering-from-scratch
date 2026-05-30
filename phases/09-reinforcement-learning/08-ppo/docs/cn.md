# 近端策略优化（PPO）

> A2C 在一次更新后就丢弃每条展开数据。PPO 将策略梯度包裹在裁剪的重要性比值中，让你可以对同一数据进行 10+ 轮次的训练而不会使策略崩溃。Schulman et al.（2017）。2026 年依然是默认的策略梯度算法。

**类型：** 构建
**语言：** Python
**前置知识：** 第9阶段第06课（REINFORCE）、第9阶段第07课（Actor-Critic）
**预计时间：** 约75分钟

## 问题背景

A2C（第07课）是在策略的：梯度 `E_{π_θ}[A · ∇ log π_θ]` 需要从*当前* `π_θ` 采样的数据。做一次更新后，`π_θ` 就变了；你使用的数据现在变成离策略的了。复用它，梯度就有偏差。

展开数据很昂贵。在 Atari 上，跨 8 个环境 × 128 步的一次展开 = 1024 次转移，花费十几秒环境时间。在一次梯度步骤后就扔掉太浪费了。

信任域策略优化（TRPO，Schulman 2015）是第一个修复方案：约束每次更新，使新旧策略之间的 KL 散度保持在 `δ` 以下。理论上清晰，但每次更新需要求解共轭梯度。2026 年没有人运行 TRPO。

PPO（Schulman et al. 2017）用一个简单的裁剪目标替换了硬信任域约束。一行额外代码，每次展开十轮次，不需要共轭梯度，理论保证足够好。九年后，它仍然是从 MuJoCo 到 RLHF 一切场景的默认策略梯度算法。

## 核心概念

![PPO 裁剪代理目标：在 1 ± ε 处进行比值裁剪](../assets/ppo.svg)

**重要性比值。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略与收集数据的策略之间的似然比。`r_t = 1` 意味着无变化。`r_t = 2` 意味着新策略采取 `a_t` 的可能性是旧策略的两倍。

**裁剪代理目标。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两项逻辑：

- 如果优势 `A_t > 0` 且比值试图超过 `1 + ε`，裁剪使梯度为零——不要将好的动作推得比旧概率高出 `+ε` 以上。
- 如果优势 `A_t < 0` 且比值试图超过 `1 - ε`（意味着我们会让坏动作相对其裁剪后的减少变得更可能），裁剪封顶梯度——不要将坏动作推得低于 `-ε`。

`min` 处理另一方向：如果比值已经向*有利*方向移动，你仍然获得梯度（不裁剪对你有益的一侧）。

典型 `ε = 0.2`。将目标绘制为 `r_t` 的函数：一个分段线性函数，在"好的一侧"有平坦的顶部，在"坏的一侧"有平坦的底部。

**完整的 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与 A2C 相同的 Actor-Critic 结构。三个系数，通常 `c_v = 0.5`，`c_e = 0.01`，`ε = 0.2`。

**训练循环。**

1. 在 `N` 个并行环境中，每个收集 `T` 步，共 `N × T` 次转移。
2. 计算优势（GAE），将其冻结为常数。
3. 将 `π_{θ_old}` 冻结为当前 `π_θ` 的快照。
4. 进行 `K` 个轮次，对每个 `(s, a, A, V_target, log π_old(a|s))` 的 minibatch：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 值损失 + 熵。
   - 梯度步骤。
5. 丢弃展开数据。返回第1步。

`K = 10`，minibatch 大小为 64 是标准超参数组合。PPO 鲁棒：具体数字在 ±50% 范围内通常影响不大。

**KL 惩罚变体。** 原始论文提出了一种使用自适应 KL 惩罚的替代方案：`L = L^{PG} - β · KL(π_θ || π_old)`，`β` 根据观测到的 KL 调整。裁剪版本占据主导；KL 变体在 RLHF 中得以保留（在 RLHF 中，对参考策略的 KL 是你无论如何都想要的单独约束）。

## 动手实现

### 第一步：在展开时捕获 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照在展开时获取一次，在更新轮次期间不变。

### 第二步：计算 GAE 优势（第07课）

与 A2C 相同。在批次上归一化。

### 第三步：裁剪代理更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # 反向传播 -surrogate，加上值损失，减去熵
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # 已裁剪
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"裁剪 → 零梯度"模式是 PPO 的核心。如果新策略在有利方向上已经漂移太远，更新就停止。

### 第四步：值和熵

向 Critic 目标添加标准 MSE，向 Actor 添加熵奖励，与 A2C 相同。

### 第五步：诊断

每次更新时观察三件事：

- **均值 KL** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]`。如果超过 `0.1`，减少 `K_EPOCHS` 或 `LR`。
- **裁剪比例**——比值落在 `[1-ε, 1+ε]` 之外的样本比例。应为约 `0.1-0.3`。如果约 `0`，裁剪从不触发 → 提高 `LR` 或 `K_EPOCHS`。如果约 `0.5+`，你在过拟合展开数据 → 降低它们。
- **解释方差** `1 - Var(V_target - V_pred) / Var(V_target)`。Critic 质量指标。随着 Critic 学习，应趋向 1。

## 常见陷阱

- **裁剪系数调整不当。** `ε = 0.2` 是事实标准。降到 `0.1` 使更新过于保守；`0.3+` 会引发不稳定。
- **轮次过多。** `K > 20` 经常导致不稳定，因为策略偏离 `π_old` 太远。限制轮次数，对大型网络尤其如此。
- **没有奖励归一化。** 大的奖励尺度会侵蚀裁剪范围。在计算优势之前归一化奖励（运行标准差）。
- **忘记优势归一化。** 每批次零均值/单位标准差归一化是标准做法。跳过它会在大多数基准上破坏 PPO。
- **学习率未衰减。** PPO 受益于线性 LR 衰减到零。固定 LR 通常更差。
- **重要性比值计算错误。** 为了数值稳定性，始终用 `exp(log_new - log_old)` 而不是 `new / old`。
- **梯度符号错误。** 最大化代理 = *最小化* `-L^{CLIP}`。符号翻转是最常见的 PPO bug。

## 工程应用

PPO 是 2026 年在意外多种领域的默认 RL 算法：

| 使用场景 | PPO 变体 |
|---------|---------|
| MuJoCo / 机器人控制 | 带高斯策略、GAE(0.95) 的 PPO |
| Atari / 离散游戏 | 带分类策略、滚动 128 步展开的 PPO |
| LLM 的 RLHF | 带对参考模型 KL 惩罚的 PPO，响应末尾的 RM 奖励 |
| 大规模游戏智能体 | IMPALA + PPO（AlphaStar、OpenAI Five） |
| 推理 LLM | GRPO（第12课）——无 Critic 的 PPO 变体 |
| 仅偏好数据 | DPO——PPO+KL 的闭式折叠，无需在线采样 |

PPO 的*损失形状*——裁剪代理 + 值 + 熵——是 DPO、GRPO 和几乎每个 RLHF 流水线的脚手架。

## 交付物

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. N envs × T steps.
2. Update schedule. K epochs, minibatch size, LR schedule.
3. Surrogate params. ε (clip), c_v, c_e, advantage normalization on.
4. Advantage. GAE(λ) with explicit γ and λ.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse K > 30 or ε > 0.3 (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 练习

1. **（简单）** 在 4×4 网格世界上用 `ε=0.2, K=4` 运行 PPO。在相同环境步数下，与 A2C（每次展开一轮次）比较样本效率。
2. **（中等）** 扫描 `K ∈ {1, 4, 10, 30}`。绘制回报 vs 环境步数，并跟踪每次更新的均值 KL。在这个任务上，`K` 为多少时 KL 会爆炸？
3. **（困难）** 将裁剪代理替换为自适应 KL 惩罚（如果 `KL > 2·target`，`β` 翻倍；如果 `KL < target/2`，`β` 减半）。比较最终回报、稳定性和无裁剪性。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 重要性比值 (Importance ratio) | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`；与收集数据的策略的偏差。 |
| 裁剪代理 (Clipped surrogate) | "PPO 的主要技巧" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在有利一侧超过裁剪后梯度变平。 |
| 信任域 (Trust region) | "TRPO / PPO 的意图" | 限制每次更新的 KL 以保证单调改进。 |
| KL 惩罚 (KL penalty) | "软信任域" | 替代 PPO：`L - β · KL(π_θ \|\| π_old)`。自适应 `β`。 |
| 裁剪比例 (Clip fraction) | "裁剪触发频率" | 诊断指标——应为 0.1-0.3；超出范围意味着调整不当。 |
| 多轮次训练 (Multi-epoch training) | "数据复用" | 每次展开 K 个轮次；用方差成本换取样本效率。 |
| 接近在策略 (On-policy-ish) | "大致在策略" | PPO 名义上在策略，但 K>1 轮次使用略微离策略的数据，仍然安全。 |
| PPO-KL | "另一个 PPO" | KL 惩罚变体；在 RLHF 中使用，因为对参考策略的 KL 本来就是约束。 |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 原始论文
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO，PPO 的前身
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 每个 PPO 超参数的消融实验
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT；PPO 在 RLHF 中的配方
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — 带 PyTorch 的清晰现代讲解
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — 许多论文使用的参考单文件 PPO
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 语言模型 PPO 的生产配方；与第09课（RLHF）配合阅读
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — "37 个代码级优化"论文；哪些 PPO 技巧是关键的，哪些是民间传说

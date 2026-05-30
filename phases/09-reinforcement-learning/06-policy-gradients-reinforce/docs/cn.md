# 策略梯度——从头实现 REINFORCE

> 停止估计值函数。直接参数化策略，计算期望回报的梯度，向上爬坡。Williams（1992）用一个定理写出了它。这就是 PPO、GRPO 以及每个 LLM RL 循环存在的原因。

**类型：** 构建
**语言：** Python
**前置知识：** 第3阶段第03课（反向传播）、第9阶段第03课（蒙特卡洛）、第9阶段第04课（TD 学习）
**预计时间：** 约75分钟

## 问题背景

Q学习和 DQN 参数化的是*值*函数。你通过 `argmax Q` 选择动作。对于离散动作和离散状态来说这没问题。但当动作是连续的（如何对 10 维力矩做 `argmax`？）或者你想要随机策略（`argmax` 本质上是确定性的）时就会失败。

策略梯度则直接参数化*策略*。`π_θ(a | s)` 是一个输出动作分布的神经网络。从中采样来执行动作。计算期望回报对 `θ` 的梯度。向上爬坡。不需要 `argmax`，不需要贝尔曼递推。只是对 `J(θ) = E_{π_θ}[G]` 做梯度上升。

REINFORCE 定理（Williams 1992）告诉你这个梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。运行一个片段，计算回报，在每步乘以 `∇ log π_θ(a | s)`，取平均，梯度上升，完成。

2026 年的每个 LLM RL 算法——PPO、DPO、GRPO——都是 REINFORCE 的改进版。在手上烂熟这个算法，是本阶段其余内容以及第10阶段第07课（RLHF 实现）和第10阶段第08课（DPO）的前提。

## 核心概念

![策略梯度：softmax 策略，log-π 梯度，回报加权更新](../assets/policy-gradient.svg)

**策略梯度定理。** 对于任意由 `θ` 参数化的策略 `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从步骤 `t` 开始的折扣回报。期望是对从 `π_θ` 采样的完整轨迹 `τ` 求的。

**证明很简洁。** 在期望下对 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求导。使用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（对数导数技巧）。分解 `log P(τ; θ) = Σ log π_θ(a_t | s_t) + 不依赖 θ 的环境项`。环境项消失。两行代数给出定理。

**方差减少技巧。** 原始 REINFORCE 方差极高——回报有噪声，`∇ log π` 有噪声，它们的乘积噪声非常大。两种标准修复：

1. **基线减法。** 将 `G_t` 替换为 `G_t - b(s_t)`，其中基线 `b(s_t)` 不依赖于 `a_t`。无偏，因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型选择：由 Critic 学习的 `b(s_t) = V̂(s_t)` → Actor-Critic（第07课）。
2. **未来回报。** 将 `Σ_t G_t · ∇ log π_θ(a_t | s_t)` 替换为 `Σ_t G_t^{从t起} · ∇ log π_θ(a_t | s_t)`。对于给定动作，只有未来的回报才有意义——过去的奖励贡献零均值噪声。

合并后得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这就是带基线的 REINFORCE——A2C（第07课）和 PPO（第08课）的直接祖先。

**Softmax 策略参数化。** 对于离散动作的标准选择：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是任何输出每个动作分数的神经网络。梯度有简洁的形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

即所取动作的分数减去其在策略下的期望值。

**连续动作的高斯策略。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有封闭形式。这就是第9阶段第07课 SAC 所需的全部。

## 动手实现

### 第一步：Softmax 策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

对表格环境使用线性策略（每个动作一个权重向量）。对于 Atari，换入 CNN 并保留 softmax 头。

### 第二步：采样和对数概率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 第三步：记录对数概率的展开

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 第四步：REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（`a` 的 one-hot 减去概率向量）是 softmax 策略梯度的核心。把它刻进肌肉记忆。

### 第五步：基线

近期片段的 `G` 运行均值作为基线，足以为 4×4 网格世界提供方差减少；大约 500 个片段后收敛。将基线升级为学习到的 `V̂(s)` 就得到了 Actor-Critic。

## 常见陷阱

- **梯度爆炸。** 回报可能很大。在乘以 `∇ log π` 之前，始终将批次中的 `G` 归一化到约 `N(0, 1)`。
- **熵崩溃。** 策略过早收敛到接近确定性的动作，停止探索，陷入困境。修复：在目标中加入熵奖励 `β · H(π(·|s))`。
- **高方差。** 原始 REINFORCE 需要数千个片段。Critic 基线（第07课）或 TRPO/PPO 的信任域（第08课）是标准修复方案。
- **样本低效率。** 在策略意味着每次更新后抛弃所有转移。通过重要性采样离策略修正可以取回数据，代价是方差增大（PPO 的比值就是一个被裁剪的 IS 权重）。
- **非稳态梯度。** 100 个片段前的相同梯度使用旧的 `π`。在策略方法每隔几次展开就更新，正是为此。
- **信用分配。** 没有未来回报，过去的奖励只会贡献噪声。始终使用未来回报。

## 工程应用

2026 年，REINFORCE 很少直接运行，但其梯度公式无处不在：

| 使用场景 | 派生方法 |
|---------|---------|
| 连续控制 | 带高斯策略的 PPO / SAC |
| LLM RLHF | 带 KL 惩罚的 PPO，在 token 级策略上运行 |
| LLM 推理（DeepSeek） | GRPO——带群体相对基线的 REINFORCE，无需 Critic |
| 多智能体 | 集中 Critic 的 REINFORCE（MADDPG、COMA） |
| 离散动作机器人 | A2C、A3C、PPO |
| 仅偏好设置 | DPO——将 REINFORCE 重写为偏好似然损失，无需采样 |

当你在 2026 年的训练脚本中读到 `loss = -advantage * log_prob` 时，那就是带基线的 REINFORCE。整篇论文（DPO、GRPO、RLOO）都是在这一行代码上的方差减少技巧。

## 交付物

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned V̂(s), or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with β = 0 and observed policy entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **（简单）** 在 4×4 网格世界上用线性 softmax 策略实现 REINFORCE。无基线训练 1000 个片段。绘制学习曲线；测量方差（回报的标准差）。
2. **（中等）** 添加运行均值基线，重新训练。将样本效率和方差与不加基线的运行进行比较。基线能减少多少收敛所需的步数？
3. **（困难）** 添加熵奖励 `β · H(π)`。扫描 `β ∈ {0, 0.01, 0.1, 1.0}`。绘制最终回报和策略熵。在这个任务上，甜蜜点在哪里？

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 策略梯度 (Policy gradient) | "直接训练策略" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`；由对数导数技巧推导。 |
| REINFORCE | "原始 PG 算法" | Williams（1992）；蒙特卡洛回报乘以对数策略梯度。 |
| 对数导数技巧 (Log-derivative trick) | "评分函数估计器" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；使期望的梯度可处理。 |
| 基线 (Baseline) | "方差减少" | 任何从 `G` 中减去的 `b(s)`；无偏，因为 `E[b · ∇ log π] = 0`。 |
| 未来回报 (Reward-to-go) | "只有未来回报才重要" | 用 `G_t^{从t起}` 替代完整的 `G_0`；正确且方差更低。 |
| 熵奖励 (Entropy bonus) | "鼓励探索" | `+β · H(π(·\|s))` 项防止策略崩塌。 |
| 在策略 (On-policy) | "在你刚看到的内容上训练" | 梯度期望是关于当前策略的——不能直接复用旧数据。 |
| 优势 (Advantage) | "比平均好多少" | `A(s, a) = G(s, a) - V(s)`；带基线的 REINFORCE 乘以的有符号量。 |

## 延伸阅读

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — 原始 REINFORCE 论文
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 带函数近似的现代策略梯度定理
- [Sutton & Barto (2018). 第13章 — 策略梯度方法](http://incompleteideas.net/book/RLbook2020.pdf) — 教材讲解
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — 带 PyTorch 代码的清晰教学讲解
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — 方差减少和自然梯度视角，连接 REINFORCE 与信任域家族（TRPO、PPO）

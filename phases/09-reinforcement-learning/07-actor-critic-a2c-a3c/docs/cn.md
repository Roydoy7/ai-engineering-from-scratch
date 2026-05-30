# Actor-Critic——A2C 与 A3C

> REINFORCE 噪声大。添加一个学习 `V̂(s)` 的 Critic，从回报中减去它，就得到了期望值相同但方差远低的优势函数。这就是 Actor-Critic。A2C 同步运行它；A3C 跨线程运行它。两者都是每个现代深度 RL 方法的心智模型。

**类型：** 构建
**语言：** Python
**前置知识：** 第9阶段第04课（TD 学习）、第9阶段第06课（REINFORCE）
**预计时间：** 约75分钟

## 问题背景

原始 REINFORCE 有效，但方差极高。蒙特卡洛回报 `G_t` 在不同片段之间可能相差 10 倍以上。将这种噪声乘以 `∇ log π` 后取平均，产生的梯度估计器需要数千个片段才能使策略移动同样的距离，而 DQN 更新少得多就能做到。

方差来源于使用原始回报。如果减去一个基线 `b(s_t)`——状态的任意函数，包括学习到的值——期望不变而方差下降。最佳的可处理基线是 `V̂(s_t)`。现在乘以 `∇ log π` 的量是*优势*：

`A(s, a) = G - V̂(s)`

动作好，意味着回报高于平均；差，则低于平均。带学习 Critic 的 REINFORCE 就是 *Actor-Critic*。Critic 给 Actor 提供低方差的教学信号。这就是 2015 年后的每个深度策略方法（A2C、A3C、PPO、SAC、IMPALA）。

## 核心概念

![Actor-Critic：策略网络加值网络，TD 残差作为优势](../assets/actor-critic.svg)

**两个网络，一个联合损失：**

- **Actor** `π_θ(a | s)`：策略。通过采样执行动作。用策略梯度训练。
- **Critic** `V_φ(s)`：估计从状态出发的期望回报。训练以最小化 `(V_φ(s) - target)²`。

**优势函数。** 两种标准形式：

- *MC 优势：* `A_t = G_t - V_φ(s_t)`。无偏，方差较高。
- *TD 优势：* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（使用 `V_φ`），方差低得多。也称为 *TD 残差* `δ_t`。

**n 步优势。** 在两者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD。`n = ∞` 是 MC。大多数实现对 Atari 使用 `n = 5`，对 MuJoCo 上的 PPO 使用 `n = 2048`。

**广义优势估计（GAE）。** Schulman et al.（2016）提出了对所有 n 步优势的指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差，高偏差）。`λ = 1` 是 MC（高方差，无偏）。`λ = 0.95` 是 2026 年的默认值——调节直到偏差/方差的平衡达到你想要的位置。

**A2C：同步优势 Actor-Critic。** 在 `N` 个并行环境中收集 `T` 步。计算每步的优势。在合并批次上更新 Actor 和 Critic。重复。A3C 更简单、更易扩展的同胞。

**A3C：异步优势 Actor-Critic。** Mnih et al.（2016）。启动 `N` 个工作线程，每个运行一个环境。每个工作者在自己的展开上本地计算梯度，然后异步应用到共享参数服务器。不需要回放缓冲区——工作者通过运行不同轨迹来解相关。A3C 证明了可以在 CPU 上规模化训练。2026 年，基于 GPU 的 A2C（批量并行环境）占主导，因为 GPU 需要大批次。

**联合损失。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三项：策略梯度损失、值回归、熵奖励。`c_v ~ 0.5`，`c_e ~ 0.01` 是典型起始值。

## 动手实现

### 第一步：Critic

用 MSE 更新的线性 Critic `V_φ(s) = w · features(s)`：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格环境中，Critic 在几百个片段内收敛。在 Atari 上，用共享 CNN 主干 + 值头替换线性 Critic。

### 第二步：n 步优势

给定长度为 `T` 的展开和自举的最终值 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是 Critic 的目标。`advantages` 是乘以 `∇ log π` 的量。

### 第三步：联合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

在策略，每次展开一次更新，Actor 和 Critic 有独立的学习率。

### 第四步：并行化（A3C vs A2C）

- **A3C：** 启动 `N` 个线程，每个运行自己的环境和前向传播。定期将梯度更新推送到共享主节点。主节点上不加锁——竞态是可以的，只是增加一些噪声。
- **A2C：** 在单个进程中运行 `N` 个环境实例，将观测堆叠为 `[N, obs_dim]` 批次，批量前向传播，批量反向传播。更高的 GPU 利用率，确定性，更易推理。2026 年的默认选择。

我们的玩具代码为清晰起见是单线程的；改写为批量 A2C 只需三行 numpy。

## 常见陷阱

- **Actor 梯度之前的 Critic 偏差。** 如果 Critic 是随机的，其基线没有信息价值，你在纯噪声上训练 Actor。预热 Critic 几百步后再打开策略梯度，或者使用较慢的 Actor 学习率。
- **优势归一化。** 将每批次的优势归一化为零均值/单位标准差。几乎零成本，却大幅稳定训练。
- **共享主干。** 在图像输入上，Actor 和 Critic 使用共享特征提取器。分离的头部。共享特征从两个损失中受益。
- **在策略契约。** A2C 对每次展开只更新一次。更多次则梯度有偏（PPO 增加的就是重要性采样修正）。
- **熵崩溃。** 没有 `c_e > 0`，策略在几百次更新内变得接近确定性并停止探索。
- **奖励尺度。** 优势幅度取决于奖励尺度。归一化奖励（例如用运行标准差除）以在各任务间保持一致的梯度幅度。

## 工程应用

A2C/A3C 在 2026 年很少是最终选择，但它们是后续所有方法改进的架构基础：

| 方法 | 与 A2C 的关系 |
|------|------------|
| PPO | A2C + 裁剪重要性比值，用于多轮次更新 |
| IMPALA | A3C + V-trace 离策略修正 |
| SAC（第9阶段第07课） | 带软值 Critic 的离策略 A2C（下一课） |
| GRPO（第9阶段第12课） | 无 Critic 的 A2C——群体相对优势 |
| DPO | 折叠为偏好排名损失的 A2C，无需采样 |
| AlphaStar / OpenAI Five | 带联赛训练 + 模仿预训练的 A2C |

如果你在 2026 年的论文中看到"优势"，就想到 Actor-Critic。

## 交付物

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. c_v (value), c_e (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with c_e = 0 and observed entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **（简单）** 在 4×4 网格世界上用 MC 优势（`G_t - V(s_t)`）训练 Actor-Critic。与第06课的带运行均值基线的 REINFORCE 比较样本效率。
2. **（中等）** 切换到 TD 残差优势（`r + γ V(s') - V(s)`）。测量优势批次的方差。下降了多少？
3. **（困难）** 实现 GAE(λ)。扫描 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`。绘制最终回报 vs 样本效率的曲线。这个任务上偏差/方差的甜蜜点在哪里？

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| Actor | "策略网络" | `π_θ(a\|s)`，用策略梯度更新。 |
| Critic | "值网络" | `V_φ(s)`，用 MSE 回归到回报/TD 目标进行更新。 |
| 优势 (Advantage) | "比平均好多少" | `A(s, a) = Q(s, a) - V(s)` 或其估计器。`∇ log π` 的乘数。 |
| TD 残差 (TD residual) | "δ" | `δ_t = r + γ V(s') - V(s)`；一步优势估计。 |
| GAE | "插值旋钮" | n 步优势的指数加权和，由 `λ` 参数化。 |
| A2C | "同步 Actor-Critic" | 跨环境批量化；每次展开一次梯度步骤。 |
| A3C | "异步 Actor-Critic" | 工作线程将梯度推送到共享参数服务器。原始论文；2026 年较少使用。 |
| 自举 (Bootstrap) | "在视野末端使用 V" | 截断展开，加上 `γ^n V(s_{t+n})` 来封闭求和。 |

## 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C，原始异步 Actor-Critic 论文
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE
- [Sutton & Barto (2018). 第13章 — Actor-Critic 方法](http://incompleteideas.net/book/RLbook2020.pdf) — 基础；当 Critic 是神经网络时，与第9章（函数近似）配对阅读
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — 带 V-trace 离策略修正的可扩展分布式 Actor-Critic
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 值得阅读的生产级 A2C/PPO 实现
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 两时间尺度 Actor-Critic 分解的基础收敛结果

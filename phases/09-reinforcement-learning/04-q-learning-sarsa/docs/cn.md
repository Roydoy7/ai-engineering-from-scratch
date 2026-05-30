# 时序差分——Q学习与 SARSA

> 蒙特卡洛等待片段结束才更新。TD 在每一步之后就通过自举下一个值估计来更新。Q学习是离策略且乐观的；SARSA 是在策略且保守的。两者都是一行代码，都支撑着本阶段每个深度 RL 方法。

**类型：** 构建
**语言：** Python
**前置知识：** 第9阶段第01课（MDP）、第9阶段第02课（动态规划）、第9阶段第03课（蒙特卡洛）
**预计时间：** 约75分钟

## 问题背景

蒙特卡洛有效，但它有两个昂贵的要求：需要能终止的片段，且只在最终回报出来后才更新。如果你的片段有 1000 步，MC 要等 1000 步才能更新任何东西。高方差、低偏差、实践中速度慢。

动态规划恰好相反——零方差的自举备份——但需要已知模型。

时序差分（TD）学习取中间路线。从单次转移 `(s, a, r, s')` 出发，构造一步目标 `r + γ V(s')` 并将 `V(s)` 向它靠拢。不需要模型，不需要完整片段。使用近似 `V` 作为右侧时有偏差，但方差比 MC 大幅降低，且从第一步就开始在线更新。

这是所有现代 RL——DQN、A2C、PPO、SAC——赖以转动的枢轴。第9阶段的后续内容，都是在本课写出的一步 TD 更新之上叠加函数近似和技巧。

## 核心概念

![Q学习 vs SARSA：离策略的 max 与在策略的 Q(s', a')](../assets/td.svg)

**V 的 TD(0) 更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

方括号内的量是 TD 误差 `δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的在线类比。收敛要求 `α` 满足 Robbins-Monro 条件（`Σ α = ∞`，`Σ α² < ∞`）且所有状态被无限次访问。

**Q学习。** 用于控制的离策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 假设从 `s'` 开始将遵循*贪婪*策略，无论智能体实际采取什么动作。这种解耦使 Q学习在智能体通过 ε-贪婪探索时学习 `Q*`。Mnih et al.（2015）将其转化为 Atari 上的深度 Q学习（第05课）。

**SARSA。** 在策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名称来自五元组 `(s, a, r, s', a')`。SARSA 使用智能体*实际*采取的下一个动作 `a'`，而非贪婪的 `argmax`。收敛到当前运行的 ε-贪婪策略 `π` 的 `Q^π`，在极限 `ε → 0` 时趋向 `Q*`。

**悬崖行走的差异。** 在经典悬崖行走任务中（掉入悬崖 = 奖励 -100），Q学习学到沿悬崖边缘的最优路径，但探索期间偶尔会受到惩罚。SARSA 学到离悬崖远一步的更安全路径，因为它将探索噪声纳入了 Q 值计算。经过充分训练，两者在 `ε → 0` 时都达到最优。实践中这很重要：如果部署时真的在进行探索，SARSA 的行为更保守。

**期望 SARSA。** 用 `π` 下的期望值替换 `Q(s', a')`：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

方差比 SARSA 更低（不对 `a'` 采样），目标与在策略相同。现代教材中常作为默认选择。

**n 步 TD 与 TD(λ)。** 通过等待 `n` 步再自举，在 TD(0) 和 MC 之间插值。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 以几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 取平均。大多数深度 RL 使用 `n` 在 3 到 20 之间。

## 动手实现

### 第一步：ε-贪婪策略上的 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行代码。与 Q学习的*唯一*区别是目标那一行。

### 第二步：Q学习

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标与行为解耦。这一个符号就是在策略与离策略的差别。

### 第三步：学习曲线

跟踪每 100 个片段的平均回报。Q学习在简单确定性网格世界上收敛更快；SARSA 在悬崖行走上更保守。在 `code/main.py` 的 4×4 网格世界上，使用 `α=0.1, ε=0.1`，两者在约 2000 个片段后接近最优。

### 第四步：与 DP 真值对比

运行值迭代（第02课）得到 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。健康的表格 TD 智能体在 10,000 个片段后，在 4×4 网格世界上误差约为 `0.5`。

## 常见陷阱

- **Q 的初始值很重要。** 乐观初始化（负奖励任务中 `Q = 0`）鼓励探索。悲观初始化可能使贪婪策略永远陷入困境。
- **α 的调度。** 固定 `α` 适用于非稳态问题。理论上衰减 `α_n = 1/n` 保证收敛，但实践中太慢——将 `α` 固定在 `[0.05, 0.3]` 范围内，监控学习曲线。
- **ε 的调度。** 从高值（`ε=1.0`）开始，衰减到 `ε=0.05`。"GLIE"（无限探索下的极限贪婪）是收敛条件。
- **Q学习的最大化偏差。** 当 `Q` 有噪声时，`max` 算子向上偏置。导致过估计——Hasselt 的双 Q学习（第05课 DDQN 使用）通过两个 Q 表来修复。
- **非终止片段。** TD 可以在没有终止状态的情况下学习，但需要限制步数或在限制处正确处理自举。标准做法：将步数上限视为非终止，继续自举。
- **状态哈希。** 如果状态是元组/张量，使用可哈希的键（元组而非列表；四舍五入的浮点元组，而非原始值）。

## 工程应用

2026 年的 TD 技术栈：

| 任务 | 方法 | 原因 |
|------|------|------|
| 小型表格环境 | Q学习 | 直接学习最优策略 |
| 在策略的安全关键场景 | SARSA / 期望 SARSA | 探索期间更保守 |
| 高维状态空间 | DQN（第9阶段第05课） | 带回放和目标网络的神经网 Q 函数 |
| 连续动作空间 | SAC / TD3（第9阶段第07课） | Q 网络上的 TD 更新；策略网络发出动作 |
| LLM RL（基于奖励模型） | PPO / GRPO（第9阶段第08、12课） | 带 GAE TD 风格优势的 Actor-Critic |
| 离线 RL | CQL / IQL（第9阶段第08课） | 带保守正则化的 Q学习 |

2026 年论文中你读到的 90% 的"RL"都是 Q学习或 SARSA 的某种变体。在深入阅读之前，先把表格更新烂熟于心。

## 交付物

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, |Q - Q*| check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 练习

1. **（简单）** 在 4×4 网格世界上实现 Q学习和 SARSA。绘制 2000 个片段的学习曲线（每 100 个片段的平均回报）。哪个收敛更快？
2. **（中等）** 构建悬崖行走环境（4×12，最后一行是悬崖，奖励 -100 并重置到起点）。比较 Q学习和 SARSA 的最终策略。截图记录每种方法的路径，哪个更靠近悬崖边缘？
3. **（困难）** 实现双 Q学习。在有噪声奖励的网格世界（每步奖励加高斯噪声 σ=5）上，证明 Q学习对 `V*(0,0)` 的过估计有实质性偏差，而双 Q学习没有。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| TD 误差 (TD error) | "更新信号" | `δ = r + γ V(s') - V(s)`，自举残差。 |
| TD(0) | "一步 TD" | 每次转移后使用下一个状态的估计立即更新。 |
| Q学习 (Q-learning) | "离策略 RL 101" | 带下一状态动作 `max` 的 TD 更新；无论行为策略如何，都学习 `Q*`。 |
| SARSA | "在策略的 Q学习" | 使用实际下一个动作的 TD 更新；学习当前 ε-贪婪策略 `π` 的 `Q^π`。 |
| 期望 SARSA (Expected SARSA) | "低方差的 SARSA" | 用策略 π 下的期望值替换采样的 `a'`。 |
| GLIE | "正确的探索调度" | 无限探索下的极限贪婪；Q学习收敛的必要条件。 |
| 自举 (Bootstrapping) | "在目标中使用当前估计" | 区分 TD 与 MC 的关键。引入偏差，但大幅降低方差。 |
| 最大化偏差 (Maximization bias) | "Q学习过估计" | 对有噪声估计取 `max` 向上偏置；双 Q学习修复了这个问题。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文和收敛证明
- [Sutton & Barto (2018). 第6章 — 时序差分学习](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q学习、期望 SARSA
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 最大化偏差的修复
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — 期望 SARSA 的动机
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 首创 SARSA 的论文（当时称为"修正的连接主义 Q学习"）
- [Sutton & Barto (2018). 第7章 — n 步自举](http://incompleteideas.net/book/RLbook2020.pdf) — 将 TD(0) 推广到 TD(n)，从 Q学习到资格迹，再到 PPO 中 GAE 的路径

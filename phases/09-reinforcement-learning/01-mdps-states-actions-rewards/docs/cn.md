# 马尔可夫决策过程：状态、动作与奖励

> 马尔可夫决策过程由五个要素构成：状态、动作、转移、奖励、折扣因子。强化学习中的一切——Q学习、PPO、DPO、GRPO——都在这个框架上优化。学会它一次，强化学习的其余部分就全打通了。

**类型：** 学习
**语言：** Python
**前置知识：** 第1阶段第06课（概率与分布）、第2阶段第01课（机器学习分类）
**预计时间：** 约45分钟

## 问题背景

你在写一个象棋机器人、库存规划器、交易代理，或者训练推理模型的 PPO 循环。四个完全不同的领域，一个令人惊讶的事实：它们都归结为同一个数学对象。

监督学习给你 `(x, y)` 配对，要求你拟合一个函数。强化学习不给你标签——只有一连串的状态、你采取的动作，以及一个标量奖励。这步棋赢了吗？补货决策省了钱吗？交易盈利了吗？LLM 刚生成的这个 token 能让评判者给出更高的奖励吗？

在正式化之前，你无法从这个数据流中学习。"我看到了什么"、"我做了什么"、"接下来发生了什么"、"那有多好"——每一个都必须变成可以推理的对象。这个正式化就是马尔可夫决策过程（MDP）。本阶段的每个 RL 算法，包括最后的 RLHF 和 GRPO 循环，都在这个框架上进行优化。

## 核心概念

![马尔可夫决策过程：状态、动作、转移、奖励、折扣](../assets/mdp.svg)

**五个要素。**

- **状态** `S`。智能体决策所需的一切信息。在网格世界中是当前格子，在象棋中是棋盘状态，在 LLM 中是上下文窗口加任何记忆。
- **动作** `A`。可做的选择。向上/下/左/右移动、走一步棋、发出一个 token。
- **转移** `P(s' | s, a)`。给定状态 `s` 和动作 `a`，下一个状态的分布。象棋中是确定性的，库存管理中是随机的，LLM 解码中几乎是确定性的。
- **奖励** `R(s, a, s')`。标量信号。赢棋 = +1，输棋 = -1。收入减成本。GRPO 中的对数似然比项。
- **折扣因子** `γ ∈ [0, 1)`。未来奖励相对当前奖励的权重。`γ = 0.99` 对应约 100 步的有效视野；`γ = 0.9` 对应约 10 步。

**马尔可夫性质** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来只取决于当前状态。如果不满足，说明状态表示不完整——这不是方法的失败，而是状态定义的失败。

**策略与回报。** 策略 `π(a | s)` 将状态映射到动作分布。回报 `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` 是未来奖励的折扣求和。值函数 `V^π(s) = E[G_t | s_t = s]` 是在策略 `π` 下从状态 `s` 出发的期望回报。Q 值 `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` 是以特定动作开始的期望回报。每个 RL 算法都估计这两者之一，然后相应地改进 `π`。

**贝尔曼方程。** 本阶段所有算法都用到的不动点方程：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

这些方程将期望回报分解为"当前步的奖励"加"落点折扣值"。递归结构。第9阶段的每个算法要么迭代此方程至收敛（动态规划），要么对其采样（蒙特卡洛），要么单步自举（时序差分）。

## 动手实现

### 第一步：构建一个小型确定性 MDP

4×4 网格世界。智能体从左上角出发，终止状态在右下角，每步奖励 -1，动作集合 `{上, 下, 左, 右}`。见 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

五行代码，这就是完整的环境。确定性转移，固定步惩罚，吸收终止状态。

### 第二步：展开一个策略

策略是从状态到动作分布的函数。最简单的：均匀随机。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

运行随机策略 1000 次。在这块 4×4 的棋盘上，平均回报大约是 -60 到 -80。最优回报是 -6（直线往右下走）。弥合这个差距就是第9阶段的全部内容。

### 第三步：通过贝尔曼方程精确计算 `V^π`

对于小型 MDP，贝尔曼方程是一个线性方程组。枚举状态，对期望值求和，迭代直到值不再变化。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这就是迭代策略评估。它是 Sutton & Barto 中的第一个算法，也是后续每个 RL 方法的理论基础。

### 第四步：`γ` 是有物理含义的超参数

有效视野大约是 `1 / (1 - γ)`。`γ = 0.9` → 10 步；`γ = 0.99` → 100 步；`γ = 0.999` → 1000 步。

太低则智能体行为短视。太高则信用分配变得嘈杂，因为许多早期步骤共享对遥远奖励的责任。LLM 的 RLHF 通常使用 `γ = 1`，因为片段较短且有界。控制任务使用 0.95–0.99。长视野策略游戏使用 0.999。

## 常见陷阱

- **非马尔可夫状态。** 如果你需要最近三次观测才能决策，"状态"就不仅仅是当前观测。修复：叠帧（DQN 在 Atari 上叠 4 帧）或使用循环状态（在观测上使用 LSTM/GRU）。
- **稀疏奖励。** 只有胜利才有奖励的设置在大状态空间中几乎无法学习。塑造奖励（中间信号）或用模仿学习引导（第9阶段第09课）。
- **奖励黑客。** 优化代理奖励往往产生病态行为。OpenAI 的赛艇代理永远绕圈收集道具，而不是完成比赛。始终根据目标结果而非代理来定义奖励。
- **折扣设置错误。** 无限视野任务使用 `γ = 1` 会让所有值变成无穷大。始终用有限视野或 `γ < 1` 来约束。
- **奖励尺度。** 奖励 {+100, -100} vs {+1, -1} 给出相同的最优策略，但梯度幅度差异巨大。在传入 PPO/DQN 之前，归一化到 `[-1, 1]` 左右。

## 工程应用

2026 年的技术栈在接触代码之前，将每个 RL 流水线简化为一个 MDP：

| 场景 | 状态 | 动作 | 奖励 | γ |
|------|------|------|------|---|
| 控制（运动、操控） | 关节角度 + 速度 | 连续力矩 | 任务特定的塑造奖励 | 0.99 |
| 游戏（象棋、围棋、扑克） | 棋盘 + 历史 | 合法动作 | 赢=+1 / 输=-1 | 1.0（有限） |
| 库存/定价 | 库存 + 需求 | 订购数量 | 收入 - 成本 | 0.95 |
| LLM 的 RLHF | 上下文 token | 下一个 token | 末尾的奖励模型分数 | 1.0（片段约 200 token） |
| 推理的 GRPO | 提示 + 部分响应 | 下一个 token | 末尾验证器 0/1 | 1.0 |

在写任何训练循环之前，先写出五元组。大多数"RL 不起作用"的 bug 都源于纸面上就已经破损的 MDP 形式化。

## 交付物

保存为 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any γ ≥ 1.0 on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## 练习

1. **（简单）** 在 `code/main.py` 中实现 4×4 网格世界和随机策略展开。运行 10,000 个片段，报告回报的均值和标准差。与最优回报 (-6) 比较。
2. **（中等）** 对均匀随机策略用 `γ ∈ {0.5, 0.9, 0.99}` 运行 `policy_evaluation`。以 4×4 网格形式打印每种情况下的 `V`。解释为什么较大的 `γ` 使终止状态附近的状态值增长更快。
3. **（困难）** 将网格世界改为随机性的：每次动作以概率 `p = 0.1` 滑向相邻方向。重新评估均匀策略。`V[起点]` 变好还是变差？为什么？

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| MDP | "强化学习设置" | 满足马尔可夫性质的五元组 `(S, A, P, R, γ)`。 |
| 状态 (State) | "智能体看到的东西" | 在所选策略类下，对未来动态的充分统计量。 |
| 策略 (Policy) | "智能体的行为" | 条件分布 `π(a \| s)` 或确定性映射 `s → a`。 |
| 回报 (Return) | "总奖励" | 从当前步开始的折扣求和 `Σ γ^t r_t`。 |
| 值函数 (Value) | "一个状态有多好" | 在策略 `π` 下从状态 `s` 出发的期望回报。 |
| Q 值 (Q-value) | "一个动作有多好" | 在策略 `π` 下从状态 `s` 以动作 `a` 开始的期望回报。 |
| 贝尔曼方程 (Bellman equation) | "动态规划递推" | 将值/Q 分解为单步奖励加折扣后继值的不动点方程。 |
| 折扣因子 γ (Discount γ) | "未来与当前的权衡" | 对遥远奖励的几何权重；有效视野约 `1/(1-γ)`。 |

## 延伸阅读

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 权威教材，第3章介绍 MDP 和贝尔曼方程，第1章阐述奖励假设
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — 贝尔曼方程的起源
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 从深度 RL 角度出发的简洁 MDP 入门
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — MDP 及精确求解方法的运筹学参考
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — 将 MDP 作为动态规划特例的最清晰推导

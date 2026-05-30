# 蒙特卡洛方法——从完整片段中学习

> 动态规划需要模型。蒙特卡洛只需要片段。运行策略，观察回报，取平均值。这是强化学习中最简单的思想——也是打开后续一切的那把钥匙。

**类型：** 构建
**语言：** Python
**前置知识：** 第9阶段第01课（MDP）、第9阶段第02课（动态规划）
**预计时间：** 约75分钟

## 问题背景

动态规划很优雅，但它假设你能对每个状态和动作查询 `P(s' | s, a)`。现实世界中几乎没有这样的情况。机器人无法解析计算关节力矩后的相机像素分布。定价算法无法对每种可能的客户反应求积分。LLM 无法枚举一个 token 之后的所有可能延续。

你需要一种只需从环境中*采样*的方法。运行策略，得到轨迹 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`，用它来估计值函数。这就是蒙特卡洛。

从 DP 到 MC 的转变在哲学上意义重大：我们从*已知模型 + 精确备份*转向*采样展开 + 平均回报*。方差增大了，但适用范围大幅扩展。本课之后的每个 RL 算法——TD、Q学习、REINFORCE、PPO、GRPO——本质上都是蒙特卡洛估计器，有时在其上叠加了自举。

## 核心概念

![蒙特卡洛：展开，计算回报，取平均；首次访问 vs 每次访问](../assets/monte-carlo.svg)

**核心思想，一行公式：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`，其中 `G^{(i)}(s)` 是在策略 `π` 下访问状态 `s` 之后观测到的回报。

**首次访问 vs 每次访问 MC。** 给定一个多次访问状态 `s` 的片段，首次访问 MC 只计算第一次访问后的回报；每次访问 MC 计算所有访问。两者在极限情况下都是无偏的。首次访问更易分析（独立同分布样本）；每次访问每个片段使用更多数据，实践中通常收敛更快。

**增量均值。** 不用存储所有回报，而是更新运行均值：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重排：`V_new = V_old + α · (target - V_old)`，其中 `α = 1/n`。将 `1/n` 换成固定步长 `α ∈ (0, 1)`，就得到能跟踪 `π` 变化的非稳态 MC 估计器。这个转变就是从 MC 到 TD 再到每个现代 RL 算法的全部跨越。

**探索现在是个问题。** DP 通过枚举触及了每个状态。MC 只看到策略访问的状态。如果 `π` 是确定性的，整个状态空间中大片区域永远不会被采样，其值估计永远停在零。按历史顺序排列的三种修复方案：

1. **探索性起始。** 每个片段从随机的 (s, a) 对开始。保证覆盖，但实践中不现实（不能把机器人"重置"到任意状态）。
2. **ε-贪婪。** 对当前 Q 贪婪行动，但以概率 `ε` 选择随机动作。所有状态-动作对最终都会被渐近地采样。
3. **离策略 MC。** 在行为策略 `μ` 下收集数据，通过重要性采样学习目标策略 `π`。方差高，但它是通往 DQN 等回放缓冲区方法的桥梁。

**蒙特卡洛控制。** 评估 → 改进 → 评估，就像策略迭代一样，但评估是基于采样的：

1. 运行 `π`，得到一个片段。
2. 从观测回报更新 `Q(s, a)`。
3. 使 `π` 对 `Q` ε-贪婪。
4. 重复。

在温和条件下（每个配对被无限次访问，`α` 满足 Robbins-Monro），以概率 1 收敛到 `Q*` 和 `π*`。

## 动手实现

### 第一步：展开 → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

不需要模型，只需 `env.reset()` 和 `env.step(s, a)`。接口与 gym 环境相同，但更精简。

### 第二步：计算回报（反向扫描）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

单次扫描，`O(T)` 复杂度。反向递推 `G_t = r_{t+1} + γ G_{t+1}` 避免了重复求和。

### 第三步：首次访问 MC 评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

三行完成核心工作：首次访问时标记状态，增加计数，更新运行均值。

### 第四步：ε-贪婪 MC 控制（在策略）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 第五步：与 DP 金标准对比

你对 `V^π` 的 MC 估计随着片段数量趋向无穷大，应当收敛到第02课的 DP 结果。实践上：在 4×4 网格世界上运行 50,000 个片段，可以使结果与 DP 答案相差约 `0.1`。

## 常见陷阱

- **无限片段。** MC 要求片段能够*终止*。如果你的策略可以无限循环，设置 `max_steps` 上限，将超时视为隐式失败。随机策略的网格世界经常超时——这是正常的，但要确保正确计入。
- **高方差。** MC 使用完整回报。对于长片段，方差巨大——末尾的一个不幸奖励会以相同幅度影响 `V(s_0)`。TD 方法（第04课）通过自举来减少这个问题。
- **状态覆盖。** 在新的 Q 上执行贪婪 MC，遇到平局时只会尝试一个动作。你*必须*探索（ε-贪婪、探索性起始、UCB）。
- **非稳态策略。** 如果 `π` 在变化（如 MC 控制中），旧的回报来自不同的策略。固定步长 MC 能处理这种情况，而样本平均 MC 不能。
- **离策略重要性采样。** 权重 `π(a|s)/μ(a|s)` 沿轨迹相乘。随视野增大，方差爆炸。使用逐决策加权 IS 来限制，或切换到 TD。

## 工程应用

2026 年蒙特卡洛方法的作用：

| 使用场景 | 为何用 MC |
|---------|---------|
| 短视野游戏（21点、扑克） | 片段自然终止；回报干净清晰 |
| 离线评估已记录的策略 | 对存储轨迹上的折扣回报取平均 |
| 蒙特卡洛树搜索（AlphaZero） | 从树叶节点的 MC 展开指导选择 |
| LLM RL 评估 | 计算给定策略下采样完成的平均奖励 |
| PPO 中的基线估计 | 优势目标 `A_t = G_t - V(s_t)` 使用 MC 的 `G_t` |
| RL 教学 | 实际有效的最简单算法——去掉自举以看清核心 |

现代深度 RL 算法（PPO、SAC）通过 n 步回报或 GAE 在纯 MC（完整回报）和纯 TD（单步自举）之间插值。两个端点都是同一估计器的实例。

## 交付物

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## 练习

1. **（简单）** 对 4×4 网格世界的均匀随机策略实现首次访问 MC 评估。运行 10,000 个片段，以片段计数为横轴，绘制 `V(0,0)` 对比 DP 答案的曲线。
2. **（中等）** 用 `ε ∈ {0.01, 0.1, 0.3}` 实现 ε-贪婪 MC 控制。比较 20,000 个片段后的平均回报。曲线看起来如何？偏差-方差权衡在哪里体现？
3. **（困难）** 实现*离策略* MC 与重要性采样：在均匀随机策略 `μ` 下收集数据，估计确定性最优策略 `π` 的 `V^π`。比较普通 IS vs 逐决策 IS vs 加权 IS。哪个方差最小？

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 蒙特卡洛 (Monte Carlo) | "随机采样" | 通过对分布的独立同分布样本取平均来估计期望值。 |
| 回报 G_t (Return) | "未来奖励" | 从步骤 `t` 到片段结束的折扣奖励之和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| 首次访问 MC (First-visit MC) | "每个状态只计一次" | 片段中只有第一次访问对值估计有贡献。 |
| 每次访问 MC (Every-visit MC) | "利用所有访问" | 每次访问都有贡献；略有偏差但样本效率更高。 |
| ε-贪婪 (ε-greedy) | "探索噪声" | 以概率 `1-ε` 执行贪婪动作；以概率 `ε` 执行随机动作。 |
| 重要性采样 (Importance sampling) | "纠正从错误分布采样" | 用 `π(a\|s)/μ(a\|s)` 的乘积重加权回报，从 `μ` 数据估计 `V^π`。 |
| 在策略 (On-policy) | "从自己的数据学习" | 目标策略 = 行为策略。普通 MC、PPO、SARSA。 |
| 离策略 (Off-policy) | "从别人的数据学习" | 目标策略 ≠ 行为策略。重要性采样 MC、Q学习、DQN。 |

## 延伸阅读

- [Sutton & Barto (2018). 第5章 — 蒙特卡洛方法](http://incompleteideas.net/book/RLbook2020.pdf) — 权威处理
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — 首次访问 vs 每次访问分析
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — 离策略 MC 与方差控制
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — 现代低方差 IS 估计器
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — MC/TD 自对弈收敛到超人水平的第一次大规模实证展示；本阶段后半部分每课的概念前驱

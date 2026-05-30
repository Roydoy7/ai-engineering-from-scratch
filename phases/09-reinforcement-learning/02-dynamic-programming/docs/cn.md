# 动态规划——策略迭代与值迭代

> 动态规划是"开卷考试"版的强化学习。你已经知道转移函数和奖励函数，只需迭代贝尔曼方程直到 `V` 或 `π` 停止变化。它是每个基于采样的方法所要追赶的基准。

**类型：** 构建
**语言：** Python
**前置知识：** 第9阶段第01课（MDP）
**预计时间：** 约75分钟

## 问题背景

你有一个已知模型的 MDP：可以查询任意状态-动作对的 `P(s' | s, a)` 和 `R(s, a, s')`。库存管理器知道需求分布，棋盘游戏有确定性转移，网格世界只需四行 Python。你有*模型*。

无模型 RL（Q学习、PPO、REINFORCE）是为没有模型、只能从环境中采样的情况发明的。但当你确实有模型时，有更快、更好的方法：动态规划。贝尔曼在 1957 年设计了它们，至今仍定义着正确性——当人们说"这个 MDP 的最优策略"时，就是指 DP 会返回的策略。

2026 年你需要它们的三个原因：第一，RL 研究中的每个表格环境（GridWorld、FrozenLake、CliffWalking）都用 DP 求解以产生金标准策略。第二，精确的值函数让你能*调试*采样方法——如果 Q学习对 `V*(s_0)` 的估计与 DP 答案相差 30%，那 Q学习就有 bug。第三，现代离线 RL 和规划方法（MCTS、AlphaZero 的搜索、第9阶段第10课的基于模型的 RL）都在学习到的或给定的模型上迭代贝尔曼备份。

## 核心概念

![策略迭代与值迭代的并排对比](../assets/dp.svg)

**两种算法，都是对贝尔曼方程的不动点迭代。**

**策略迭代。** 交替进行两步，直到策略不再改变。

1. *评估：* 给定策略 `π`，通过反复应用 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` 直到收敛来计算 `V^π`。
2. *改进：* 给定 `V^π`，使 `π` 对 `V^π` 贪婪：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛是有保证的，因为：(a) 每次改进步骤要么保持 `π` 不变，要么在某个状态上严格增大 `V^π`；(b) 确定性策略的空间是有限的。即使对于较大的状态空间，通常也在约 5-20 次外层迭代内收敛。

**值迭代。** 将评估和改进合并为一次扫描。应用贝尔曼*最优性*方程：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复直到 `max_s |V_{new}(s) - V(s)| < ε`。最后通过取贪婪动作提取策略。每次迭代速度更快（无内层评估循环），但通常需要更多迭代次数才能收敛。

**广义策略迭代（GPI）。** 统一的框架。值函数和策略被锁在双向改进循环中；任何驱动两者朝向相互一致的方法（异步值迭代、修正策略迭代、Q学习、Actor-Critic、PPO）都是 GPI 的实例。

**为何 `γ < 1` 很重要。** 贝尔曼算子在上确界范数意义下是 `γ`-压缩的：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。压缩保证唯一不动点和几何收敛。去掉 `γ < 1`，保证就消失了——你需要有限视野或吸收终止状态。

## 动手实现

### 第一步：构建网格世界 MDP 模型

使用第01课相同的 4×4 网格世界。我们添加一个随机变体：以概率 `0.1` 智能体滑向随机的垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 返回 `(s', r, p)` 的列表，这就是完整的模型。

### 第二步：策略评估

给定策略 `π(s) = {动作: 概率}`，迭代贝尔曼方程直到 `V` 停止变化：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 第三步：策略改进

将 `π` 替换为对 `V` 贪婪的策略。如果 `π` 没有改变，返回——我们已达到最优。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 第四步：组合在一起

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # 任意初始策略
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4 网格世界的典型收敛：4-6 次外层迭代。输出 `V*(0,0) ≈ -6` 和严格减少步数的策略。

### 第五步：值迭代（单循环版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

相同的不动点，更少的代码。

## 常见陷阱

- **忘记处理终止状态。** 如果对吸收状态应用贝尔曼方程，它仍会选出一个什么也不改变的"最优动作"。用 `if s == terminal: V[s] = 0` 来保护。
- **上确界范数 vs L2 收敛。** 使用 `max |V_new - V|`，而不是平均值。理论保证是基于上确界范数的。
- **原地更新 vs 同步更新。** 原地更新 `V[s]`（高斯-赛德尔风格）比用单独的 `V_new` 字典（雅可比风格）收敛更快。生产代码使用原地更新。
- **策略平局。** 如果两个动作有相等的 Q 值，`argmax` 可能在每次迭代中以不同方式打破平局，导致"策略稳定"检查来回振荡。使用稳定的平局处理方式（固定顺序中的第一个动作）。
- **状态空间爆炸。** DP 每次扫描的复杂度是 `O(|S| · |A|)`，适用于约 10⁷ 个状态以内。超过此规模，需要函数近似（第9阶段第05课起）。

## 工程应用

2026 年，DP 是正确性基准和规划器的内循环：

| 使用场景 | 方法 |
|---------|------|
| 精确求解小型表格 MDP | 值迭代（更简单）或策略迭代（外层步骤更少） |
| 验证 Q学习/PPO 实现 | 与玩具环境上 DP 最优的 V* 对比 |
| 基于模型的 RL（第9阶段第10课） | 在学习到的转移模型上执行贝尔曼备份 |
| AlphaZero/MuZero 规划 | 蒙特卡洛树搜索 = 异步贝尔曼备份 |
| 离线 RL（CQL、IQL） | 保守 Q 迭代——带有 OOD 动作惩罚的 DP |

每当有人说"最优值函数"，就意味着"DP 的不动点"。当你在论文中看到 `V*` 或 `Q*` 时，脑海中浮现的应该是这个循环。

## 交付物

保存为 `outputs/skill-dp-solver.md`：

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## 练习

1. **（简单）** 对 4×4 网格世界用 `γ ∈ {0.9, 0.99}` 运行值迭代。达到 `max |ΔV| < 1e-6` 需要多少次扫描？以 4×4 网格形式打印 `V*`。
2. **（中等）** 在*随机性*网格世界（滑动概率 `0.1`）上比较策略迭代与值迭代。记录：扫描次数、挂钟时间、最终 `V*(0,0)`。哪个在迭代次数上收敛更快？在挂钟时间上呢？
3. **（困难）** 构建修正策略迭代：在评估步骤中，只运行 `k` 次扫描而不是运行到收敛。对 `k ∈ {1, 2, 5, 10, 50}` 绘制 `V*(0,0)` 误差 vs `k` 的曲线。曲线告诉你评估/改进权衡的什么信息？

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 策略迭代 (Policy iteration) | "DP 算法" | 交替进行评估（`V^π`）和改进（对 `V^π` 贪婪的 `π`），直到策略不再改变。 |
| 值迭代 (Value iteration) | "更快的 DP" | 在一次扫描中应用贝尔曼最优备份；几何收敛到 `V*`。 |
| 贝尔曼算子 (Bellman operator) | "那个递推" | `(T V)(s) = max_a Σ P (r + γ V(s'))`；在上确界范数下是 `γ`-压缩的。 |
| 压缩 (Contraction) | "DP 为何收敛" | 任何满足 `||T x - T y|| ≤ γ ||x - y||` 的算子都有唯一不动点。 |
| GPI | "一切都是 DP" | 广义策略迭代：任何驱动 `V` 和 `π` 朝向相互一致的方法。 |
| 同步更新 (Synchronous update) | "雅可比风格" | 在整个扫描中使用旧的 `V`；分析简洁但速度较慢。 |
| 原地更新 (In-place update) | "高斯-赛德尔风格" | 在更新过程中使用当前 `V`；实践中收敛更快。 |

## 延伸阅读

- [Sutton & Barto (2018). 第4章 — 动态规划](http://incompleteideas.net/book/RLbook2020.pdf) — 策略迭代和值迭代的权威讲解
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — 压缩映射论证的严格处理
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 修正策略迭代及其收敛分析
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 策略迭代的原始论文
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — 从 DP 到近似 DP/深度 RL 的桥梁，后续每课都用到

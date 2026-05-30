# 多智能体强化学习

> 单智能体 RL 假设环境是静止的。将两个学习中的智能体放入同一世界，这个假设就被打破了：每个智能体都是另一个的环境的一部分，而两者都在变化。多智能体 RL 是在马尔可夫假设不再成立时让学习收敛的一组技巧。

**类型：** 构建
**语言：** Python
**前置知识：** 第9阶段第04课（Q学习）、第9阶段第06课（REINFORCE）、第9阶段第07课（Actor-Critic）
**预计时间：** 约45分钟

## 问题背景

机器人学习在房间中导航是单智能体 RL 问题。足球队不是。AlphaStar vs StarCraft 对手不是。出价智能体的市场不是。两辆车在十字路口协商不是。多对多的现实问题不是。

在每个多智能体场景中，从任何单个智能体的视角来看，其他智能体*就是*环境的一部分。随着它们学习并改变行为，环境变成非稳态的。马尔可夫性质——"下一个状态只取决于当前状态和我的动作"——被违反了，因为下一个状态还取决于*其他*智能体的选择，而它们的策略是移动中的靶子。

这破坏了表格收敛证明（Q学习的保证假设环境稳定）。也破坏了朴素深度 RL：智能体相互追赶陷入循环，永远无法收敛到稳定策略。你需要多智能体专用技术：集中训练/分布式执行、反事实基线、联赛博弈、自对弈。

2026 年的应用：机器人群体、交通路由、自动驾驶车队、市场模拟器、多智能体 LLM 系统（第16阶段），以及任何有不止一个智能玩家的游戏。

## 核心概念

![四种 MARL 体制：独立、集中 Critic、自对弈、联赛](../assets/marl.svg)

**形式化：马尔可夫博弈。** MDP 的推广：状态 `S`，联合动作 `a = (a_1, …, a_n)`，转移 `P(s' | s, a)`，以及每个智能体的奖励 `R_i(s, a, s')`。每个智能体 `i` 在自己的策略 `π_i` 下最大化自己的回报。如果奖励相同，是**完全合作**的。如果零和，是**对抗性**的。如果混合，是**一般和**的。

**核心挑战：**

- **非稳态性。** 从智能体 `i` 的视角，`P(s' | s, a_i)` 取决于 `π_{-i}`，而后者在变化。
- **信用分配。** 共享奖励时，是哪个智能体造成的？
- **探索协调。** 智能体必须探索互补策略，而不是重复探索相同状态。
- **可扩展性。** 联合动作空间随 `n` 指数增长。
- **部分可观测性。** 每个智能体只看到自己的观测；全局状态是隐藏的。

**四种主导体制：**

**1. 独立 Q学习 / 独立 PPO（IQL、IPPO）。** 每个智能体学习自己的 Q 或策略，将其他智能体视为环境的一部分。简单，有时有效（特别是经验回放起到平滑智能体建模作用时）。理论收敛：无。实践中：对松耦合任务没问题，对紧耦合任务很差。

**2. 集中训练/分布式执行（CTDE）。** 最常见的现代范式。每个智能体有自己的*策略* `π_i`，以局部观测 `o_i` 为条件——部署时标准的分布式执行。*训练*期间，集中的 Critic `Q(s, a_1, …, a_n)` 以完整的全局状态和联合动作为条件。示例：
- **MADDPG**（Lowe et al. 2017）：每个智能体带集中 Critic 的 DDPG。
- **COMA**（Foerster et al. 2017）：反事实基线——问"如果我采取动作 `a'` 代替，我的奖励会是多少？"——隔离我的贡献。
- **MAPPO** / 带共享 Critic 的 **IPPO**（Yu et al. 2022）：带集中值函数的 PPO。2026 年合作 MARL 的主导方法。
- **QMIX**（Rashid et al. 2018）：值分解——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`，带单调混合。

**3. 自对弈。** 同一智能体的两个副本相互对弈。对手的策略*就是*我过去某个快照的策略。AlphaGo / AlphaZero / MuZero，OpenAI Five。对零和博弈效果最好；训练信号对称。

**4. 联赛博弈。** 自对弈对一般和/对抗性环境的扩展：保留一组过去和当前策略，从联赛中采样对手进行训练。添加利用者（专门击败当前最优）和主利用者（专门击败利用者）。AlphaStar（星际争霸 II）。当博弈存在"石头剪刀布"式策略循环时需要。

**通信。** 允许智能体相互发送学习到的消息 `m_i`。适用于合作场景。Foerster et al.（2016）证明了可微的智能体间通信可以端到端训练。今天基于 LLM 的多智能体系统（第16阶段）本质上是用自然语言通信。

## 动手实现

本课使用一个有两个合作智能体的 6×6 网格世界。它们从对角出发，必须到达共同目标。共享奖励：任一智能体还在移动时每步 `-1`，两者都到达时 `+10`。见 `code/main.py`。

### 第一步：多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # 两个智能体

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间是 `|A|² = 16`。全局状态是两个位置。

### 第二步：独立 Q学习

每个智能体运行自己以联合状态为键的 Q 表。每步：两者都选择 ε-贪婪动作，收集联合转移，各自用共享奖励更新自己的 Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

在这个任务上有效，因为奖励密集且一致。在紧耦合任务（例如一个智能体必须*等待*另一个）上会失败。

### 第三步：带分解值更新的集中 Q

在联合动作 `Q(s, a_1, a_2)` 上使用一个 Q。用共享奖励更新。通过边缘化实现分布式执行：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。用指数级联合动作空间换来*正确*的全局视角。

### 第四步：简单自对弈（对抗性两智能体）

相同智能体，两个角色。训练智能体 A 对抗智能体 B；每 `K` 个片段后，将 A 的权重复制到 B。对称训练，一致进步。AlphaZero 配方的微型版本。

## 常见陷阱

- **非稳态回放。** 独立智能体的经验回放比单智能体更差，因为旧转移是由现在已过时的对手生成的。修复：重新标注或按近期性加权。
- **信用分配歧义。** 长片段后的共享奖励；无法明确说明哪个智能体做了贡献。修复：反事实基线（COMA），或按智能体进行奖励塑造。
- **策略漂移/追赶。** 每个智能体的最优响应随其他智能体的更新而改变。修复：集中 Critic、慢学习率，或交替固定更新。
- **通过协调进行奖励黑客。** 智能体找到设计者未预料到的协调利用。拍卖智能体收敛到零出价。修复：仔细的奖励设计、行为约束。
- **探索冗余。** 两个智能体探索相同的状态-动作对。修复：每个智能体的熵奖励，或角色条件化。
- **联赛循环。** 纯自对弈可能陷入支配循环。修复：带多样化对手的联赛博弈。
- **样本爆炸。** `n` 个智能体 × 状态空间 × 联合动作。用函数近似来近似；分解动作空间（每个智能体一个策略输出头）。

## 工程应用

2026 年的 MARL 应用图谱：

| 领域 | 方法 | 备注 |
|-----|------|------|
| 合作导航/操控 | MAPPO / QMIX | CTDE；共享 Critic + 分布式 Actor |
| 双人博弈（象棋、围棋、扑克） | 带 MCTS 的自对弈（AlphaZero） | 零和；对称训练 |
| 复杂多人游戏（Dota、星际争霸） | 联赛博弈 + 模仿预训练 | OpenAI Five、AlphaStar |
| 自动驾驶车队 | 带注意力的 CTDE MAPPO / PPO | 部分可观测；可变团队规模 |
| 拍卖市场 | 博弈论均衡 + RL | `n → ∞` 时的平均场 RL |
| LLM 多智能体系统（第16阶段） | 自然语言通信 + 角色条件化 | RL 循环在智能体规划层 |

2026 年，MARL 最大的增长领域是基于 LLM 的：成群的语言模型智能体相互协商、辩论、构建软件。RL 以对*轨迹级*输出的偏好优化形式出现，而非 token 级（第16阶段第03课）。

## 交付物

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with n agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 练习

1. **（简单）** 在 2 智能体合作网格世界上训练独立 Q学习。需要多少个片段使平均回报 > 0？绘制联合学习曲线。
2. **（中等）** 添加"协调"任务：只有两个智能体在同一回合踩上目标时才算达到。独立 Q 还能收敛吗？什么地方崩溃了？
3. **（困难）** 为 MAPPO 风格训练实现集中 Critic，并在协调任务上与独立 PPO 比较收敛速度。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 马尔可夫博弈 (Markov game) | "多智能体 MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个智能体有自己的奖励。 |
| CTDE | "集中训练，分布式执行" | 训练时使用联合 Critic；每个智能体的策略只使用局部观测。 |
| IPPO | "独立 PPO" | 每个智能体独立运行 PPO。简单基线；常被低估。 |
| MAPPO | "多智能体 PPO" | 带以全局状态为条件的集中值函数的 PPO。 |
| QMIX | "单调值分解" | `Q_tot = f_单调(Q_1, …, Q_n)`，允许分布式 argmax。 |
| COMA | "反事实多智能体" | 优势 = 我的 Q 减去对我的动作边缘化后的期望 Q。 |
| 自对弈 (Self-play) | "智能体 vs 过去的自己" | 单智能体，两个角色；零和博弈的标准方法。 |
| 联赛博弈 (League play) | "种群训练" | 缓存过去的策略，从池中采样对手；处理策略循环。 |

## 延伸阅读

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 带集中 Critic 的 CTDE
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — 信用分配的反事实基线
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 带单调性的值分解
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO 在 MARL 中出人意料的强大表现
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 大规模联赛博弈
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 零和博弈中的纯自对弈
- [Sutton & Barto (2018). 第15章和第17章](http://incompleteideas.net/book/RLbook2020.pdf) — 教材对多智能体场景和 CTDE 旨在解决的非稳态性问题的简要处理
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 涵盖合作、竞争和混合 MARL 及收敛结果的综述

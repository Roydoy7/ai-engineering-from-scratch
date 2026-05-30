# 深度 Q 网络（DQN）

> 2013年：Mnih 在原始像素上训练了一个 Q学习网络，在七款 Atari 游戏上击败了所有经典 RL 智能体。2015年：扩展到 49 款游戏，发表在 Nature 上，开启了深度 RL 时代。DQN 是 Q学习加三个让函数近似稳定的技巧。

**类型：** 构建
**语言：** Python
**前置知识：** 第3阶段第03课（反向传播）、第9阶段第04课（Q学习、SARSA）
**预计时间：** 约75分钟

## 问题背景

表格 Q学习需要为每个（状态，动作）对存储一个独立的 Q 值。象棋棋盘有约 10⁴³ 种状态。Atari 游戏帧是 210×160×3 = 100,800 个特征。表格 RL 在状态数达到数千时就失效了，更别说数十亿。

修复方法事后看来显而易见：用神经网络 `Q(s, a; θ)` 替换 Q 表。但显而易见的东西花了数十年才实现。在 Q学习中使用朴素函数近似会在"致命三角"下发散——函数近似 + 自举 + 离策略学习。Mnih et al.（2013、2015）发现了三个工程技巧来稳定学习：

1. **经验回放**使转移解相关。
2. **目标网络**冻结自举目标。
3. **奖励裁剪**归一化梯度幅度。

DQN 在 Atari 上是第一次用同一架构、同一套超参数从原始像素解决数十种控制问题。此后构建的所有"深度 RL"——DDQN、Rainbow、对抗网络、分布式、R2D2、Agent57——都堆叠在这三个技巧的基础上。

## 核心概念

![DQN 训练循环：环境、回放缓冲区、在线网络、目标网络、贝尔曼 TD 损失](../assets/dqn.svg)

**目标函数。** DQN 最小化神经 Q 函数上的一步 TD 损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络，每步通过梯度下降更新。`θ^-` = 目标网络，定期从 `θ` 复制（约每 10,000 步）。`D` = 过去转移的回放缓冲区。

**按重要性排列的三个技巧：**

**经验回放。** 约 10⁶ 次转移的环形缓冲区。每个训练步骤随机均匀采样一个 minibatch。这打破了时间相关性（连续帧几乎相同），让网络能多次从稀有的有奖励转移中学习，并使连续梯度更新解相关。没有它，Atari 上带神经网络的在策略 TD 会发散。

**目标网络。** 在贝尔曼方程两侧使用同一网络 `Q(·; θ)` 会使目标随每次更新而移动——"追着自己的尾巴跑"。修复方案：保留一个权重冻结的第二网络 `Q(·; θ^-)`。每 `C` 步复制 `θ → θ^-`。这使回归目标在数千次梯度步骤内保持稳定。软更新 `θ^- ← τ θ + (1-τ) θ^-`（用于 DDPG、SAC）是更平滑的变体。

**奖励裁剪。** Atari 奖励幅度从 1 到 1000+ 不等。裁剪到 `{-1, 0, +1}` 防止任何单个游戏主导梯度。当奖励幅度本身有意义时不适用；对于只有符号重要的 Atari 来说是合适的。

**双重 DQN（DDQN）。** Hasselt（2016）修复了最大化偏差：用在线网络*选择*动作，目标网络*评估*它。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

即插即用的替换，始终更好。生产中默认使用。

**其他改进（Rainbow，2017）：** 优先级回放（更频繁地采样高 TD 误差的转移）、对抗架构（分离 `V(s)` 和优势头）、噪声网络（学习探索）、n 步回报、分布式 Q（C51/QR-DQN）、多步自举。每个贡献几个百分点；收益大致可叠加。

## 动手实现

这里的代码只用标准库，无需 numpy——我们在小型连续网格世界上使用手写的单隐藏层 MLP，因此每个训练步骤在微秒内完成。算法与规模扩展到 Atari 的 DQN 完全相同。

### 第一步：回放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 需要约 50,000 容量；我们的玩具环境 5,000 就够了。

### 第二步：小型 Q 网络（手动 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：线性 → ReLU → 线性。这就是完整的网络。

### 第三步：DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

形式与第04课的 Q学习相同，有两点区别：(a) 通过可微的 `Q(·; θ)` 反向传播，而不是索引表格；(b) 目标使用 `Q(·; θ^-)`。

### 第四步：外层循环

每个片段，对 `Q(·; θ)` ε-贪婪行动，将转移推入缓冲区，采样 minibatch，进行梯度步骤，定期同步 `θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们的 16 维独热状态小型网格世界上，智能体在约 500 个片段内学到接近最优的策略。在 Atari 上，扩展到 2 亿帧并添加 CNN 特征提取器。

## 常见陷阱

- **致命三角。** 函数近似 + 离策略 + 自举可能导致发散。DQN 通过目标网络 + 回放来缓解；不要移除任何一个。
- **探索。** ε 必须衰减，通常在前约 10% 的训练中从 1.0 到 0.01。没有足够的早期探索，Q 网络会收敛到局部极小值。
- **过估计。** 对有噪声的 Q 取 `max` 向上偏置。生产中始终使用双重 DQN。
- **奖励尺度。** 裁剪或归一化奖励；梯度幅度与奖励幅度成正比。
- **回放缓冲区冷启动。** 在缓冲区有几千个转移之前不要训练。在约 20 个样本上的早期梯度会过拟合。
- **目标同步频率。** 太频繁 ≈ 没有目标网络；太不频繁 ≈ 目标过时。Atari DQN 每 10,000 步同步一次。经验法则：每约 1/100 的训练视野同步一次。
- **观测预处理。** Atari DQN 叠 4 帧使状态满足马尔可夫性。任何需要速度信息的环境都需要叠帧或循环状态。

## 工程应用

2026 年，DQN 很少是最先进的，但仍是离策略算法的参考实现：

| 任务 | 首选方法 | 为何不用 DQN？ |
|------|---------|-------------|
| 离散动作的 Atari 类 | Rainbow DQN 或 Muesli | 同一框架，更多技巧 |
| 连续控制 | SAC / TD3（第9阶段第07课） | DQN 没有策略网络 |
| 在策略 / 高吞吐量 | PPO（第9阶段第08课） | 无回放缓冲区；更易扩展 |
| 离线 RL | CQL / IQL / Decision Transformer | 保守 Q 目标，无自举爆炸 |
| 大型离散动作空间（推荐系统） | 带动作嵌入的 DQN，或 IMPALA | 可以用；细节很重要 |
| LLM RL | PPO / GRPO | 序列级而非步级；不同的损失 |

这些教训依然通用。回放和目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的自对弈缓冲区以及每个离线 RL 方法中。奖励裁剪在 PPO 中以优势归一化的形式延续。这个架构就是蓝图。

## 交付物

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 练习

1. **（简单）** 运行 `code/main.py`。绘制每片段回报曲线。需要多少个片段，运行均值才能超过 -10？
2. **（中等）** 禁用目标网络（贝尔曼目标两侧都使用在线网络）。测量训练不稳定性——回报是否振荡或发散？
3. **（困难）** 添加双重 DQN：用在线网络选择 `argmax a'`，目标网络评估。在有噪声奖励的网格世界上，比较 1000 个片段后有无双重 DQN 时 `Q(s_0, best_a)` 对 `V*(s_0)` 的偏差。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| DQN | "深度 Q学习" | 带神经 Q 函数、回放缓冲区和目标网络的 Q学习。 |
| 经验回放 (Experience replay) | "打乱的转移" | 每次梯度步骤均匀采样的环形缓冲区；使数据解相关。 |
| 目标网络 (Target network) | "冻结的自举" | 贝尔曼目标中使用的 Q 的周期性副本；稳定训练。 |
| 致命三角 (Deadly triad) | "RL 为何发散" | 函数近似 + 自举 + 离策略 = 无收敛保证。 |
| 双重 DQN (Double DQN) | "最大化偏差的修复" | 在线网络选择动作，目标网络评估。 |
| 对抗 DQN (Dueling DQN) | "V 和 A 两个头" | 分解 Q = V + A - mean(A)；相同输出，更好的梯度流。 |
| Rainbow | "所有技巧的集合" | DDQN + PER + 对抗 + n步 + 噪声 + 分布式，合为一体。 |
| PER | "优先级回放" | 按 TD 误差幅度的比例采样转移。 |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 启动深度 RL 的 2013 年 NeurIPS 研讨论文
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature 论文，49 款游戏的 DQN
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — 双重 DQN
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — 对抗 DQN
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — 技巧叠加论文
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 清晰的现代讲解
- [Sutton & Barto (2018). 第9章 — 带近似的在策略预测](http://incompleteideas.net/book/RLbook2020.pdf) — "致命三角"的教材处理（目标网络和回放缓冲区正是为了驯服它而设计）
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 用于消融研究的参考单文件 DQN；适合与本课的从头实现对照阅读

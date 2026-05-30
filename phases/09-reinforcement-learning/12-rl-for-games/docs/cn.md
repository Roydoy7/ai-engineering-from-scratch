# 博弈强化学习——AlphaZero、MuZero 与 LLM 推理时代

> 1992年：TD-Gammon 用纯 TD 在西洋双陆棋上击败人类冠军。2016年：AlphaGo 击败李世石。2017年：AlphaZero 从头主宰国际象棋、将棋和围棋。2024年：DeepSeek-R1 证明同样的配方——用 GRPO 替换 PPO——在推理上同样有效。游戏是推动本阶段每项突破的基准。

**类型：** 构建
**语言：** Python
**前置知识：** 第9阶段第05课（DQN）、第9阶段第08课（PPO）、第9阶段第09课（RLHF）、第9阶段第10课（多智能体 RL）
**预计时间：** 约120分钟

## 问题背景

游戏拥有 RL 想要的一切。清晰的奖励（胜/负）。无限片段（自对弈重置）。完美仿真（游戏本身就是仿真器）。离散或小型连续动作空间。强制对抗鲁棒性的多智能体结构。

游戏是每项重大 RL 突破的测试场地。TD-Gammon（西洋双陆棋，1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2，2019）。AlphaStar（星际争霸 II，2019）。MuZero（学习模型，2019）。AlphaTensor（矩阵乘法，2022）。AlphaDev（排序算法，2023）。DeepSeek-R1（数学推理，2025）——最新展示博弈 RL 技术在文本上同样有效。

这个收官课通过一个统一视角——**自对弈 + 搜索 + 策略改进**——综述三个里程碑架构：AlphaZero、MuZero 和 GRPO。每个都推广了前一个；GRPO 尤其是将 AlphaZero 的配方应用于 LLM 推理，以 token 为动作，数学验证为胜利信号。

## 核心概念

![AlphaZero ↔ MuZero ↔ GRPO：相同的循环，不同的环境](../assets/rl-games.svg)

**统一循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # 自对弈游戏
    policy_target = search.improved_policy(trajectory) # 搜索改进原始策略
    policy_net.update(policy_target, value_target)     # 监督学习搜索输出
```

**AlphaZero（2017）。** Silver et al. 给定已知规则的游戏（国际象棋、将棋、围棋）：

- 策略-值网络：一个主干 `f_θ(s) → (p, v)`。`p` 是合法动作的先验。`v` 是预期的游戏结果。
- 蒙特卡洛树搜索（MCTS）：每次落子时，展开可能延续的树。用 `(p, v)` 作为先验 + 自举。通过 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自对弈：智能体之间对弈。在第 `t` 步，MCTS 访问分布 `π_t` 成为策略训练目标。
- 损失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是游戏结果（+1 / 0 / -1）。

零人类知识。零手工启发式。一个配方，各在数千万次自对弈游戏后主宰国际象棋、将棋和围棋。

**MuZero（2019）。** Schrittwieser et al. 去除了必须知道规则的要求。

- 不需要固定环境，而是学习一个*潜在动力学模型* `(h, g, f)`：
  - `h(s)`：将观测编码为潜在状态。
  - `g(s_latent, a)`：预测下一个潜在状态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 值。
- MCTS 在*学习到的潜在空间*中运行。相同的搜索，相同的训练循环。
- 适用于围棋、国际象棋、将棋*和* Atari——一个算法，不需要规则知识。

**随机 MuZero（2022）。** 添加随机动力学和机会节点；扩展到西洋双陆棋类游戏。

**Muesli、Gumbel MuZero（2022–2024）。** 在样本效率和确定性搜索上的改进。

**GRPO（2024–2025）。** DeepSeek-R1 配方。相同的 AlphaZero 形状循环，应用于语言模型推理：

- "游戏"：回答数学/编程/推理问题。"胜利" = 验证器（测试用例通过，数值答案匹配）返回 1。
- 策略：LLM。动作：token。状态：提示词 + 到目前为止的回复。
- 无 Critic（PPO 风格的 V_φ）。对每个提示词，从策略采样 `G` 个完成。计算每个的奖励。用**群体相对优势** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 风格更新的信号。
- 对参考策略的 KL 惩罚防止漂移（与 RLHF 类似）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

无奖励模型，无 Critic，无 MCTS。群体相对基线替代了所有三者。在推理基准上以 PPO-RLHF 质量的一小部分计算量匹配或超越后者。

**完整的 R1 配方。** DeepSeek-R1（DeepSeek 2025）是一篇论文中的两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基础模型开始。无 SFT。直接用 GRPO 和两个奖励组件：*准确性奖励*（基于规则——最终答案是否解析到正确数字/代码是否通过单元测试）和*格式奖励*（完成是否将思维链包裹在 `<think>…</think>` 标签中）。经过数千步，平均响应长度从约 100 增长到约 10,000 个 token，数学基准分数攀升至接近 o1-preview 水平。模型从头学会推理。缺点：其思维链通常不可读，混合语言，缺乏风格打磨。
- **R1。** 用四阶段流水线修复 R1-Zero 的可读性问题：
  1. **冷启动 SFT。** 收集几千个格式整洁的长链式思维演示。在其上对基础模型进行监督微调。提供可读的起始点。
  2. **以推理为导向的 GRPO。** 用准确性+格式奖励以及*语言一致性*奖励应用 GRPO，防止代码混用。
  3. **拒绝采样 + 第二轮 SFT。** 从 RL 检查点采样约 60 万条推理轨迹，只保留最终答案正确且链式思维可读的那些，并与约 20 万条非推理 SFT 示例（写作、问答、自我认知）合并。再次微调基础模型。
  4. **全谱 GRPO。** 最后一轮 RL，涵盖推理（基于规则的奖励）和通用对齐（基于有帮助/无害偏好的奖励）。

结果在开放权重下在 AIME 和 MATH-500 上媲美 o1，且小到足以蒸馏。同一篇论文还发布了六个蒸馏的密集模型（Qwen-1.5B 到 Llama-70B），通过在 R1 的推理轨迹上进行 SFT——学生不需要 RL。强大 RL 教师的蒸馏在学生规模上始终优于从头开始的 RL。

**为何对推理用 GRPO 而非 PPO。** DeepSeekMath 论文（2024 年 2 月）中的三个原因：(1) 不需要训练值网络，内存减半；(2) 群体基线自然处理推理任务产生的稀疏端到轨迹奖励；(3) 每提示词的归一化使各种难度问题的优势可比，而 PPO 的单个 Critic 做不到这一点。

**无搜索 vs 基于搜索。** 游戏已经分叉：

- *长视野完全信息博弈*（围棋、国际象棋）：仍然基于搜索。AlphaZero / MuZero 占主导。
- *LLM 推理*：目前无 MCTS 生产；完整展开的 GRPO，推理计算的 Best-of-N。过程奖励模型（PRMs）暗示步骤级搜索正在重新加入。

## 动手实现

`code/main.py` 中的代码实现了**微型 GRPO**——一个有多组样本的 bandit。算法与在 LLM 上的完全相同；只有策略和环境更简单。它教授*损失*和*群体相对优势*，这是 2025 年的创新。

### 第一步：小型验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真实 GRPO 中，验证器运行单元测试或检查数学等式。

### 第二步：策略：每个提示词 K 个答案 token 的 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于以提示词为条件的 LLM 的最终层输出。

### 第三步：群体采样和群体相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL 惩罚：将 theta 拉向参考策略
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

群体相对优势是 2024 年 DeepSeek 的技巧。不需要 Critic。"基线"是群体均值，归一化使用群体标准差。

### 第四步：与无值函数的 REINFORCE 基线比较

相同设置，相同计算，普通 REINFORCE。GRPO 收敛更快更稳定。

### 第五步：观察熵和 KL

与 RLHF 相同的诊断：对参考策略的均值 KL、策略熵、随时间变化的奖励。一旦这些稳定，训练就完成了。

## 常见陷阱

- **通过验证器博弈进行奖励黑客。** GRPO 继承了 RLHF 的风险：如果验证器有误或可被利用，LLM 会找到漏洞。鲁棒的验证器（多个测试用例、形式证明）很重要。
- **群体大小太小。** 群体基线的方差类似于 `1/√G`。`G < 4` 时，优势信号有噪声；标准选择是 `G = 8` 到 `64`。
- **长度偏差。** 不同长度的 LLM 完成有不同的对数概率。按 token 数归一化，或使用序列级对数概率，或截断到最大长度。
- **纯自对弈循环。** AlphaZero 风格的训练在一般和博弈上可能陷入支配循环。通过多样化对手池（联赛博弈，第10课）缓解。
- **搜索-策略不匹配。** AlphaZero 训练策略模仿搜索输出。如果策略网络太小无法表示搜索的分布，训练就会停滞。
- **计算门槛。** MuZero / AlphaZero 需要大量计算。一次消融通常需要数百 GPU 时。存在微型演示（如四子棋上的 AlphaZero）用于学习。
- **验证器覆盖率。** 通过有 bug 解决方案的单元测试会强化 bug。设计能捕获边缘情况的验证器。

## 工程应用

2026 年的博弈 RL 技术图谱，按领域：

| 领域 | 主导方法 |
|-----|---------|
| 两人零和棋盘游戏（围棋、国际象棋、将棋） | AlphaZero / MuZero / KataGo |
| 不完全信息纸牌游戏（扑克） | CFR + 深度学习（DeepStack、Libratus、Pluribus） |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多人策略游戏（Dota、星际争霸） | PPO + 自对弈 + 联赛（OpenAI Five、AlphaStar） |
| LLM 数学/代码推理 | GRPO（DeepSeek-R1、Qwen-RL、开放复现） |
| LLM 对齐 | DPO / RLHF-PPO（非 GRPO；验证器是偏好而非可验证的） |
| 机器人 | PPO + DR（非博弈 RL，但使用相同策略梯度工具） |
| 组合优化问题 | AlphaZero 变体（AlphaTensor、AlphaDev） |

*配方*——自对弈、搜索增强改进、策略蒸馏——跨越文本、像素和物理控制。GRPO 是最新的实例；更多即将到来。

## 交付物

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 练习

1. **（简单）** 在 `code/main.py` 中实现 GRPO bandit。在 2 个提示词 × 每个 4 个答案 token 上训练。用 `G=8` 在 < 1000 次更新内收敛。
2. **（中等）** 插入 PPO（裁剪）和普通 REINFORCE。在相同 bandit 上与 GRPO 比较样本效率和奖励方差。
3. **（困难）** 扩展到长度为 2 的"推理链"：智能体发出两个 token，验证器对这对 token 给出奖励。测量 GRPO 如何处理两步序列的信用分配。（提示：按*完整序列*计算群体优势，传播到两个 token 位置。）

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| MCTS | "带学习网络的树搜索" | 蒙特卡洛树搜索；带学习 `(p, v)` 先验的 UCB1/PUCT 选择。 |
| AlphaZero | "自对弈 + MCTS" | 训练以匹配 MCTS 访问次数和游戏结果的策略-值网络。 |
| MuZero | "学习模型的 AlphaZero" | 相同循环，但通过学习动力学在潜在空间中运行。 |
| GRPO | "无 Critic 的 PPO" | 群体相对策略优化；带群体均值基线 + KL 的 REINFORCE。 |
| PUCT | "AlphaZero 的 UCB" | `Q + c · p · √N / (1 + N_a)`——平衡值估计与先验。 |
| 自对弈 (Self-play) | "智能体 vs 过去的自己" | 零和博弈的标准；对称训练信号。 |
| 联赛博弈 (League play) | "基于种群的自对弈" | 过去 + 当前 + 利用者作为对手采样。 |
| 验证器奖励 (Verifier reward) | "可验证 RL" | 奖励来自确定性检查器（测试通过，答案匹配）。 |
| 过程奖励 (Process reward) | "PRM" | 对每个推理步骤评分，而非只对最终答案。 |

## 延伸阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4)
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — 引入 GRPO 和群体相对基线的论文
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 完整四阶段 R1 配方以及 R1-Zero 消融
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — 大规模 CFR + 深度学习
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 开创一切的论文
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 用自定义奖励函数应用 GRPO 的生产参考
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — 多种规模下 R1 配方的开放复现
- [Sutton & Barto (2018). 第17章 — 强化学习的前沿](http://incompleteideas.net/book/RLbook2020.pdf) — 教材对 R1 在 LLM 规模上实例化的自对弈、搜索和"设计奖励"的框架

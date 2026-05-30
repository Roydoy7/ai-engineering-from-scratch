# 仿真到真实迁移

> 在仿真器中训练但在硬件上失败的策略，是一个记住了仿真器的策略。域随机化、域适应和系统辨识是让学习到的控制器跨越现实差距的三个工具。

**类型：** 学习
**语言：** Python
**前置知识：** 第9阶段第08课（PPO）、第2阶段第10课（偏差/方差）
**预计时间：** 约45分钟

## 问题背景

训练真实机器人既慢、危险又昂贵。双足机器人需要数百万个训练片段才能学会行走；真实的双足机器人一旦摔倒就会损坏硬件。仿真提供了无限次重置、确定性可复现性、并行环境，以及无物理损伤。

但仿真器是不准确的。轴承摩擦力比 MuJoCo 模型更大。相机有仿真器未包含的镜头畸变。电机有延迟、间隙和饱和，99% 的仿真模型都跳过了这些。风、灰尘和可变光照会破坏在无菌渲染上训练的策略。**现实差距**——仿真分布与真实分布之间的系统性差异——是机器人部署 RL 的核心问题。

你需要一个*对仿真到真实分布偏移具有鲁棒性*的策略。三种历史方法：随机化仿真器（域随机化）、用少量真实数据适应策略（域适应/微调），或辨识真实系统参数并匹配它（系统辨识）。2026 年，主导配方将这三者与大规模并行仿真（Isaac Sim、Isaac Lab、GPU 上的 MuJoCo MJX）结合在一起。

## 核心概念

![三种仿真到真实体制：域随机化、适应、系统辨识](../assets/sim-to-real.svg)

**域随机化（DR）。** Tobin et al. 2017，Peng et al. 2018。训练期间，随机化真实机器人上可能不同的每个仿真参数：质量、摩擦系数、电机 PD 增益、传感器噪声、相机位置、光照、纹理、接触模型。策略学习"今天在哪个仿真中"的条件分布，并在整个范围内泛化。如果真实机器人落在训练包络内，策略就有效。

- **优点：** 不需要真实数据。一种配方，多种机器人。
- **缺点：** 过度随机化的训练产生"通用"但过于保守的策略。噪声太多 ≈ 正则化太多。

**系统辨识（SI）。** 在训练前将仿真器的参数拟合到真实世界数据。如果你能测量真实机器人的关节摩擦力，就将其插入仿真。然后训练一个期望这些值的策略。需要访问真实系统，但直接减少现实差距。

- **优点：** 精确，低噪声的训练目标。
- **缺点：** 残差模型误差对策略不可见；未辨识的小影响（如电机死区）仍然会破坏部署。

**域适应。** 在仿真中训练，用少量真实数据微调。两种方式：

- **Real2Sim2Real：** 使用真实展开学习残差仿真器 `f(s, a, z) - f_sim(s, a)`，在修正后的仿真中训练。用不多的真实数据弥合差距。
- **观测适应：** 训练一个通过学习到的特征提取器将真实观测 → 仿真样式观测的策略（如 GAN 像素到像素）。控制器保持在仿真中。

**特权学习/师生蒸馏。** Miki et al. 2022（ANYmal 四足机器人）。在仿真中训练一个能访问特权信息的*教师*（地面真实摩擦、地形高度、IMU 漂移）。将*学生*蒸馏为只看真实传感器观测。学生学会从历史中推断特权特征，在物理参数间具有鲁棒性。

**大规模并行仿真。** 2024–2026。Isaac Lab、MuJoCo MJX、Brax 在单个 GPU 上运行数千个并行机器人。带 4096 个并行人形机器人的 PPO 在数小时内积累数年的经验。随着训练分布扩大，"现实差距"缩小；当这 4096 个环境各有不同随机化参数时，DR 几乎是免费的。

**2026 年真实世界配方（四足机器人行走示例）：**

1. 带域随机化重力、摩擦力、电机增益、有效载荷的大规模并行仿真。
2. 用特权信息训练教师策略（地形图、体速度地面真实值）。
3. 仅使用本体感知（腿部关节编码器）从教师蒸馏学生策略。
4. 可选的通过真实 IMU 自动编码器进行观测适应。
5. 部署。在 10+ 种环境中零样本泛化。如果失败，用安全约束 PPO 进行数分钟真实世界微调。

## 动手实现

本课代码是在有*噪声*转移的网格世界上域随机化的小型演示。我们训练一个在"仿真"中经历随机化滑动概率的策略，并在"真实"中以训练期间从未见过的滑动水平评估。该形状直接映射到 MuJoCo 到硬件的迁移。

### 第一步：参数化仿真

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是仿真器暴露的参数。在真实机器人中，它可以是摩擦力、质量、电机增益——任何在仿真和真实之间偏移的东西。

### 第二步：用 DR 训练

在每个片段开始时，采样 `slip ~ Uniform[0.0, 0.4]`。训练 PPO / Q学习 / 任何算法。对许多片段这样做。

### 第三步：在"真实"滑动上零样本评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个在训练支持内；`0.5` 和 `0.7` 在外。DR 训练的策略应该在支持内保持接近最优，在支持外优雅退化。固定滑动训练的策略在其训练滑动之外会很脆弱。

### 第四步：与窄训练比较

用 `slip = 0.0` 单独训练第二个策略。在相同的 `slip` 扫描上评估。你应该看到一旦真实滑动 > 0 就出现灾难性下降。

## 常见陷阱

- **随机化过多。** 在 `slip ∈ [0, 0.9]` 上训练，策略会如此厌恶风险以至于从不尝试最优路径。匹配*预期*的真实世界分布，而不是"什么都可能发生"。
- **随机化不足。** 在薄片段上训练，策略根本无法泛化。使用自适应课程（自动域随机化），随着策略改进扩宽分布。
- **错误辨识参数空间。** 随机化错误的东西（当真实差距是电机延迟时随机化相机色调），DR 没有帮助。首先分析真实机器人。
- **特权信息泄露。** 使用全局状态进行动作（而非仅观测）的教师可能产生学生无法赶上的情况。确保教师的策略在给定观测历史的情况下对学生是可实现的。
- **仿真到仿真迁移失败。** 如果你的策略对更难的仿真变体不具有鲁棒性，对真实世界也不会。在部署前始终在留出的仿真变体上测试。
- **无真实世界安全包络。** 在仿真中有效并在真实中"有效"而没有低层安全屏障的策略仍然可能损坏硬件。在非学习控制器中添加速率限制、力矩限制、关节限制。

## 工程应用

2026 年的仿真到真实技术栈：

| 领域 | 技术栈 |
|-----|--------|
| 腿式运动（ANYmal、Spot、人形） | Isaac Lab + DR + 特权教师/学生 |
| 操控（灵巧手、拾放） | Isaac Lab + DR + 视觉 DR-GAN |
| 自动驾驶 | CARLA / NVIDIA DRIVE Sim + DR + 真实微调 |
| 无人机竞速 | RotorS / Flightmare + DR + 在线适应 |
| 手指/手内操控 | OpenAI Dactyl（前所未有规模的 DR） |
| 工业机械臂 | MuJoCo-Warp + SI + 少量真实微调 |

对各种规模的控制，工作流是一致的：尽可能拟合仿真，对无法拟合的进行随机化，训练大型策略，蒸馏，带安全屏障部署。

## 交付物

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## 练习

1. **（简单）** 在固定滑动网格世界（slip=0.0）上训练 Q学习智能体。在 `slip ∈ {0.0, 0.1, 0.3, 0.5}` 上评估。绘制回报 vs 滑动的曲线。
2. **（中等）** 采样 `slip ~ Uniform[0, 0.3]` 训练 DR Q学习智能体。评估相同的扫描。DR 在 slip=0.5（分布外）时增益多少？
3. **（困难）** 实现课程：从 slip=0.0 开始，每当策略达到最优的 90% 时扩宽 DR 范围。测量达到 slip=0.3 零样本所需的总环境步数 vs 固定 DR 基线。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 现实差距 (Reality gap) | "仿真到真实的差异" | 训练与部署物理/感知之间的分布偏移。 |
| 域随机化 (Domain randomization, DR) | "在随机仿真中训练" | 训练期间随机化仿真参数，使策略泛化。 |
| 系统辨识 (System identification, SI) | "测量真实并拟合仿真" | 估计真实物理参数；设置仿真以匹配。 |
| 域适应 (Domain adaptation) | "在真实数据上微调" | 仿真训练后少量真实世界微调；可能适应观测或动力学。 |
| 特权信息 (Privileged info) | "教师的地面真实" | 只有仿真才有的信息；学生必须从观测历史中推断。 |
| 教师/学生 (Teacher/student) | "将特权信息蒸馏到可观测" | 教师用捷径训练；学生学会在没有捷径的情况下模仿。 |
| ADR | "自动域随机化" | 随着策略改进扩宽 DR 范围的课程。 |
| Real2Sim | "用真实数据弥合差距" | 学习残差，使仿真模仿真实展开。 |

## 延伸阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 原始 DR 论文（机器人视觉）
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — 动力学 DR，四足运动
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl，大规模 ADR
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — ANYmal 的教师-学生方法
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — 推动 2025–2026 年部署的大规模并行仿真
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR 课程方法
- [Sutton & Barto (2018). 第8章 — 基于表格方法的规划和学习](http://incompleteideas.net/book/RLbook2020.pdf) — 支撑现代仿真到真实流水线的 Dyna 框架（用模型进行规划 + 展开）
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep RL for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — 带基准结果的仿真到真实方法分类

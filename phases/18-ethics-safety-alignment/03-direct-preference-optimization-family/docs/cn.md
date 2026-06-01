# 直接偏好优化系列（The Direct Preference Optimization Family）

> Rafailov 等人（2023 年）证明了 RLHF 的最优解可以用偏好数据的封闭形式表达，因此可以跳过显式奖励模型，直接优化策略。这一洞见催生了一个系列——IPO、KTO、SimPO、ORPO、BPO——每一个都修复了 DPO 的一种失败模式。2026 年，直接对齐算法（DAA）的前沿后训练运行次数多于 PPO。但第 2 课的过度优化曲线仍然适用：DAA 并没有逃脱古德哈特，它们只是改变了问题咬到你的地方。

**类型：** 学习  
**语言：** Python（标准库，六种变体偏好损失比较器）  
**前置知识：** Phase 18 · 01（InstructGPT）、Phase 18 · 02（奖励黑客）、Phase 10 · 08（DPO 基础）  
**预计时间：** 约 75 分钟

## 学习目标

- 从带 KL 的 RLHF 最优解推导 DPO 封闭形式。
- 说出 IPO、KTO、SimPO、ORPO、BPO 各自修复了 DPO 的哪种失败模式。
- 区分"隐式奖励差距"和"偏好强度"，并解释为什么 IPO 的恒等映射很重要。
- 解释为什么 Rafailov 等人（NeurIPS 2024）证明了 DAA 在没有显式 RM 的情况下仍然存在过度优化。

## 问题所在

RLHF 目标（第 1 课）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有一个已知的最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

所以奖励由最优策略与参考策略的比率隐式定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

将此代入 Bradley-Terry 偏好似然，配分函数 `Z(x)` 会消掉，因为它只依赖于 `x`。剩下的是只有策略参数的损失——不需要奖励模型。这就是 DPO。

问题在于：推导假设最优解是可达的，偏好数据在分布内，参考策略是真正的模式锚点。这些条件没有一个完全成立。每个系列成员修复了一个不同的被违反的假设。

## 核心概念

### DPO（Rafailov 等人，2023 年）

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出错的地方：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 是无界的。微小的偏好可以产生任意大的差距。
- 损失以相反方向驱动选定和被拒绝的对数概率。只要被拒绝的下降更快，它就可以将选定的绝对对数概率向下推。这就是"选定响应退化"现象。
- 分布外偏好（罕见配对 vs 罕见配对）产生任意隐式奖励。

### IPO（Azar 等人，2024 年）

恒等偏好优化（Identity Preference Optimization）将对数-sigmoid 替换为对偏好概率的恒等映射。损失变成有界目标上的平方误差：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

边距由 `1/(2 beta)` 限定。偏好强度和隐式奖励差距成比例。不会爆炸。

### KTO（Ethayarajh 等人，2024 年）

Kahneman-Tversky 优化（Kahneman-Tversky Optimization）完全放弃成对结构。给定单个标注输出和二元"可取"或"不可取"信号，映射到展望理论效用：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

对收益和损失有不同权重（损失厌恶）。优势：可以使用非配对数据，这类数据更丰富。

### SimPO（Meng 等人，2024 年）

简单偏好优化（Simple Preference Optimization）将训练信号与生成对齐。完全去掉参考策略并按长度归一化对数似然：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

带有稳定化边距 `gamma`。长度归一化去除了利用 DPO 长度偏见失败模式的激励（较长的 `y_w` 从定义上就给出更大的对数概率差距）。

### ORPO（Hong 等人，2024 年）

优势比偏好优化（Odds-Ratio Preference Optimization）在标准 SFT 负对数似然中添加偏好项：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

没有参考策略——SFT 项是正则化项。从基础模型到对齐模型单阶段训练。不需要单独的 SFT 检查点。

### BPO（ICLR 2026 投稿，OpenReview id=b97EwMUWu7）

识别了"选定响应退化"问题：DPO 保留了 `y_w > y_l` 的排名，但 `y_w` 的绝对对数概率可能下降。BPO 添加了一个单行修正，惩罚选定响应的向下移动。报告称在 Llama-3.1-8B-Instruct 数学推理上比 DPO 提升了 +10.1% 准确率。

### 普遍结果：DAA 仍然存在过度优化

Rafailov 等人"直接对齐算法中奖励模型过度优化的缩放定律"（NeurIPS 2024）在多个数据集上训练了不同 KL 预算的 DPO、IPO、SLiC 策略。黄金奖励 vs KL 曲线具有相同的 Gao 等人峰值-崩溃形状。隐式奖励在训练期间查询分布外样本；KL 正则化不能稳定这一点。

DAA 并没有逃脱古德哈特。它们将问题咬到你的表面从"奖励模型过度优化"变成"参考策略比率过度优化"。普遍修复——更好的数据、集成、早停——对两者都适用。

### 如何在它们之间选择（2026 年）

- 如果你有大量成对偏好数据：保守 beta 的 DPO，如果长度偏见明显则用 SimPO。
- 如果你有非配对二元反馈：KTO。
- 如果你想要从基础模型的单阶段流水线：ORPO。
- 如果你在 DPO 日志中看到选定对数概率退化：BPO。
- 如果偏好强度差异很大且 DPO 正在饱和：IPO。

每家实验室都在测试套件上运行全部五种，并按任务选择赢者。没有理由认为数学推理和安全性的最优是相同的。

## 使用它

`code/main.py` 在玩具偏好数据集上比较六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO），其中真实偏好强度因对而异。每种损失都针对相同的 500 对样本使用小型 softmax 策略进行优化。绘制每种方法的最终胜率、选定对数概率漂移和隐式奖励分布。

## 交付它

本课产出 `outputs/skill-preference-loss-selector.md`。给定数据集统计数据（成对 vs 非配对，可变 vs 均匀偏好强度，长度分布）和目标（单阶段或 SFT-然后-偏好），推荐一种偏好损失并报告它所防范的失败模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 的最终选定对数概率下降。BPO 应该保留更高的选定绝对概率——验证这一点。

2. 修改偏好数据使所有对具有相同强度。六种方法中哪种最鲁棒？哪种退化？解释 IPO 在这里的优势。

3. 使被拒绝响应平均比选定响应长 2 倍。在不改变其他任何内容的情况下，数值地展示 DPO 的长度利用以及 SimPO 的修复。

4. Rafailov 等人（NeurIPS 2024）声称 DAA 存在过度优化。复现单点版本：绘制选定-减去-被拒绝的 KL 散度，并观察大 beta 时 DPO 的过度优化。

5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写下 BPO 添加到 DPO 中的单行修正。对照 `code/main.py` 中的实现确认。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| DPO | "没有奖励模型的 RLHF" | 从 RLHF 封闭形式最优解推导的损失；只有策略参数 |
| Implicit reward（隐式奖励） | "对数比率" | `beta * log(pi(y\|x) / pi_ref(y\|x))` — DPO 隐含的奖励 |
| IPO | "有界 DPO" | 用恒等映射替换对数-sigmoid；隐式奖励差距由 `1/(2 beta)` 限定 |
| KTO | "非配对 DPO" | 带损失厌恶的单标签展望理论效用 |
| SimPO | "无参考 DPO" | 长度归一化对数似然 + 边距；没有参考策略 |
| ORPO | "单阶段 DPO" | NLL + 优势比偏好项；从基础模型一次通过训练 |
| BPO | "保留选定的 DPO" | DPO 加上对降低选定响应绝对对数概率的惩罚 |
| Degraded Chosen（选定退化） | "选定向下" | DPO 只要被拒绝下降更快就降低选定对数概率 |
| DAA | "直接对齐算法" | 任何跳过显式 RM 的偏好损失方法 |

## 延伸阅读

- [Rafailov 等人 — 直接偏好优化（NeurIPS 2023，arXiv:2305.18290）](https://arxiv.org/abs/2305.18290)
- [Azar 等人 — 理解从人类偏好学习的通用理论框架（AISTATS 2024，arXiv:2310.12036）](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh 等人 — KTO：模型对齐作为展望理论优化（arXiv:2402.01306）](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO（NeurIPS 2024，arXiv:2405.14734）](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO（EMNLP 2024，arXiv:2403.07691）](https://arxiv.org/abs/2403.07691)
- [BPO — 行为保留优化（ICLR 2026 OpenReview b97EwMUWu7）](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov 等人 — DAA 中 RM 过度优化的缩放定律（NeurIPS 2024，arXiv:2406.02900）](https://arxiv.org/abs/2406.02900)

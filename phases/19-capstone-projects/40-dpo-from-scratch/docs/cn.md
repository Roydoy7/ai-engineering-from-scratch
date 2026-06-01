# 压轴项目第 40 课：从头开始的直接偏好优化（Capstone Lesson 40: Direct Preference Optimization from Scratch）

> 奖励模型和 PPO 是经典的 RLHF 栈。DPO 将该栈压缩为单个监督损失，直接针对偏好对拟合策略。本课从奖励差异恒等式推导 DPO 损失，提供工作的参考模型加策略模型，计算每 token 对数概率，并在选中和被拒绝补全的偏好夹具上训练小型 Transformer。测试固定损失数学和梯度方向，使你知道实现与论文匹配。

**类型：** 构建  
**语言：** Python（torch，numpy）  
**前置知识：** Phase 19 第 30-37 课（NLP LLM Track：分词器，嵌入表，注意力块，Transformer 主体，预训练循环，检查点，生成，困惑度）  
**预计时间：** 约 90 分钟

## 学习目标

- 将 DPO 损失推导为对缩放对数比率差的 sigmoid 并将其与隐式奖励联系。
- 构建带冻结参考和可训练策略的参考模型 + 策略模型对。
- 计算两个模型下的序列级对数概率，掩码提示词 token。
- 在 `(prompt, chosen, rejected)` 三元组上训练策略，观察 chosen 对数概率相对于 rejected 上升。
- 用损失数学、梯度符号和参考不变性的测试固定行为。

## 核心概念

从 Bradley-Terry 模型开始。给定提示词 `x` 和两个补全 `y_w`（选中）和 `y_l`（被拒绝），人类更喜欢 `y_w` 的概率是

```text
P(y_w > y_l | x) = sigmoid( r(x, y_w) - r(x, y_l) )
```

其中 `r` 是某个潜在奖励函数。RLHF 首先从偏好拟合 `r`，然后训练策略 `pi` 以 KL 锚点最大化 `r`。

DPO 推导观察到，在此目标下最优策略 `pi*` 有一个封闭形式可以消去 `log Z(x)` 项（它不依赖于 `y`），从而得到：

```text
L_DPO(theta) = - E_{(x, y_w, y_l)} [
  log sigmoid( beta * ( log pi_theta(y_w|x) - log pi_ref(y_w|x)
                       - log pi_theta(y_l|x) + log pi_ref(y_l|x) ) )
]
```

这就是损失。它是每个样例单个标量上的 sigmoid，从四个对数概率计算得到。没有单独的奖励模型。没有 PPO。没有损失中的 KL 项；KL 约束被融入封闭形式推导中。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| DPO loss（DPO 损失） | "偏好损失" | log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))；单次前向传递，无 RM |
| Reference model（参考模型） | "SFT 锚点" | 冻结的 SFT 模型；策略偏离参考时对数比率增大 |
| Log-ratio diff（对数比率差） | "隐式奖励差" | log pi(y_w|x) - log pi_ref(y_w|x) - log pi(y_l|x) + log pi_ref(y_l|x) |
| Chosen / rejected（选中 / 被拒绝） | "偏好对" | 人类对同一提示词标注哪个补全更好 |
| beta（beta） | "KL 惩罚系数" | 控制策略偏离参考多远；越大越保守 |
| Reference invariance（参考不变性） | "参考不移动" | 参考参数的 requires_grad=False；对数概率在整个训练过程中固定 |

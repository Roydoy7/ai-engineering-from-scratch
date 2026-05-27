# 采样方法

> 采样是AI探索可能性空间的方式。

**类型：** 构建实现
**语言：** Python
**前置知识：** 第一阶段，第06-07课（概率、贝叶斯定理）
**时间：** 约120分钟

## 学习目标

- 仅使用均匀随机数从零实现逆CDF采样、拒绝采样和重要性采样
- 为语言模型词元生成构建温度采样、top-k和top-p（核）采样
- 解释重参数化技巧，以及为什么它能在VAE中实现通过采样的反向传播
- 运行Metropolis-Hastings MCMC从未归一化的目标分布中采样

## 问题背景

语言模型处理完你的提示，产生一个包含5万个logit的向量——词汇表中每个词元对应一个。现在它必须选择一个。怎么选？

如果它总是选择概率最高的词元，每次响应都完全相同——确定性的、无聊的。如果随机均匀选择，输出是乱码。答案在这两个极端之间，而那个"之间"由采样控制。

采样不局限于文本生成。强化学习通过采样轨迹来估计策略梯度。VAE通过从学习到的分布中采样并通过随机性反向传播来学习潜在表示。扩散模型通过采样噪声并迭代去噪来生成图像。蒙特卡洛方法估计没有封闭形式解的积分。MCMC算法探索不可能枚举的高维后验分布。

**每个生成式AI系统都是一个采样系统。** 采样策略决定了输出的质量、多样性和可控性。本课从零构建每种主要的采样方法，从均匀随机数出发，直至驱动现代LLM和生成模型的技术。

## 核心概念

### 为什么采样很重要

采样在AI和机器学习中扮演四种基本角色：

**生成。** 语言模型、扩散模型和GAN都通过采样产生输出。采样算法直接控制创造力、连贯性和多样性。温度、top-k和核采样是工程师每天调节的旋钮。

**训练。** 随机梯度下降采样小批量数据。Dropout采样要停用的神经元。数据增强采样随机变换。重要性采样在强化学习（PPO、TRPO）中重新加权样本以减少梯度方差。

**估计。** ML中许多量没有封闭形式的解：在数据分布上的期望损失、能量模型的配分函数、贝叶斯推断中的证据。蒙特卡洛估计通过对样本取平均来近似所有这些。

**探索。** MCMC算法探索贝叶斯推断中的后验分布。进化策略采样参数扰动。汤普森采样在老虎机问题中平衡探索与利用。

核心挑战：你只能直接从简单分布（均匀、正态）采样。对于其他所有分布，你需要一种方法将简单样本转换为目标分布的样本。

### 均匀随机采样

每种采样方法都从这里开始。均匀随机数生成器产生[0, 1)中的值，每个等长子区间的概率相等。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    对于 0 <= a <= b <= 1

性质：
  E[U] = 0.5
  Var(U) = 1/12
```

从n个离散集合中均匀采样：生成U并返回floor(n * U)。从连续范围[a, b]采样：计算a + (b - a) * U。

关键洞察：单个均匀随机数恰好包含从任何分布产生一个样本所需的正确随机量。技巧在于找到正确的变换。

### 逆CDF方法（逆变换采样）

累积分布函数（CDF）将值映射到概率：

```
F(x) = P(X <= x)

性质：
  F是非递减的
  F(-inf) = 0
  F(+inf) = 1
  F将实数线映射到[0, 1]
```

逆CDF将概率映射回值。若U ~ Uniform(0, 1)，则X = F_inverse(U)服从目标分布。

```
算法：
  1. 生成 u ~ Uniform(0, 1)
  2. 返回 F_inverse(u)

为什么有效：
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**指数分布示例：**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

求解 F(x) = u 得 x：
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

由于(1 - U)和U具有相同的分布：
  x = -ln(u) / lambda
```

当你能写出封闭形式的F_inverse时，这完美有效。对于正态分布，没有封闭形式的逆CDF，所以我们使用其他方法（Box-Muller或数值近似）。

**离散版本：** 对于离散分布，构建CDF作为累积和，生成U，找到累积和第一次超过U的索引。这就是第06课中`sample_categorical`的工作方式。

### 拒绝采样

当你无法反转CDF但能对目标PDF进行（可能不归一化的）评估时，拒绝采样有效。

```
目标分布: p(x)  （可以评估，可能未归一化）
提案分布: q(x)  （可以采样）
边界: M，使得对所有x有 p(x) <= M * q(x)

算法：
  1. 采样 x ~ q(x)
  2. 采样 u ~ Uniform(0, 1)
  3. 若 u < p(x) / (M * q(x))，接受x
  4. 否则，拒绝并转到步骤1

接受率 = 1/M
```

边界M越紧，接受率越高。在低维（1-3维）中，拒绝采样效果很好。在高维中，接受率指数级下降，因为大部分提案体积被拒绝——这是拒绝采样的维数灾难。

**示例：从截断正态分布采样。** 在截断范围上使用均匀提案，包络M是该范围内正态PDF的最大值。

**示例：从半圆采样。** 在边界矩形内均匀提案，若点落在半圆内则接受。这就是蒙特卡洛计算pi的方式：接受率等于面积比pi/4。

### 重要性采样

有时你不需要从目标分布p(x)采样，你需要在p(x)下估计期望值，而你有来自不同分布q(x)的样本。

```
目标: 估计 E_p[f(x)] = ∫ f(x) * p(x) dx

改写：
  E_p[f(x)] = ∫ f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

其中 w(x) = p(x) / q(x)  是重要性权重。

估计量：
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    其中 x_i ~ q(x)
```

这在强化学习中至关重要。在PPO（近端策略优化）中，你在旧策略pi_old下收集轨迹，但想优化新策略pi_new。重要性权重是pi_new(a|s) / pi_old(a|s)。PPO裁剪这些权重以防止新策略与旧策略偏离太远。

重要性采样估计量的方差取决于q与p的相似程度。若q与p差异很大，少数样本获得巨大权重并主导估计。自归一化重要性采样通过除以权重之和来减少这一问题：

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### 蒙特卡洛估计

蒙特卡洛估计通过对随机样本取平均来近似积分，大数定律保证收敛。

```
目标: 估计 I = ∫_D g(x) dx

方法：
  1. 从D中均匀采样 x_1, ..., x_N
  2. I ~ (D的体积 / N) * sum(g(x_i))

误差: O(1 / sqrt(N))   与维度无关
```

误差率与维度无关。这就是为什么蒙特卡洛方法在高维中占主导，而基于网格的积分不可能实现。

**估计pi：**

```
从[-1, 1] x [-1, 1]均匀采样(x, y)
统计落在单位圆内的点数: x^2 + y^2 <= 1
pi ~ 4 * (圆内数量) / (总数量)
```

**估计期望：**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    其中 x_i ~ p(x)

样本均值收敛到真实期望。
估计量的方差 = Var(f(X)) / N
```

### 马尔可夫链蒙特卡洛（MCMC）：Metropolis-Hastings

MCMC构建一个平稳分布为目标分布p(x)的马尔可夫链。足够多步后，链上的样本（近似）就是p(x)的样本。

```
目标: p(x)  （已知到归一化常数）
提案: q(x'|x)  （给定当前状态，如何提案下一状态）

Metropolis-Hastings算法：
  1. 从某个x_0出发
  2. 对 t = 1, 2, ..., T:
     a. 提案 x' ~ q(x'|x_t)
     b. 计算接受率:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. 以概率min(1, alpha)接受:
        - 若 u < alpha（u ~ Uniform(0,1)）: x_{t+1} = x'
        - 否则: x_{t+1} = x_t
  3. 丢弃前B个样本（预热期）
  4. 返回剩余样本
```

对于对称提案（q(x'|x) = q(x|x')），接受率简化为p(x')/p(x)，这是原始Metropolis算法。

**为什么有效。** 接受规则确保细致平衡：处于x并移动到x'的概率等于处于x'并移动到x的概率。细致平衡意味着p(x)是链的平稳分布。

**实践注意事项：**
- 预热期（Burn-in）：在链达到均衡之前丢弃早期样本
- 稀疏化（Thinning）：每k个样本保留一个以减少自相关
- 提案尺度：太小则链移动缓慢（高接受率，慢探索）；太大则大多数提案被拒绝（低接受率，原地踏步）
- 高维高斯提案的最优接受率约为0.234

### Gibbs采样

Gibbs采样是多元分布MCMC的特例。它不是同时在所有维度上提案移动，而是每次从条件分布中更新一个变量。

```
目标: p(x_1, x_2, ..., x_d)

算法：
  对每次迭代t：
    采样 x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    采样 x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    采样 x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs采样要求你能从每个条件分布p(x_i | x_{-i})采样。对许多模型这很简单：
- 贝叶斯网络：条件分布来自图结构
- 高斯混合模型：条件分布是高斯的
- Ising模型：每个自旋的条件分布只依赖其邻居

接受率始终为1（每个提案都被接受），因为从精确条件分布采样自动满足细致平衡。

**局限性。** 当变量高度相关时，Gibbs采样混合缓慢，因为每次只更新一个变量无法在分布中做大的对角移动。

### 温度采样（用于LLM）

语言模型为词汇表中的每个词元输出logit值z_1, ..., z_V。Softmax将其转换为概率。温度在softmax之前重缩放logit：

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: 标准softmax（原始分布）
T -> 0:  argmax（确定性，总是选择最高logit）
T -> inf: 均匀（所有词元等概率）
T < 1.0: 使分布变尖锐（更自信，多样性更少）
T > 1.0: 使分布变平坦（不那么自信，多样性更多）
```

**为什么有效。** 将logit除以T < 1会放大logit之间的差异。若z_1 = 2，z_2 = 1，除以T = 0.5得z_1/T = 4，z_2/T = 2，差距更大。经过softmax后，最高logit词元获得更大的概率份额。

**实践设置：**
- T = 0.0：贪婪解码，最适合事实问答
- T = 0.3-0.7：略有创意，适合代码生成
- T = 0.7-1.0：平衡，适合一般对话
- T = 1.0-1.5：创意写作、头脑风暴
- T > 1.5：越来越随机，很少有用

温度不改变哪些词元可能出现，它改变分配给每个词元的概率质量。

### Top-k采样

Top-k采样将候选集限制为概率最高的k个词元，然后重新归一化并从该受限集合中采样。

```
算法：
  1. 计算所有V个词元的softmax概率
  2. 按概率降序排列词元
  3. 只保留前k个词元
  4. 重新归一化: p_i' = p_i / sum(p_j for j in top-k)
  5. 从重新归一化的分布中采样

k = 1:  贪婪解码
k = V:  无过滤（标准采样）
k = 40: 典型设置，移除不太可能词元的长尾
```

Top-k防止模型选择词汇分布长尾中极不可能的词元（拼写错误、乱码）。问题：k是固定的，不管上下文如何。当模型很自信（一个词元有95%概率）时，k=40仍允许39个替代。当模型不确定（概率分散在1000个词元上）时，k=40截断了合理选项。

### Top-p（核）采样

Top-p采样动态调整候选集大小。它不保留固定数量的词元，而是保留累积概率超过p的最小词元集合。

```
算法：
  1. 计算所有V个词元的softmax概率
  2. 按概率降序排列词元
  3. 找到最小的k，使得前k个词元的概率之和 >= p
  4. 只保留这k个词元
  5. 重新归一化并采样

p = 0.9:  保留覆盖90%概率质量的词元
p = 1.0:  无过滤
p = 0.1:  非常限制性，接近贪婪
```

当模型很自信时，核采样保留很少的词元（可能2-3个）。当模型不确定时，保留很多（可能200个）。这种自适应行为是核采样通常比top-k产生更好文本的原因。

**常见组合：**
- 温度0.7 + top-p 0.9：良好的通用设置
- 温度0.0（贪婪）：确定性任务的最佳选择
- 温度1.0 + top-k 50：Fan等人（2018）原始论文设置

Top-k和top-p可以组合使用，先应用top-k，再在剩余集合上应用top-p。

### 重参数化技巧（用于VAE）

变分自编码器（VAE）通过将输入编码为潜在空间中的分布、从该分布采样，然后将样本解码回来进行学习。问题：**你无法通过采样操作进行反向传播。**

```
标准采样（不可微）：
  z ~ N(mu, sigma^2)

  随机性阻断了梯度流。
  d/d_mu [从N(mu, sigma^2)采样] = ???
```

重参数化技巧将随机性与参数分离：

```
重参数化采样：
  epsilon ~ N(0, 1)          （固定随机噪声，无参数）
  z = mu + sigma * epsilon   （参数的确定性函数）

  现在z是mu和sigma的确定性可微函数。
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  梯度流过mu和sigma。
```

这之所以有效，是因为N(mu, sigma^2)与mu + sigma * N(0, 1)具有相同的分布。关键洞察：将随机性移到无参数的来源（epsilon），然后将样本表示为参数的可微变换。

**在VAE训练循环中：**
1. 编码器为每个输入输出mu和log(sigma^2)
2. 采样 epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. 解码z重建输入
5. 反向传播经过步骤4、3、2、1（可能，因为步骤3是可微的）

没有重参数化技巧，VAE就无法用标准反向传播训练。这个单一洞察使VAE变得实用。

### Gumbel-Softmax（可微分类采样）

重参数化技巧适用于连续分布（高斯）。对于离散分类分布，我们需要不同的方法。Gumbel-Softmax提供了分类采样的可微近似。

**Gumbel-Max技巧（不可微）：**

```
从对数概率为log(p_1), ..., log(p_k)的分类分布采样：
  1. 对每个类别采样 g_i ~ Gumbel(0, 1)
     （g = -log(-log(u))，其中 u ~ Uniform(0, 1)）
  2. 返回 argmax(log(p_i) + g_i)

这产生精确的分类样本。
```

**Gumbel-Softmax（可微近似）：**

```
用软softmax替换硬argmax：
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau（温度）控制近似程度：
  tau -> 0:  接近独热向量（硬分类）
  tau -> inf: 接近均匀分布（1/k, 1/k, ..., 1/k）
  tau = 1.0: 软近似
```

Gumbel-Softmax产生离散样本的连续松弛。输出是概率向量（软独热），而不是硬独热。梯度流经softmax。在训练的前向传播中，可以使用"直通"估计器：前向传播使用硬argmax，反向传播使用软Gumbel-Softmax梯度。

**应用：**
- VAE中的离散潜变量
- 神经架构搜索（选择离散操作）
- 硬注意力机制
- 离散动作的强化学习

### 分层采样

标准蒙特卡洛采样可能因偶然在样本空间留下空缺。分层采样通过将空间分为层次并从每层采样来强制均匀覆盖。

```
标准蒙特卡洛：
  从[0, 1]均匀采样N个点
  某些区域可能有簇，其他区域有空缺

分层采样：
  将[0, 1]分为N个等层次：[0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  在每层内均匀采样一个点
  x_i = (i + u_i) / N   其中 u_i ~ Uniform(0, 1)，i = 0, ..., N-1
```

分层采样的方差始终小于等于标准蒙特卡洛：

```
Var(分层) <= Var(标准蒙特卡洛)

f(x)平滑变化时改善最大。
对于分段常数函数，分层采样是精确的。
```

**应用：**
- 数值积分（准蒙特卡洛）
- 训练数据划分（确保每折中的类别平衡）
- 带分层的重要性采样（结合两种技术）
- NeRF（神经辐射场）沿相机光线使用分层采样

### 与扩散模型的联系

扩散模型通过采样过程生成图像。正向过程在T步内向图像添加高斯噪声，直到变成纯噪声。逆向过程学习去噪，逐步恢复原始图像。

```
正向过程（已知）：
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  其中 epsilon ~ N(0, I)

  T步后: x_T ~ N(0, I)  （纯噪声）

逆向过程（学习的）：
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  其中 z ~ N(0, I)

  每一步去噪都是一个采样步骤。
```

与本课方法的联系：
- 每一步去噪使用重参数化技巧（采样噪声，应用确定性变换）
- 噪声调度{alpha_t}控制一种温度退火形式
- 训练使用蒙特卡洛估计来近似ELBO（证据下界）
- 扩散模型中的祖先采样是马尔可夫链（每步只依赖当前状态）

整个图像生成过程是迭代采样：从噪声开始，每步基于学习的去噪模型采样稍微不那么嘈杂的版本。

## 动手实现

### 步骤1：均匀和逆CDF采样

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成10,000个指数样本，验证均值为1/lambda。

### 步骤2：拒绝采样

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

使用拒绝采样从截断正态分布中采样，通过直方图验证形状。

### 步骤3：重要性采样

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

使用均匀提案估计正态分布下的E[X^2]，与已知答案(mu^2 + sigma^2)对比。

### 步骤4：蒙特卡洛估计pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 步骤5：Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从双峰分布（两个高斯的混合）中采样，可视化链的轨迹。

### 步骤6：Gibbs采样

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 步骤7：温度采样

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示温度如何改变一组词元logit的输出分布。

### 步骤8：Top-k和top-p采样

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 步骤9：重参数化技巧

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

演示梯度通过重参数化样本流动，但不通过直接采样流动。

### 步骤10：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示降低温度如何使输出接近独热向量。

完整实现（含所有可视化）见`code/sampling.py`。

## 应用示例

使用NumPy和SciPy的生产版本：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"指数均值: {exponential_samples.mean():.4f}（期望2.0）")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"1.96处的CDF: {normal.cdf(1.96):.4f}")
print(f"0.975处的逆CDF: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"采样的词元索引: {token}")
```

对于大规模MCMC，使用专用库：
- PyMC：带NUTS（自适应HMC）的完整贝叶斯建模
- emcee：集成MCMC采样器
- NumPyro/JAX：GPU加速的MCMC

你从零构建了这些，现在你知道库调用在做什么了。

## 练习

1. 为柯西分布实现逆CDF采样。CDF是F(x) = 0.5 + arctan(x)/pi。生成10,000个样本，将直方图与真实PDF对比。注意重尾（距中心很远的极端值）。

2. 使用Uniform(0, 1)提案，用拒绝采样从Beta(2, 5)分布生成样本。将接受的样本与真实Beta PDF对比。理论接受率是多少？

3. 使用1,000、10,000和100,000个样本用蒙特卡洛估计sin(x)从0到pi的积分。比较每个规模下的误差，验证误差按O(1/sqrt(N))缩放。

4. 实现Metropolis-Hastings从正比于exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2)的2D分布中采样。绘制样本和链轨迹，实验不同的提案标准差。

5. 构建完整的文本生成演示：给定10个词语的词汇表和logit值，使用(a)贪婪，(b)温度=0.7，(c)top-k=3，(d)top-p=0.9生成长度为20的词元序列。在5次运行中比较输出的多样性。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| 采样（Sampling） | "抽取随机值" | 根据概率分布生成值。所有生成式AI背后的机制 |
| 均匀分布（Uniform distribution） | "都等可能" | [a, b]中的每个值都有等概率密度1/(b-a)。所有采样方法的起点 |
| 逆CDF（Inverse CDF） | "概率变换" | F_inverse(U)将均匀样本转换为具有已知CDF的任意分布的样本。精确且高效 |
| 拒绝采样（Rejection sampling） | "提案并接受/拒绝" | 从简单提案中生成，以与目标/提案比成比例的概率接受。精确但浪费样本 |
| 重要性采样（Importance sampling） | "重新加权样本" | 通过给每个样本加权p(x)/q(x)，用来自q(x)的样本估计p(x)下的期望。RL中PPO的核心 |
| 蒙特卡洛（Monte Carlo） | "对随机样本取平均" | 将积分近似为样本平均值。误差O(1/sqrt(N))与维度无关 |
| MCMC | "收敛的随机游走" | 构建平稳分布为目标的马尔可夫链，Metropolis-Hastings是基础算法 |
| Metropolis-Hastings | "上坡接受，有时下坡也接受" | 提案移动，基于密度比接受。细致平衡保证收敛到目标分布 |
| Gibbs采样（Gibbs sampling） | "一次一个变量" | 保持其他变量固定，从每个变量的条件分布更新。100%接受率 |
| 温度（Temperature） | "置信度旋钮" | 在softmax前将logit除以T。T<1使分布变尖锐（更自信），T>1使分布变平坦（更多样） |
| Top-k采样 | "保留k个最佳" | 将除概率最高的k个词元外的所有词元归零，重新归一化，采样。固定候选集大小 |
| 核采样（top-p） | "保留可能的那些" | 保留累积概率超过p的最小词元集合。自适应候选集大小 |
| 重参数化技巧（Reparameterization trick） | "将随机性移到外面" | 写成z = mu + sigma * epsilon，其中epsilon ~ N(0,1)。使采样可微。VAE训练的基础 |
| Gumbel-Softmax | "软分类采样" | 使用Gumbel噪声+带温度的softmax的分类采样可微近似 |
| 分层采样（Stratified sampling） | "强制覆盖" | 将采样空间分为层次，从每层采样。方差始终低于朴素蒙特卡洛 |
| 预热期（Burn-in） | "预热阶段" | 链达到平稳分布之前丢弃的早期MCMC样本 |
| 细致平衡（Detailed balance） | "可逆性条件" | p(x) * T(x->y) = p(y) * T(y->x)。p是马尔可夫链平稳分布的充分条件 |
| 扩散采样（Diffusion sampling） | "迭代去噪" | 通过从噪声开始并应用学习的去噪步骤来生成数据。每步是一个条件采样操作 |

## 延伸阅读

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) — MCMC基础的详细教程
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) — 原始Gumbel-Softmax论文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — 核（top-p）采样论文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — 引入重参数化技巧的VAE论文
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM将采样与图像生成联系起来

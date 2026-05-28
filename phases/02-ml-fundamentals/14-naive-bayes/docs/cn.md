# 朴素贝叶斯（Naive Bayes）

> "朴素"假设从数学上来说是错的，但它照样好用——这正是它的魅力所在。

**类型：** 动手实现
**语言：** Python
**前置知识：** 第二阶段第1-7课（分类、贝叶斯定理）
**预计时间：** 约75分钟

## 学习目标

- 从零实现带 Laplace 平滑的多项式朴素贝叶斯，用于文本分类
- 解释为什么"朴素独立假设"在数学上是错的，但在实践中仍能得出正确的类别排名
- 对比多项式（Multinomial）、伯努利（Bernoulli）和高斯（Gaussian）三种朴素贝叶斯变体，为不同特征类型选择合适的版本
- 在高维稀疏数据上对比朴素贝叶斯和逻辑回归，理解背后的偏差-方差权衡

## 问题背景

你需要对文本分类——把邮件分成垃圾邮件和正常邮件，把用户评论分成正面和负面，把工单分到不同类别。你有数千个特征（每个词一个），训练数据却有限。

大多数分类器在这里会卡死。逻辑回归需要足够多的样本才能可靠地估计成千上万个权重；决策树每次只在一个词上分裂，会疯狂过拟合；KNN 在一万个维度里毫无意义，因为每个点到其他每个点的距离都差不多。

朴素贝叶斯对付这类问题游刃有余。它做了一个数学上明显错误的假设（在已知类别的条件下，每个特征与其他所有特征相互独立），然而在文本分类上，尤其是训练集较小时，它仍然胜过很多"更聪明"的模型。它只需一次遍历数据即可训练完毕，可以扩展到数百万个特征，还能输出概率估计（尽管由于独立假设，概率往往校准不准）。

理解"为什么一个错误的假设能导出正确的预测"，会教给你一个关于机器学习的根本道理：**最好的模型不是最正确的那个，而是偏差-方差权衡最优的那个。**

## 核心概念

### 贝叶斯定理（快速回顾）

贝叶斯定理用于"翻转"条件概率：

```
P(类别 | 特征) = P(特征 | 类别) × P(类别) / P(特征)
```

我们想要的是 `P(类别 | 特征)`——给定文档里出现的词，该文档属于某类别的概率。我们可以通过以下三项计算出来：
- `P(特征 | 类别)`：在该类别的文档中观察到这些词的可能性（似然）
- `P(类别)`：类别的先验概率（总体上垃圾邮件有多常见？）
- `P(特征)`：证据项，对所有类别相同，比较时可以忽略

`P(类别 | 特征)` 最高的类别获胜。

### 朴素独立假设

精确计算 `P(特征 | 类别)` 需要估计所有特征的联合概率。词汇表有一万个词，就需要估计 2^10000 种可能组合上的分布——根本不可能。

朴素假设：**在已知类别的条件下，每个特征与其他特征条件独立。**

```
P(w1, w2, ..., wn | 类别) = P(w1 | 类别) × P(w2 | 类别) × ... × P(wn | 类别)
```

一个不可能的联合分布，变成了 n 个简单的单特征分布，每个只需要一个频次计数。

这个假设显然是错的。"机器"和"学习"在任何文档里都不是独立的。但分类器并不需要正确的概率估计——它只需要正确的排名，即哪个类别概率最高。独立假设会引入系统性误差，但这些误差对所有类别的影响方式相似，所以排名依然正确。

### 为什么它依然有效

三个原因：

1. **排名比校准更重要。** 分类只需要排名第一的类别是正确的。就算 P(垃圾邮件) = 0.99999，而真实概率只有 0.7，分类器仍然正确地选了垃圾邮件。我们不需要准确的概率，只需要赢家正确。

2. **高偏差，低方差。** 独立假设是一个很强的先验，它对模型的约束力很强，从而防止过拟合。训练数据有限时，一个略微偏差但非常稳定的模型，胜过一个理论上正确却极不稳定的模型——这正是偏差-方差权衡的体现。

3. **特征冗余相互抵消。** 相关特征提供冗余证据。分类器会重复计算这些证据，但重复的是支持正确类别的证据。如果"机器"和"学习"总是同时出现，两个词都为"技术"类提供证据。朴素贝叶斯数了两次，但数对了。

还有一个实际原因：朴素贝叶斯极快。训练就是一次遍历数据统计频次。预测就是一次矩阵乘法。在百万级文档上训练只需几秒。这种速度让你能更快地迭代、尝试更多特征组合、跑更多实验。

### 逐步推导

让我们用一个具体例子来过一遍计算流程。假设两个类别：垃圾邮件和正常邮件，词汇表只有三个词："free"、"money"、"meeting"。

训练数据：
- 垃圾邮件：出现 "free" 80 次、"money" 60 次、"meeting" 10 次（共 150 个词）
- 正常邮件：出现 "free" 5 次、"money" 10 次、"meeting" 100 次（共 115 个词）
- 垃圾邮件占 40%，正常邮件占 60%

带 Laplace 平滑（alpha=1）：

```
P(free | 垃圾)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | 垃圾)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | 垃圾) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | 正常)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | 正常)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | 正常) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

新邮件包含："free" 出现 2 次，"money" 出现 1 次，"meeting" 出现 0 次。

```
log P(垃圾 | 邮件) = log(0.4) + 2×log(0.529) + 1×log(0.399) + 0×log(0.072)
                   = -0.916 + 2×(-0.637) + (-0.919) + 0
                   = -3.109

log P(正常 | 邮件) = log(0.6) + 2×log(0.051) + 1×log(0.093) + 0×log(0.856)
                   = -0.511 + 2×(-2.976) + (-2.375) + 0
                   = -8.838
```

垃圾邮件以绝对优势获胜。"free" 出现两次是强烈的垃圾邮件信号。注意："meeting" 未出现时对两个对数得分都贡献 0（0 × log(P) = 0）——在多项式朴素贝叶斯里，缺席的词没有影响；而伯努利朴素贝叶斯则会显式建模词的缺席。

### 三种变体

朴素贝叶斯有三种风味，每种对 `P(特征 | 类别)` 的建模方式不同。

#### 多项式朴素贝叶斯（Multinomial NB）

把每个特征建模为一个计数。最适合特征是词频或 TF-IDF 值的文本数据。

```
P(词_i | 类别) = (类别中词_i 的计数 + alpha) / (类别中总词数 + alpha × 词汇表大小)
```

`alpha` 是 Laplace 平滑参数（下面会详细讲）。这是文本分类的主力变体。

#### 高斯朴素贝叶斯（Gaussian NB）

把每个特征建模为一个正态分布。最适合连续特征。

```
P(x_i | 类别) = (1 / sqrt(2π × 方差)) × exp(-(x_i - 均值)² / (2 × 方差))
```

每个类别的每个特征都有自己的均值和方差。当每个类别内的特征确实近似服从正态分布时效果很好。

#### 伯努利朴素贝叶斯（Bernoulli NB）

把每个特征建模为二值（出现或缺席）。最适合短文本或二值特征向量。

```
P(词_i | 类别) = (包含词_i 的类别文档数 + alpha) / (类别总文档数 + 2×alpha)
```

与多项式不同，伯努利会显式惩罚词的缺席。如果"free"通常出现在垃圾邮件里，而当前邮件没有出现，伯努利会把这算作反对垃圾邮件的证据。

### 怎么选变体

| 变体 | 特征类型 | 最适合 | 典型场景 |
|------|---------|--------|---------|
| 多项式 | 计数或频率 | 文本分类、词袋模型 | 邮件过滤、主题分类 |
| 高斯 | 连续值 | 近似正态分布的表格数据 | 鸢尾花分类、传感器数据 |
| 伯努利 | 二值（0/1） | 短文本、二值特征向量 | 短信过滤、特征出现/缺席 |

### Laplace 平滑

如果某个词出现在测试数据里，但在某个类别的训练数据里从未出现过，会怎样？

不加平滑：`P(词 | 类别) = 0/N = 0`。整个连乘式里只要乘了一个零，`P(类别 | 特征) = 0`——不管其他特征给出了多少证据，一个没见过的词就能让整个预测归零。

Laplace 平滑给每个特征计数加一个小值 `alpha`（通常为 1）：

```
P(词_i | 类别) = (count(词_i, 类别) + alpha) / (类别总词数 + alpha × 词汇表大小)
```

加了 alpha=1 之后，每个词都至少有一点点概率。从未见过的词"discombobulate"出现在测试邮件里，也不会让垃圾邮件概率归零了。

平滑还有贝叶斯解释：等价于在词分布上放置一个均匀的 Dirichlet 先验。

alpha 越大，平滑越强（分布越均匀）；alpha 越小，模型越信任数据。alpha 是一个需要调的超参数。

| Alpha | 效果 | 适用场景 |
|-------|------|---------|
| 0.001 | 几乎不平滑，完全信任数据 | 超大训练集，几乎不会出现未见特征 |
| 0.1 | 轻微平滑 | 大训练集 |
| 1.0 | 标准 Laplace 平滑 | 默认起点 |
| 10.0 | 强平滑，分布趋于均匀 | 训练集极小，预期未见特征很多 |

### 对数空间计算

将数百个概率（每个都小于 1）连乘，会导致浮点下溢——真实值是一个很小的正数，在浮点运算里却变成了零。

解决方案：在对数空间里工作。连乘变成连加：

```
log P(类别 | x1, x2, ..., xn) = log P(类别) + Σ_i log P(xi | 类别)
```

这把预测变成了一个点积：

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

矩阵乘法——这就是为什么朴素贝叶斯预测如此之快：它本质上和单层线性模型的运算完全相同。

### 朴素贝叶斯 vs 逻辑回归

两者都是文本的线性分类器，差别在于建模方式：

| 维度 | 朴素贝叶斯 | 逻辑回归 |
|------|-----------|---------|
| 类型 | 生成式（建模 P(X\|Y)） | 判别式（建模 P(Y\|X)） |
| 训练 | 统计频次 | 优化损失函数 |
| 小数据 | 更好（强先验有帮助） | 更差（样本不足以估计权重） |
| 大数据 | 更差（错误假设成为瓶颈） | 更好（可以学习灵活的决策边界） |
| 特征相关性 | 假设独立 | 能处理相关特征 |
| 速度 | 单次遍历，极快 | 迭代优化 |
| 概率校准 | 概率不准 | 概率更准 |

经验法则：先用朴素贝叶斯。如果数据够多而且 NB 遇到瓶颈，再换逻辑回归。

### 分类流程图

```mermaid
flowchart LR
    A[原始文本] --> B[分词]
    B --> C[构建词汇表]
    C --> D[统计词频]
    D --> E[施加平滑]
    E --> F[计算对数概率]
    F --> G[预测：argmax P(类别|词)]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

在实践中，我们在对数空间操作以避免浮点下溢，把连乘转化为连加：

```
log P(类别 | 特征) = log P(类别) + Σ_i log P(特征_i | 类别)
```

## 动手实现

`code/naive_bayes.py` 中包含从零实现的 MultinomialNB 和 GaussianNB。

### MultinomialNB 的从零实现

三个核心步骤：

1. **fit(X, y)**：对每个类别统计每个特征的频次，加 Laplace 平滑，计算对数概率，保存类别先验（类别频率的对数）。

2. **predict_log_proba(X)**：对每个样本，计算所有类别的 `log P(类别) + Σ log P(特征_i | 类别)`，这是一次矩阵乘法：`X @ log_probs.T + log_priors`。

3. **predict(X)**：返回对数概率最高的类别。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键洞察：拟合完成后，预测就是矩阵乘法加偏置——这就是朴素贝叶斯如此之快的原因。

### GaussianNB 的从零实现

对连续特征，为每个类别的每个特征估计均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9  # 防止方差为零
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测时使用高斯概率密度函数，对每个特征分别计算（在对数空间里累加）。

### 演示：文本分类

代码生成模拟两类文章（科技 vs 体育）的词袋数据。每类有不同的词频分布：

- 第 0-39 个词在科技文章里高频，在体育文章里低频
- 第 80-119 个词在体育文章里高频，在科技文章里低频
- 第 40-79 个词两类都是中等频率（噪声词）

这制造了一个真实的场景：有些词是强类别指标，其他词是噪声。MultinomialNB 用词频来分类。

### 演示：连续特征

代码生成类鸢尾花数据（3 类，4 个特征，高斯分布的簇）。GaussianNB 用每类的均值和方差来分类。每个类别有不同的中心点（均值向量）和不同的扩散程度（方差），模拟了测量值在不同类别间系统性差异的真实场景。

代码还演示了：
- **平滑对比**：用不同 alpha 值训练，展示平滑强度对准确率的影响
- **训练规模实验**：准确率如何随训练样本从 20 增长到 1600 而提升——NB 用极少样本就能达到不错的准确率，这是它的核心优势
- **混淆矩阵**：逐类别的精确率、召回率和 F1，直观看出 NB 在哪里犯错

### 预测速度

朴素贝叶斯的预测就是矩阵乘法。设有 n 个样本、d 个特征、k 个类别：
- MultinomialNB：一次矩阵乘法 (n×d) @ (d×k) = O(n×d×k)
- GaussianNB：n×k 次高斯 PDF 计算，每次对 d 个特征 = O(n×d×k)

两者对所有维度都是线性复杂度。对比 KNN（需要计算到所有训练点的距离）或带 RBF 核的 SVM（需要对所有支持向量计算核函数），朴素贝叶斯在预测时快了不止一个数量级。

## 实际使用

用 sklearn，两种变体都是一行代码：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

文本分类的完整 sklearn 流水线：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

`naive_bayes.py` 中会把从零实现的版本和 sklearn 版本跑在同一数据集上，验证结果一致。

### TF-IDF + 朴素贝叶斯

原始词频对每次出现赋予相同权重。但像"the"、"is"这样的高频词在每个类别里都频繁出现——它们没有区分信息。TF-IDF 降低常见词的权重、提升罕见但有区分力的词的权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF 值非负，可以直接用于 MultinomialNB。**TF-IDF + MultinomialNB 是文本分类最强的基线组合之一**，在训练样本不足一万条时，经常能打败更复杂的模型。

### BernoulliNB 用于短文本

对于短文本（推文、短信、聊天消息），BernoulliNB 有时能超越 MultinomialNB。短文本词数少，MultinomialNB 依赖的频率信息噪声很大；BernoulliNB 只关心词是否出现，在短文本上更可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),  # 把所有计数转成0/1
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

CountVectorizer 的 `binary=True` 把所有计数转成 0/1。不加这个参数 BernoulliNB 也能运行，但它接收的是计数，而不是它设计时期望的二值特征。

### 校准朴素贝叶斯的概率

NB 的概率校准较差——它说 P(垃圾邮件) = 0.95 时，真实概率可能只有 0.7。如果你需要可靠的概率估计（比如设置阈值，或与其他模型融合），可以用 sklearn 的 CalibratedClassifierCV：

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

它通过交叉验证，在 NB 原始输出上拟合一个逻辑回归。校准后的概率与真实类别频率更为接近。

### 常见坑

1. **特征值为负。** MultinomialNB 要求非负特征。如果有负值（某些设置下的 TF-IDF 或标准化特征），改用 GaussianNB，或把特征平移为正数。

2. **方差为零。** GaussianNB 需要用方差做除法。如果某个特征在某个类别里方差为零（所有值相同），概率计算就会出问题。代码里给所有方差加了一个小平滑项（1e-9）来避免这种情况。

3. **类别不均衡。** 如果 99% 的邮件是正常邮件，先验 P(正常) = 0.99 会强到掩盖似然证据。可以手动设置类别先验，或使用 sklearn 的 `class_prior` 参数。

4. **特征缩放。** MultinomialNB 不需要缩放（基于计数运算）；GaussianNB 也不需要（它会自己估计每个特征的统计量）。这是相对于逻辑回归和 SVM 的一个优势，后两者对特征尺度很敏感。

## 成果

本课产出：
- `outputs/skill-naive-bayes-chooser.md` —— 选择正确 NB 变体的决策技能提示词
- `code/naive_bayes.py` —— MultinomialNB 和 GaussianNB 从零实现，含 sklearn 对比验证

### 朴素贝叶斯在哪里会失败

当独立假设导致排名错误（不只是概率不准）时，NB 会失败。这发生在：

1. **强特征交互。** 如果类别依赖于两个特征的组合而非单个特征（类似 XOR 模式），NB 完全无法捕捉。每个特征单独来看没有证据，NB 也不能非线性地组合它们。

2. **高度相关且证据对立的特征。** 如果特征 A 说"垃圾邮件"，特征 B 说"正常邮件"，但 A 和 B 完全相关（现实中它们总是一致的），NB 会看到冲突证据，而实际上并没有冲突。

3. **非常大的训练集。** 数据足够多时，判别模型（如逻辑回归）能学到真实决策边界，超越 NB。小数据时有帮助的独立假设，此时反而成了制约。

在实践中，这些失败模式在文本分类里很少见。文本特征多、单个特征信号弱，独立假设的误差倾向于相互抵消。对于特征少且强相关的表格数据，则优先考虑逻辑回归或基于树的模型。

## 练习

1. **平滑实验。** 用 alpha = 0.01、0.1、1.0、10.0、100.0 分别训练 MultinomialNB。画出准确率 vs alpha 的图。准确率在哪里达到峰值？为什么 alpha 过大会伤害性能？

2. **特征独立性检验。** 找一个真实文本数据集，挑两个明显相关的词（比如"machine"和"learning"）。计算 P(词1 | 类别) × P(词2 | 类别) 并与 P(词1 AND 词2 | 类别) 对比。独立假设有多错？它影响分类准确率了吗？

3. **实现 BernoulliNB。** 扩展代码，实现一个 BernoulliNB 类。把词袋转成二值（出现/缺席），在文本数据上与 MultinomialNB 对比准确率。什么时候伯努利赢？

4. **NB vs 逻辑回归。** 两者都在文本数据上训练。从 100 个训练样本开始，逐步增加到 10000 个。画出两者准确率随训练集大小的变化曲线。在哪个点逻辑回归开始超越朴素贝叶斯？

5. **垃圾邮件过滤器。** 构建一个完整的垃圾邮件分类器：对原始邮件文本分词、构建词汇表、生成词袋特征、训练 MultinomialNB，用精确率和召回率（而不只是准确率）评估——为什么要用这两个指标？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 朴素贝叶斯（Naive Bayes） | "简单的概率分类器" | 应用贝叶斯定理并假设特征在已知类别条件下相互独立的分类器 |
| 条件独立（Conditional independence） | "特征互不影响" | P(A, B\|C) = P(A\|C) × P(B\|C)——在已知 C 的条件下，B 对 A 不提供额外信息 |
| Laplace 平滑（Laplace smoothing） | "加一平滑" | 给每个特征计数加一个小值，防止零概率主导预测 |
| 先验（Prior） | "观察数据之前的信念" | P(类别)——在观察任何特征之前，每个类别的概率 |
| 似然（Likelihood） | "数据与模型的吻合程度" | P(特征\|类别)——在已知类别的条件下，观察到这些特征的概率 |
| 后验（Posterior） | "观察数据之后的信念" | P(类别\|特征)——观察到特征后，类别的更新概率 |
| 生成式模型（Generative model） | "建模数据的生成过程" | 学习 P(X\|Y) 和 P(Y)，然后用贝叶斯定理求 P(Y\|X) 的模型 |
| 判别式模型（Discriminative model） | "建模决策边界" | 直接学习 P(Y\|X) 而不建模 X 如何生成的模型 |
| 对数概率（Log probability） | "防止下溢" | 用 log P 代替 P，防止很多小概率连乘在浮点数里变成零 |

## 延伸阅读

- [scikit-learn Naive Bayes 文档](https://scikit-learn.org/stable/modules/naive_bayes.html) —— 三种变体的详细数学说明
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) —— 多项式 vs 伯努利的经典对比论文
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) —— 改进文本 NB 的研究
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) —— 证明了 NB 用更少数据就能收敛，而 LR 最终达到更高精度

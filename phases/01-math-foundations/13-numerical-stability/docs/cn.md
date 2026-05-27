# 数值稳定性

> 浮点运算是一种会泄漏的抽象。它会在训练过程中咬你一口，而你完全不会预料到。

**类型：** 构建实现
**语言：** Python
**前置知识：** 第一阶段，第01-04课
**时间：** 约120分钟

## 学习目标

- 使用最大值减法技巧实现数值稳定的softmax和log-sum-exp
- 识别浮点计算中的溢出、下溢和灾难性消去
- 使用居中有限差分法对比数值梯度验证解析梯度
- 解释为何bfloat16在训练中优于float16，以及损失缩放如何防止梯度下溢

## 问题背景

你的模型训练了三个小时，然后损失值变成了NaN。你加了一条打印语句。第9000步时logit值正常，第9001步时变成了`inf`，到第9002步每个梯度都是`nan`，训练彻底崩溃。

或者：你的模型训练完成，但准确率比论文声称的低2%。你检查了所有东西——架构、超参数、数据都对应。问题在于论文使用float32而你使用了没有正确缩放的float16。32位累积的舍入误差悄悄吃掉了你的准确率。

或者：你从头实现交叉熵损失，在小logit值上运行正常，但当logit超过100时返回`inf`。softmax溢出了，因为`exp(100)`超过了float32能表示的范围。每个ML框架都用两行代码处理这个问题，而你不知道这个技巧的存在。

**数值稳定性不是理论问题，而是训练成功与悄悄失败之间的差别。** 你将要调试的每一个严重ML bug，最终都会归结到浮点运算。

## 核心概念

### IEEE 754：计算机如何存储实数

计算机按照IEEE 754标准将实数存储为浮点值。一个浮点数由三部分组成：符号位、指数和尾数（有效位）。

```
Float32布局（共32位）：
[1位符号] [8位指数] [23位尾数]

值 = (-1)^符号 * 2^(指数 - 127) * 1.尾数
```

尾数决定精度（有效数字的位数），指数决定范围（数字可以有多大或多小）。

```
格式       位数   指数  尾数  十进制有效位  范围（约）
float64    64     11    52    ~15-16       +/- 1.8e308
float32    32     8     23    ~7-8         +/- 3.4e38
float16    16     5     10    ~3-4         +/- 65,504
bfloat16   16     8     7     ~2-3         +/- 3.4e38
```

float32提供约7位有效十进制数字，可以区分1.0000001和1.0000002，但不能区分1.00000001和1.00000002。超过7位之后，一切都是舍入噪声。

float16只有约3位有效数字，最大可表示的数是65,504——对于ML来说小得令人担忧，因为logit、梯度和激活值经常超过这个值。

bfloat16是Google针对float16范围问题的解决方案。它有与float32相同的8位指数（相同的范围，最大3.4e38），但只有7位尾数（精度比float16更低）。对于训练神经网络，范围比精度更重要，所以bfloat16通常更优。

### 为什么 0.1 + 0.2 != 0.3

0.1无法在二进制浮点数中精确表示。在二进制中，它是一个无限循环小数：

```
0.1 的二进制表示 = 0.0001100110011001100110011... （无限循环）
```

Float32将其截断为23位尾数。存储的值约为0.100000001490116，类似地0.2约为0.200000002980232，它们的和是0.300000004470348，而不是0.3。

```
在Python中：
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这在ML中很重要，因为：

1. `if loss < threshold`这样的损失比较可能给出错误答案
2. 累加许多小值（数千步的梯度更新）会偏离真实的和
3. 如果用`==`比较浮点数，校验和和可重现性测试会失败

**修复方法：永远不要用`==`比较浮点数。使用`abs(a - b) < epsilon`或`math.isclose()`。**

### 灾难性消去

当你减去两个几乎相等的浮点数时，有效数字相互抵消，你得到的结果中舍入噪声被提升到了高位。

```
a = 1.0000001    （在float32中存储为 1.00000011920929）
b = 1.0000000    （在float32中存储为 1.00000000000000）

真实差值:  0.0000001
计算结果:  0.00000011920929

相对误差: 19.2%
```

仅一次减法就产生了19%的相对误差。在ML中，这种情况发生在：

- 计算均值较大的数据的方差：当E[x]很大时，用`E[x^2] - E[x]^2`
- 减去几乎相等的对数概率
- 用过小的epsilon计算有限差分梯度

**修复方法：重新排列公式，避免减去大的近似相等的数。** 对于方差，使用Welford算法或先对数据中心化。对于对数概率，全程在对数空间中操作。

### 溢出和下溢

溢出发生在结果太大而无法表示时，下溢发生在结果太小（比最小可表示正数更接近零）时。

```
Float32的边界：
  最大值:          3.4028235e+38
  最小正规数:      1.175e-38
  最小非规格化数:  1.401e-45
  溢出：           任何 > 3.4e38 的值变成 inf
  下溢：           任何 < 1.4e-45 的值变成 0.0
```

`exp()`函数是ML中溢出的主要来源：

```
exp(88.7)  = 3.40e+38   （勉强适合float32）
exp(89.0)  = inf         （溢出）
exp(-87.3) = 1.18e-38   （刚好在下溢阈值之上）
exp(-104)  = 0.0         （下溢为零）
```

`log()`函数则面临另一方向的问题：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      （正常）
log(1e-46) = -inf        （输入下溢为0，然后log(0) = -inf）
```

在ML中，`exp()`出现在softmax、sigmoid和概率计算中；`log()`出现在交叉熵、对数似然和KL散度中。没有正确技巧的情况下，`log(exp(x))`的组合是个雷区。

### Log-Sum-Exp技巧

直接计算`log(sum(exp(x_i)))`在数值上是危险的。如果某个`x_i`很大，`exp(x_i)`会溢出；如果所有`x_i`都很小，每个`exp(x_i)`会下溢为零，`log(0)`就是`-inf`。

**技巧：在求幂之前减去最大值。**

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么有效：减去`max(x)`后，最大的指数是`exp(0) = 1`，不可能溢出。求和中至少有一项是1，所以总和至少是1，`log(1) = 0`，不可能下溢为`-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    （加减c）
= log(sum(exp(x_i - c) * exp(c)))               （exp(a+b) = exp(a)*exp(b)）
= log(exp(c) * sum(exp(x_i - c)))               （提取exp(c)）
= c + log(sum(exp(x_i - c)))                    （log(a*b) = log(a) + log(b)）
```

令`c = max(x)`，溢出被消除。

这个技巧在ML中无处不在：
- Softmax归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型
- 变分推断

### 为什么Softmax需要最大值减法技巧

Softmax将logit转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

没有技巧时，logit值[100, 101, 102]会导致溢出：

```
exp(100) = inf（在float32中，exp(88.7)已达到极限）
```

使用技巧，减去max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
总和 = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率结果完全相同，计算是安全的。这不是优化，而是**正确性的要求**。

### NaN和Inf：检测与预防

`nan`（非数字）和`inf`（无穷大）像病毒一样在计算中传播。梯度更新中的一个`nan`会使权重变成`nan`，进而使每个后续输出变成`nan`。训练在一步内就会崩溃。

`inf`的来源：
- 对大正数求`exp()`
- 除以零：`1.0 / 0.0`
- float32累加溢出

`nan`的来源：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 对负数求`sqrt()`
- 对负数求`log()`
- 涉及已有`nan`的任何运算

检测方法：

```python
import math

math.isnan(x)       # 若x为nan则为True
math.isinf(x)       # 若x为+inf或-inf则为True
math.isfinite(x)    # 若x既非nan又非inf则为True
```

预防策略：

1. 对`exp()`的输入截断：`exp(clamp(x, -80, 80))`
2. 在分母中加epsilon：`x / (y + 1e-8)`
3. 在`log()`内加epsilon：`log(x + 1e-8)`
4. 使用稳定实现（log-sum-exp、稳定softmax）
5. 梯度裁剪以防止权重爆炸
6. 调试期间每次前向传播后检查`nan`/`inf`

### 数值梯度检验

解析梯度（来自反向传播）可能有bug。数值梯度检验通过有限差分计算梯度来验证它们。

居中差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这是O(h^2)精度，远优于前向差分`(f(x+h) - f(x)) / h`的O(h)精度。

选择h的原则：太大则近似不准确，太小则灾难性消去破坏答案。典型值`h = 1e-5`到`1e-7`。

检验方法：计算解析梯度与数值梯度之间的相对差异。

```
相对误差 = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验规则：
- 相对误差 < 1e-7：完美，梯度正确
- 相对误差 < 1e-5：可接受，可能正确
- 相对误差 > 1e-3：有问题
- 相对误差 > 1：梯度完全错误

实现新层或损失函数时始终检查梯度。PyTorch提供`torch.autograd.gradcheck()`。

### 混合精度训练

现代GPU有专用硬件（Tensor Cores），计算float16矩阵乘法比float32快2-8倍。混合精度训练利用了这一点：

```
1. 维护float32主权重副本
2. 前向传播使用float16（快速）
3. 在float32中计算损失（防止溢出）
4. 反向传播使用float16（快速）
5. 将梯度缩放到float32
6. 更新float32主权重
```

纯float16训练的问题：梯度通常很小（1e-8或更小）。float16会将低于约6e-8的任何值下溢为零。你的模型停止学习，因为所有梯度更新都变成了零。

**修复方法是损失缩放：**

```
1. 将损失乘以一个大的缩放因子（例如1024）
2. 反向传播计算(损失 * 1024)的梯度
3. 所有梯度都放大了1024倍（推到float16可表示范围之上）
4. 在更新权重之前将梯度除以1024
5. 净效果：相同的更新，但没有下溢
```

动态损失缩放自动调整缩放因子：从大值（65536）开始，如果梯度溢出为`inf`则减半，如果N步没有溢出则加倍。

### bfloat16 vs float16：为什么bfloat16在训练中更优

```
float16:   [1位符号] [5位指数]  [10位尾数]
bfloat16:  [1位符号] [8位指数]  [7位尾数]
```

float16精度更高（10位尾数vs 7位），但范围有限（最大约65,504）。bfloat16精度较低但与float32相同的范围（最大约3.4e38）。

对于训练神经网络：
- 训练峰值期间，激活值和logit经常超过65,504。float16溢出；bfloat16能处理。
- float16需要损失缩放，bfloat16通常不需要，因为它的范围覆盖了梯度量级范围。
- bfloat16是float32的简单截断：丢弃尾数的低16位。转换简单，指数部分无损。

float16在推理时更优（值有边界，精度更重要），bfloat16在训练时更优（范围更重要）。这就是TPU和现代NVIDIA GPU（A100、H100）原生支持bfloat16的原因。

### 梯度裁剪

梯度爆炸在梯度通过许多层指数级增长时发生（常见于RNN、深度网络和Transformer）。单个大梯度可以在一步内破坏所有权重。

两种裁剪方式：

**按值裁剪：** 独立截断每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单，但可能改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量，使其范数不超过阈值。

```
如果 ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保留梯度方向，这就是`torch.nn.utils.clip_grad_norm_()`的做法，是标准选择。

典型值：Transformer使用`max_norm=1.0`，强化学习使用`max_norm=0.5`，简单网络使用`max_norm=5.0`。

梯度裁剪不是黑客手段，而是**安全机制**。没有它，单个异常批次产生的梯度就足以毁掉数周的训练。

### 归一化层作为数值稳定器

批归一化、层归一化和RMS归一化通常被认为是帮助训练收敛的正则化工具，但它们同样是数值稳定器。

没有归一化，激活值可能通过各层指数级增长或缩减：

```
第1层: 值在 [0, 1]
第5层: 值在 [0, 100]
第10层: 值在 [0, 10,000]
第50层: 值在 [0, inf]
```

归一化在每层对激活值重新中心化和缩放：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常1e-5）防止所有激活值相同时除以零。学习参数`gamma`和`beta`允许网络恢复它需要的任何尺度。

这使整个网络中的值保持在数值安全范围内，既防止前向传播中的溢出，也防止反向传播中的梯度爆炸。

### 常见ML数值Bug

**Bug：训练几个epoch后损失变成NaN。**
原因：logit增长过大，softmax溢出。或者学习率太高，权重发散。
修复：使用稳定softmax（最大值减法），降低学习率，添加梯度裁剪。

**Bug：损失卡在log(类别数)。**
原因：模型输出接近均匀概率。通常意味着梯度消失或模型根本没有学习。
修复：检查数据标签是否正确，验证损失函数，检查死亡ReLU。

**Bug：验证准确率比预期低1-3%。**
原因：混合精度训练没有正确的损失缩放。梯度下溢悄悄地将小更新归零。
修复：启用动态损失缩放，或改用bfloat16。

**Bug：某些层的梯度范数为0.0。**
原因：死亡ReLU神经元（所有输入为负），或float16下溢。
修复：使用LeakyReLU或GELU，使用梯度缩放，检查权重初始化。

**Bug：模型在一个GPU上正常，在另一个GPU上结果不同。**
原因：非确定性浮点累加顺序。GPU并行归约在不同硬件上以不同顺序求和，而浮点加法不满足结合律。
修复：接受小差异（1e-6），或设置`torch.use_deterministic_algorithms(True)`并接受速度损失。

**Bug：`exp()`在损失计算中返回`inf`。**
原因：原始logit直接传递给`exp()`，没有使用最大值减法技巧。
修复：使用`torch.nn.functional.log_softmax()`，它内部实现了log-sum-exp。

**Bug：从float32切换到float16后训练发散。**
原因：float16无法表示低于6e-8的梯度量级或高于65,504的激活值。
修复：使用带损失缩放的混合精度（AMP），或改用bfloat16。

## 动手实现

### 步骤1：演示浮点精度限制

```python
print("=== 浮点精度 ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"差值: {(0.1 + 0.2) - 0.3:.2e}")
```

### 步骤2：实现朴素与稳定softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"朴素:  {softmax_naive(safe_logits)}")
print(f"稳定:  {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"稳定:  {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) 会返回 [nan, nan, nan]
```

### 步骤3：实现稳定log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"朴素:  {logsumexp_naive(safe):.6f}")
print(f"稳定:  {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"稳定:  {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) 返回 inf
```

### 步骤4：实现稳定交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"朴素:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"稳定:  {cross_entropy_stable(true_class, logits):.6f}")
```

### 步骤5：梯度检验

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  参数 {i}: 解析={a:.8f} 数值={n:.8f} "
              f"相对误差={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 应用示例

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"原始范数: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"裁剪后范数: {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"方向保持: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"警告 {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("正常", [1.0, 2.0, 3.0])
check_tensor("含nan", [1.0, float('nan'), 3.0])
check_tensor("含inf", [1.0, float('inf'), 3.0])
```

完整实现（包含所有边界情况）见`code/numerical.py`。

## 产出物

本课程产出：
- `code/numerical.py` — 包含稳定softmax、log-sum-exp、交叉熵、梯度检验和混合精度模拟的完整实现
- `outputs/prompt-numerical-debugger.md` — 用于诊断训练中NaN/Inf和数值问题的提示模板

这些稳定实现会在第三阶段构建训练循环和第四阶段实现注意力机制时再次用到。

## 练习

1. **灾难性消去。** 用朴素公式`E[x^2] - E[x]^2`在float32中计算[1000000.0, 1000001.0, 1000002.0]的方差。然后用Welford在线算法计算。将两者与真实方差（0.6667）对比误差。

2. **精度探索。** 找到最小的正float32值`x`，使得Python中`1.0 + x == 1.0`。这就是机器精度。验证它与`numpy.finfo(numpy.float32).eps`是否一致。

3. **Log-sum-exp边界情况。** 用以下输入测试你的`logsumexp_stable`：(a) 所有值相等；(b) 一个值远大于其他；(c) 所有值很小（-1000）。验证它在朴素版本失败的地方给出正确结果。

4. **对神经网络层进行梯度检验。** 实现一个单线性层`y = Wx + b`及其解析反向传播。使用`numerical_gradient`验证3×2权重矩阵的正确性。

5. **损失缩放实验。** 模拟float16训练：创建范围在[1e-9, 1e-3]的随机梯度，转换为float16，测量变为零的比例。然后应用损失缩放（乘以1024），转换为float16，缩放回来，再次测量零比例。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| IEEE 754 | "浮点标准" | 定义二进制浮点格式、舍入规则和特殊值（inf、nan）的国际标准。每个现代CPU和GPU都实现它。 |
| 机器精度（Machine epsilon） | "精度极限" | 使给定浮点格式中1.0 + e != 1.0成立的最小值e。float32约为1.19e-7。 |
| 灾难性消去（Catastrophic cancellation） | "减法引起的精度损失" | 减去近似相等的浮点数时，有效数字抵消，舍入噪声主导结果。 |
| 溢出（Overflow） | "数字太大" | 结果超过最大可表示值，变成inf。exp(89)使float32溢出。 |
| 下溢（Underflow） | "数字太小" | 结果比最小可表示正数更接近零，变成0.0。exp(-104)使float32下溢。 |
| Log-sum-exp技巧 | "先减去最大值" | 通过提取exp(max(x))来计算log(sum(exp(x)))，防止溢出和下溢。用于softmax、交叉熵和对数概率运算。 |
| 稳定softmax（Stable softmax） | "不爆炸的softmax" | 在求幂之前减去max(logits)。数值上等价的结果，不可能溢出。 |
| 梯度检验（Gradient checking） | "验证反向传播" | 将反向传播的解析梯度与有限差分的数值梯度对比，以发现实现bug。 |
| 混合精度（Mixed precision） | "float16前向，float32反向" | 对计算密集操作使用低精度浮点，对数值敏感操作使用高精度浮点。典型加速2-3倍。 |
| 损失缩放（Loss scaling） | "防止梯度下溢" | 反向传播前将损失乘以大常数，使梯度保持在float16可表示范围内，更新权重前再除以同一常数。 |
| bfloat16 | "Brain浮点" | Google的16位格式，有8位指数（与float32相同的范围）和7位尾数（精度低于float16）。训练首选。 |
| 梯度裁剪（Gradient clipping） | "限制梯度范数" | 缩放梯度向量使其范数不超过阈值。防止爆炸梯度破坏权重。 |
| NaN | "非数字" | 来自未定义操作（0/0、inf-inf、sqrt(-1)）的特殊浮点值，在所有后续运算中传播。 |
| Inf | "无穷大" | 来自溢出或除以零的特殊浮点值，可以组合产生NaN（inf - inf、inf * 0）。 |
| 数值梯度（Numerical gradient） | "暴力求导" | 通过求值f(x+h)和f(x-h)并除以2h来近似导数。慢但可靠，用于验证。 |

## 延伸阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) — 权威参考，内容密集但全面
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) — 引入float16训练损失缩放的NVIDIA论文
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) — PyTorch混合精度的实践指南
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) — Google为TPU选择此格式的原因
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) — 减少浮点求和舍入误差的算法

# 图像分类

> 分类器是从像素到类别概率分布的函数。其他一切都是管道工程。

**类型：** 动手实现
**语言：** Python
**前置知识：** Phase 2 第9课（模型评估），Phase 3 第10课（迷你框架），Phase 4 第3课（CNN）
**预计时间：** ~75分钟

## 学习目标

- 在 CIFAR-10 上构建端到端的图像分类流水线：数据集、数据增强、模型、训练循环、评估
- 解释每个组件（数据加载器、损失函数、优化器、调度器、数据增强）的作用，并预测破坏其中任何一个在损失曲线上的表现
- 从零实现 mixup、cutout 和标签平滑，并说明各自何时值得添加
- 阅读混淆矩阵和逐类精确率/召回率表，诊断超出综合准确率之外的数据集和模型问题

## 问题所在

每个实际发布的视觉任务，在某种程度上都归结为图像分类。检测是对区域分类，分割是对像素分类，检索是按与类别中心的相似度排名。把分类做对——数据集循环、数据增强策略、损失函数、评估——是能迁移到这个 Phase 所有其他任务的技能。

大多数分类 bug 不在模型里，而活在流水线中：错误的归一化、未打乱的训练集、扭曲标签的数据增强、被训练数据污染的验证集分割、在第 30 轮后悄悄发散的学习率。一个正确设置下能在 CIFAR-10 上达到 93% 的 CNN，在破损的设置下通常只能得到 70-75%，而损失曲线看起来一直都还正常。

本课手工连接整个流水线，使每个部分都可检查。我们不会使用 `torchvision.datasets` 中任何可能隐藏 bug 的东西。

## 核心概念

### 分类流水线

```mermaid
flowchart LR
    A["数据集<br/>（图像+标签）"] --> B["数据增强<br/>（随机变换）"]
    B --> C["归一化<br/>（均值/标准差）"]
    C --> D["数据加载器<br/>（批处理+打乱）"]
    D --> E["模型<br/>（CNN）"]
    E --> F["Logits<br/>（N, C）"]
    F --> G["交叉熵损失"]
    F --> H["Argmax<br/>（评估时）"]
    G --> I["反向传播"]
    I --> J["优化器步进"]
    J --> K["调度器步进"]
    K --> E

    style A fill:#dbeafe,stroke:#2563eb
    style E fill:#fef3c7,stroke:#d97706
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#dcfce7,stroke:#16a34a
```

这个循环中的每一行都可能住着 bug。交叉熵接受原始 logits，不是 softmax 输出，所以损失之前的任何 `model(x).softmax()` 都会悄悄计算错误的梯度。数据增强只应用于输入，不应用于标签——除了 mixup，它同时混合两者。`optimizer.zero_grad()` 必须在每一步之前调用一次；跳过它会累积梯度，看起来像极不稳定的学习率。这些 bug 中的每一个都会让学习曲线变平，但不会报错。

### 交叉熵、Logits 与 Softmax

分类器对每张图像产生 `C` 个数字，称为 logits。对它们应用 softmax 将其转换为概率分布：

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

交叉熵测量正确类别的负对数概率：

```
CE(z, y) = -log( softmax(z)_y )
         = -z_y + log( sum_j exp(z_j) )
```

右边的形式是数值稳定的（log-sum-exp）。PyTorch 的 `nn.CrossEntropyLoss` 将 softmax + NLL 融合在一个操作中，直接接受原始 logits。自己先应用 softmax 几乎总是一个 bug——你会计算 log(softmax(softmax(z)))，这是一个无意义的量。

### 为什么数据增强有效

CNN 对平移有归纳偏置（来自权重共享），但对裁剪、翻转、颜色抖动或遮挡没有内置的不变性。教它这些不变性的唯一方法是让它看到能练习这些变换的像素。训练时的每个随机变换，都是在说：「这两张图像有相同的标签；学习忽略差异的特征。」

```
原始裁剪:   「朝左的狗」
翻转:       「朝右的狗」       <- 相同标签，不同像素
旋转(+15°): 「轻微倾斜的狗」
颜色抖动:   「在更暖光线下的狗」
随机擦除:   「有一块缺失的狗」
```

规则：数据增强必须保持标签。对数字做 cutout 和旋转可能把「6」变成「9」；对于那个数据集，你要用更小的旋转范围，并选择尊重数字特定不变性的增强。

### Mixup 与 Cutmix

普通数据增强变换像素但保持标签为 one-hot。**Mixup** 和 **Cutmix** 通过同时插值两者来打破这一点。

```
Mixup:
  lambda ~ Beta(a, a)
  x = lambda * x_i + (1 - lambda) * x_j
  y = lambda * y_i + (1 - lambda) * y_j

Cutmix:
  将 x_j 的一个随机矩形区域粘贴到 x_i 中
  y = y_i 和 y_j 按面积加权混合
```

为什么有效：模型停止记忆尖锐的 one-hot 目标，学习在类别之间插值。训练损失上升，测试准确率上升。这是对任何分类器最便宜的单一鲁棒性升级。

### 标签平滑

Mixup 的近亲。不是对 `[0, 0, 1, 0, 0]` 训练，而是对 `[eps/C, eps/C, 1-eps, eps/C, eps/C]` 训练，`eps` 取一个小值如 0.1。阻止模型产生任意尖锐的 logits，以几乎零代价改善校准。自 PyTorch 1.10 起内置于 `nn.CrossEntropyLoss(label_smoothing=0.1)`。

### 超越准确率的评估

综合准确率隐藏了不平衡问题。一个总是预测多数类的 90-10 二分类器得分 90%。真正告诉你发生了什么的工具：

- **逐类准确率** — 每个类别一个数字；立即显示表现不佳的类别。
- **混淆矩阵** — C×C 网格，第 i 行 j 列 = 真实类别为 i 但预测为 j 的计数；对角线是正确的，非对角线是你的模型「活着」的地方。
- **Top-1 / Top-5** — 正确类别是否在前 1 或前 5 个预测中；Top-5 对 ImageNet 很重要，因为「诺里奇梗」vs「诺福克梗」这样的类别本质上有歧义。
- **校准（ECE）** — 0.8 置信度的预测有 80% 的时间是正确的吗？现代网络系统性地过于自信；用温度缩放或标签平滑修复。

## 动手实现

### 第1步：确定性的合成数据集

CIFAR-10 存在于磁盘上。为了使本课可复现且快速，我们构建一个合成数据集，它看起来像 CIFAR——32×32 的 RGB 图像，带有模型必须学习的类别特定结构。完全相同的流水线可以不加修改地用于真实的 CIFAR-10。

```python
import numpy as np
import torch
from torch.utils.data import Dataset


def synthetic_cifar(num_per_class=1000, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    Y = []
    for c in range(num_classes):
        centre = rng.uniform(0, 1, (3,))
        freq = 2 + c
        for _ in range(num_per_class):
            yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
            r = np.sin(xx * freq) * 0.5 + centre[0]
            g = np.cos(yy * freq) * 0.5 + centre[1]
            b = (xx + yy) * 0.5 * centre[2]
            img = np.stack([r, g, b], axis=-1)
            img += rng.normal(0, 0.08, img.shape)
            img = np.clip(img, 0, 1)
            X.append(img.astype(np.float32))
            Y.append(c)
    X = np.stack(X)
    Y = np.array(Y)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


class ArrayDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, int(self.Y[i])
```

每个类别有自己的颜色调色板和频率模式，加上高斯噪声迫使模型学习信号而不是记住像素。十个类别，每类一千张图像，随机打乱。

### 第2步：归一化与数据增强

每个视觉流水线都必须有的两种变换。

```python
def standardize(mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    def _fn(img):
        return (img - mean) / std
    return _fn


def random_hflip(p=0.5):
    def _fn(img):
        if np.random.random() < p:
            return img[:, ::-1, :].copy()
        return img
    return _fn


def random_crop(pad=4):
    def _fn(img):
        h, w = img.shape[:2]
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        y = np.random.randint(0, 2 * pad)
        x = np.random.randint(0, 2 * pad)
        return padded[y:y + h, x:x + w, :]
    return _fn


def compose(*fns):
    def _fn(img):
        for fn in fns:
            img = fn(img)
        return img
    return _fn
```

裁剪前用反射填充，而不是零填充，因为黑色边框是一种信号，模型会学着以无用的方式忽略它。

### 第3步：Mixup

在训练步骤中混合两张图像和两个标签。作为批变换实现，使其靠近前向传播而不是在数据集内部。

```python
def mixup_batch(x, y, num_classes, alpha=0.2):
    if alpha <= 0:
        return x, torch.nn.functional.one_hot(y, num_classes).float()
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_onehot = torch.nn.functional.one_hot(y, num_classes).float()
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x_mixed, y_mixed


def soft_cross_entropy(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()
```

`soft_cross_entropy` 是对软标签分布的交叉熵。当目标恰好是 one-hot 时，它退化为通常的形式。

### 第4步：训练循环

完整方案：对数据进行一次遍历，每批次计算一次梯度，每轮次步进一次调度器。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, device, num_classes, use_mixup=True):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x_m, y_soft = mixup_batch(x, y, num_classes)
            logits = model(x_m)
            loss = soft_cross_entropy(logits, y_soft)
        else:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        # 当 mixup 开启时，训练准确率是针对未混合标签 y 的近似值
        # （模型看到的是软目标，而不是 y）。把它作为进度信号；
        # 真正的性能依赖验证准确率。
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        for t, p in zip(y.cpu(), pred.cpu()):
            cm[t, p] += 1
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        correct += (pred == y).sum().item()
    return loss_sum / total, correct / total, cm
```

每次写训练循环都要检查的五个不变量：

1. 训练前 `model.train()`，评估前 `model.eval()`——切换 dropout 和 batchnorm 的行为。
2. `.backward()` 之前调用 `.zero_grad()`。
3. 累积指标时用 `.item()`，防止任何东西保持计算图存活。
4. 评估时用 `@torch.no_grad()`——节省内存和时间，防止微妙的意外。
5. 对原始 logits 做 argmax，不对 softmax——结果相同，少一个操作。

### 第5步：组合在一起

使用前一课的 `TinyResNet`，训练几轮，进行评估。

```python
from main import synthetic_cifar, ArrayDataset
from main import standardize, random_hflip, random_crop, compose
from main import mixup_batch, soft_cross_entropy
from main import train_one_epoch, evaluate
# TinyResNet 来自前一课（03-cnns-lenet-to-resnet）
from cnns_lenet_to_resnet import TinyResNet  # 示例占位符

X, Y = synthetic_cifar(num_per_class=500)
split = int(0.9 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
train_tf = compose(random_hflip(), random_crop(pad=4), standardize(mean, std))
eval_tf = standardize(mean, std)

train_ds = ArrayDataset(X_train, Y_train, transform=train_tf)
val_ds = ArrayDataset(X_val, Y_val, transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyResNet(num_classes=10).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, 10, use_mixup=True)
    va_loss, va_acc, _ = evaluate(model, val_loader, device, 10)
    scheduler.step()
    print(f"epoch {epoch:2d}  lr {scheduler.get_last_lr()[0]:.4f}  "
          f"train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
```

在合成数据集上，这在五轮内就能达到接近完美的验证准确率，这正是重点：流水线是正确的，模型能学习可学的东西。换成真实的 CIFAR-10，同样的循环在不做任何修改的情况下可以训练到约 90%。

### 第6步：读取混淆矩阵

仅靠准确率永远不会告诉你模型在哪里失败，混淆矩阵会。

```python
def print_confusion(cm, labels=None):
    c = cm.shape[0]
    labels = labels or [str(i) for i in range(c)]
    print(f"{'':>6}" + "".join(f"{l:>5}" for l in labels))
    for i in range(c):
        row = cm[i].tolist()
        print(f"{labels[i]:>6}" + "".join(f"{v:>5}" for v in row))
    print()
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    prec = tp / (tp + fp).clamp_min(1)
    rec = tp / (tp + fn).clamp_min(1)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
    for i in range(c):
        print(f"{labels[i]:>6}  prec {prec[i]:.3f}  rec {rec[i]:.3f}  f1 {f1[i]:.3f}")

_, _, cm = evaluate(model, val_loader, device, 10)
print_confusion(cm)
```

行是真实类别，列是预测类别。第 3 类和第 5 类之间的非对角线计数聚集，意味着模型混淆了这两个类，为你提供了有针对性的数据收集或类别特定数据增强的起点。

## 实际使用

`torchvision` 将上面的所有内容封装成惯用组件。对于真实的 CIFAR-10，完整流水线是四行加一个训练循环。

```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
train_tf = Compose([
    RandomCrop(32, padding=4, padding_mode="reflect"),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean, std),
])
eval_tf = Compose([ToTensor(), Normalize(mean, std)])

train_ds = CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
val_ds   = CIFAR10(root="./data", train=False, download=True, transform=eval_tf)
```

两点需要注意：均值/标准差是**数据集特定的**——在 CIFAR-10 训练集上计算，而非 ImageNet——而反射填充是社区默认的裁剪策略。在这里复制粘贴 ImageNet 统计量是约 1% 的准确率泄漏，没人发现，直到有人对模型进行分析。

## 练习

1. **(简单)** 在合成数据集上，有 mixup 和无 mixup 各训练同一模型五轮。绘制两者的训练和验证损失曲线。解释为什么 mixup 的训练损失更高，而验证准确率却相似或更好。
2. **(中等)** 实现 Cutout——在每张训练图像中随机清零一个 8×8 的方块——并进行对比实验：无增强、hflip+crop、hflip+crop+cutout、hflip+crop+mixup。报告每种的验证准确率。
3. **(困难)** 构建一个 CIFAR-100 流水线（100 个类别，相同输入尺寸），并复现 ResNet-34 训练结果在已发布准确率的 1% 以内。附加：扫描三个学习率和两个权重衰减，记录到本地 CSV，生成最终的混淆矩阵-最高混淆对表格。

## 关键术语

| 术语 | 通常的说法 | 准确含义 |
|------|-----------|---------|
| Logits | 「原始输出」 | 每张图像的 C 个 softmax 前的数值；交叉熵期望这些值，而不是 softmax 后的值 |
| 交叉熵 (Cross-entropy) | 「损失函数」 | 正确类别的负对数概率；在一个稳定的操作中融合 log-softmax 和 NLL |
| 数据加载器 (DataLoader) | 「批处理器」 | 包裹数据集，提供打乱、批处理和（可选的）多工作进程加载；是一半训练 bug 的背锅侠 |
| 数据增强 (Augmentation) | 「随机变换」 | 训练时保持标签不变的任何像素级变换；教导 CNN 本身没有的不变性 |
| Mixup / Cutmix | 「混合两张图像」 | 同时混合输入和标签，让分类器学习平滑插值而不是硬边界 |
| 标签平滑 (Label smoothing) | 「更软的目标」 | 将 one-hot 替换为 (1-eps, eps/(C-1), ...)；改善校准并轻微提升准确率 |
| Top-k 准确率 (Top-k accuracy) | 「Top-5」 | 正确类别在 k 个最高概率预测中；用于具有本质歧义类别的数据集 |
| 混淆矩阵 (Confusion matrix) | 「错误住在哪里」 | C×C 表，条目 (i,j) 计数真实类别为 i 但预测为 j 的图像数；对角线是正确的，非对角线告诉你要修什么 |

## 延伸阅读

- [CS231n: Training Neural Networks](https://cs231n.github.io/neural-networks-3/) — 在一页内对训练流水线最清晰的介绍
- [Bag of Tricks for Image Classification (He et al., 2019)](https://arxiv.org/abs/1812.01187) — 合在一起在 ImageNet 上为 ResNet 准确率增加 3-4% 的每个小技巧
- [mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)](https://arxiv.org/abs/1710.09412) — 原始 mixup 论文；三页理论加令人信服的实验
- [Why temperature scaling matters (Guo et al., 2017)](https://arxiv.org/abs/1706.04599) — 证明现代网络校准不足并用一个标量参数修复它的论文

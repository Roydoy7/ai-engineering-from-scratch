# JAX 入门

> PyTorch 修改张量，TensorFlow 构建图，JAX 编译纯函数。最后这个会改变你对深度学习的思考方式。

## 核心问题

你知道如何用 PyTorch 构建神经网络：定义 `nn.Module`，调用 `.backward()`，步进优化器。它能用，数百万人在用。

但 PyTorch 的 DNA 里有一个约束：它在 Python 中即时地、一次一个地追踪操作。每个 `tensor + tensor` 都是一个单独的内核启动，每个训练步骤都重新解释相同的 Python 代码。这在你需要在 2048 块 TPU 上训练 5400 亿参数模型之前都没问题。那时候额外开销会杀死你。

Google DeepMind 在 JAX 上训练 Gemini。Anthropic 在 JAX 上训练 Claude。这些不是小规模运算——它们是地球上最大的神经网络训练。它们选择 JAX，因为它把你的训练循环当作一个可编译的程序，而不是一系列 Python 调用。

JAX 是具有三种超能力的 NumPy：自动微分、JIT 编译到 XLA 和自动向量化。你写一个处理单个样本的函数，JAX 给你一个处理批次、计算梯度、编译到机器码、跨多个设备运行的函数。所有这些无需修改原始函数。

---

## JAX 哲学

JAX 是一个**函数式框架**。没有类，没有可变状态，没有 `.backward()` 方法。相反：

| PyTorch | JAX |
|---------|-----|
| 有状态的 `nn.Module` 类 | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 即时执行 | 通过 XLA 的 JIT 编译 |
| `for x in batch:` 手动循环 | `jax.vmap(f)` 自动向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自动并行 |
| 可变的 `model.parameters()` | 不可变的数组 pytree |

这不是风格偏好——这是编译器约束。JIT 编译要求纯函数：相同输入总是产生相同输出，没有副作用。正是这个限制使 100 倍加速成为可能。

---

## jax.numpy：熟悉的接口

JAX 在加速器上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

相同的函数名，相同的广播规则，相同的切片语义。但数组存在于 GPU/TPU 上，每个操作都可以被编译器追踪。

**一个关键区别：JAX 数组是不可变的。** 没有 `a[0] = 5`。代替它：`a = a.at[0].set(5)`。这开始感觉奇怪，一周后就豁然开朗了——不可变性正是让 `grad`、`jit`、`vmap` 可以组合的原因。

---

## jax.grad：函数式自动微分

PyTorch 把梯度附加到张量（`.grad`）。JAX 把梯度附加到函数。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)  # 返回 6.0
```

`jax.grad` 接受一个函数并返回一个计算梯度的新函数。没有 `.backward()` 调用，没有存储在张量上的计算图，梯度只是另一个你可以调用、组合或 JIT 编译的函数。

**可以任意组合：**

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)  # 二阶导数
```

二阶导数、三阶导数、雅可比矩阵、海森矩阵——全部通过组合 `grad` 实现。PyTorch 也可以做到（`torch.autograd.functional.hessian`），但那是"螺栓拧上去的"。在 JAX 中，这是基础。

**约束：** `grad` 只能作用于纯函数。函数内部不能有 print 语句（它们在追踪时运行，不在执行时），不能修改外部状态，不能在没有显式密钥管理的情况下生成随机数。

---

## jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss
```

第一次调用时，JAX **追踪**函数——记录哪些操作发生，不实际执行。然后把追踪结果交给 XLA（加速线性代数），Google 的 TPU 和 GPU 编译器。XLA 融合操作，消除多余的内存拷贝，生成优化的机器码。

后续调用完全跳过 Python——编译好的代码以 C++ 的速度在加速器上运行。

**JIT 有帮助时：**
- 训练步骤（重复数千次的相同计算）
- 推理（相同模型，不同输入）
- 任何用相似形状输入调用多次的函数

**JIT 有害时：**
- 包含依赖值的 Python 控制流（`if x > 0`，其中 x 是被追踪的数组）
- 一次性计算（编译开销超过运行时间）
- 调试（追踪隐藏了实际执行）

控制流限制是真实的。`jax.lax.cond` 替代 `if/else`，`jax.lax.scan` 替代 `for` 循环。这不是可选的——这是编译的代价。

---

## vmap：自动向量化

你写一个处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 把它提升为处理批次：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 的意思是：不对 `params` 批处理（共享），对 `x` 的第 0 轴批处理。没有手动的 `for` 循环，没有变形，没有批次维度的传递。JAX 自动识别批次维度并向量化整个计算。

这不是语法糖。`vmap` 生成融合的向量化代码，比 Python 循环快 10-100 倍。而且它与 `jit` 和 `grad` 可以组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

逐样本梯度，一行代码。在 PyTorch 中这几乎不可能不用黑科技实现。

---

## pmap：跨设备数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 在所有可用设备（GPU/TPU）上复制函数并分割批次。在函数内部，`jax.lax.pmean` 和 `jax.lax.psum` 在设备间同步梯度。

Google 用 `pmap`（及其后继者 `shard_map`）在数千块 TPU v5e 上训练 Gemini。编程模型：写单设备版本，用 `pmap` 包装，完成。

---

## Pytree：通用数据结构

JAX 操作的是"pytree"——列表、元组、字典和数组的嵌套组合。你的模型参数是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每个 JAX 变换——`grad`、`jit`、`vmap`——都知道如何遍历 pytree。`jax.tree.map(f, tree)` 把 `f` 应用到每个叶节点。这就是优化器如何一次更新所有参数：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()` 方法，没有参数注册。树结构就是模型。

---

## JAX 生态系统

JAX 提供基础原语，库提供人体工程学：

| 库 | 角色 | 风格 |
|----|------|------|
| **Flax**（Google） | 神经网络层 | 有显式状态的 `nn.Module` |
| **Equinox**（Patrick Kidger） | 神经网络层 | 基于 pytree，更 Pythonic |
| **Optax**（DeepMind） | 优化器 + 学习率调度 | 可组合的梯度变换 |
| **Orbax**（Google） | 检查点 | 保存/恢复 pytree |

Optax 是标准优化器库。它把梯度变换（Adam、SGD、裁剪）与参数更新分离，使组合变得简单：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

---

## JAX vs PyTorch：如何选择

| 因素 | JAX | PyTorch |
|------|-----|---------|
| TPU 支持 | 一等公民（Google 同时构建了两者） | 社区维护（torch_xla） |
| GPU 支持 | 良好（通过 XLA 的 CUDA） | 最佳（原生 CUDA） |
| 调试 | 困难（追踪 + 编译） | 简单（即时，逐行） |
| 生态系统 | 研究导向（Flax, Equinox） | 庞大（HuggingFace, torchvision等） |
| 招聘 | 小众（Google/DeepMind/Anthropic） | 主流（随处可见） |
| 大规模训练 | 优越（XLA, pmap, mesh） | 良好（FSDP, DeepSpeed） |
| 原型速度 | 较慢（函数式开销） | 较快（改了就跑） |
| 主要用户 | DeepMind（Gemini）、Anthropic（Claude） | Meta（LLaMA）、OpenAI（GPT）、Stability AI |

**诚实的答案：** 除非有特定原因，否则用 PyTorch。那些原因是：TPU 访问、需要逐样本梯度、大规模多设备训练，或者在 Google/DeepMind/Anthropic 工作。

---

## JAX 中的随机数

JAX 没有全局随机状态。每个随机操作都需要一个显式的 PRNG 密钥：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

一开始很烦人，但它保证了跨设备和编译的可重现性——这是 PyTorch 的 `torch.manual_seed` 在多 GPU 设置中无法保证的。

---

## 从零实现：在 MNIST 上训练 MLP

### 第一步：设置和数据

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 第二步：初始化参数

没有类，只是一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    params = {
        'layer1': {
            'w': jnp.sqrt(2.0 / 784) * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': jnp.sqrt(2.0 / 256) * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': jnp.sqrt(2.0 / 128) * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

手动的 Kaiming 初始化，一个种子分裂出三个 PRNG 密钥。每个权重都是嵌套字典中的不可变数组。

### 第三步：前向传播

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数。参数进来，预测出去。没有 `self`，没有存储状态。

### 第四步：JIT 编译的训练步骤

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 在一次遍历中同时返回损失值和梯度。`@jax.jit` 装饰器把两个函数都编译到 XLA。第一次调用后，每个训练步骤不再触碰 Python。

### 第五步：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"第 {epoch + 1:2d} 轮 | 损失: {epoch_loss / n_batches:.4f} | "
          f"训练准确率: {train_acc:.4f} | 测试准确率: {test_acc:.4f}")
```

注意缺少了什么：没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个更新是一个组合的函数调用——梯度被计算、被 Adam 变换、被应用到参数，全部在 `train_step` 内部完成。

---

## Flax：Google 标准

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

与 PyTorch 结构相同，但 `params` 与模型分离。`model.init()` 创建参数，`model.apply(params, x)` 运行前向传播。模型对象没有状态。

## Optax：可组合优化器

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热、权重衰减——全部作为变换链组合。每个变换看到梯度，修改它，传给下一个。没有单体优化器类。

---

## 安装

```bash
pip install jax jaxlib optax flax

# GPU 支持
pip install jax[cuda12]

# TPU（Google Cloud）
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能注意事项：**

- 第一次 JIT 调用很慢（编译）。基准测试前先预热。
- 避免在 JIT 内部对 JAX 数组做 Python 循环，用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 在 JIT 内部有效，普通 `print()` 无效。
- JAX 默认预分配 75% 的 GPU 内存，设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 禁用。

---

## 关键术语

| 术语 | 英文 | 含义 |
|------|------|------|
| XLA | XLA | 加速线性代数——融合操作并从计算图生成优化 GPU/TPU 内核的编译器 |
| JIT | JIT | 即时编译——第一次调用时追踪函数，编译到 XLA，后续调用运行编译版本 |
| 纯函数 | Pure Function | 输出只依赖输入的函数——没有全局状态、没有突变、没有无显式密钥的随机数 |
| vmap | vmap | 把处理单个样本的函数变换为处理批次，无需重写 |
| pmap | pmap | 在多个设备上复制函数并分割输入批次 |
| Pytree | Pytree | JAX 可以遍历和变换的列表、元组、字典和数组的任意嵌套结构 |
| 追踪 | Tracing | JAX 用抽象值执行函数来构建计算图，不计算实际结果 |
| 函数式自动微分 | Functional Autodiff | 通过变换函数来计算导数，而不是在张量上附加梯度存储 |
| Optax | Optax | JAX 的可组合梯度变换库：Adam、SGD、裁剪、调度，可以链式组合 |
| Flax | Flax | Google 为 JAX 提供的神经网络库，增加层抽象同时保持状态显式 |

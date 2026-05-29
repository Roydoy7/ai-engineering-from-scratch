# 音频分类——从 k-NN+MFCC 到 AST 和 BEATs

> 从"狗叫 vs 警报声"到"这是哪种语言"，都是音频分类。特征是 Mel 频谱，架构每十年换一代，评估指标始终是 AUC、F1 和逐类召回率。

**类型：** 构建
**语言：** Python
**前置知识：** 第6阶段第2课（频谱图与 Mel）、第3阶段第6课（CNN）、第5阶段第8课（文本的 CNN 与 RNN）
**预计时间：** 约75分钟

## 问题背景

你拿到一段 10 秒的音频，想知道："这是什么？"城市声音（警报、钻机、狗叫）、语音指令（yes/no/stop）、语言识别（中/英/阿）、说话人情绪（愤怒/中性），还是环境声（室内/室外、嘈杂）？这些都是*音频分类*，2026 年基线架构已经成熟：对数 Mel → CNN 或 Transformer → softmax。

核心难点不在网络，在数据。音频数据集有严重的类别不平衡、强烈的领域偏移（干净 vs 嘈杂），以及标注噪声（谁来决定"城市嘈杂"和"餐厅噪声"的区别？）。80% 的问题来自数据整理、数据增强和评估设计，而不是把 CNN 换成 Transformer。

## 核心概念

**k-NN + MFCC（1990 年代基线）**：对每个音频片段展平 MFCC，与标注样本库计算余弦相似度，取 top-K 多数投票。在干净的小数据集（Speech Commands、ESC-50）上出奇地强，无需 GPU。

**2D CNN + 对数 Mel（2015-2019）**：把 `(T, n_mels)` 对数 Mel 当作图像，套 ResNet-18 或 VGG 风格，在时间轴做全局平均池化，然后 softmax。2026 年大多数 Kaggle 竞赛的基线仍然是这个。

**音频频谱图 Transformer，AST（2021-2024）**：对对数 Mel 做分块（如 16×16），加位置嵌入，送入 ViT。在监督学习中是 AudioSet 的最高水平（mAP 0.485）。

**BEATs 和 WavLM-base（2024-2026）**：在数百万小时数据上做自监督预训练，用你原本需要的 1-10% 监督数据微调即可达标。2026 年非语音音频的默认起点。BEATs-iter3 在 AudioSet 上比 AST 多 1-2 mAP，同时只用四分之一的计算量。

**Whisper 编码器作为冻结骨干（2024）**：取 Whisper 的编码器，去掉解码器，接一个线性分类器。在语言识别和简单事件分类上接近 SOTA，且无需任何音频增强。这是最容易拿到的"免费午餐"基线。

### 类别不平衡才是真正的挑战

ESC-50：50 类，每类 40 个片段——均衡，简单。UrbanSound8K：10 类，10:1 不平衡。AudioSet：632 类，长尾比例达 100000:1。有效的技术：

- 训练时做均衡采样（评估时不做）。
- Mixup：对两段音频（及其标签）做线性插值作为数据增强。
- SpecAugment：随机遮蔽时间轴和频率轴的随机区间，简单但关键。

### 评估指标

- 多类互斥（Speech Commands）：top-1 准确率、top-5 准确率。
- 多类多标签（AudioSet、UrbanSound 风格）：平均精度均值（mAP）。
- 严重不平衡：逐类召回率 + 宏平均 F1。

2026 年你应该知道的数字：

| 基准 | 基线 | 2026 SOTA | 来源 |
|------|------|-----------|------|
| ESC-50 | 82%（AST） | 97.0%（BEATs-iter3） | BEATs 论文（2024） |
| AudioSet mAP | 0.485（AST） | 0.548（BEATs-iter3） | HEAR 排行榜 2026 |
| Speech Commands v2 | 98%（CNN） | 99.0%（Audio-MAE） | HEAR v2 结果 |

## 动手实现

### 第一步：特征提取

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 第二步：固定长度摘要

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单但有效：跨时间轴的均值+方差给出一个 26 维的固定长度嵌入（13 个 MFCC 系数）。瞬间计算完成。早在 2017 年就能击败 ESC-50 上的 SOTA 神经网络基线。

### 第三步：k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 第四步：升级为对数 Mel + CNN

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

300 万参数，在单张 RTX 4090 上训练 ESC-50 约需 10 分钟，准确率 80%+。

### 第五步：2026 年默认方案——微调 BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

BEATs 用 `microsoft/BEATs-base` 通过 `beats` 库调用，transformers API 形状相同。

## 工程应用

2026 年技术栈：

| 情况 | 起点 |
|------|------|
| 小数据集（<1000 片段） | MFCC 均值 k-NN（你的基线）+ 音频增强 |
| 中等数据集（1K-100K） | BEATs 或 AST 微调 |
| 大数据集（>100K） | 从头训练或微调 Whisper 编码器 |
| 实时、边缘部署 | 40-MFCC CNN，int8 量化（KWS 风格） |
| 多标签（AudioSet） | BEATs-iter3 + BCE 损失 + Mixup + SpecAugment |
| 语言识别 | MMS-LID，SpeechBrain VoxLingua107 基线 |

决策规则：**从冻结骨干开始，而不是全新模型**。微调 BEATs 的分类头只需几小时就能达到 SOTA 的 95%，而不是几周。

## 交付物

保存为 `outputs/skill-classifier-designer.md`。根据给定的音频分类任务，选择架构、数据增强策略、类别平衡方案和评估指标。

## 练习

1. **（简单）** 运行 `code/main.py`，它在一个 4 类合成数据集（不同音高的纯音）上训练 k-NN MFCC 基线，报告混淆矩阵。
2. **（中等）** 将 `summarize` 替换为 [均值、方差、偏度、峰度]，4 阶矩池化是否比均值+方差在相同合成数据集上表现更好？
3. **（困难）** 用 `torchaudio` 在 ESC-50 第 1 折上训练 2D CNN，报告 5 折交叉验证准确率，再加上 SpecAugment（时间遮蔽 = 20，频率遮蔽 = 10），报告准确率变化。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| AudioSet | "音频界的 ImageNet" | Google 的 200 万片段、632 类弱标注 YouTube 数据集 |
| ESC-50 | "小型分类基准" | 50 类 × 40 个环境声片段 |
| AST | "音频频谱图 Transformer" | 对对数 Mel 分块的 ViT，2021 年 SOTA |
| BEATs | "自监督音频模型" | 微软模型，iter3 截至 2026 年领跑 AudioSet |
| Mixup | "配对增强" | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2` |
| SpecAugment | "遮蔽增强" | 在频谱图上随机置零时间和频率区间 |
| mAP | "主要多标签指标" | 跨类别和阈值的平均精度均值 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021-2024 年的主流架构
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024 年以来的默认起点
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 主流音频增强方法
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 延续至今的 50 类基准
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632 类 YouTube 分类体系，仍是黄金标准

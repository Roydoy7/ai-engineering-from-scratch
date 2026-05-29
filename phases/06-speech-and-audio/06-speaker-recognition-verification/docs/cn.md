# 说话人识别与验证

> ASR 回答"他们说了什么"，说话人识别回答"是谁说的"。数学形式看起来相同——嵌入加余弦相似度——但每一个生产决策都取决于一个数字：EER。

**类型：** 构建
**语言：** Python
**前置知识：** 第6阶段第2课（频谱图与 Mel）、第5阶段第22课（嵌入模型深入）
**预计时间：** 约45分钟

## 问题背景

用户说一句口令。你想知道：这个人是否就是他声称的那个人（*验证*，1:1）？还是在注册库中的某个人（*识别*，1:N）？或者两者都不是——这是一个未知说话人（*开放集*）？

2018 年前：GMM-UBM + i-vector，EER 尚可但对信道偏移（手机 vs 笔记本）和情绪变化脆弱。2018–2022 年：x-vector（TDNN 骨干 + 角度间隔训练）。2022 年至今：ECAPA-TDNN 和 WavLM-large 嵌入。到 2026 年，该领域由三个模型和一个指标主导。

这个指标就是 **EER**——等错误率。设置决策阈值使误接受率 = 误拒绝率，交叉点即为 EER。每篇论文、每个排行榜、每次采购评估都用它。

## 核心概念

**流水线**。注册：录制目标说话人 5–30 秒的音频，计算固定维度嵌入（ECAPA-TDNN 为 192 维，WavLM-large 为 256 维）。验证：计算测试话语的嵌入，计算余弦相似度，与阈值比较。

**ECAPA-TDNN（2020 年，2026 年仍主流）**。强调通道注意力、传播与聚合的时延神经网络。带 squeeze-excitation 的一维卷积块，多头注意力池化，再接一个线性层输出 192 维。在 VoxCeleb 1+2（2700 个说话人，110 万话语）上用 AAM-softmax 损失训练。

**WavLM-SV（2022 年至今）**。在预训练的 WavLM-large SSL 骨干上用 AAM 损失微调。质量更高但速度更慢——300+ MB vs 15 MB。

**x-vector（基线）**。TDNN + 统计池化。经典方案；在 CPU/边缘端仍有用武之地。

**AAM-softmax**。在角度空间加入间隔 `m` 的标准 softmax：对正确类别使用 `cos(θ + m)`，强制类间角度分离。典型参数：`m=0.2`，缩放 `s=30`。

### 打分

- 注册嵌入与测试嵌入之间的**余弦相似度**，基于阈值做决策。
- **PLDA（概率 LDA）**。将嵌入投影到潜空间，在该空间中同一说话人与不同说话人有封闭形式的似然比。叠加在余弦相似度之上可降低 EER 10–20%。2020 年前是标准方法，现在仅用于闭集场景。
- **分数归一化**。`S-norm` 或 `AS-norm`：用一组冒充者的均值和标准差对每个分数归一化。跨域评估必不可少。

### 2026 年你应该知道的数字

| 模型 | VoxCeleb1-O EER | 参数量 | 吞吐量（A100） |
|------|-----------------|--------|--------------|
| x-vector（经典） | 3.10% | 500 万 | 400× 实时 |
| ECAPA-TDNN | 0.87% | 1500 万 | 200× 实时 |
| WavLM-SV large | 0.42% | 3.16 亿 | 20× 实时 |
| Pyannote 3.1 分割+嵌入 | 0.65% | 600 万 | 100× 实时 |
| ReDimNet（2024） | 0.39% | 2400 万 | 100× 实时 |

### 说话人分离

在多说话人片段中判断"谁在什么时候说话"。流水线：VAD → 分段 → 对每段计算嵌入 → 聚类（凝聚层次聚类或谱聚类）→ 平滑边界。现代技术栈：`pyannote.audio` 3.1，将说话人分割、嵌入、聚类封装在一个调用背后。2026 年在 AMI 数据集上的 SOTA 说话人错误率（DER）约为 15%（2022 年为 23%）。

## 动手实现

### 第一步：基于 MFCC 统计的简单嵌入

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

与 SOTA 差距极大——仅用于教学。`code/main.py` 在合成说话人数据上用它做概念验证。

### 第二步：余弦相似度 + 阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 第三步：从相似度对计算 EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回 (eer, 对应阈值)，两者都要报告。

### 第四步：用 SpeechBrain 进行生产级验证

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### 第五步：用 pyannote 做说话人分离

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 工程应用

2026 年技术栈：

| 场景 | 选型 |
|------|------|
| 闭集 1:1 验证、边缘端 | ECAPA-TDNN + 余弦阈值 |
| 开放集验证、云端 | WavLM-SV + AS-norm |
| 说话人分离（会议、播客） | `pyannote/speaker-diarization-3.1` |
| 防欺骗（回放/深度伪造检测） | AASIST 或 RawNet2 |
| 微型嵌入端（关键词识别 + 注册） | Titanet-Small（NeMo） |

## 常见陷阱

- **信道不匹配**。在 VoxCeleb（网络视频）上训练的模型 ≠ 电话音频。务必在目标信道上评估。
- **短话语**。测试音频短于 3 秒时，EER 会急剧恶化。
- **含噪音的注册样本**。一个有噪音的注册样本会毒化整个锚点。使用 ≥ 3 个干净样本并取平均。
- **跨条件使用固定阈值**。始终在目标领域的留出开发集上调整阈值。
- **非归一化嵌入上计算余弦**。先做 L2 归一化，否则向量模长会主导结果。

## 交付物

保存为 `outputs/skill-speaker-verifier.md`。选择模型、注册协议、阈值调整方案和防欺诈措施。

## 练习

1. **（简单）** 运行 `code/main.py`，构建合成"说话人"（不同音调特征），注册，在 100 对测试集上计算 EER。
2. **（中等）** 在 30 段 VoxCeleb1 话语（5 个说话人 × 6 段）上使用 SpeechBrain ECAPA，对比余弦相似度 vs PLDA 的 EER。
3. **（困难）** 用 `pyannote.audio` 构建完整的注册 → 分离 → 验证流水线，在 AMI 开发集上评估 DER。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| EER（等错误率） | "那个核心指标" | 误接受率 = 误拒绝率时的阈值处误差率 |
| 验证 (Verification) | "1:1" | "这是 Alice 吗？" |
| 识别 (Identification) | "1:N" | "谁在说话？" |
| 开放集 (Open-set) | "有未知" | 测试集可能包含未注册说话人 |
| 注册 (Enrollment) | "登记" | 计算说话人的参考嵌入 |
| AAM-softmax | "那个损失函数" | 带加性角度间隔的 softmax，强制聚类分离 |
| PLDA | "经典打分" | 概率 LDA，在嵌入之上做似然比打分 |
| DER（说话人错误率） | "分离指标" | 漏检 + 误报 + 混淆之和 |

## 延伸阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 经典深度嵌入论文
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — 2020–2026 年主流架构
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — SV 和说话人分离的 SSL 骨干
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — 生产级说话人分离+嵌入技术栈
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — 各模型当前 EER 排名

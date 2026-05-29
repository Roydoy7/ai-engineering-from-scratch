# 声音防欺骗与音频水印——ASVspoof 5、AudioSeal、WaveVerify

> 声音克隆的发货速度快于防御措施。2026 年的生产语音系统需要两样东西：能区分真实与伪造语音的检测器（AASIST、RawNet2），以及能在压缩和编辑后存活的水印（AudioSeal）。两者都要发货，否则不要发布声音克隆功能。

**类型：** 构建
**语言：** Python
**前置知识：** 第6阶段第6课（说话人识别）、第6阶段第8课（声音克隆）
**预计时间：** 约75分钟

## 问题背景

三种相关防御措施：

1. **防欺骗/深度伪造检测**。给定一段音频，它是合成的还是真实的？ASVspoof 基准（ASVspoof 2019 → 2021 → 5）是金标准。
2. **音频水印**。在生成音频中嵌入不可感知的信号，检测器之后可以提取。AudioSeal（Meta）和 WavMark 是开源选项。
3. **认证来源**。音频文件的加密签名 + 元数据。C2PA/内容真实性倡议。

检测针对不合作的攻击者。水印处理合规问题——AI 生成的音频应该可被识别。两者在 2026 年都是必须的。

## 核心概念

### ASVspoof 5——2024–2025 年基准

与以往版本相比的最大变化：

- **众包数据**（非录音室级干净）——真实条件。
- **约 2000 个说话人**（之前约 100 个）。
- **32 种攻击算法**。TTS + 声音转换 + 对抗性扰动。
- **两个赛道**。反制（CM）独立检测；防欺骗鲁棒 ASV（SASV）用于生物特征系统。

ASVspoof 5 最新水平：约 7.23% EER。旧的 ASVspoof 2019 LA：0.42% EER。真实世界部署：预计在野采集片段上 5–10% EER。

### AASIST 和 RawNet2——检测模型家族

**AASIST**（2021，更新至 2026）。在频谱特征上使用图注意力。ASVspoof 5 反制任务的当前 SOTA。

**RawNet2**。在原始波形上的卷积前端 + TDNN 骨干。更简单的基线，通过微调仍有竞争力。

**NeXt-TDNN + SSL 特征**。2025 年变体：ECAPA 风格 + WavLM 特征 + focal loss。在 ASVspoof 2019 LA 上达到 0.42% EER。

### AudioSeal——2024 年水印默认方案

Meta 的 **AudioSeal**（2024 年 1 月，2024 年 12 月 v0.2）。核心设计：

- **局部化**。以 16 kHz 采样分辨率（1/16000 秒）逐帧检测水印。
- **生成器+检测器联合训练**。生成器学习嵌入不可听信号；检测器学习在数据增强中找到它。
- **鲁棒性**。能在 MP3/AAC 压缩、EQ、±10% 速度偏移、+10 dB SNR 噪声混合后存活。
- **速度**。检测器以 485 倍实时速度运行；比 WavMark 快 1000 倍。
- **容量**。每次话语可嵌入 16 位有效载荷（可编码模型 ID、生成时间戳、用户 ID）。

### WavMark

AudioSeal 之前的开源基线。可逆神经网络，32 位/秒。问题：

- 同步暴力破解速度慢。
- 可通过高斯噪声或 MP3 压缩去除。
- 不适合实时处理。

### WaveVerify（2025 年 7 月）

解决 AudioSeal 的弱点——特别是时间操作（反转、变速）。使用基于 FiLM 的生成器 + 专家混合检测器。在标准攻击上与 AudioSeal 相当，能处理时间编辑。

### 攻击者利用的漏洞

来自 AudioMarkBench："在音高偏移下，所有水印的比特恢复准确率都低于 0.6，表明几乎被完全去除。"**音高偏移是万能攻击**。2026 年没有任何水印能完全抵抗激进的音高修改。这就是为什么你需要在水印之外同时使用检测（AASIST）。

### C2PA/内容真实性倡议

不是机器学习技术——是一种元数据格式。音频文件携带关于创建工具、作者、日期的加密签名元数据。Audobox/Seamless 使用它。适合溯源；如果恶意方重新编码并剥离元数据则无效。

## 动手实现

### 第一步：简单的频谱特征检测器（玩具）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成语音通常具有异常平坦的高频能量。生产检测器使用 AASIST，而不是这个。但直觉是对的。

### 第二步：AudioSeal 嵌入与检测

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### 第三步：评估——EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 第四步：生产集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每次生成都附带：(1) 水印，(2) 签名清单，(3) 符合保留政策的审计日志。

## 工程应用

| 用例 | 防御措施 |
|------|---------|
| 发布 TTS/声音克隆 | 每个输出嵌入 AudioSeal（不可商量） |
| 生物特征声音解锁 | AASIST + ECAPA 集成；活体验证挑战 |
| 呼叫中心欺诈检测 | AASIST 对 20% 抽样来电 |
| 播客真实性 | 上传时 C2PA 签名，AI 生成则加 AudioSeal |
| 研究/训练检测器 | ASVspoof 5 训练/验证/评估集 |

## 常见陷阱

- **嵌入水印但检测器从不运行**。毫无意义。在 CI 中部署检测器。
- **检测无校准**。在 ASVspoof LA 上训练的 AASIST 过拟合，真实世界准确率下降。在你的领域上校准。
- **音高偏移漏洞**。激进的音高偏移去除大多数水印。要有检测回退方案。
- **元数据剥离和重新托管**。C2PA 通过重新编码即可绕过。始终将加密防御和感知防御（水印）一起部署。
- **活体检测作为检测方案**。让用户说一个随机短语。防止回放攻击，但不防实时克隆。

## 交付物

保存为 `outputs/skill-spoof-defender.md`。为声音生成部署选择检测模型、水印、来源清单和操作手册。

## 练习

1. **（简单）** 运行 `code/main.py`，在合成音频上演示玩具检测器 + 玩具水印嵌入/检测。
2. **（中等）** 安装 `audioseal`，在 TTS 输出中嵌入 16 位有效载荷并重新解码。用噪声损坏音频，测量比特恢复准确率。
3. **（困难）** 在 ASVspoof 2019 LA 上微调 RawNet2 或 AASIST，测量 EER，在 F5-TTS 生成片段的留出集上测试，观察分布外检测如何退化。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| ASVspoof | "那个基准" | 两年一次的挑战赛，2024 年 = ASVspoof 5 |
| CM（反制） | "检测器" | 分类器：真实语音 vs 合成/转换语音 |
| SASV | "说话人验证 + CM" | 集成生物特征 + 欺骗检测 |
| AudioSeal | "Meta 水印" | 局部化，16 位有效载荷，比 WavMark 快 485 倍 |
| 比特恢复准确率 | "水印存活率" | 攻击后恢复的有效载荷比特分数 |
| C2PA | "来源清单" | 关于创建/作者身份的加密元数据 |
| AASIST | "检测器家族" | 基于图注意力的防欺骗 SOTA |

## 延伸阅读

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 当前基准
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — 水印默认方案
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — 针对时间攻击的 MoE 检测器
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — SOTA 检测骨干
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — 鲁棒性评估
- [C2PA specification](https://c2pa.org/specifications/specifications/) — 来源清单格式

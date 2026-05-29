# 音频评估——WER、MOS、UTMOS、MMAU、FAD 与开放排行榜

> 无法衡量的就无法发货。本课整理了 2026 年每个音频任务的指标：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、ASR 回路 WER）、音频语言模型（MMAU、LongAudioBench）、音乐（FAD、CLAP）和说话人（EER）。以及你拿来比较的排行榜。

**类型：** 学习
**语言：** Python
**前置知识：** 第6阶段第04、06、07、09、10课；第2阶段第9课（模型评估）
**预计时间：** 约60分钟

## 问题背景

每个音频任务都有多个指标，各自衡量不同维度。使用错误的指标，会让你发出一个在仪表盘上看起来很好、在生产中表现很差的模型。2026 年的标准列表：

| 任务 | 主要指标 | 次要指标 |
|------|---------|---------|
| ASR | WER | CER · RTFx · 首 token 延迟 |
| TTS | MOS / UTMOS | SECS · ASR 回路 WER · CER · TTFA |
| 声音克隆 | SECS（ECAPA 余弦） | MOS · CER |
| 说话人验证 | EER | minDCF · 操作点处的 FAR/FRR |
| 说话人分离 | DER | JER · 说话人混淆 |
| 音频分类 | top-1 · mAP | 宏平均 F1 · 逐类召回率 |
| 音乐生成 | FAD | CLAP · 试听组 MOS |
| 音频语言模型 | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| 流式 S2S | 延迟 P50/P95 | WER · MOS |

## 核心概念

### ASR 指标

**WER（词错误率）**。`(S + D + I) / N`。评分前转小写、去标点、规范化数字。使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。< 5% = 朗读语音达到人类水平。

**CER（字符错误率）**。相同公式，字符级别。用于词语分割不明确的声调语言（普通话、广东话）。

**RTFx（实时因子倒数）**。每壁钟秒处理的音频秒数。越高越好。Parakeet-TDT 达到 3380×，Whisper-large-v3 约 30×。

**首 token 延迟**。从音频输入到第一个文字 token 的壁钟时间。流式场景关键。Deepgram Nova-3：约 150 ms。

### TTS 指标

**MOS（平均主观意见分）**。1-5 分人工评分。金标准但速度慢。每个样本收集 20+ 评分者，每个模型 100+ 样本。

**UTMOS（2022–2026）**。学习型 MOS 预测器。在标准基准上与人工 MOS 的相关性约 0.9。F5-TTS：UTMOS 3.95；真实录音：4.08。

**SECS（说话人编码器余弦相似度）**。用于声音克隆。参考和克隆输出之间的 ECAPA 嵌入余弦相似度。> 0.75 = 可识别的克隆。

**ASR 回路 WER**。对 TTS 输出运行 Whisper，计算相对输入文字的 WER。捕获可懂度退化。2026 年 SOTA：< 2% CER。

**TTFA（首音延迟）**。壁钟延迟。Kokoro-82M：约 100 ms；F5-TTS：约 1 s。

### 声音克隆专项

**SECS + MOS + CER** 三合一。SECS 高但 MOS 低意味着音色正确但不自然；反之意味着声音自然但说话人不对。

### 说话人验证

**EER（等错误率）**。误接受率等于误拒绝率时的阈值。ECAPA 在 VoxCeleb1-O 上：0.87%。

**minDCF（最小检测代价）**。在选定操作点（通常 FAR=0.01）的加权代价。比 EER 更贴近生产实际。

### 说话人分离

**DER（说话人错误率）**。`(FA + Miss + 混淆) / 总说话人时间`。漏检语音 + 误报语音 + 说话人混淆，各作为分数。AMI 会议：10–20% DER 是现实水平。pyannote 3.1 + Precision-2 商业：录制良好的音频 DER < 10%。

**JER（Jaccard 错误率）**。DER 的替代指标，对短片段偏差鲁棒。

### 音频分类

多标签：在所有类别上的 **mAP（平均精度均值）**。AudioSet：BEATs-iter3 的 mAP 为 0.548。

多类互斥：**top-1、top-5 准确率**。Speech Commands v2：Audio-MAE 达到 99.0% top-1。

类别不平衡：**宏平均 F1** + **逐类召回率**。要按类别报告——聚合准确率会掩盖哪些类别失败。

### 音乐生成

**FAD（弗雷歇音频距离）**。真实与生成音频 VGGish 嵌入分布之间的距离。MusicGen-small 在 MusicCaps 上：4.5。MusicLM：4.0。越低越好。

**CLAP 分数**。使用 CLAP 嵌入的文字-音频对齐分数。> 0.3 = 合理对齐。

**试听组 MOS**。消费级音乐的最终裁判。Suno v5 在 TTS Arena 上 ELO 1293（来自配对人类偏好）。

### 音频语言模型基准

**MMAU（大规模多音频理解）**。1 万个音频问答对。

**MMAU-Pro**。1800 个难题，四个类别：语音/声音/音乐/多音频。四选一的随机水平 25%。Gemini 2.5 Pro 总体约 60%；所有模型的多音频约 22%。

**LongAudioBench**。多分钟音频配语义查询。Audio Flamingo Next 超越 Gemini 2.5 Pro。

**AudioCaps / Clotho**。字幕基准，使用 SPICE、CIDEr、FENSE 指标。

### 流式语音到语音

**延迟 P50/P95/P99**。从用户话语结束到首个可听回应的壁钟时间。Moshi：200 ms；GPT-4o Realtime：300 ms。

**WER/MOS** 针对输出。

**插话响应性**。从用户中断到助手静音的时间。目标 < 150 ms。

### 2026 年排行榜

| 排行榜 | 赛道 | URL |
|--------|------|-----|
| Open ASR Leaderboard（HF） | 英语+多语言+长音频 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena（HF） | 英语 TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT，ELO 来自配对投票 | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM 推理 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 说话人识别 | `voxsrc.github.io` |
| MMAU 音乐子集 | 音乐 LALM | （MMAU 内） |
| HEAR benchmark | 自监督音频 | `hearbenchmark.com` |

## 动手实现

### 第一步：带归一化的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 第二步：TTS 回路 WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 第三步：声音克隆的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 第四步：音乐生成的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 第五步：说话人验证的 EER（与第6课相同）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 工程应用

每次部署都配备一套固定的评估框架，在每次模型更新时运行。三条基本原则：

1. **评分前先归一化**。转小写、去标点、展开数字。报告归一化规则。
2. **报告分布，而非均值**。延迟用 P50/P95/P99。分类用逐类召回率。MMAU 用逐类别报告。
3. **跑一个权威的公开基准**。即使你的生产数据不同，在 Open ASR/TTS Arena/MMAU 上报告也让审阅者能进行苹果对苹果的比较。

## 常见陷阱

- **UTMOS 外推**。在 VCTK 风格干净语音上训练；对嘈杂/克隆/情绪化音频打分偏差。
- **MOS 评审团偏差**。20 个亚马逊众包工作者 ≠ 20 个目标用户。高风险场景需要支付领域专家评估费用。
- **FAD 依赖参考集**。跨模型对比时，使用相同的参考分布。
- **聚合 WER**。5% 的总体 WER 可能掩盖带口音语音 30% 的 WER。按人口学切片报告。
- **公开基准饱和**。大多数前沿模型在标准基准上已接近天花板。构建一个反映你自己流量的内部留出集。

## 交付物

保存为 `outputs/skill-audio-evaluator.md`。为任意音频模型发布选择指标、基准和报告格式。

## 练习

1. **（简单）** 运行 `code/main.py`，在玩具输入上计算 WER/CER/EER/SECS/类 FAD/类 MMAU。
2. **（中等）** 构建 TTS 回路 WER 框架。将 Kokoro 或 F5-TTS 输出通过 Whisper 处理，在 50 个提示上计算 WER，标记 WER > 10% 的提示。
3. **（困难）** 在 MMAU-Pro 语音 + 多音频子集（各 50 条）上评测第10课选择的 LALM，报告逐类别准确率并与已发布数字对比。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| WER | "ASR 分数" | 归一化后词级 `(S+D+I)/N` |
| CER | "字符 WER" | 用于声调语言或字符级系统 |
| MOS | "人工意见" | 1-5 分；20+ 评分者 × 100 样本 |
| UTMOS | "机器学习 MOS 预测器" | 学习型模型，与人工 MOS 相关性约 0.9 |
| SECS | "声音克隆相似度" | 参考与克隆之间的 ECAPA 余弦相似度 |
| EER | "说话人验证分数" | FAR = FRR 时的阈值 |
| DER | "说话人分离分数" | (FA + 漏检 + 混淆) / 总计 |
| FAD | "音乐生成质量" | VGGish 嵌入上的弗雷歇距离 |
| RTFx | "吞吐量" | 每壁钟秒处理的音频秒数 |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) — 带归一化工具的 WER/CER 库
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — 学习型 MOS 预测器
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — 音乐生成标准
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 年实时排名
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 人工投票 TTS 排行榜
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM 推理排行榜
- [HEAR benchmark](https://hearbenchmark.com/) — 音频 SSL 基准

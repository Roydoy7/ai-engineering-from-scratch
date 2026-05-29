# 声音克隆与声音转换

> 声音克隆用别人的声音朗读你的文字。声音转换在保留说话内容的同时，将你的声音重写为他人的声音。两者依赖同一种分解：将说话人身份与内容分离。

**类型：** 构建
**语言：** Python
**前置知识：** 第6阶段第6课（说话人识别）、第6阶段第7课（TTS）
**预计时间：** 约75分钟

## 问题背景

2026 年，5 秒音频片段就足以用消费级 GPU 克隆任何人的高质量声音。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 都提供零样本或少样本克隆。这项技术是福音（无障碍 TTS、配音、辅助声音），也是武器（诈骗电话、政治深度伪造、知识产权盗窃）。

两个密切相关的任务：

- **声音克隆（TTS 侧）**：文字 + 5 秒参考声音 → 该声音朗读的音频。
- **声音转换（语音侧）**：A 说 X 的音频 + B 的参考声音 → B 说 X 的音频。

两者都将波形分解为（内容、说话人、韵律），然后将一个来源的内容与另一个来源的说话人重新组合。

2026 年你需要遵守的关键约束：**欧盟《人工智能法案》（2026 年 8 月起执行）和加利福尼亚 AB 2905（2025 年生效）在法律上要求添加水印和知情同意验证**。你的流水线必须嵌入不可听水印，并拒绝未经同意的克隆请求。

## 核心概念

**零样本克隆**。将 5 秒片段传入一个已在数千个说话人上训练过的模型。说话人编码器将片段映射为说话人嵌入；TTS 解码器以该嵌入加文字为条件生成音频。

使用方：F5-TTS（2024）、YourTTS（2022）、XTTS v2（2024）、OpenVoice v2（2024）。

**少样本微调**。录制目标声音 5–30 分钟，用 LoRA 微调基础模型约一小时，质量从"还行"跃升到"难以区分"。Coqui 和 ElevenLabs 都支持这种模式；社区也用 F5-TTS 实现。

**声音转换（VC）**。两大流派：

- **识别-合成**。运行类 ASR 模型提取内容表示（如软音素后验概率、PPG），再用目标说话人嵌入重新合成。对语言和口音鲁棒，KNN-VC（2023）、Diff-HierVC（2023）采用此路线。
- **解耦**。训练一个在瓶颈层将内容、说话人、韵律分离的自编码器，推理时交换说话人嵌入。质量较低但速度更快，AutoVC（2019）、VITS-VC 变体采用此路线。

**基于神经编解码器的克隆（2024 年至今）**。VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox——将音频视为 SoundStream/EnCodec 的离散 token，在编解码器 token 上训练大型自回归或流匹配模型。短提示下质量与 ElevenLabs 相当。

### 伦理不是附加功能

**水印**。PerTh（Perth）和 SilentCipher（2024）在音频中不可察觉地嵌入约 16–32 位 ID，能在重编码、流传输和常规编辑后存活。生产可用的开源实现。

**知情同意门控**。每个克隆输出必须与可验证的知情同意记录配对。"我，Rohit，于 2026-04-22 授权将此声音用于 X 目的。"存储在防篡改日志中。

**检测**。AASIST、RawNet2 和 Wav2Vec2-AASIST 可作为检测器发货。ASVspoof 2025 挑战赛发布的数据显示：SOTA 检测器对 ElevenLabs、VALL-E 2 和 Bark 输出的 EER 为 0.8–2.3%。

### 数字（2026 年）

| 模型 | 零样本？ | SECS（目标相似度） | WER（可懂度） | 参数量 |
|------|---------|------------------|-------------|--------|
| F5-TTS | 是 | 0.72 | 2.1% | 3.35 亿 |
| XTTS v2 | 是 | 0.65 | 3.5% | 4.70 亿 |
| OpenVoice v2 | 是 | 0.70 | 2.8% | 2.20 亿 |
| VALL-E 2 | 是 | 0.77 | 2.4% | 3.70 亿 |
| VoiceBox | 是 | 0.78 | 2.1% | 3.30 亿 |

SECS > 0.70 对大多数听众而言与目标声音难以区分。

## 动手实现

### 第一步：识别-合成分解（code/main.py 中的代码示例）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念简单；实现的主体在 `tts_model` 和说话人编码器中。

### 第二步：用 F5-TTS 进行零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考文字必须与参考音频完全匹配，包括标点；不匹配会破坏对齐。

### 第三步：用 KNN-VC 进行声音转换

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 运行 WavLM 提取源和目标池的逐帧嵌入，然后将每个源帧替换为池中最近邻帧。非参数化，只需一分钟的目标语音即可工作。

### 第四步：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

约 32 位有效载荷，经 MP3 重编码和轻度噪声后仍可检测。

### 第五步：知情同意门控

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 工程应用

2026 年技术栈：

| 场景 | 选型 |
|------|------|
| 5 秒零样本克隆、开源 | F5-TTS 或 OpenVoice v2 |
| 商业生产级克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 声音转换（重写） | KNN-VC 或 Diff-HierVC |
| 多说话人微调 | StyleTTS 2 + 说话人适配器 |
| 跨语言克隆 | XTTS v2 或 VALL-E X |
| 深度伪造检测 | Wav2Vec2-AASIST |

## 常见陷阱

- **参考文字不对齐**。F5-TTS 等模型要求参考文字与参考音频精确匹配，包括标点。
- **有混响的参考音频**。回声会破坏克隆效果。使用干声、近距离麦克风录音。
- **情绪不匹配**。参考音频"欢快"会导致所有克隆输出都很欢快。让参考音频的情绪与目标用途匹配。
- **语言泄漏**。克隆英语说话人后让模型说法语，往往仍带着英语口音；使用跨语言模型（XTTS、VALL-E X）。
- **没有水印**。从 2026 年 8 月起，在欧盟无水印即违法。

## 交付物

保存为 `outputs/skill-voice-cloner.md`。设计包含知情同意门控、水印和质量目标的克隆或转换流水线。

## 练习

1. **（简单）** 运行 `code/main.py`，通过计算两个"说话人"在交换前后的余弦相似度来演示说话人嵌入交换。
2. **（中等）** 用 OpenVoice v2 克隆自己的声音，测量参考和克隆之间的 SECS，通过 Whisper 计算 CER。
3. **（困难）** 对 20 个克隆输出应用 SilentCipher 水印，经 128 kbps MP3 编解码后检测有效载荷，报告比特准确率。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 零样本克隆 | "5 秒就够了" | 预训练模型 + 说话人嵌入，无需训练 |
| PPG | "音素后验概率图" | 逐帧 ASR 后验概率，用作语言无关的内容表示 |
| KNN-VC | "最近邻转换" | 将每个源帧替换为目标池中最近邻帧 |
| 神经编解码器 TTS | "VALL-E 那种" | 在 EnCodec/SoundStream token 上训练的 AR 模型 |
| 水印 | "不可听签名" | 嵌入音频中的比特，经重编码后仍可存活 |
| SECS | "克隆保真度" | 目标和克隆说话人嵌入之间的余弦相似度 |
| AASIST | "深度伪造检测器" | 防欺骗模型，检测合成语音 |

## 延伸阅读

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 开源 SOTA 零样本克隆
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 和 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — 神经编解码器 TTS
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — 基于解耦的声音转换
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — 基于检索的声音转换
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — 生产可用的 32 位音频水印
- [ASVspoof 2025 results](https://www.asvspoof.org/) — 检测器与合成器的军备竞赛，2026 年更新

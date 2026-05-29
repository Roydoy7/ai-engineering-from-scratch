# 音乐生成——MusicGen、Stable Audio、Suno 与版权地震

> 2026 年音乐生成现状：Suno v5 和 Udio v4 主导商业赛道；MusicGen、Stable Audio Open 和 ACE-Step 领跑开源。技术问题基本已解决，法律问题（华纳音乐 5 亿美元和解、环球音乐和解）在 2025–2026 年重塑了整个领域。

**类型：** 构建
**语言：** Python
**前置知识：** 第6阶段第2课（频谱图）、第4阶段第10课（扩散模型）
**预计时间：** 约75分钟

## 问题背景

文字 → 30 秒到 4 分钟的音乐片段，包含歌词、人声和结构。三个子问题：

1. **器乐生成**。"lo-fi 嘻哈鼓点配温暖键盘"这样的文字 → 音频。MusicGen、Stable Audio、AudioLDM。
2. **完整歌曲生成（带人声和歌词）**。"关于德克萨斯雨夜的乡村歌曲"→ 完整歌曲。Suno、Udio、YuE、ACE-Step。
3. **条件可控生成**。续接现有片段、重新生成过渡段、切换风格、分离音轨或局部修复。Udio 的局部修复 + 音轨分离是 2026 年的对标功能。

## 核心概念

### 基于神经编解码器 token 的 Token 语言模型

Meta 的 **MusicGen**（2023，MIT 协议）及其众多变体：以文字/旋律嵌入为条件，自回归预测 EnCodec token（32 kHz，4 个码本），用 EnCodec 解码。3 亿–33 亿参数。强劲基线，但超过 30 秒时表现变差。

**ACE-Step**（开源，2026 年 4 月发布 4B XL 版本）将此扩展为支持歌词条件的完整歌曲生成，是开源社区最接近 Suno 的方案。

### 基于 Mel 或潜变量的扩散模型

**Stable Audio（2023）** 和 **Stable Audio Open（2024）**：在压缩音频上做潜变量扩散。擅长循环、音效设计和环境音，但不擅长有结构的完整歌曲。

**AudioLDM / AudioLDM2**：通过 T2I 风格的潜变量扩散实现文字转音频，推广到音乐、音效和语音。

### 混合架构（商业产品）——Suno、Udio、Lyria

权重不公开。推测是 AR 编解码器语言模型 + 带专用人声/鼓/旋律头的扩散声码器。Suno v5（2026）以 ELO 1293 领跑质量榜。Udio v4 新增局部修复 + 音轨分离（低音、鼓、人声独立下载）。

### 评估指标

- **FAD（弗雷歇音频距离）**。生成音频与真实音频分布之间的嵌入级距离，使用 VGGish 或 PANNs 特征，越低越好。MusicGen small 在 MusicCaps 上 FAD 为 4.5；SOTA 约为 3.0。
- **音乐性（主观）**。人类偏好打分。Suno v5 ELO 1293 领先。
- **文字-音频对齐**。提示与输出之间的 CLAP 分数。
- **音乐性瑕疵**。节拍过渡错位、人声短语漂移、超过 30 秒后结构失散。

## 2026 年模型全景

| 模型 | 参数量 | 时长 | 人声 | 协议 |
|------|--------|------|------|------|
| MusicGen-large | 33 亿 | 30 秒 | 无 | MIT |
| Stable Audio Open | 12 亿 | 47 秒 | 无 | Stability 非商业 |
| ACE-Step XL（2026 年 4 月） | 40 亿 | >2 分钟 | 有 | Apache-2.0 |
| YuE | 70 亿 | >2 分钟 | 有，多语言 | Apache-2.0 |
| Suno v5（闭源） | ? | 4 分钟 | 有，ELO 1293 | 商业 |
| Udio v4（闭源） | ? | 4 分钟 | 有 + 音轨 | 商业 |
| Google Lyria 3（闭源） | ? | 实时 | 有 | 商业 |
| MiniMax Music 2.5 | ? | 4 分钟 | 有 | 商业 API |

## 法律格局（2025–2026 年）

- **华纳音乐 vs Suno 和解**。5 亿美元。WMG 现已取得对 Suno 上 AI 相似性、音乐版权和用户生成曲目的监督权。Udio 亦有类似的环球音乐和解。
- **欧盟《人工智能法案》** + **加利福尼亚 SB 942**：AI 生成音乐必须披露。
- **Riffusion / MusicGen** 基于 MIT 协议，无合规负担，但也无商业人声。

安全发货模式：

1. 仅生成器乐（MusicGen、Stable Audio Open，MIT/CC0 输出）。
2. 使用商业 API（Suno、Udio、ElevenLabs Music），按次授权。
3. 在自有或已授权的曲库上训练（大多数企业最终走这条路）。
4. 为生成内容打上水印和元数据标签。

## 动手实现

### 第一步：用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三种规模：`small`（3 亿，速度快）、`medium`（15 亿）、`large`（33 亿）。Small 足以验证创意是否可行。

### 第二步：旋律条件生成

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 获取色度图并保留旋律，同时切换音色。适合"把这段旋律改成弦乐四重奏"这样的用途。

### 第三步：FAD 评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算 VGGish 嵌入距离。适用于流派级回归测试，不能替代人类听众。

### 第四步：结合 LLM 的音乐工作流

结合第7、8课的思路：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 工程应用

| 目标 | 技术栈 |
|------|--------|
| 器乐音效设计 | Stable Audio Open |
| 游戏/自适应音乐 | Google Lyria RealTime（闭源） |
| 带人声完整歌曲（商业） | 带明确授权的 Suno v5 或 Udio v4 |
| 带人声完整歌曲（开源） | ACE-Step XL 或 YuE |
| 短广告 Jingle | MusicGen 旋律条件 + 哼唱参考 |
| 音乐视频背景 | MusicGen + Stable Video Diffusion |

## 2026 年仍在发货的陷阱

- **版权洗白提示**。"泰勒·斯威夫特风格的歌曲"——商业 Suno/Udio 现已过滤，开源模型不过滤。自行添加过滤词单。
- **超过 30 秒后重复/漂移**。AR 模型会循环。交叉淡出多段生成，或使用 ACE-Step 保持结构连贯性。
- **节拍漂移**。模型会偏离 BPM。在提示中加 BPM 标签，并用 librosa 的 `beat_track` 做后处理过滤。
- **人声可懂度差**。Suno 表现出色；开源模型的歌词往往含混不清。如果歌词重要，使用商业 API 或微调。
- **单声道输出**。开源模型生成单声道或假立体声，使用合适的立体声重建工具（ezst、Cartesia 的立体声扩散）升级。

## 交付物

保存为 `outputs/skill-music-designer.md`。为音乐生成部署选择模型、授权策略、时长/结构方案和披露元数据。

## 练习

1. **（简单）** 运行 `code/main.py`，以 ASCII 符号形式生成"创意"和弦进行 + 鼓点节奏，如需要可通过任意 MIDI 渲染器播放。
2. **（中等）** 安装 `audiocraft`，用 MusicGen-small 针对 4 种风格提示各生成 10 秒片段，与参考风格集对比测量 FAD。
3. **（困难）** 使用 ACE-Step（或 MusicGen-melody），用不同音色提示生成同一旋律的三个变体，计算与提示的 CLAP 相似度以验证对齐效果。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| FAD | "音频 FID" | 真实与生成音频嵌入分布之间的弗雷歇距离 |
| 色度图 (Chromagram) | "旋律音高表示" | 12 维逐帧向量，作为旋律条件的输入 |
| 音轨 (Stems) | "乐器轨道" | 分离后的低音/鼓/人声/旋律 WAV 文件 |
| 局部修复 (Inpainting) | "重生成某段" | 遮蔽时间窗口，模型仅重新生成该部分 |
| CLAP | "文字-音频 CLIP" | 对比音频-文字嵌入，评估文字-音频对齐 |
| EnCodec | "音乐编解码器" | Meta 的神经编解码器，MusicGen 使用，32 kHz，4 个码本 |

## 延伸阅读

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — 开源自回归基准
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — 音效设计默认方案
- [ACE-Step](https://github.com/ace-step/ACE-Step) — 开源 40 亿参数完整歌曲生成器，2026 年 4 月
- [Suno v5 平台文档](https://suno.com) — 商业质量领头羊
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — 音乐+音效的潜变量扩散
- [WMG-Suno 和解报道](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025 年 11 月先例

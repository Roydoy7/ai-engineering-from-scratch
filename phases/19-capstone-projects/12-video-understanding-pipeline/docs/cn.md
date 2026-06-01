# 压轴项目 12——视频理解管道（场景、问答、搜索）（Capstone 12 — Video Understanding Pipeline: Scene, QA, Search）

> Twelve Labs 将 Marengo + Pegasus 产品化。VideoDB 发布了视频 CRUD API。AI2 的 Molmo 2 发布了开源 VLM 检查点。Gemini 长上下文原生处理小时级视频。TimeLens-100K 在规模上定义了时间定位。2026 年的管道已确立：场景分割、每场景字幕 + 嵌入、转录对齐、多向量索引，以及带（开始，结束）时间戳加帧预览的查询答案。本压轴项目是摄入 100 小时视频，在公开基准上测量，并在计数和动作问题上测量幻觉。

**类型：** 压轴项目  
**语言：** Python（管道），TypeScript（UI）  
**前置知识：** Phase 4（CV）、Phase 6（语音）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 12（多模态）、Phase 17（基础设施）  
**涉及的阶段：** P4 · P6 · P7 · P11 · P12 · P17  
**预计时间：** 30 小时

## 问题所在

长格式视频问答是 2026 年规模下带宽需求最高的多模态问题。Gemini 2.5 Pro 可以原生读取 2 小时的视频，但将 100 小时的视频摄入可查询语料库仍然需要场景级索引。生产形态结合了场景分割（TransNetV2 或 PySceneDetect）、VLM 每场景字幕（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、转录对齐（带词级时间戳的 Whisper-v3-turbo）和并排存储字幕、帧嵌入和转录的多向量索引。查询管道用（开始，结束）时间戳加帧预览回答。

基准是公开的（ActivityNet-QA、NeXT-GQA）加上你自己的 100 个问题自定义集。计数和动作类型问题的幻觉是已知的困难失败类别；本压轴项目明确测量它。

## 核心概念

在摄入时，三个管道并行运行。**场景分割**将视频切割成场景。**VLM 字幕**为每个场景生成字幕和来自关键帧的帧嵌入。**ASR 对齐**生成词级时间戳。三个流通过（scene_id，时间范围）连接。每个场景在多向量索引（Qdrant）中获得三种向量类型：字幕嵌入、关键帧嵌入、转录嵌入。

在查询时，自然语言问题对所有三个向量触发；结果用 RRF 合并；时间定位适配器（TimeLens 风格）在顶部场景内细化（开始，结束）窗口。VLM 合成器（Gemini 2.5 Pro 或 Qwen3-VL-Max）接受查询 + 顶部场景 + 裁剪帧，并用引用的时间戳和帧预览回答。

幻觉测量很重要。计数（"有多少人进入房间？"）和动作类型（"厨师是先倒还是先搅拌？"）问题是众所周知不可靠的。分别报告准确率与描述性问题。

## 架构

```
视频文件 / URL
      |
      v
PySceneDetect / TransNetV2（场景分割）
      |
      +--- 每场景关键帧 --- VLM 字幕 + 帧嵌入
      |                     (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- 音频通道 --- Whisper-v3-turbo ASR + 词级时间戳
      |
      v
多向量 Qdrant：{caption_emb, keyframe_emb, transcript_emb}
      |
查询：
  对所有三个进行密集查询 -> RRF 合并 -> top-k 场景
      |
      v
TimeLens / VideoITG 时间定位（在场景内细化开始/结束）
      |
      v
VLM 合成：查询 + 顶部场景 + 帧预览
      |
      v
答案 + (开始, 结束) 时间戳 + 帧缩略图 + 引用
```

## 技术栈

- 场景分割：TransNetV2（2024-26 最先进）或 PySceneDetect
- ASR：通过 faster-whisper 使用 Whisper-v3-turbo，带词级时间戳
- VLM 字幕 + 回答者：Gemini 2.5 Pro 或 Qwen3-VL-Max 或 Molmo 2
- 时间定位：TimeLens-100K 训练的适配器或 VideoITG
- 索引：带多向量支持的 Qdrant（字幕/帧/转录）
- UI：Next.js 15，带 HTML5 视频播放器和场景缩略图
- 评估：ActivityNet-QA、NeXT-GQA、自定义 100 个问题手工标注集
- 幻觉基准：带手工标注的计数和动作类型子集

## 构建它

1. **摄入遍历器。** 接受 YouTube URL 或本地 MP4。如需降至 720p。持久化 `{video_id, file_path}`。

2. **场景分割。** 运行 TransNetV2 或 PySceneDetect 生成 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标 100 小时：约 6k-8k 场景。

3. **ASR 通道。** 对音频运行 Whisper-v3-turbo；导出词级时间戳；分割成每场景转录切片。

4. **VLM 字幕。** 每个场景，用关键帧和简短字幕模板调用 Gemini 2.5 Pro（或 Qwen3-VL-Max）。生成字幕 + 帧嵌入。

5. **多向量索引。** 带三个命名向量的 Qdrant 集合。有效负载：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询。** 自然语言问题触发三个密集查询；用倒数排名融合合并；top-k=5 场景。

7. **时间定位。** 在顶部场景上运行 TimeLens 风格适配器以细化场景内的（开始，结束）窗口。

8. **VLM 合成。** 用查询 + 顶部 3 个场景片段（作为图像或短片）+ 转录调用 Gemini 2.5 Pro。要求 `(video_id, start_ms, end_ms)` 引用。

9. **评估。** 运行 ActivityNet-QA 和 NeXT-GQA。构建 100 个问题的自定义集。报告总体准确率 + 按类别分解（计数、动作、描述）。

## 使用它

```
$ video-qa ask --url=https://youtube.com/watch?v=X "第一分钟通过路口的汽车有多少辆？"
[场景]    检测到 23 个场景
[asr]     转录完成，4 分 12 秒
[索引]    写入 69 个向量（23 个场景 x 3）
[查询]    顶部场景：场景 3 [01:32-01:54]，置信度 0.84
[定位]    细化窗口：[00:12-00:58]
[合成]    gemini 2.5 pro，1.4 秒
答案：    00:12 到 00:58 之间有 5 辆汽车通过路口。
引用：[场景 3: 00:12-00:58]
          [00:14, 00:27, 00:44, 00:51, 00:57 的帧预览]
```

## 交付它

`outputs/skill-video-qa.md` 是可交付成果。给定 YouTube URL 或上传的视频，管道索引场景并用带时间戳引用的答案回答问题。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 时间定位 IoU | 保留定位集上的交并比 |
| 20 | 问答准确率 | NeXT-GQA 和自定义 100 个问题 |
| 20 | 摄入吞吐量 | 每美元的视频小时数 |
| 20 | UI 和引用用户体验 | 时间戳链接、缩略图条、跳转到帧 |
| 15 | 幻觉率 | 计数和动作类型准确率单独报告 |
| **100** | | |

## 练习

1. 在字幕通道上将 Gemini 2.5 Pro 换为 Qwen3-VL-Max。在人工评分的 50 个场景样本上报告字幕质量差值。

2. 将每场景帧嵌入减少到一个池化向量而非多向量。测量检索回归。

3. 构建"严格计数"模式：合成器用时间戳提取每个被计数的实例，用户点击验证。测量用户验证是否减少幻觉。

4. 基准摄入成本：三种 VLM 选择的每美元视频小时数。选择最佳点。

5. 添加说话人分离的转录：对音频运行 pyannote 说话人分离并嵌入每说话人转录。演示"Alice 说了什么关于 X 的话？"查询。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Scene segmentation（场景分割） | "镜头检测" | 在镜头边界将视频切割成场景 |
| Multi-vector index（多向量索引） | "字幕 + 帧 + 转录" | 每个表示带命名向量的 Qdrant 集合 |
| Temporal grounding（时间定位） | "确切在什么时候发生" | 为查询答案细化（开始，结束）窗口 |
| Frame embedding（帧嵌入） | "视觉表示" | 关键帧的向量嵌入；用于场景视觉相似性 |
| RRF fusion（RRF 融合） | "倒数排名融合" | 跨多个排名列表的合并策略；经典混合检索技巧 |
| Counting hallucination（计数幻觉） | "错误计数" | VLM 在"有多少个 X"问题上的已知失败模式 |
| ActivityNet-QA | "视频问答基准" | 长格式视频问答准确率基准 |

## 延伸阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 开源 VLM 检查点
- [TimeLens（CVPR 2026）](https://github.com/TencentARC/TimeLens) — 规模上的时间定位
- [Gemini 视频长上下文](https://deepmind.google/technologies/gemini) — 托管参考
- [VideoDB](https://videodb.io) — 视频 CRUD API 参考
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商业参考
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 场景分割模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 经典开源替代
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 参考评估基准

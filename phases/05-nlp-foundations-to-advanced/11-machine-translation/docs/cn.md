# 机器翻译

> 翻译是三十年来为 NLP 研究买单的任务，现在还在继续买单。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段第10课（注意力机制）、第5阶段第4课（GloVe、FastText、子词）
**预计时间：** 约75分钟

## 问题背景

一个模型读取一种语言的句子，输出另一种语言的句子。长度会变，词序会变，有些源语言词对应多个目标语言词，反之亦然。成语拒绝一一对应的翻译——"I miss you"在法语中是"tu me manques"，字面意思是"你欠缺在我这里"。任何词级对齐都无法在这里存活。

机器翻译是逼着 NLP 发明编码器-解码器、注意力、Transformer，最终催生整个 LLM 范式的那个任务。每一步进展的到来，都是因为翻译质量可以量化，而人机之间的差距顽固地存在。

本课跳过历史课，直接讲 2026 年的工作流水线：预训练多语言编码器-解码器（NLLB-200 或 mBART）、子词分词、集束搜索、BLEU 和 chrF 评估，以及那几个至今仍在生产中漏网的失败模式。

## 核心概念

现代机器翻译是一个在平行文本上训练的 Transformer 编码器-解码器。编码器按源语言的分词方式读取输入，解码器通过交叉注意力（第10课）逐个子词生成目标语言输出，解码时使用集束搜索避免贪心解码陷阱，最后对输出做反分词、反大写正规化，并与参考译文打分。

三个运营选择决定真实世界的 MT 质量：

- **分词器**：在混合语言语料上训练的 SentencePiece BPE。跨语言共享词汇表是 NLLB 支持零样本语言对的基础。
- **模型大小**：NLLB-200 蒸馏版 600M 能跑在笔记本电脑上；NLLB-200 3.3B 是已发布的生产默认；54.5B 是研究上界。
- **解码方式**：一般内容用集束宽度 4-5；长度惩罚防止输出过短；需要术语一致性时用约束解码。

## 动手实现

### 第一步：调用预训练 MT 模型

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

三件事很重要：`src_lang` 告诉分词器应用哪种脚本和分割方式；`forced_bos_token_id` 告诉解码器生成哪种语言；两者都是 NLLB 特有的技巧，mBART 和 M2M-100 有各自的约定，它们不可互换。

### 第二步：BLEU 和 chrF

BLEU 衡量输出与参考译文之间的 n-gram 重叠。四种 n-gram 粒度（1-4）的精确率取几何平均，加上对过短输出的简短惩罚。分数在 [0, 100] 范围内。广泛使用，但令人沮丧的是解读方式：30 BLEU 是"可用"，40 是"良好"，50 是"优秀"，1 BLEU 以内的差异是噪声。

chrF 衡量字符级 F 分数，对形态丰富的语言更敏感，BLEU 会低估这类语言的匹配数，通常与 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

务必使用 `sacrebleu`，它对分词方式做了归一化，使得分数在不同论文间可比。自己实现 BLEU 计算是误导性基准的来源。

### 2026 年的三层评估体系

现代机器翻译评估使用三类互补的指标家族，至少要用其中两种：

- **启发式方法**（BLEU、chrF）：快速、基于参考、可解释、对改写不敏感。用于历史对比和回归检测。
- **学习型方法**（COMET、BLEURT、BERTScore）：在人类判断数据上训练的神经模型，比较翻译与源文本和参考译文的语义相似度。COMET 自 2023 年以来与 MT 研究关联最高，是 2026 年质量优先场景的生产默认。
- **LLM 作为裁判**（无参考）：用大模型对翻译的流畅性、充分性、语气、文化适当性打分。设计好评分标准时，GPT-4-as-judge 与人类评分的一致率约 80%。用于没有参考译文的开放性内容。

2026 年实践栈：`sacrebleu` 算 BLEU 和 chrF，`unbabel-comet` 算 COMET，最后用提示 LLM 给出面向用户的最终信号。在信任任何指标用于生产数据之前，都要用 50-100 条人工标注样本对其进行校准。

无参考指标（COMET-QE、BLEURT-QE、LLM-as-judge）允许在没有参考译文的情况下评估翻译，这对不存在参考译文的长尾语言对非常重要。

### 第三步：生产中会出什么问题

上面的工作流水线 80% 的时候会流畅翻译，剩余 20% 会悄无声息地失败。已知失败模式：

- **幻觉**：模型生成源文本中不存在的内容。在陌生领域词汇上常见。症状：输出流畅，但声称源文本没有陈述的事实。缓解：对领域术语做约束解码、对受监管内容做人工审核、监控比输入长得多的输出。
- **目标语言错误**：模型翻译成了错误的语言。NLLB 在罕见语言对上出乎意料地容易出这个问题。缓解：检查 `forced_bos_token_id`，始终在输出上做语言 ID 检测。
- **术语漂移**："Sign up"在文档 1 变成"s'inscrire"，在文档 2 变成"créer un compte"。对界面文本和面向用户的字符串，一致性比原始质量更重要。缓解：词汇表约束解码或后编辑词典。
- **语气不匹配**：法语的"tu"和"vous"、日语的敬语级别。模型会选训练时更常见的形式，对面向客户的内容通常是错的。缓解：如果模型支持，用语气前缀 token 提示；或者在正式语料上微调小模型。
- **短输入时长度爆炸**：非常短的输入句子经常产生过长的翻译，因为长度惩罚在低于约 5 个源 token 时急剧下降。缓解：设置与源长度成比例的硬性最大长度上限。

### 第四步：面向特定领域微调

预训练模型是通才。法律、医疗或游戏对话翻译通过在领域平行数据上微调可以获得显著提升。方法并不复杂：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千条高质量平行样本胜过几十万条从网上爬来的噪声样本。训练数据质量是生产中最大的单一杠杆。

## 工程应用

2026 年 MT 的生产技术栈：

| 使用场景 | 推荐起点 |
|---------|---------|
| 任意到任意，200 种语言 | `facebook/nllb-200-distilled-600M`（笔记本）或 `nllb-200-3.3B`（生产） |
| 以英语为中心，高质量，50 种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短文本运行，便宜推理，英-法/德/西 | Helsinki-NLP / Marian 模型 |
| 延迟关键的浏览器端 | ONNX 量化 Marian（约 50 MB） |
| 最高质量，愿意付费 | GPT-4 / Claude / Gemini 加翻译提示词 |

截至 2026 年，LLM 在多个语言对上已经超过专用 MT 模型，尤其是在惯用语内容和长上下文上。权衡是每 token 成本和延迟。当上下文长度、文体一致性或通过提示词做领域适应比吞吐量更重要时，选 LLM。

## 交付物

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 练习

1. **（简单）** 用 `nllb-200-distilled-600M` 将一段 5 句英文翻译成法语，再翻译回英语，测量来回翻译后与原文的接近程度，应该能看到语义保留但用词有所漂移。
2. **（中等）** 使用 `fasttext lid.176` 或 `langdetect` 对翻译输出做语言 ID 检测，集成到 MT 调用中，在返回前捕获目标语言错误的输出。
3. **（困难）** 在自选的 5000 对领域语料上微调 `nllb-200-distilled-600M`，在保留集上对比微调前后的 BLEU，报告哪些类型的句子有所提升，哪些有所退步。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| BLEU | "翻译分数" | 带简短惩罚的 n-gram 精确率，范围 [0, 100] |
| chrF | "字符 F 分数" | 字符级 F 分数，对形态丰富语言更敏感 |
| NMT（神经机器翻译） | "神经翻译" | 在平行文本上训练的 Transformer 编码器-解码器，2017 年后的默认方案 |
| NLLB | "不落下任何语言" | Meta 的 200 语言 MT 模型家族 |
| 约束解码 (Constrained decoding) | "受控输出" | 强制特定 token 或 n-gram 出现/不出现在输出中 |
| 幻觉 (Hallucination) | "凭空捏造" | 源文本中不存在的模型输出内容 |

## 延伸阅读

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB 论文
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — 为什么 `sacrebleu` 是报告 BLEU 的唯一正确方式
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF 论文
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — 实用微调逐步指南

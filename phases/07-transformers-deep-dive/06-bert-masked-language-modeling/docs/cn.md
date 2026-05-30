# BERT——掩码语言建模

> GPT 预测下一个词。BERT 预测缺失的词。一句话的区别——带来了五年来所有与嵌入相关的成就。

**类型：** 构建
**语言：** Python
**前置知识：** 第7阶段第05课（完整 Transformer）、第5阶段第02课（文本表示）
**预计时间：** 约45分钟

## 问题背景

2018 年，每个 NLP 任务——情感分析、命名实体识别、问答、蕴含——都在各自的标注数据上从头训练自己的模型。没有可以微调的预训练"理解英语"检查点。ELMo（2018）证明可以用双向 LSTM 预训练上下文嵌入；有所帮助，但泛化能力有限。

BERT（Devlin et al. 2018）提出了一个问题：如果我们用 Transformer 编码器、在互联网上所有句子上训练，并强迫它从两侧上下文预测缺失词呢？然后在下游任务上微调一个头。参数效率令人叹为观止。

结果是：在 18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）统治了所有现存的 NLP 排行榜。到 2020 年，地球上每一个搜索引擎、内容审核流水线和语义搜索系统内部都有一个 BERT。

2026 年，仅编码器模型仍然是分类、检索和结构化抽取的正确工具——它们每 token 运行速度比解码器快 5–10 倍，其嵌入是所有现代检索栈的骨干。ModernBERT（2024 年 12 月）将架构推进到 8K 上下文，采用 Flash Attention + RoPE + GeGLU。

## 核心概念

### 训练信号

取一个句子：`the quick brown fox jumps over the lazy dog`

随机掩码 15% 的 token：

```
输入：  the [MASK] brown fox jumps [MASK] the lazy dog
目标：  the  quick  brown fox jumps  over  the lazy dog
```

训练模型预测掩码位置的原始 token。因为编码器是双向的，在位置 1 预测 `[MASK]` 时可以使用位置 2+ 的 `brown fox jumps`。这是 GPT 做不到的事情。

### BERT 的掩码规则

在被选中用于预测的 15% token 中：

- 80% 替换为 `[MASK]`
- 10% 替换为随机 token
- 10% 保持不变

为什么不总是用 `[MASK]`？因为 `[MASK]` 在推理时从不出现。如果 100% 的掩码位置都用 `[MASK]`，会在预训练和微调之间造成分布偏移。10% 随机 + 10% 不变让模型保持"诚实"。

### 下一句预测（NSP）——以及为什么被抛弃

原始 BERT 也训练了 NSP：给定两个句子 A 和 B，预测 B 是否紧随 A。RoBERTa（2019）对其做了消融实验，证明 NSP 有害无益。现代编码器跳过了它。

### 2026 年的变化：ModernBERT

2024 年 ModernBERT 论文用 2026 年的原语重建了块：

| 组件 | 原始 BERT（2018） | ModernBERT（2024） |
|------|-----------------|------------------|
| 位置编码 | 学习型绝对编码 | RoPE |
| 激活函数 | GELU | GeGLU |
| 归一化 | LayerNorm | Pre-norm RMSNorm |
| 注意力 | 完整密集注意力 | 交替局部（128）+ 全局 |
| 上下文长度 | 512 | 8192 |
| 分词器 | WordPiece | BPE |

与 2018 年的堆栈不同，它原生支持 Flash Attention。在序列长度 8K 时，推理速度比 DeBERTa-v3 快 2–3 倍，GLUE 分数更高。

### 2026 年仍选择编码器的场景

| 任务 | 编码器优于解码器的原因 |
|------|------------------|
| 检索/语义搜索嵌入 | 双向上下文 = 每 token 嵌入质量更高 |
| 分类（情感、意图、毒性） | 单次前向传播；无生成开销 |
| NER / Token 标注 | 按位置输出，天然双向 |
| 零样本蕴含（NLI） | 编码器顶部加分类器头 |
| RAG 重排序器 | 交叉编码器打分，比 LLM 重排序器快 10 倍 |

## 动手实现

### 第一步：掩码逻辑

见 `code/main.py`。函数 `create_mlm_batch` 接受一个 token ID 列表、词表大小和掩码概率。返回输入 ID（已应用掩码）和标签（仅在掩码位置，其余为 -100——PyTorch 的忽略索引约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: 保持原始 token
    return input_ids, labels
```

### 第二步：在小型语料库上运行 MLM 预测

在 20 个词的词表、200 个句子上训练 2 层编码器 + MLM 头。不使用梯度——做前向传播的正确性检查。完整训练需要 PyTorch。

### 第三步：比较掩码类型

展示三路规则如何让模型在没有 `[MASK]` 的情况下仍可用。对未掩码句子和掩码句子分别预测。两者都应产生合理的 token 分布，因为模型在训练中见过这两种模式。

### 第四步：微调头

将 MLM 头替换为玩具情感数据集上的分类头。只有头在训练；编码器被冻结。这是每个 BERT 应用都遵循的模式。

## 工程应用

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**嵌入模型就是微调过的 BERT。** `sentence-transformers` 中的模型如 `all-MiniLM-L6-v2` 是用对比损失训练的 BERT。编码器相同，损失变了。

**交叉编码器重排序器也是微调过的 BERT。** 在 `[CLS] query [SEP] doc [SEP]` 上做对分类。query 和 doc 之间的双向注意力正是交叉编码器比双编码器质量更高的原因。

**2026 年不选 BERT 的情况。** 任何需要生成的场景。编码器没有合理的方式自回归地生成 token。另外：任何在 10 亿参数以下、小型解码器能以更大灵活性匹配质量的场景（Phi-3-Mini、Qwen2-1.5B）。

## 交付物

见 `outputs/skill-bert-finetuner.md`。该技能为新的分类或抽取任务规划 BERT 微调方案（骨架选择、头规格、数据、评估、停止条件）。

## 练习

1. **（简单）** 运行 `code/main.py`，在 10,000 个 token 上打印掩码分布。确认约 15% 被选中，其中约 80% 变为 `[MASK]`。
2. **（中等）** 实现全词掩码：如果一个词被分词为多个子词，则一起掩码或不掩码所有子词。测量这是否提升了在 500 句语料库上的 MLM 准确率。
3. **（困难）** 在公开数据集的 10,000 个句子上训练一个微型（2 层，d=64）BERT。对 `[CLS]` token 微调用于 SST-2 情感分析。与参数量相当的仅解码器基线对比——哪个赢？

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| MLM（掩码语言建模） | "掩码语言建模" | 训练信号：随机替换 15% 的 token 为 `[MASK]`，预测原始 token |
| 双向 (Bidirectional) | "两侧都看" | 编码器注意力没有因果掩码——每个位置看到所有其他位置 |
| `[CLS]` | "池化 token" | 每个序列前置的特殊 token；其最终嵌入用作句子级表示 |
| `[SEP]` | "段分隔符" | 分隔成对序列（如 query/doc、句子 A/B） |
| NSP（下一句预测） | "下一句预测" | BERT 的第二个预训练任务；RoBERTa 证明无用，2019 年后被抛弃 |
| 微调 (Fine-tuning) | "适配任务" | 编码器基本冻结；在顶部训练小型头用于下游任务 |
| 交叉编码器 (Cross-encoder) | "重排序器" | 将 query 和 doc 一起作为输入、输出相关性分数的 BERT |
| ModernBERT | "2024 年的更新" | 用 RoPE、RMSNorm、GeGLU、交替局部/全局注意力、8K 上下文重建的编码器 |

## 延伸阅读

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 原始论文
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — 如何正确训练 BERT；终结 NSP
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 替换 token 检测在相同计算下优于 MLM
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 论文
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 典范编码器参考实现

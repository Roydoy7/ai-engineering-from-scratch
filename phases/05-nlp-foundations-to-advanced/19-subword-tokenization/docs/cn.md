# 子词分词——BPE、WordPiece、Unigram、SentencePiece

> 词级分词器对未见词无能为力。字符级分词器让序列长度爆炸。子词分词器折中了两者。每个现代 LLM 都用其中一种。

**类型：** 学习
**语言：** Python
**前置知识：** 第5阶段第1课（文本处理）、第5阶段第4课（GloVe/FastText/子词）
**预计时间：** 约60分钟

## 问题背景

你的词汇表有 5 万个词，用户输入"untokenizable"，分词器返回 `[UNK]`，模型对这个词没有任何信号。更糟糕的是：你语料库中 90 百分位的文档有 40 个罕见词，意味着每篇文档丢失 40 比特信息。

子词分词解决了这个问题。常见词保持为单个 token，罕见词分解成有意义的片段：`untokenizable` → `un`、`token`、`izable`。训练数据覆盖一切，因为任何字符串归根结底都是字节序列。

2026 年的每个前沿 LLM 都使用三种算法之一（BPE、Unigram、WordPiece），封装在三个库之一（tiktoken、SentencePiece、HF Tokenizers）中，没有选择就无法发布语言模型。

## 核心概念

**BPE（字节对编码）**：从字符级词汇表开始，统计每对相邻字符，合并最高频的对成为新 token，重复直到达到目标词汇表大小。主导算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级 BPE**：同样的算法，但在原始字节（256 个基本 token）而非 Unicode 字符上操作。保证零 `[UNK]` token——任何字节序列都能编码。GPT-2 使用 50,257 个 token（256 字节 + 50,000 次合并 + 1 个特殊 token）。

**Unigram**：从庞大的词汇表开始，给每个 token 分配一元组概率，迭代地剪除那些删除后对语料库对数似然增加最小的 token。推理时可以采样分词结果（通过子词正则化用于数据增强），被 T5、mBART、ALBERT、XLNet、Gemma 使用。

**WordPiece**：合并能最大化训练语料库似然的词对，而非原始频率。被 BERT、DistilBERT、ELECTRA 使用。

**SentencePiece vs tiktoken**：SentencePiece 是直接在原始 Unicode 文本上**训练**词汇表的库（BPE 或 Unigram），将空格编码为 `▁`；tiktoken 是 OpenAI 针对预构建词汇表的快速**编码器**，不做训练。

经验法则：
- **训练新词汇表**：SentencePiece（多语言，无需预分词）或 HF Tokenizers
- **针对 GPT 词汇表的快速推理**：tiktoken（cl100k_base、o200k_base）
- **两者都要**：HF Tokenizers——一个库，训练+服务

## 动手实现

### 第一步：从零实现 BPE

核心循环：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

算法编码了三件事：`</w>` 标记词尾，使"low"（后缀）和"lower"（前缀）保持不同；频率加权让高频对先赢；合并列表是有序的——推理按训练顺序应用合并。

### 第二步：用学到的合并规则编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素的 O(n·|merges|)。生产实现（tiktoken、HF Tokenizers）使用带优先队列的合并排名查找，以近线性时间运行。

### 第三步：SentencePiece 实践

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # 或 "unigram"
    character_coverage=0.9995, # CJK 字符可以调低（如日语用 0.995）
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：不需要预分词，空格编码为 `▁`，`character_coverage` 控制罕见字符被保留还是映射到 `<unk>` 的激进程度。

### 第四步：tiktoken 用于 OpenAI 兼容词汇表

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

只做编码，速度快（Rust 后端），与 GPT-4/5 的分词精确匹配，用于字节计数、成本估算、上下文窗口预算。

## 2026 年仍在发货的陷阱

- **分词器漂移**：在词汇表 A 上训练，部署时用词汇表 B。Token ID 不同，模型输出垃圾。在 CI 中检查 `tokenizer.json` 哈希值。
- **空格歧义**：BPE 中"hello"和" hello"（带前导空格）产生不同 token。始终明确指定 `add_special_tokens` 和 `add_prefix_space`。
- **多语言训练不足**：以英语为主的语料库产生的词汇表将非拉丁字符集分割成 5-10 倍的 token 数。在 GPT-3.5 上相同的提示词，日语/阿拉伯语成本是英语的 5-10 倍。`o200k_base` 部分修复了这个问题。
- **emoji 分割**：一个 emoji 可能占 5 个 token，在预算上下文时要检查 emoji 处理。

## 工程应用

2026 年技术栈：

| 情况 | 选择 |
|------|------|
| 从零训练单语言模型 | HF Tokenizers（BPE） |
| 训练多语言模型 | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 服务 OpenAI 兼容 API | tiktoken（`o200k_base` 用于 GPT-4+） |
| 特定领域词汇（代码、数学、蛋白质） | 在领域语料上训练自定义 BPE，与基础词汇合并 |
| 边缘推理，小模型 | Unigram（较小词汇表效果更好） |

词汇表大小是一个缩放决策，不是常数。粗略经验：<1B 参数用 32k，1-10B 用 50-100k，多语言/前沿模型用 200k+。

## 交付物

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 练习

1. **（简单）** 在小型语料上训练 500 次合并的 BPE，对三个保留词编码，有多少恰好产生 1 个 token，多少产生超过 1 个 token？
2. **（中等）** 对 100 个英语维基百科句子，比较 `cl100k_base`、`o200k_base` 和你用 vocab=32k 训练的 SentencePiece BPE 的 token 数量，报告每种方法的压缩比。
3. **（困难）** 在同一语料上分别用 BPE、Unigram 和 WordPiece 训练，测量用各自分词器在小型情感分类器上的下游准确率，三者之间的差距超过 1 个 F1 点吗？

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| BPE（字节对编码） | "字节对编码" | 贪心合并最高频字符对直到达到目标词汇表大小 |
| 字节级 BPE | "永远没有未知词" | 在原始 256 字节上做 BPE，GPT-2/Llama 使用 |
| Unigram | "概率分词器" | 用对数似然从大候选集中剪枝，T5、Gemma 使用 |
| SentencePiece | "处理空格的那个" | 在原始文本上训练 BPE/Unigram 的库，空格编码为 `▁` |
| tiktoken | "快速的那个" | OpenAI 基于 Rust 的 BPE 编码器，针对预构建词汇表，不做训练 |
| 合并列表 (Merge list) | "魔法数字" | 有序的 `(a, b) → ab` 合并列表，推理时按顺序应用 |
| 字符覆盖率 (Character coverage) | "多罕见才算太罕见" | 分词器必须覆盖训练语料中字符的比例，约 0.9995 为典型值 |

## 延伸阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 论文
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram 论文
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — SentencePiece 库
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 简明参考
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — 使用手册和编码列表

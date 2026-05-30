# 从零构建分词器

> 第01课给了你一个玩具，这课给你一件武器。

**类型：** 构建
**语言：** Python
**前置知识：** 第10阶段第01课（分词器：BPE、WordPiece、SentencePiece）
**预计时间：** 约90分钟

## 学习目标

- 构建处理Unicode、空白规范化和特殊token的生产级BPE分词器
- 实现字节级回退，使分词器能对任何输入（包括emoji、中日韩字符和代码）编码而不产生未知token
- 添加预分词正则表达式模式，在应用BPE合并前按词边界切分文本
- 在语料上训练自定义分词器，并与tiktoken在多语言文本上的压缩比进行对比评估

## 问题背景

你在第01课实现的BPE分词器能处理英文文本。现在扔进去日文试试，或者emoji，或者混杂着制表符和空格的Python代码。

它会崩溃。

不是因为BPE有问题——而是实现不完整。生产级分词器要处理任意编码的原始字节、在切分前规范化Unicode、管理永远不参与BPE合并的特殊token、将预分词与子词切分串联起来，并且要足够快，不能成为处理15万亿token训练流水线的瓶颈。

GPT-2的分词器有50,257个token，Llama 3有128,256个，GPT-4大约10万个。这些不是玩具数字。背后的合并表是在数百GB文本上训练出来的，而围绕它的那套机制——规范化、预分词、特殊token注入、对话模板格式化——才是区分"能处理'hello world'"和"能处理整个互联网"的分词器的关键。

你将要构建这套机制。

## 核心概念

### 完整流水线

生产级分词器不是单一算法，而是五个阶段的流水线，每个阶段解决一个不同的问题。

```mermaid
graph LR
    A[原始文本] --> B[规范化]
    B --> C[预分词]
    C --> D[BPE合并]
    D --> E[特殊Token]
    E --> F[Token ID]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段有其具体职责：

| 阶段 | 作用 | 重要性 |
|------|------|--------|
| 规范化 | NFKC Unicode，可选小写化，可选去除重音 | "fi"连字（U+FB01）变成"fi"（两个字符）。没有这步，同一个词会产生不同token。 |
| 预分词 | BPE前将文本切分成块 | 防止BPE跨词边界合并。"the cat"永远不应该产生token"e c"。 |
| BPE合并 | 对字节序列应用学习到的合并规则 | 核心压缩步骤，将原始字节转换为子词token。 |
| 特殊Token | 注入[BOS]、[EOS]、[PAD]、对话模板标记 | 这些token有固定ID，永远不参与BPE合并，模型需要它们来理解结构。 |
| ID映射 | 将token字符串转换为整数ID | 模型看到的是整数，不是字符串。 |

### 字节级BPE

第01课的分词器操作UTF-8字节，这个方向是对的。但我们跳过了一个重要的问题：如果这些字节不是有效的UTF-8怎么办？

字节级BPE通过将每个可能的字节值（0-255）都视为有效token来解决这个问题。基础词表恰好是256个条目。任何文件——文本、二进制、损坏的——都可以分词，不会产生未知token。

GPT-2加了一个技巧：将每个字节映射到可打印的Unicode字符，让词表保持人类可读。字节0x20（空格）在他们的映射中变成字符"G"。这纯粹是展示效果，算法本身不关心。

真正的威力在于：字节级BPE能处理地球上所有语言。中文字符每个是3个UTF-8字节，日文可以是3-4个字节，阿拉伯语、天城文、emoji——全都只是字节序列。BPE算法在这些字节序列中寻找模式，和在英语ASCII字节中寻找模式完全一样。

### 预分词

在BPE接触你的文本之前，需要先把它切成块。这防止合并算法产生跨越词边界的token。

GPT-2使用正则表达式切分文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这个模式对缩写词（"don't"变成"don"+"'t"）、带可选前导空格的单词、数字、标点和空白进行切分。前导空格保持附着在单词上——所以"the cat"变成 [" the", " cat"]，而不是 ["the", " ", "cat"]。

Llama使用SentencePiece，完全跳过正则表达式。它把原始字节流当成一个长序列，让BPE算法自己决定边界。这更简单，但给BPE更多自由来创建跨词token。

选择很重要。GPT-2的正则表达式防止分词器学到一个词末尾的"the"和下一个词开头的"the"应该合并。SentencePiece允许这种情况，有时产生更高效的压缩，但token可解释性较差。

### 特殊Token

每个生产级分词器都为结构标记保留了token ID：

| Token | 用途 | 使用模型 |
|-------|------|---------|
| `[BOS]` / `<s>` | 序列开始 | Llama 3、GPT |
| `[EOS]` / `</s>` | 序列结束 | 所有模型 |
| `[PAD]` | 批次对齐填充 | BERT、T5 |
| `[UNK]` | 未知token（字节级BPE消除了这个需求） | BERT、WordPiece |
| `<\|im_start\|>` | 对话消息边界开始 | ChatGPT、Qwen |
| `<\|im_end\|>` | 对话消息边界结束 | ChatGPT、Qwen |
| `<\|user\|>` | 用户轮次标记 | Llama 3 |
| `<\|assistant\|>` | 助手轮次标记 | Llama 3 |

特殊token永远不会被BPE切分。它们在合并算法运行前就被精确匹配，替换为固定ID，周围的文本则正常分词。

### 对话模板

这是大多数人感到困惑、大多数实现出错的地方。

当你向对话模型发送消息时，API接受消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看到的不是JSON，而是一个扁平的token序列。对话模板用特殊token把消息转换成这个扁平序列。每个模型做法不同：

```
Llama 3：
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT：
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

模板用错了，模型就会产生垃圾。它是在一种精确格式上训练的，任何偏差——少一个换行、换了个token、多了个空格——都会把输入推出训练分布。

### 速度

Python对于生产级分词来说太慢了。

tiktoken（OpenAI）用Rust编写，提供Python绑定。HuggingFace tokenizers也是Rust，SentencePiece是C++。这些实现比纯Python快10-100倍。

作个对比：以100万token/秒（快速Python）的速度对Llama 3预训练的15万亿token分词，需要174天；以1亿token/秒（Rust）的速度，只需要1.7天。

你在Python中构建是为了理解算法。在生产中，你会使用编译实现，只接触Python包装器。

## 动手实现

### 第一步：字节级编码

基础工作。把任意字符串转换为字节序列，为了展示方便将每个字节映射为可打印字符，并实现逆向转换。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

在多语言文本上测试，观察字节数：

```python
texts = [
    ("英语", "hello"),
    ("中文", "你好"),
    ("Emoji", "🔥"),
    ("混合", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)}个字符 -> {len(b)}个字节 -> {b}")
```

"hello"是5个字节，"你好"是6个字节（每个字符3个），火焰emoji是4个字节。字节级分词器不关心是什么语言，字节就是字节。

### 第二步：带正则表达式的预分词器

用GPT-2正则表达式模式把文本切分成块，每块由BPE独立处理。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex`模块支持Unicode属性转义（`\p{L}`表示字母，`\p{N}`表示数字）。标准库`re`不支持，所以我们回退到ASCII字符类。生产多语言分词器需要安装`regex`。

试一下：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格保持附着在单词上，缩写词在撇号处切分，标点变成独立的块。BPE永远不会跨这些边界合并token。

### 第三步：在字节序列上运行BPE

第01课的核心算法，但现在独立对每个预分词块操作。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 第四步：特殊Token处理

特殊token需要精确匹配和固定ID，完全绕过BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 第五步：完整分词器类

将所有步骤串联：规范化、在特殊token处切分、预分词、BPE合并、映射到ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 第六步：多语言测试

真正的考验。扔进去英语、中文、emoji和代码。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"输入：   {text}")
    print(f"Token数：{len(ids)}个ID")
    print(f"解码：   {decoded}")
    print()
```

中文字符每个产生3个字节，emoji产生4个字节。这些都不会使分词器崩溃，都不会产生未知token。这就是字节级BPE的威力。

## 使用生产工具

### 比较真实分词器

加载Llama 3、GPT-4和Mistral的实际分词器，看它们如何处理同一段多语言文字。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4（{len(tokens)}个token）：{pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name}（{len(tokens)}个token）：{pieces[:20]}...")
```

同一段文本会产生不同的token数。Llama 3有128K词表，对常见模式的合并更激进。GPT-4有10万个token，居中。Mistral只有32K，产生更多token，但嵌入层更小。

权衡永远如此：词表越大，序列越短，但参数越多。

## 交付物

本课产出用于构建和调试生产级分词器的提示词，见 `outputs/prompt-tokenizer-builder.md`。

## 练习

1. **（简单）** 添加 `get_token_bytes(id)` 方法，显示任意token ID对应的原始字节。用它检查你最常用的合并token实际上代表什么。
2. **（中等）** 实现Llama风格的预分词器，按空白和数字切分但保留前导空格。在同一语料上对比它与GPT-2正则表达式方式产生的词表差异。
3. **（困难）** 添加对话模板方法，接受 `{"role": ..., "content": ...}` 消息列表，产生Llama 3对话格式的正确token序列。与HuggingFace实现进行对比测试。

## 关键术语

| 术语 | 常见说法 | 真正含义 |
|------|---------|---------|
| 字节级BPE（Byte-level BPE） | "在字节上工作的分词器" | 以256个字节值为基础词表的BPE——处理任何输入都不产生未知token |
| 预分词（Pre-tokenization） | "BPE前的切分" | 基于正则表达式或规则的切分，防止BPE跨词边界合并 |
| NFKC规范化（NFKC normalization） | "Unicode清理" | 规范分解后接兼容性组合——"fi"连字变成"fi"，全角"A"变成"A" |
| 对话模板（Chat template） | "消息变成token的方式" | 将角色/内容消息列表转换为扁平token序列的精确格式——模型专用，必须与训练格式完全匹配 |
| 特殊Token（Special tokens） | "控制token" | 绕过BPE的保留token ID——[BOS]、[EOS]、[PAD]、对话标记——在合并前精确匹配 |
| 生育率（Fertility） | "每个词的token数" | 输出token数与输入词数之比——GPT-4英文是1.3，韩文是2-3，越高意味着上下文浪费越多 |
| tiktoken | "OpenAI分词器" | 带Python绑定的Rust BPE实现——比纯Python快10-100倍 |
| 合并表（Merge table） | "词表" | 训练期间学习到的有序字节对合并列表——这就是分词器的全部学习知识 |

## 延伸阅读

- [OpenAI tiktoken源码](https://github.com/openai/tiktoken) — GPT-3.5/4使用的Rust BPE实现
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) — 支持BPE、WordPiece、Unigram的Rust分词器库
- [Llama 3论文（Meta, 2024）](https://arxiv.org/abs/2407.21783) — 关于128K词表和分词器训练的详细信息
- [SentencePiece（Kudo & Richardson, 2018）](https://arxiv.org/abs/1808.06226) — 语言无关分词
- [GPT-2分词器源码](https://github.com/openai/gpt-2/blob/master/src/encoder.py) — 最初的字节到Unicode映射

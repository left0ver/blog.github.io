---
title: 如何使用tokenizers库训练自己的tokenizer
mathjax: true
date: 2025-07-23 18:04:53
tags:
  - AI
---

所有代码均可在[train_my_tokenizer.py](https://github.com/left0ver/study-transformer/blob/main/train_my_tokenizer.py)中找到

# 如何使用tokenizers库训练自己的tokenizer

tokenizers 包含五个组件，分别是Normalizers，Pre-tokenizers，Models，Post-Processors，Decoders

## Normalizers

`normalizers.NFC`: 会将基础字符和附加字符组合成一个单一的预组合字符，例如`e` + `´` → `é`

`normalizers.NFD`: NFC逆过来，将一些预组合字符分解为基础字符和附加字符，如：`é` → `e` + `´`。

`normalizers.Lowercase`：转为小写

`normalizers.Strip`：去除左右两边的空格（可以配置只去除左边或者只去除右边），但是对于llm模型来说，通常会需要处理代码问题，这时候空格就变得比较重要了，因此大多数时候不会去除空格

`normalizers.StripAccents`:移除字符上的重音或变音符号。例如：`"déjà vu"` → `"deja vu"`（通常先使用NFD进行分解，然后去除重音符号）

`normalizers.Replace`：使用正则表达式/字符串替换文本

`normalizers.Replace`: 在字符串开头添加前缀

`normalizers.ByteLevel`:将字符串转为字节序列（一般是UTF-8编码），byte—bpe会用到

<!-- more -->

**实践**

>测试了一下想要使用StripAccents去除重音符号，必须先使用NFD()将重音字符分解，即 é` → `e` + `´ ,然后StripAccents 可以去除`´`

```python
tokenizer.normalizer = normalizers.Sequence([normalizers.Strip(), normalizers.StripAccents(),normalizers.Lowercase()])

print(tokenizer.normalizer.normalize_str("  Hello, my friend, how are you?Ġ  ")) # hello, my friend, how are you?ġ

tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Strip(),
        normalizers.Lowercase(),
        normalizers.NFD(),
        normalizers.StripAccents(),
    ]
)
print(tokenizer.normalizer.normalize_str("  Hello, my friend, Héllò hôw are ü?résuméĠ  ")) # hello, my friend, hello how are u?resumeg
```

## Pre-tokenizers

通过一组规则对输入进行拆分，即将输入的文本切分成小块，后续的model不过扩多个块构建token，例如我们按空格切分.

`hello world` 切分为`[hello,world]`,构建token的时候会将hello 和 world别算作一个词来构建词表

有以下的pre_tokenizer：

`ByteLevel`: 在空格处进行分割，使用utf-8编码并将词元转为字节流。`hello my friend, how are you? -> [hello,Ġmy,Ġfriend,",",Ġhow,Ġare,Ġyou,?]`  

>add_prefix_space=True 在句子前面加上空格

>Ġ代表空格

>因此我们可以使用256个byte来表示任何token，因此可以不需要unk token

`Whitespace`: 使用空格和所有不是字母、数字或下划线的字符进行分割。 `hello world！-> [hello,world,!]`

`WhitespaceSplit`: 按最常见的空格字符划分.`hello world! -> [hello,world!]`

`Digits`: 将数字分离出来.`hello123world -> [hello,123,world]`

`Punctuation`:将所有标点符号分离出来。`hello-world! -> [hello,-,world,!]`

`CharDelimiterSplit`：根据所给的字符分割。例如根据x分割，`helloxworld -> [hello,world]`

`Split`: 根据所给的`pattern`(字符串/正则表达式)拆分, 拆分之后

假设我们设置`pattern = "-"`

- removed：找到分隔符进行拆分，然后分隔符丢弃。`hello-world ->[hello,world]`
- isolated: 分隔符切分完文本之后，分隔符会作为一个独立的词。 `hello-world -> [hello,-,world]`
- merged_with_previous: 和前一个词合并。 `hello-world -> [hello-,world]`
- merged_with_next: 和后一个词合并。 `hello-world -> [hello,-world]`
- contiguous: 用来处理多个分隔符连续出现的情况，将连续出现的分隔符合并为一个单独的词元。 `hello--world ->[hello,--,world]`,和isolated 的行为有点点差别

```python
PreTokenizer = pre_tokenizers.PreTokenizer
BertPreTokenizer = pre_tokenizers.BertPreTokenizer
ByteLevel = pre_tokenizers.ByteLevel
CharDelimiterSplit = pre_tokenizers.CharDelimiterSplit
Digits = pre_tokenizers.Digits
FixedLength = pre_tokenizers.FixedLength
Metaspace = pre_tokenizers.Metaspace
Punctuation = pre_tokenizers.Punctuation
Sequence = pre_tokenizers.Sequence
Split = pre_tokenizers.Split
UnicodeScripts = pre_tokenizers.UnicodeScripts
Whitespace = pre_tokenizers.Whitespace
WhitespaceSplit = pre_tokenizers.WhitespaceSplit
```

接下来就实践一下：

```python
tokenizer = Tokenizer(model =  models.BPE(byte_fallback =True))
tokenizer.normalizer = normalizers.Sequence([normalizers.Strip(), normalizers.StripAccents(),normalizers.Lowercase()])

tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test pre-tokenization!")) # [("Let's", (0, 5)), ('test', (6, 10)), ('pre-tokenization!', (11, 28))] 

tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test pre-tokenization!")) # [('Let', (0, 3)), ("'", (3, 4)), ('s', (4, 5)), ('test', (6, 10)), ('pre', (11, 14)), ('-', (14, 15)), ('tokenization', (15, 27)), ('!', (27, 28))]


tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test pre-tokenization!")) # [('Let', (0, 3)), ("'s", (3, 5)), ('Ġtest', (5, 10)), ('Ġpre', (10, 14)), ('-', (14, 15)), ('tokenization', (15, 27)), ('!', (27, 28))]


tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)# add_prefix_space=True 在句子前面加上空格
print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test pre-tokenization!"))# [('ĠLet', (0, 3)), ("'s", (3, 5)), ('Ġtest', (5, 10)), ('Ġpre', (10, 14)), ('-', (14, 15)), ('tokenization', (15, 27)), ('!', (27, 28))]
```

## Models

models 即用来tokenizer的算法，通常由bpe，wordpiece（用于bert等），Unigram ，WordLevel

- WordLevel：直接将pre-tokenizer切分之后的词映射到对应的ids，不会做其他任何操作

models通常在训练的时候会用到

```python
trainer = trainers.BpeTrainer(
        show_progress=True,
        min_frequency=2,
        # 因为我们使用的是ByteLevel，所以不需要添加特殊的token
        special_tokens=["[SOS]", "[EOS]", "[PAD]"],
    )

    tokenizer.train_from_iterator(
        get_all_sentences(ds_raw, config["lang_src"]),
        trainer=trainer,
    )
    tokenizer.save(str(tokenizer_path))
```



## Post-Processors

有时候我们想要将tokenizer的字符串在输入模型之前插入一些特殊的token，例如bert中就会在开头和末尾分别插入[CLS]和[SEP]

```python
post_processor = processors.TemplateProcessing(single="[SOS] $A [EOS]",pair="[SOS] $A [EOS] $B [EOS]",special_tokens=("[SOS]", "[EOS]"))
# input:("hello world","how are you")
#output:("[SOS] hello world [EOS] how are you [EOS]")
```



**实践：** 


### TemplateProcessing

设置Template，这个template跟bert的很类似，只是特殊token不一样

```python
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[SOS]:0 $A:0 [EOS]:0",
        pair=f"[SOS]:0 $A:0 [EOS]:0 $B:1 [EOS]:1",
        special_tokens=[
            ("[SOS]", tokenizer.token_to_id("[SOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
```



- 单个句子

```python
sentence = "Let's test this tokenizer."
encoding = tokenizer.encode(sentence)
print(encoding) # 无post_processor ，tokens = ['let', "'s", 'Ġtest', 'Ġthis', 'Ġtoken', 'iz', 'er', '.']

tokenizer.post_processor  = processors.TemplateProcessing(single="[SOS] $A [EOS]",pair="[SOS] $A [EOS] $B [EOS]",special_tokens=[("[SOS]",0), ("[EOS]",1)])

post_processor_res =tokenizer.post_processor.process(encoding)

print(post_processor_res) # ['[SOS]', 'let', "'s", 'Ġtest', 'Ġthis', 'Ġtoken', 'iz', 'er', '.', '[EOS]']


# 尝试了一下
```




- 一对句子（save的时候没有设置post_processor,然后从文件中加载tokenizer，再设置post_processor，先调用encode，然后再调用process，测了一下会有问题）

```python
encoding = tokenizer.encode("hello world", "Let's test this tokenizer.")
print(encoding.tokens)
post_processor_res = tokenizer.post_processor.process(encoding)
print(post_processor_res.tokens)
```
- 一对句子，如果提前设置好了post_processor,调用encode时候会自动调用后处理的方法，然后得到如下的结果

```python
encoding = tokenizer.encode("hello world", "Let's test this tokenizer.")
print(encoding.tokens)
# encoding.tokens
#['[SOS]', 'he', 'll', 'o', 'Ġworld', '[EOS]', 'let', "'s", 'Ġtest', 'Ġthis', 'Ġtoken', 'iz', 'er', '.', '[EOS]']
#encoding.type_ids
#[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

### ByteLevel

```python
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
encoding = tokenizer.encode("Let's test this tokenizer.")
# encoding.tokens
#['let', "'s", 'Ġtest', 'Ġthis', 'Ġtoken', 'iz', 'er', '.']
# start, end = encoding.offsets[3] 
# sentence[start:end]
# ' this'

tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
encoding = tokenizer.encode("Let's test this tokenizer.")
# encoding.tokens
#['let', "'s", 'Ġtest', 'Ġthis', 'Ġtoken', 'iz', 'er', '.']
# start, end = encoding.offsets[3] 
# sentence[start:end]
# 'this'
```



## Decoder

decoder的作用就是将ids转为text

`decoders.ByteLevel`:  将字节序列转为原始的utf-8文本

```python
sentence = "Let's test this tokenizer."
encoding = tokenizer.encode(sentence)
print(encoding.tokens) # Let's test this tokenizer.
tokenizer.decoder = decoders.ByteLevel()
decoding = tokenizer.decode( encoding.ids)
print(decoding)# Let's test this tokenizer.
```



## 封装到PreTrainedTokenizerFast类中

要在transformers中使用tokenizer，只要封装到`PreTrainedTokenizerFast`中即可

```python
tokenizer = PreTrainedTokenizerFast(tokenizer_object = tokenizer)
output = tokenizer.tokenize(sentence)
tokenizer.save_pretrained("./my_tokenizer")
```

使用`PreTrainedTokenizerFast`加载

```python
tokenizer =PreTrainedTokenizerFast.from_pretrained("./my_tokenizer")
print(tokenizer.tokenize(sentence))
```

# 有关tokenization的一些问题

1. 为什么大模型对于一些简单的任务做的不好,例如一些拼写问题（star）、简单的算术 、 将字符串反转
   - 例如strawberry有多少个r?我们使用[Tiktokenizer](https://tiktokenizer.vercel.app/?model=gpt-4)可视化strawberry的分词结果可以看出，strawberry这个单词被拆分成了三部分，即3个token，而不是一个token
   <img src="https://img.leftover.cn/img-md/202507221805350.png" alt="Snipaste_2025-07-22_18-05-06"  />
   - 例如让chatgpt将`.DefaultCellStyle`反转，直接让他进行反转就会得到错误答案，但如果我们先让它使用空格将每个字符分开，再让他进行反转操作就可以答对

> 这些问题其实并不是大语言模型本身的限制导致的，而是tokenizer

2. 为什么不使用unicode编码作为vocabulary
   1. unicode编码虽然可以表示所有的字符，但是他太大的，有15w个单词，这会导致训练的Embedding层很大，并且在最后面进行softmax的时候计算量很大
   2. 并且unicode编码在不断扩大，因为他不是固定不变的，如果使用Unicode编码作为vocabulary的话，这会导致你需要频繁的该模型结构
   3. 如果使用Unicode编码作为vocabulary，这时候空格表示一个token，在面对编程语言等问题的时候，编程语言通常会包含大量的空格，这样的话就会导致序列长度很长，从而模型效果差。因为通常我们会将多个空格编码为一个token，这样就可以避免序列长度很差的问题

3. 为什么不使用UTF-16或者UTF-32编码，而使用UTF-8编码

   `UTF-32`是定长编码，使用4个字节来存储，如果使用UTF-32的来进行编码的话会产生大量的0，尤其是对于英文来说，浪费空间且BPE的合并效果也不好

   `UTF-16`使用两个字节/四个字节来存储，同样相对于UTF-8编码来说，会产生大量的0，浪费空间并且BPE的合并效果不好

   `UTF-8`是变成编码，并且兼容ASCII码，因此使用UTF-8编码是最节省空间的并且效果也是最好的

4. 为什么GPT在一些小语种上的效果很差？例如老挝语、泰语
   - 从tokenizer角度回答：因为互联网上大多数是英文语料，而老挝语、泰语的语料很少，这会导致在使用BPE算法训练tokenizer的时候，只有少量的老挝语的token会被合并，因此LLM在进行回答的时候，如果是老挝语的语言，他的输入的token sequence就会更长，自然模型的效果不好。因为在训练tokenizer的时候很多是英文的语料，这就导致大量的英文相关的token被合并，所以如果输入的是英语，这时候输入就会token sequence就会更短，因此模型的效果很会更好
   - 从模型训练的角度回答：LLM在预训练的时候，是使用互联网的语料使用自回归的方式进行训练的，而老挝语等语言的语料少，自然对于老挝语的训练效果差，因此在使用老挝语等语言输入LLM的时候，回答效果就不好

5. 怎么设置vocab_size

   这通常是一个经验的超参数，一般在1w或者10w左右

6. 我怎样增加vocabulary的大小
   - 修改Embedding层，为新加的词汇初始化对应的向量，可以使用0初始化，也可以随机初始化，当然也有一些其他的更加高级的初始化方法
   - 其次，修改模型最后面的Projection_layer(投影层),修改对应的词表的大小

```python

class ProjectionLayer(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, vocab_size)
```
7. 为什么vocab_size不能设置为特别大？
   - 从上面可以看到，如果我们的vocab_size变大，那么投影层的参数量和Embedding层的参数量也会变大，导致模型需要更大的算力和显存
   - 其次，我们vocab_size很大的话，那么每个token在训练中出现的频率就会变低，这可能导致模型欠拟合，
   - 同时vocab_size很大的话就表明有很多小的token合并为了一个大的token，因此对于一个大的token，它包含的语义信息比较多，模型可能不能完全学到对应的语义信息

8. 为什么vocab_size不能设置为特别小？
   - vocab_size很小的话，会导致输入模型的时候序列长度很大，同时seq_len很长，导致其捕捉不到token之间的语义关系。
   - 同时vocab_size很小的话，会导致一句话里面有大量相同的token，尤其是输入是代码的时候，包含大量的空格等信息



# Reference
1. [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
2. [模块化构建 tokenizer](https://huggingface.co/learn/llm-course/zh-CN/chapter6/8?fw=pt)
3. [tokenizer的文档](https://huggingface.co/docs/tokenizers/components)


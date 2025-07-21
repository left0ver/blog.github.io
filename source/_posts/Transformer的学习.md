---
title: Transformer的学习
mathjax: true
date: 2025-07-21 21:10:22
tags:
  - AI
---
# 手写transformer

## Input_Embedding

通常我们首先会构建词表，然后每个词对应一个idx，即编号，然后通过embedding层转换为词向量

>需要注意的是：这里需要指定一个padding_idx,即padding字符的对应编号是多少，转换为词向量的时候，它的词向量为0

>论文3.4中提到，在embedding 要乘以$\sqrt{d_{model}}$

![](https://img.leftover.cn/img-md/202507151436699.png)

```python
class Input_Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

    def forward(self, x):
        # (batch_size,seq_len) -> (batch_size,seq_len,embedding_dim)
        return self.embedding(x) * math.sqrt(self.embedding_dim)
```

<!-- more -->
## PositionalEncoding

位置编码公式：
$$
PE_{(pos,2i)} = \sin (pos / 10000^{\frac{2i}{d_{model}}})
PE_{(pos,2i+1)} = \cos (pos / 10000^{\frac{2i}{d_{model}}})
$$
在self-attention中，在处理句子的时候，会计算句子中每个词和其他所有词之间的"相关性",但是这个过程是**顺序无关的**，但是在语言中词的顺序不同表达的含义也不同。例如

“我打了你” vs “你打了我”

因此如果模型无法感知词语的位置和顺序，就无法理解这些句子的根本区别。

而想LSTM 和 GRU之类的时序模型，是按顺序进行计算的，天生就带有位置的信息

而位置编码就是为**模型提供了词语在序列中位置的信息**

>在transformer中，位置编码是通过正余弦函数生成的，不参与模型的训练
>
>后面也出现了一些其他的方法，例如bert提出了可学习的位置编码，即将位置编码作为模型的一部分进行训练

```python
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # 位置编码不需要训练
        pe = torch.zeros(seq_len, embedding_dim)
        # pos 表示位置 (seq_len,)
        # i 表示维度 (embedding_dim,)
        # (seq_len,1)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # (embedding_dim/2,)
        _2i = torch.arange(0, embedding_dim, step=2, dtype=torch.float)
        # 偶数项
        # (seq_len.embedding_dim/2)
        pe[:, 0::2] = torch.sin(pos / 10000 ** (_2i / embedding_dim))
        # 奇数项
        # (seq_len.embedding_dim/2)
        pe[:, 1::2] = torch.cos(pos / 10000 ** (_2i / embedding_dim))

        pe = pe.unsqueeze(0)  # (1,seq_len,embedding_dim)
		# 位置编码不需要训练，但是它是模型状态的一部分，我们可以使用register_buffer来定义它
        self.register_buffer("pe", pe)

    def forward(self, x):
        # (batch_size,seq_len,embedding_dim)
        x = x + self.pe[:, : x.shape[1], :]

        return self.dropout(x)
```

## 多头注意力

- 这一步先将q，k，v分别通过一个全连接层进行线性变换得到query、key、value，之后将query、key、value分为多个头,
- 然后query和key进行注意力分数的计算

$$
attention(Q,K,V) =  softmax(\frac{QK^T}{\sqrt{d_k}})V
$$



>需要注意的是：在计算注意力的时候，如果我们不希望某个词和其他词产生联系，就需要将矩阵中两词对应的位置的值写为负无穷，在代码中，我们通过会给一个特别小的值，例如-1e-9，在进行softmax计算之后，矩阵中该位置的值几乎为0，也就屏蔽掉了这两个词的注意力
>
>mask==0的表示mask掉，不计算注意力

- 再将得到的注意力分数和 value 相乘，最后将结果变换回(batch,seq_len,embedding_dim)，再经过一个全连接层(w_o)输出

<img src="https://img.leftover.cn/img-md/202507151507242.png" alt="Snipaste_2025-07-15_15-05-44" style="zoom: 50%;" />

```python
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, n_head: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.embedding_dim = embedding_dim
        assert embedding_dim % n_head == 0, "embedding_dim 不能被n_head整除"
        self.d_k = embedding_dim // n_head
        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)
        self.w_o = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout=None):
        # 计算注意力分数
        # 即 d_k = embedding_dim// n_head
        d_k = query.shape[-1]
        # (batch,n_head,seq_len,d_k) -> transpose ->(batch,n_head,d_k,seq_len) -[matmul] -> (batch,n_head,seq_len,seq_len)
        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            # mask ==0,表示mask掉，这时候我们会给一个特别小的一个分数
            attention_scores = torch.masked_fill(attention_scores, mask == 0, -1e-9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            #  (batch,n_head,seq_len,seq_len) *  (batch,n_head,seq_len,d_k)  =  (batch,n_head,seq_len,d_k)
            # attention_scores 是为了可视化注意力分数
        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        # 把k,q,v分成多个头
        # (batch,seq_len,embedding_dim) -> (batch,seq_len,n_haed,d_k) -> (batch,n_head,seq_len,d_k)
        query = query.view(
            query.shape[0], query.shape[1], self.n_head, self.d_k
        ).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_head, self.d_k).transpose(
            1, 2
        )
        value = value.view(
            value.shape[0], value.shape[1], self.n_head, self.d_k
        ).transpose(1, 2)
        # 计算注意力分数并将其乘以vlaue
        # x: (batch,n_head,seq_len,d_k)
        # attention_scores:(batch,n_head,seq_len,seq_len)
        x, attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        # (batch,seq_len,embedding_dim)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_head * self.d_k)
        return self.w_o(x)

```

## LayerNorm

- **为什么要Normalization ？**

  Normalization就是把数据拉回标准正态分布，在神经网络中，我们经过计算，可能值会越来越大，我们通过Normalization把值拉回正态分布，以提高数值稳定性

- Normalization通常分为 batch_Normalization   和layer_Normalization,操作的维度不一样，但是其目的都是相同的，即把值拉回正态分布

- 在Layer_Normalization是对每个样本的所有特征做归一化，而batch_Normalization 则是对一个batch_size样本内的每个特征分别做归一化

> 需要注意的是: 
>
> - 当batch_size较小的时候，BN的效果就比较差,因为BN是对一个batch样本内的每个特征分别做归一化
> - 在RNN 和Transformer等时序问题上，通常会使用LN，这是因为在时序问题中，不同样本的长度通常不一样，而BN则需要对不同样本的同一位置特征进行归一化处理，虽然通常会padding到同一个长度，但是padding位置都是填充的0，没有意义。因此时序问题中通常会采样LN

```python
class Layer_Norm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-12):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        # （batch,seq_len,embedding_dim）->(batch,seq_len,1)
        mean = torch.mean(x, dim=-1, keepdim=True)
        #  （batch,seq_len,embedding_dim）->(batch,seq_len,1)
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        # (batch,seq_len,embedding_dim)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta
```

## FFN

简单来说，FFN由两个全连接层和一个relu激活函数组成
$$
FFN(x) = \max(0,xW_1+b_1)W_2 + b_2
$$


在FFN 中，会先进行一个升维做一个非线性变换，再降维

通过attention来解决一个序列中的长短程依赖问题，而FFN则是优化特征的权重，更好地提取特征，让不同特征进行相互地融合

```python
class FeedForwardBlock(nn.Module):
    def __init__(self, embedding_dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden)
        self.fc2 = nn.Linear(hidden, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
```



## Encoder_Block

将多头注意力，FFN 和 layer_norm整合在一起得到Encoder_Block

```python
class EncoderBlock(nn.Module):
    def __init__(
        self, embedding_dim: int, ffn_hidden: int, n_head: int, dropout: float = 0.1
    ):
        super().__init__()
        self.multi_head_attention = MultiHeadAttentionBlock(
            embedding_dim, n_head, dropout
        )
        self.layer_norm1 = Layer_Norm(embedding_dim)
        self.ffn = FeedForwardBlock(embedding_dim, ffn_hidden, dropout)
        self.layer_norm2 = Layer_Norm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        _x = x
        x = self.multi_head_attention(x, x, x, mask)
        x = self.dropout(x)
        # x+_x为残差连接
        x = self.layer_norm1(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.layer_norm2(x + _x)
        return x
```

## Encoder

将Embedding层 和 多个encoder_block组合得到Encoder

```python
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        embedding_dim: int,
        padding_idx: int,
        ffn_hidden: int,
        n_head: int,
        n_layer: int,
        dropout: float = 0.1,
    ):
        """
        n_layer :有多少个EncoderBlock
        """
        super().__init__()
        self.input_embedding = Input_Embedding(
            vocab_size=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx
        )
        self.pos_embedding = PositionalEncoding(embedding_dim, seq_len, dropout)
        self.encoder_blocks = nn.ModuleList(
            EncoderBlock(embedding_dim, ffn_hidden, n_head, dropout)
            for _ in range(n_layer)
        )

    def forward(self, x, mask=None):
        x = self.input_embedding(x)
        x = self.pos_embedding(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
        return x
```



## Decoder_Block

可以看到，在编写decoder_block的时候，第一层的注意力仍然是self-attention，而第二层则是cross-attention，Q是decoder第一层的输出，K、V则是encoder的输出

因为transformer最初是用于文本翻译任务的，`s_mask`则是源语言的mask，即encoder的mask，而`t_mask`则是目标语言的mask，即decoder的mask

<img src="https://img.leftover.cn/img-md/202507151630692.png" alt="Snipaste_2025-07-15_16-30-28" style="zoom:67%;" />

```python
class Decoder_Block(nn.Module):
    def __init__(
        self, embedding_dim: int, ffn_hidden: int, n_head: int, dropout: float = 0.1
    ):
        super().__init__()
        self.multi_head_attention1 = MultiHeadAttentionBlock(
            embedding_dim, n_head, dropout
        )
        self.layer_norm1 = Layer_Norm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attention = MultiHeadAttentionBlock(embedding_dim, n_head, dropout)
        self.layer_norm2 = Layer_Norm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = FeedForwardBlock(embedding_dim, ffn_hidden, dropout)
        self.layer_norm3 = Layer_Norm(embedding_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, t_mask, s_mask):
        """
        x : docoder的输入
        encoder_output:encoder 的输出
        t_mask: decoder的mask
        s_mask: encoder的mask
        """
        _x = x
        x = self.multi_head_attention1(x, x, x, t_mask)
        x = self.dropout1(x)
        x = self.layer_norm1(_x + x)
        _x = x
        #  第二层 Q 为decoder的输入，K，V是encoder的输出
        x = self.cross_attention(x, encoder_output, encoder_output, s_mask)
        x = self.dropout2(x)
        x = self.layer_norm2(_x + x)
        _x = x

        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.layer_norm3(x + _x)
        return x
```

## Decoder

将embedding 和 decoder_block组合在一起得到decoder

```python
class Decoder(nn.Module):
    def __init__(
        self,
        # 因为transformer最初是用于翻译任务，因此这里也会有一个decoder的vocab
        decoder_vocab_size: int,
        seq_len: int,
        embedding_dim: int,
        padding_idx: int,
        ffn_hidden: int,
        n_head: int,
        # decoder_layer的数量
        n_layer: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_embedding = Input_Embedding(
            vocab_size=decoder_vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.pos_embedding = PositionalEncoding(embedding_dim, seq_len, dropout)
        self.decoder_layers = nn.ModuleList(
            Decoder_Block(embedding_dim, ffn_hidden, n_head, dropout)
            for _ in range(n_layer)
        )
        self.proj = nn.Linear(embedding_dim, decoder_vocab_size)

    def forward(self, x, encoder_output, t_mask, s_mask):
        x = self.input_embedding(x)
        x = self.pos_embedding(x)
        for decoder_block in self.decoder_layers:
            x = decoder_block(x, encoder_output, t_mask, s_mask)
        return x

```

## Linear 和 Softmax

decoder输出维度为(batch,seq_len,embedding_dim)

我们要将得到的输出映射回词表，通过一个全连接层，将得到的输出映射到词表（因此也可以叫做Projection Layer投影层）

再做一个softmax则得到每个词的概率

```python
class ProjectionLayer(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # (batch,seq_len,embedding_dim) ->(batch,seq_len,vocab_size)
        x = self.proj(x)
        # 输出的概率
        # (batch,seq_len,vocab_size),即对于任意一个batch 和seq_len,它的每个单词的预测的概率加起来为1
        return torch.softmax(x, dim=-1)

```







# 训练

我们这里使用的是hugging face上的一个`opus_books`数据集,使用的是其中的英语和意大利语的数据集，将英语翻译为意大利语，使用datasets库来加载这个数据集

## 训练分词器

使用tokenizer这个库来训练分词器，分词方式有很多，例如`BPE`、 `Unigram`、`WordPiece`等子词的分词方式，这里我们使用最简单的分词方式，这里我们使用最简单的`Word Tokenizer`

```python
def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(config, ds_raw, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        # 设置预分词器，通过空格、标点等将词进行第一步的分割
        # WhitespaceSplit 则只会使用空格来分割
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            show_progress=True,
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
        )
        tokenizer.train_from_iterator(get_all_sentences(ds_raw, lang), trainer=trainer)
        # 保存下来，之后直接加载即可
        tokenizer.save(str(tokenizer_path))
    return tokenizer
```

## 构建数据集

1. 先下载数据集,并保存到`data/opus_books`中，获取/训练分词器

```python
   def get_ds(config):
       """
       加载数据集并创建分词器
       """
       # 在Hugging Face中下载opus_books数据集中的`en-it`子集，选择`train`部分，我们之后将自行再分出validation部分
       ds_raw = load_dataset(
           "opus_books",
           f'{config["lang_src"]}-{config["lang_tgt"]}',
           split="train",
           cache_dir="data/opus_books",
       )
       src_tokenizer = get_or_build_tokenizer(get_config(), ds_raw, config["lang_src"])
       target_tokenizer = get_or_build_tokenizer(get_config(), ds_raw, config["lang_tgt"])
   
       train_ds_size, val_ds_size = int(0.9 * len(ds_raw)), len(ds_raw) - int(
           0.9 * len(ds_raw)
       )
```

   2. 创建BilingualDataset

      这里最主要的就是`__getitem__`函数

```python
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from config import get_config
from utils import get_or_build_tokenizer
config = get_config()

class BilingualDataset(Dataset):
    def __init__(
        self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len
    ) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)   
```
我们统一使用了一个最大的seq_len=350，把每个batch的长度padding 到350，当然也可以进行动态padding，然后计算出每个样本要padding的数量

```python
   def __getitem__(self, index):
        item = self.ds[index]
        src_text = item["translation"][self.src_lang]
        tgt_text = item["translation"][self.tgt_lang]
        src_ids = self.tokenizer_src.encode(src_text).ids
        tgt_ids = self.tokenizer_tgt.encode(tgt_text).ids
        encdoer_num_padding_tokens = self.seq_len - len(src_ids) - 2
        decoder_num_padding_tokens = self.seq_len - len(tgt_ids) - 1
        assert (
            encdoer_num_padding_tokens >= 0 and decoder_num_padding_tokens >= 0
        ), "encoder 和 decoder的padding 数量应>=0"
```

添加`[SOS]`,`[EOS]`,`[PAD]`得到encoder_input

```python
  encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(src_ids, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * encdoer_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )
```

添加`[SOS]`,`[PAD]`得到decoder_input,没有`[EOS]`

```python
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(tgt_ids, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )
```

构建label，即期望的decoder的输出，没有`[SOS]`

```python
 label = torch.cat(
            [
                torch.tensor(tgt_ids, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )
```

构建decoder_mask 和 encoder_mask,得到所有的输入数据

在计算注意力的时候，（1）我们对pad是不需要计算注意力分数的（2）在decoder进行解码的时候，它只能看到它前面的词，这里我们创建了一个`generate_mask`函数来生成mask

```python
        assert len(encoder_input == self.seq_len)
        assert len(decoder_input == self.seq_len)
        assert len(label == self.seq_len)
        return {
            "encoder_input": encoder_input,
            # 进行多头注意力计算的时候，计算出的注意力分数的形状为(batch_size, num_heads, seq_len, seq_len)
            # （1，seq_len，seq_len）
            "encoder_mask": self.generate_mask(
                encoder_input == self.pad_token,
                encoder_input == self.pad_token,
                causal_mask=False,
            ),
            "decoder_input": decoder_input,
            # 这里要把padding的和未来的token都mask
          # （1，seq_len,seq_len）
            "decoder_mask": self.generate_mask(
                decoder_input == self.pad_token,
                decoder_input == self.pad_token,
                causal_mask=True,
            ),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
```

生成padding_mask,如果causal_mask=True,则还会创建causal_mask

```python
def generate_mask(
        self, q_pad: torch.Tensor, k_pad: torch.Tensor, causal_mask: bool = False
    ):
        # q_pad shape: [ q_len]
        # k_pad shape: [ k_len]
        # q_pad k_pad dtype: bool
        assert q_pad.device == k_pad.device
        q_len = q_pad.shape[0]
        k_len = k_pad.shape[0]

        mask_shape = (1, q_len, k_len)
        if causal_mask:
            mask = 1 - torch.tril(torch.ones(mask_shape))
        else:
            mask = torch.zeros(mask_shape)
        mask = mask.to(q_pad.device)
        mask[:, q_pad, :] = 1
        mask[:, :, k_pad] = 1
        # ==0 则被mask掉，1则不mask
        mask = (mask == 0).type(torch.int64)
        return mask
```



get_ds函数的全部代码：

1. 加载数据集并划分验证集和训练集
2. 构建分词器（这里我们是一个翻译任务，源语言和目标语言都要进行分词，因此要创建两个分词器）
3. 计算数据集的最大的seq_len，以便确定seq_len参数

```python
def get_ds(config):
    # 在Hugging Face中下载opus_books数据集中的`en-it`子集，选择`train`部分，我们之后将自行再分出validation部分
    ds_raw = load_dataset(
        "opus_books",
        f'{config["lang_src"]}-{config["lang_tgt"]}',
        split="train",
        cache_dir="data/opus_books",
    )
    src_tokenizer = get_or_build_tokenizer(get_config(), ds_raw, config["lang_src"])
    target_tokenizer = get_or_build_tokenizer(get_config(), ds_raw, config["lang_tgt"])

    train_ds_size, val_ds_size = int(0.9 * len(ds_raw)), len(ds_raw) - int(
        0.9 * len(ds_raw)
    )
    # target_tokenizer.token_to_id()
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    # 获取当前数据集中最大的序列长度，以便我们确定seq_len参数
    max_src_len = 0
    max_tgt_len = 0
    for item in ds_raw:
        src_ids = src_tokenizer.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = target_tokenizer.encode(item["translation"][config["lang_tgt"]]).ids
        max_src_len = max(len(src_ids), max_src_len)
        max_tgt_len = max(len(tgt_ids), max_tgt_len)

    print(f"src max_len：{max_src_len}")
    print(f"tgt max_len：{max_tgt_len}")

    train_dataset = BilingualDataset(
        train_ds_raw.dataset,
        src_tokenizer,
        target_tokenizer,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_dataset = BilingualDataset(
        val_ds_raw.dataset,
        src_tokenizer,
        target_tokenizer,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, src_tokenizer, target_tokenizer

```

## config

```python
from pathlib import Path
def get_config():
    return {
        "batch_size": 32,
        "num_epochs": 100,
      # 初始学习率
        "lr": 1e-5,
        "seq_len": 350,
        "d_model": 512,
        "n_head": 8,
        "ffn_hidden": 2048,
        "n_layer": 8,
        "dropout": 0.1,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights2",
        "model_basename": "tmodel_",
        # 预加载模型，如果不为空，则从预加载模型中加载参数（如在训练过程中程序意外中断），否则从头开始训练
        "preload": None,
        # 分词文件，如`_en.json`和`_it.json`
        "tokenizer_file": "tokenizer_{0}.json",
        #swanlab相关配置
        "experiment_name": "train_model",
        "swanlab_project": "translate_with_transformer",
        "swanlab_workspace": "leftover",
        # 是否开启swanlab记录
        "is_log":False,
    }


# 给定epoch，和模型的目录，可以获取到对应epoch的模型文件路径，可用于继续训练
def get_weights_file_path (config,epoch=str):
    model_folder = Path(config["model_folder"])
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return model_folder / model_filename

```

## 创建模型

```python
# get_model.py
from utils import get_or_build_tokenizer
from datasets import load_dataset
from model import Transformer


def get_model(config):
    ds_raw = load_dataset(
        "opus_books",
        f'{config["lang_src"]}-{config["lang_tgt"]}',
        split="train",
        cache_dir="data/opus_books",
    )
    src_tokenizer = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    target_tokenizer = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])
    src_pad_id = src_tokenizer.token_to_id("[PAD]")
    tgt_pad_id = target_tokenizer.token_to_id("[PAD]")
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = target_tokenizer.get_vocab_size()
    return Transformer(
        src_pad_id,
        tgt_pad_id,
        src_vocab_size,
        tgt_vocab_size,
        config["seq_len"],
        config["d_model"],
        config["n_head"],
        config["ffn_hidden"],
        config["n_layer"],
        config["dropout"],
    )

```



## 训练过程

加载模型、数据集、设置优化器

```python
from config import get_weights_file_path, get_config
from tqdm import tqdm
import torch
from torch import nn
from pathlib import Path
from get_datasets import get_ds
from get_model import get_model
from init_swanlab import init_swanlab
import swanlab
import warnings

def train_model(config):
    """
    训练模型
    """
    if config["is_log"]:
        init_swanlab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)


    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(
        config,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
```

如果指定了预训练权重，则加载预训练权重，继续训练

```python
 # 若指定了预加载权重，则加载预加载权重
    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
```

设置loss_fn 和 学习率调度器

需要注意的是，`[pad]`不参与loss的计算，因此需要设置`ignore_index`参数

```python
    # 定义loss函数，声明padding不参与loss计算。同时，使用label smoothing，让模型降低对计算结果的确定性，
    # 即减少本次推理结果的概率，并把减少的部分分配到其他可能的推理结果上。实测可以提升模型的泛化能力，降低过拟合。
    # 这里的label smoothing的系数设置为0.1，即将最高概率标签的概率降低0.05，再并分配给其他标签的概率。
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.05
    ).to(device)
    # 使用余弦调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["num_epochs"] * len(train_dataloader),
        eta_min=0.1 * config["lr"],
    )
```

训练循环

```python
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)  # (batch, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (batch, seq_len)
             # (batch, 1, seq_len, seq_len) 只隐藏padding tokens
            encoder_mask = batch["encoder_mask"].to(
                device
            )
             # (batch, 1, seq_len, seq_len) 隐藏padding tokens和未来的tokens 
            decoder_mask = batch["decoder_mask"].to(
                device
            ) 
            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (batch, seq_len, d_model)
            decoder_output = model.decode(
                decoder_input,
                encoder_output,
                decoder_mask,
                encoder_mask,
            )  # (batch, seq_len, d_model)
            proj_output = model.project(
                decoder_output
            )  # (batch, seq_len, tgt_vocab_size)

            label = batch["label"].to(device)  # (batch, seq_len)

            # (batch, seq_len, tgt_vocab_size) -[view]-> (batch * seq_len, tgt_vocab_size)
            # (batch, seq_len) -[view]-> (batch * seq_len)
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            if config["is_log"]:
                swanlab.log({"train loss": loss.item(),"lr": scheduler.get_lr()[0]}, step=global_step)
            loss.backward()
            # 使用梯度裁剪，保证训练的稳定
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            global_step += 1
        # 保存模型    
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )
    if config["is_log"]:
        swanlab.finish()
```





# 推理



根据贪心策略，不断生成词，（1）如果输出的词的数量超过了max_len，则不再继续生成.（2）如果当前生成的词是`[EOS]`,将词加到输出的decoder输出的后面，结束，不再继续生成（3）否则，将当前生成的词加入到末尾，继续生成

```python
def causal_mask(size):
    # 生成一个下三角矩阵：先生成单位矩阵，然后把上三角部分置为0，最后转为int
    # 0 表示mask
    return torch.tril(
        torch.ones(1, size, size, dtype=torch.int64),
    )

def greedy_decode(
    model, encoder_input, src_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    # batch,seq_len,embedding_dim
    encoder_output = model.encode(encoder_input, src_mask)
    # 这里batch_size =1
    # 填充`[SOS]`表示开始
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        # 创建causal_mask
        # 这里和train的时候不一样，因为这是推理，只需要创建causal_mask，并不需要额外的padding
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)
        # 使用decode来生成词，这里可以复用encoder_output
        out = model.decode(decoder_input, encoder_output, decoder_mask, src_mask)
        # 输出最后一个词的下一个词的概率分布.(batch,decoder_vocab_size)
        prob = model.project(out[:,-1])
        # (batch,)
        next_word = torch.argmax(prob, dim=1)
        # 将生成的词添加到decoder_input的末尾，再继续生成下一个词
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(1)], dim=1)
        # 如果生成的词是eos，则停止
        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)
```

1. 先从验证集中取出数据，这里验证集的batch_size =1 
2. 使用贪婪策略，每次取概率最大的作为下一个词（贪心策略）（这里还有其他的方法，例如beam search），transformer论文中就是使用的beam search，因为当前概率最大的词并不一定是全局最优的

```python
def run_validation(model, val_datalaoder, config, tokenizer_src, tokenizer_tgt, device):
    model.eval()
    source_texts = []
    target_texts = []
    predict_texts = []

    with torch.no_grad():
        batch_iterator = tqdm(
            val_datalaoder,
            desc="Validation",
        )
        for batch in batch_iterator:
            # batch,seq_len
            encoder_input = batch["encoder_input"].to(device)
            # batch,1,seq_len,seq_len
            encoder_mask = batch["encoder_mask"].to(device)
            src_text = batch["src_text"]
            tgt_text = batch["tgt_text"]
            # 得到生成的词
            decoder_output = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                config["seq_len"],
                device,
            )
            # 使用tokenizer_tgt进行解码
            predict_text = tokenizer_tgt.decode(decoder_output.detach().cpu().numpy())
            batch_iterator.write("-" * 80)
            batch_iterator.write(f"src_text: {src_text}")
            batch_iterator.write(f"tgt_text: {tgt_text}")
            batch_iterator.write(f"predict_text: {predict_text}")
            batch_iterator.write("-" * 80)
            source_texts.extend(src_text)
            target_texts.extend(tgt_text)
            predict_texts.extend(predict_text)
```



# Reference

本博客大量参考了下面文章和视频的内容，主要在于理解transformer架构、自回归模型的训练和推理

1. [直接带大家把Transformer手搓一遍，这次总能学会Transformer了吧！](https://www.bilibili.com/video/BV1nXjEzmEWC/?spm_id_from=333.788.videopod.episodes&vd_source=3c93d521158d3aa4f74c71c5140ba8dc&p=6)
2. [理解Transformer模型1：编写Transformer](https://www.qinzishi.tech/2024/01/06/transformer-from-scratch-1/)
3. [理解Transformer模型2：训练Transformer](https://www.qinzishi.tech/2024/01/13/transformer-from-scratch-2/)
4. [PyTorch Transformer 英中翻译超详细教程](https://zhouyifan.net/2023/06/11/20221106-transformer-pytorch/)

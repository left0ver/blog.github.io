---
title: build bert from scratch
mathjax: true
date: 2025-08-18 17:28:26
tags:
  - AI
---

# 读BERT的论文

1. bert基于`MLM` 和 "next sentence prediction" 任务进行预训练

2. 因为注意力机制的计算是双向的，跟RNN不同，RNN是单向的，因此bert是双向地提取特征

3. bert使用的是`wordPiece`的分词方式，vocab_size = 30000, 每个序列的第一个token是[CLS],然后[SEP]分隔句子

4. `[CLS]` 最终的隐藏层状态会被用来分类任务(也叫做CLS池化)，如果是token级别的任务的话，选取对应token的输出即可，然后加一些全连接层做token的分类

<!-- more -->

5. 两个任务：

   Masked LM:
   - 在训练的时候以一定的概率（15%）mask掉一些token，然后模型最后的任务就是预测这些token。在最后，被mask掉的token的隐藏层向量会输入到最终的softmax中
   - 由于预训练的时候会随机mask掉一些词，但是微调的时候其实是不需要的，这会导致微调和预训练不太一样。因此对于这要15%的mask掉的词来说，80%的概率被替换成`[MASK]`,10%的概率被替换为一个随机的token（这个情况的话其实是加入了一些噪音），10%的概率什么都不做，即把它放在那里（这里可能就是为了向微调任务靠拢一点，这样子的话在微调上效果会更好）。 但是这15%的词都会用来进行预测
  
   Next Sentence Prediction(NSP):
   - 在语料库中随机选择句子A，B。50%的情况下B是A的下一句，50%情况B是从语料库中随机选取的句子。然后训练模型让模型进行预测，以此来提高模型对两个句子之间的关系的理解
   >来提升在类似QA等任务上的效果

6. 输入:

   两个句子的输入（NSP）

```python
# 使用[SEP]分割句子
[CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]
```

   一个句子的输入（MLM任务）:

```python
my dog is hairy → my dog is [MASK] # 80%
my dog is hairy → my dog is apple # 10%替换为随机一个token
my dog is hairy → my dog is hairy # 10%什么都不变
```

7. bert 中的 `Segment_Embeddings` 是什么？

   bert中有两个任务，MLM和NSP，因此输入的句子可能是一个句子和两个句子。为了区分是第一个句子还是第二个句子，bert采用两种方式，（1）会使用`[SEP]`来分割句子.（2）bert中添加了一个segment_embedding层，这是一个可学习的embedding层。但它是vocab_size =2

```python
#transformers中的实现 src/transformers/models/bert/modeling_bert.py
self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
      # token_type_embeddings 就是论文中的segment_embedding
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
```

   <img src="https://img.leftover.cn/img-md/202507241437512.png" alt="Snipaste_2025-07-24_14-37-42" style="zoom:50%;" />



# build bert from scratch

bert和transformer一样，我们首先构建Embedding层，bert包含2个Embedding层，一个位置编码。

首先第一个Embedding层将ids转为word vector，

第二个Embedding 层叫`Segment_Embedding`,因为bert训练的时候有两个任务，MLM和NSP，NSP任务需要构建两个句子。如上面所讲，使用`[SEP]`分割，因此为了区别第一个句子和第二个句子，则构建一个Segment_Embedding，这是一个vocab_size=2的Embedding层

最后则是和transformer中一样的位置编码

```python
class Segment_Embedding(nn.Module):
    def __init__(self, token_type_vocab_size, embedding_dim, padding_idx):
        super(Segment_Embedding, self).__init__()
        self.embedding = nn.Embedding(
            token_type_vocab_size, embedding_dim, padding_idx=padding_idx
        )

    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, seq_len: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
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

        self.register_buffer("pe", pe)

    def forward(self, x):
        # (batch_size,seq_len,embedding_dim)
        x = x + self.pe[:, : x.shape[1], :]

        return x


class Bert_Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_len, padding_idx, dropout=0.1):
        super(Bert_Embedding, self).__init__()
        self.input_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.segment_embedding = Segment_Embedding(3, embedding_dim,padding_idx)
        self.positional_encoding = PositionalEncoding(embedding_dim, seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, segment_label):
        x = self.input_embedding(sequence) + self.segment_embedding(segment_label)
        x = self.positional_encoding(x)
        return self.dropout(x)
```

和transformer不同的是，bert是Encoder-Only架构，而transformer是Encoder-Decoder架构，因此bert中并没有Decoder,因此这里我们只需要像transformer中一样构EncoderLayer即可，由多个EncoderLayer组成Encoder

```python
class FeedForward(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, hidden_dim=2048, dropout=0.1):
        super(FeedForward, self).__init__()

        self.fc1 = torch.nn.Linear(d_model, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, heads, feed_forward_hidden=768 * 4, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(
            embedding_dim, heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = FeedForward(
            embedding_dim, hidden_dim=feed_forward_hidden, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        _x = x
        x = self.dropout1(self.self_attn(x, x, x, mask)[0])
        x = self.norm1(x + _x)

        _x = x

        x = self.dropout2(self.feed_forward(x))
        x = self.norm2(x + _x)
        return x
```

将Embedding 和 Encoder组合起来，我们就构建了一个bert，sequence经过bert会输出结果

```python
class BERT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        seq_len,
        padding_idx,
        feed_forward_hidden,
        num_layers=12,
        heads=12,
        dropout=0.1,
    ):
        super(BERT, self).__init__()
        self.embedding = Bert_Embedding(
            vocab_size, embedding_dim, seq_len, padding_idx, dropout
        )
        self.padding_idx = padding_idx
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(embedding_dim, heads, feed_forward_hidden, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, sequence, segment_label):
        # batch_size, seq_len - > batch_size,1 , seq_len  [repeat] -> batch_size, seq_len, seq_len [unsqueeze] -> batch_size, 1, seq_len, seq_len
        # batch,1,seq_len,seq_len
        mask = sequence == self.padding_idx
        # mask = (
        #     (sequence != self.padding_idx)
        #     .unsqueeze(1)
        #     .repeat(1, sequence.shape[1], 1)
        #     .unsqueeze(1)
        # )
        x = self.embedding(sequence, segment_label)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x
```

因为bert有两个任务，一个是MLM，一个是NSP，因此我们需要构建两个输出头，对于NSP任务（分类任务），我们取`[CLS]`对应的向量代表整个序列的特征（即CLS池化）；

对应MLM任务，我们则需要对每一个位置的token都预测一个结果（理论上来说，我们只需要预测`[MASK]`上的token即可，因此也可以只取`[MASK]`位置上的向量），这里我们制作label的时候对应的位置如果没有被mask掉，则填充`[PAD]`,计算loss的时候[PAD]不会参与计算loss，因此也可以达到上面一样的效果

>如果是在推理的时候，我们是一个MLM任务的话，则只需要取对于`[MASk]`位置的向量进入到分类头中计算即可

```python
class NextSentencePrediction(nn.Module):
    def __init__(self, embedding_dim):
        super(NextSentencePrediction, self).__init__()
        self.linear = nn.Linear(embedding_dim, 2)

    def forward(self, x):
        x = self.linear(x)
        return x


class MaskedLanguageModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(MaskedLanguageModel, self).__init__()
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.linear(x)
        return x


class BERTLM(nn.Module):
    def __init__(self, bert: BERT, vocab_size, embedding_dim):
        super(BERTLM, self).__init__()
        self.bert = bert
        self.next_sentence_prediction = NextSentencePrediction(embedding_dim)
        self.masked_language_model = MaskedLanguageModel(embedding_dim, vocab_size)

    def forward(self, sequence, segment_label):
        x = self.bert(sequence, segment_label)
        # 取出 [CLS] 的输出作为 next_sentence_prediction 的输入
        # batch,embedding_dim
        next_sentence_input = x[:, 0, :]
        next_sentence_output = self.next_sentence_prediction(next_sentence_input)

        masked_language_model_output = self.masked_language_model(x)

        return next_sentence_output, masked_language_model_output
```

```python
def build_model(
    vocab_size,
    embedding_dim,
    seq_len,
    padding_idx,
    feed_forward_hidden,
    num_layers=12,
    heads=12,
    dropout=0.1,
):
    bert = BERT(
        vocab_size,
        embedding_dim,
        seq_len,
        padding_idx,
        feed_forward_hidden,
        num_layers,
        heads,
        dropout,
    )
    model = BERTLM(bert, vocab_size, embedding_dim)
    return model

```



下载[dataset](http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip)

```python
wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
unzip -qq cornell_movie_dialogs_corpus.zip
rm cornell_movie_dialogs_corpus.zip
```

处理数据集，将句子保存到`data/text.txt`，方便之后训练tokenizer

```python
import os
from pathlib import Path
import tokenizers
import torch
import re
import random
import transformers, datasets
from tokenizers import BertWordPieceTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer
import tqdm
from torch.utils.data import Dataset, DataLoader
import itertools
import math
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from tqdm import tqdm

corpus_movie_lines = Path("data/cornell movie-dialogs corpus/movie_lines.txt")
corpus_movie_conversations = Path(
    "data/cornell movie-dialogs corpus/movie_conversations.txt"
)

sep = " +++$+++ "
lines_dic = {}

def get_data(is_save: bool = True):
    """
    只有第一次才保存为text.txt，以便训练tokenizer
    """
    with open(corpus_movie_lines, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        for line in lines:
            line_obj = line.split(sep)
            lines_dic[line_obj[0]] = line_obj[-1].removesuffix("\n")

    with open(corpus_movie_conversations, "r", encoding="utf-8", errors="ignore") as f:
        conversations = f.readlines()
    paris = []

    for conversation in conversations:
        conv_ids = eval(conversation.split(sep)[-1])
        for i in range(len(conv_ids) - 1):
            first = lines_dic[conv_ids[i]].strip()
            second = lines_dic[conv_ids[i + 1]].strip()
            paris.append((first, second))

    if is_save:
        text_data = []
        for id, text in lines_dic.items():
            text_data.append(text)

        with open("data/text.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(text_data))

    return paris

```

根据`data/text.txt`训练自己的tokenizer

```python
from tokenizers.implementations import BertWordPieceTokenizer
from transformers import BertTokenizerFast
import os

def get_tokenizer() -> BertTokenizerFast:

    text_path = "data/text.txt"
    vocab_path = "vocab.txt"
    tokenizer_dir = "bert_tokenizer"
    if os.path.exists(tokenizer_dir):
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir)
    else:
        # 训练tokenizer
        # limit_alphabet 限制初始vocab的大小
        tokenizer = BertWordPieceTokenizer()
        tokenizer.train(
            text_path, vocab_size=30000, min_frequency=2, limit_alphabet=1000
        )
        tokenizer.save_model("./")
        tokenizer = BertTokenizerFast(vocab_file = vocab_path)
        tokenizer.save_pretrained(tokenizer_dir)

    print(f"vocab_size = {tokenizer.vocab_size}")
    return tokenizer


get_tokenizer()
```

构建dataset

1. 通过`get_sentence`函数取两个句子，50%的概率是上下句，50%不是上下句
2. 取出了两个句子之后，我们通过`random_word`函数，遍历句子中的所有token，选出需要mask的15%的token，对于这15%需要被mask的token，80%的概率被替换为`[MASK]`,10%的概率被替换为vacab中的随机的一个词,10%的概率什么都不做，最后得到mask之后的token_ids和label
3. 对上个句子都通过`random_word`函数处理完了之后，最后对句子添加`[CLS]` ,`[SEP]`,最后填充`[PAD]`到一样的长度

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from bert.build_tokenizer import get_tokenizer
from bert.preprocess import get_data
import random



class BERTDataset(Dataset):
    def __init__(
        self, tokenizer: BertTokenizerFast, data_pair: list, max_length: int = 512
    ):
        super(BERTDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_pair = data_pair
        self.max_length = max_length

    def __len__(self):
        return len(self.data_pair)

    def get_random_line(self):
        idx = random.randint(0, len(self.data_pair) - 1)
        return self.data_pair[idx][1]

    def get_sentence(self, idx):
        """
        50%概率是上下句，50%概率是不是
        """
        s1, s2 = self.data_pair[idx]
        prob = random.random()
        if prob > 0.5:
            return s1, s2, 1
        else:
            return s1, self.get_random_line(), 0

    def random_word(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        token_ids = self.tokenizer(sentence)["input_ids"][1:-1]
        assert len(tokens) == len(token_ids), "tokens and token_id must be same length"
        output_ids = []
        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    output_ids.append(self.tokenizer.convert_tokens_to_ids("[MASK]"))
                elif prob < 0.9:
                    output_ids.append(
                        self.tokenizer.convert_tokens_to_ids(
                            random.choice(list(self.tokenizer.vocab.keys()))
                        )
                    )
                else:
                    # TODO:
                    output_ids.append(token_ids[i])
                output_label.append(token_ids[i])
            else:
                output_ids.append(token_ids[i])
                output_label.append(self.tokenizer.convert_tokens_to_ids("[PAD]"))
        return output_ids, output_label

    def __getitem__(self, idx):
        s1, s2, is_next_label = self.get_sentence(idx)
        s1_random, s1_label = self.random_word(s1)
        s2_random, s2_label = self.random_word(s2)
        sequence = (
            [self.tokenizer.vocab["[CLS]"]]
            + s1_random
            + [self.tokenizer.vocab["[SEP]"]]
            + s2_random
            + [self.tokenizer.vocab["[SEP]"]]
        )
        label = (
            [self.tokenizer.vocab["[PAD]"]]
            + s1_label
            + [self.tokenizer.vocab["[PAD]"]]
            + s2_label
            + [self.tokenizer.vocab["[PAD]"]]
        )
        segment_label = [1] * (len(s1_random) + 2) + [2] * (len(s2_random) + 1)
        assert (
            len(sequence) <= self.max_length
        ), f"sequence length must be less than {self.max_length}, but got {len(sequence)}"
        assert len(segment_label) == len(sequence) and len(label) == len(
            sequence
        ), f"segment_label and sequence and label must be same length, segment_label: {len(segment_label)}, sequence: {len(sequence)}, label: {len(label)}"

        num_of_padding = self.max_length - len(sequence)
        padding = [self.tokenizer.vocab["[PAD]"]] * num_of_padding
        sequence.extend(padding)
        label.extend(padding)
        segment_label.extend(padding)
        inputs = {
            "input_ids": torch.tensor(sequence, dtype=torch.long),
            "segment_label": torch.tensor(segment_label, dtype=torch.long),
            "is_next_label": torch.tensor(is_next_label, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }
        return inputs


if __name__ == "__main__":
    data_pair = get_data(is_save=False)
    tokenizer = get_tokenizer()
    bert_dataset = BERTDataset(tokenizer, data_pair)

    train_loader = DataLoader(bert_dataset, batch_size=2)

    batch = next(iter(train_loader))
    print(batch)

```

训练代码

```python
import torch
from torch import  nn
from tqdm import tqdm


class BERTTrainer:
    def __init__(
        self,
        model,
        train_loader,
        num_epochs,
        lr,
        padding_idx,
        log_step,
        device,
        test_loader=None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.device = device
        self.test_loader = test_loader
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
        self.nsp_criterion = nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, len(train_loader) * num_epochs, eta_min=0.1 * lr
        )
        self.num_epochs = num_epochs
        self.log_step = log_step

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.iteration(epoch, self.train_loader, is_train=True)

    def iteration(self, cur_epoch, data_loader, is_train):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        mode = "train" if is_train else "test"
        data_iter = tqdm(data_loader, desc=f"{mode}:{cur_epoch}")

        for i, batch in enumerate(data_iter, start=1):
            input_ids = batch["input_ids"].to(self.device)
            segment_label = batch["segment_label"].to(self.device)
            label = batch["label"].to(self.device)
            is_next_label = batch["is_next_label"].to(self.device)
            next_sent_output, mask_lm_output = self.model(input_ids, segment_label)
            next_loss = self.nsp_criterion(next_sent_output, is_next_label)

            mask_lm_loss = self.mlm_criterion(
                mask_lm_output.view(-1, mask_lm_output.size(-1)), label.view(-1)
            )

            loss = next_loss + mask_lm_loss
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
            correct = (next_sent_output.argmax(dim=-1) == is_next_label).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += is_next_label.numel()
            post_fix = {
                "epoch": cur_epoch,
                "iter": i,
                "avg_loss": avg_loss / i,
                "avg_acc": total_correct / total_element,
                "cur_acc": correct / is_next_label.numel(),
                "cur_loss": loss.item(),
            }
            if i % self.log_step == 0:
                data_iter.set_postfix(post_fix)
        print(
            f"epoch:{cur_epoch},mode:{mode},avg_loss:{avg_loss / len(data_loader)}, avg_acc:{total_correct / total_element}",flush=True
        )


if __name__ == "__main__":
    from bert.dataset import get_data, get_tokenizer, BERTDataset
    from torch.utils.data import DataLoader
    from bert.model import build_model

    data_pair = get_data(is_save=False)
    tokenizer = get_tokenizer()
    batch_size = 16
    seq_len = 512
    num_layers = 12
    ffn_hidden = 768 * 4
    n_head = 12
    embedding_dim = 768
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    num_epochs = 10
    log_step = 5
    bert_dataset = BERTDataset(tokenizer, data_pair,seq_len)

    train_loader = DataLoader(bert_dataset, batch_size=batch_size)
    init_lr = 1e-5
    model = build_model(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        seq_len=seq_len,
        padding_idx=tokenizer.vocab["[PAD]"],
        feed_forward_hidden=ffn_hidden,
        num_layers=num_layers,
        heads=n_head,
        dropout=0.1,
    )
    trainer = BERTTrainer(
        model,
        train_loader,
        num_epochs=num_epochs,
        lr=init_lr,
        padding_idx=tokenizer.vocab["[PAD]"],
        log_step=log_step,
        device=device,
    )
    trainer.train()

```

# Reference

1. [Mastering BERT Model: Building it from Scratch with Pytorch](https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891)
2. [BERT 论文逐段精读](https://www.bilibili.com/video/BV1PL411M7eQ/?spm_id_from=333.337.search-card.all.click)
3. [bert论文](https://arxiv.org/pdf/1810.04805)

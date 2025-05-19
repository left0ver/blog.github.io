---
title: NLP情感分类
mathjax: true
date: 2025-05-17 20:20:31
tags:
  - AI
---



# 使用传统的LSTM

1. 用的飞桨上的中文情感分类的[数据集](https://aistudio.baidu.com/datasetdetail/221537)

2. 进行分词，我使用了jieba对中文分词，构建词表

   >这里如果语料库比较小，可以使用jieba 的`全词模式`进行分词，这样可以得到更多的词，减少OOV的概率

   - 一般是先进行分词，再统计每个词的频率，若语料库比较大，可以根据情况过滤掉低频词（也可以不过滤）
   - 最后得到词表和stoi（词到索引的映射）、itos（索引到词的映射）
   - 通常我们会加入一些特殊的词，例如`<unk>` 表示一个词表不存在的词、`<pad> `在训练的时候进行填充来保证输入的向量维度相同
   
<!-- more -->

 ```python
   class Vocab:
       def __init__(self, train_csv_path):
           df = pd.read_csv(train_csv_path, sep="\t", encoding="utf-8")
           df.rename(columns={"text_a": "text"}, inplace=True)
   
           self.labels = df["label"].tolist()
           texts = df["text"].tolist()
   
           # 进行分词
           self.tokenized_texts = []
           for text in texts:
               self.tokenized_texts.append(jieba.lcut(text))
           special_tokens = ["<unk>", "<pad>"]
           # 这里没有限制词的频率
           all_word = special_tokens + [word for text in self.tokenized_texts for word in text] 
           self.vocab = Counter(all_word)
           self.__itos = self.vocab.keys()
           
           self.__stoi = {word: idx for idx, word in enumerate(self.__itos)}
   
           self.UNK_IDX = self.__stoi["<unk>"]  # unknown index
           self.PAD_IDX = self.__stoi["<pad>"]  # padding index
           
       def stoi(self, word):
           if word in self.__itos:
               return self.__stoi[word]
           return self.UNK_IDX
       def itos(self, idx):
           if idx <0 or idx >= len(self.__itos):
               raise IndexError("Index out of range")
           return self.__itos[idx]
       
       def __len__(self):
           return len(self.vocab)
  ```

   >需要注意的是我们只能使用训练集中的数据来构建词表

3. 将词转为词向量（这个也有很多方法，例如词频统计、TF-IDF，word2vec，BGE）
   这里我使用了gensim来训练自己的词向量，每个词使用100维的向量表示

```python
from gensim.models import Word2Vec
import jieba
import pandas as pd


train_tsv = "./data/train.tsv"

df = pd.read_csv(train_tsv, sep="\t", encoding="utf-8")
df.rename(columns={"text_a": "text"}, inplace=True)
labels = df["label"].tolist()
texts = df["text"].tolist()

# 进行分词
tokenized_train_texts = []
for text in texts:
    tokenized_train_texts.append(jieba.lcut(text))

model = Word2Vec(
    sentences=tokenized_train_texts,
    vector_size=100,       # 词向量维度
    window=5,              # 上下文窗口大小
    min_count=1,           # 最小词频阈值 (这里为了小数据集设为1)
    sg=1,                  # 使用 Skip-gram 算法
    hs=0,                  # 不使用 Hierarchical Softmax
    negative=5,            # 使用 Negative Sampling，负采样数量为5
    workers=6,             # 使用4个CPU核心
    epochs=50,             # 迭代50次 (对于小数据集，可以多迭代几次)
    seed=42
)

print("Word2Vec 模型训练完成！")
model.save("my_custom_word2vec-100.model")
print("\n模型已保存到 'my_custom_word2vec-100.model'")

# print(model.get_latest_training_loss)
print(model.wv.vectors.shape)

```

上面我们通过gensim训练了自己的词向量，得到了词向量矩阵，现在我们给定一个词，可以通过词向量矩阵得到这个词的词向量，当然word2vec有一个比较明显的缺点，就是存在OOV的词，这里可以使用fasttext等一些更先进的方法。



4. nn.Embedding： nn.Embedding是一个词向量的查找表，给定一个词的idx，就可以得到这个词的词向量，和上面的word2vec得到的矩阵类似，但是不同的是nn.Embedding可以作为我们模型的一层，跟随模型一起训练，并且可以反向传播更新词向量的参数

   >这么一来的话，上述的word2vec的工作岂不是白做了，也不一定，`我们上面通过word2vec得到了词向量，然后我们可以使用得到的词向量的权重参数来初始化nn.Embedding`。这样可以加快收敛

下面三个图是分别使用3种不同的方法得到的训练阶段的情感分类的准确率随迭代次数的图。 1. 随机初始化nn.Embedding 2. 使用word2vec得到的词向量来初始化nn.Embedding 3. 使用word2vec得到的词向量来初始化nn.Embedding 并且不对nn.Embedding进行参数更新

>可以看出第二种方法的收敛速度和准确率都是最高的，而第三种的准确率最差，从这里可以看出使用nn.Embedding相比于只是用word2vec，模型最后的效果提升还是比较明显

<img src="https://img.leftover.cn/img-md/202505172055122.png" alt="image-20250517205525004" style="zoom:33%;" />

<img src="https://img.leftover.cn/img-md/202505172055652.png" alt="image-20250517205536619" style="zoom: 33%;" />

<img src="https://img.leftover.cn/img-md/202505172056158.png" alt="image-20250517205604094" style="zoom: 33%;" />

5. 构建数据集

   由于我们使用了nn.Embedding,因此这里我们`__getitem__`函数只要返回每个样本中的词对应的idx即可
   由于我们是将好几个样本打包成一个batch进行训练，但是样本中text的长度是不一样的，因此我们通常会将该batch中长度对齐最大的那个样本，将长度不足的样本进行padding

```python

train_tsv = "./data/train.tsv"
eval_tsv = "./data/dev.tsv"
vocab =Vocab(train_tsv)
class MyDataset(Dataset):
    def __init__(self, file_tsv,is_train=True):
        self.vocab =Vocab(train_tsv)
        if is_train:
            df = pd.read_csv(file_tsv, sep="\t", encoding="utf-8")
        else:
          # 这里测试集和验证机中多了一个qid的列，这里把它删除
            df = pd.read_csv(file_tsv, sep="\t", encoding="utf-8",index_col=0)    
        df.rename(columns={"text_a": "text"}, inplace=True)
        self.labels = df["label"].tolist()
        self.texts = df["text"].tolist()
        self.tokenized_texts = [jieba.lcut(text) for text in self.texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        label = self.labels[idx]
        tokenized_text = self.tokenized_texts[idx]
        tokenized_text = [self.vocab.stoi(word) for word in tokenized_text]
        return torch.tensor(tokenized_text,dtype=torch.int32), torch.tensor(label)

# 对每个batch进行padding，padding到当前batch的最大长度
def padding_collate_fn(batch):
    input_ids, labels = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=vocab.PAD_IDX)
    labels = torch.tensor(labels)
    return input_ids,labels
```

6. 构建神经网络
   - 这里我们使用上面训练得到的词向量来初始化 nn.Embedding
   - 使用了双向lstm，隐藏层的大小为100，num_layers =2
   - 一开始我使用的是单向的lstm、隐藏层的大小为64，num_laryers =1 ,但是训练的时候loss一直不下降，好像加大隐藏层的大小也没用，后面看了李沐老师的动手学深度学习的[情感分析一章](https://zh.d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-rnn.html), 修改了模型结构，训练的效果有了明显的提升。现在看来应该是当时的模型太简单，欠拟合了。
    
    像这种文本分类、文本翻译的任务，很适合使用双向的RNN，可以得到更多的语义特征

```python
  word2vec_model_path ="gemsim_train_word2vec-100.model"
  EMBEDDING_DIM = 100
  
  
  # 使用word2vec的词向量来初始化nn.Embedding
  weights_matrix = np.zeros((len(vocab), EMBEDDING_DIM))
  word_not_found = []
  weights_matrix[0] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))
  # padding的词向量为0并且不进行更新
  weights_matrix[1] = np.zeros((EMBEDDING_DIM,))
  word2vec_model = Word2Vec.load(word2vec_model_path)
  weights_matrix[2:] = word2vec_model.wv.vectors
  
  class Net(nn.Module):
      def __init__(self, hidden_size=100):
          super(Net, self).__init__()
          # 我们可以使用已经训练好的词向量进行初始化
          # vocab.PAD_IDX 对应的向量通常为0，不进行梯度的计算
          self.embedding = torch.nn.Embedding.from_pretrained(torch.tensor(weights_matrix,dtype=torch.float32), freeze=False, padding_idx=vocab.PAD_IDX)
          # self.embedding = torch.nn.Embedding(len(vocab), EMBEDDING_DIM, padding_idx=vocab.PAD_IDX)
          self.lstm = nn.LSTM(EMBEDDING_DIM, hidden_size,num_layers=2, bidirectional=True, batch_first=True)
          self.fc = nn.Linear(4*hidden_size,2)
  
      def forward(self, x):
          x = self.embedding(x)
          output, _ = self.lstm(x)
          encoding = torch.cat((output[:,-1,:], output[:,0,:]), dim=1)
          x = self.fc(encoding)
          return x
```

7. 训练

   因为我们上面做了padding操作，在有些任务计算loss的时候需要忽略padding的元素的影响，不对其计算loss，但是这里是一个文本分类任务，这里就不用考虑
   训练了3个epoch，acc大概在0.85-0.9左右，可以提高一下模型的复杂度、训练更多的epoch，看看效果会不会更好
   
```python
   batch_size =16
   train_dataset = MyDataset(train_tsv)
   train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=padding_collate_fn)
   
   eval_dataset = MyDataset(eval_tsv,is_train=False)
   eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False,collate_fn=padding_collate_fn)
   
   # 显存不够，这里使用cpu进行train
   device = torch.device("cpu")
   
   # 有些时候计算损失的时候也需要忽略vocab.PAD_IDX，不对其进行计算loss，但是这里是一个分类任务，就不需要
   # 如果是一个文本生成任务，我们就需要忽略padding的损失，使用ignore_index参数即可
   loss_fn =nn.CrossEntropyLoss()
   net = Net()
   net.to(device)
   
   optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
   NUM_EPOCHS = 3
   five_batch_acc = []
   five_batch_right =0
   five_batch_total =0
   for epoch in range(NUM_EPOCHS):
       net.train()
       total =0
       right = 0
       for i, (input_ids, labels) in enumerate(tqdm(train_dataloader)):
           input_ids,labels = input_ids.to(device),labels.to(device)
           optimizer.zero_grad()
           outputs = net(input_ids)
           loss = loss_fn(outputs, labels)
           loss.backward()
           optimizer.step() 
           total += len(labels)
           right += (outputs.argmax(dim=1) == labels).float().sum()
           
           # 每5个step算一下acc，画图用
           five_batch_right += (outputs.argmax(dim=1) == labels).float().sum()
           five_batch_total += len(labels)
           if (i+1) % 5 == 0:
               five_batch_acc.append(five_batch_right/five_batch_total)
               five_batch_right = 0
               five_batch_total = 0
            # 每20step打印一下acc 
           if (i+1) % 20 == 0:
               print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
               print(f"epoch {epoch+1}, acc: {right/total:.4f}")
   
       # 每个epoch之后进行eval        
       net.eval()
       eval_total=0
       eval_right=0        
       for i, (input_ids, labels) in enumerate(tqdm(eval_dataloader)):
       
           input_ids,labels = input_ids.to(device),labels.to(device)
           outputs = net(input_ids)
           eval_total += len(labels)
           eval_right += (outputs.argmax(dim=1) == labels).float().sum()
       print(f"epoch {epoch+1}, eval_acc: {eval_right/eval_total:.4f}")   
   
   plt.plot(five_batch_acc, label='acc')
   plt.xlabel('Iteration')
   plt.ylabel('acc')
   plt.title('acc curve with word2vec and freeze')
   plt.legend()
   plt.show()
   torch.save(net.state_dict(), "bi_lstm.pth")
   print("\n模型参数已保存到 'bi_lstm.pth'")
```

   





# 基于 transformer 的bert的方法

这里我们使用的是hugging face生态

## 遇到的一些问题

1. bert对中文的分词是基于字的，就是单纯地把一句话的每个字分开，由于刚开始学习NLP，对很多东西不是特别了解，起初以为是bert对中文的分词效果不行（实际上是中文的分词有基于字的分词和基于词的分词，像bert就是基于字的，jieba等就是基于词的）。
2. 然后我后面用来了qwen2的tokenizer进行了尝试，qwen2使用的是B-BPE算法，是基于词的中文分词，哈哈，现在听起来有点奇怪，tokenizer使用qwen2，model使用bert。最夸张的是还跑起来了。（直接使用qwen2的tokenizer进行分词再把结果给bert是跑不起来的，因为qwen2的词表大小比bert的大，我是基于qwen2的tokenizer在自己的数据集上微调了一下，得到了一个自己的tokenizer，词表大小相对较小，比bert的小，因此能跑起来，但是有一些特殊字符也和bert的对不上）不过效果不是很好，大概只有0.85的准确率，比传统的bi-lstm的效果还差。不过正常来说，我们微调模型的时候，tokenizer应该和model保持一致，不然的话得改很多东西，因为词表的大小，还有特殊字符等很多东西都不太一样。
3. 之后使用了bert进行微调，还没跑的时候觉得基于字的分词效果肯定不行，但现实直接被打脸，直接使用bert进行情感分类的准确率就大概有0.92，微调了几个epoch之后准确率到了0.945。确实效果比传统的bi-lstm的效果好一些。这样看起来基于字的中文分词的效果好像也不是很差

## 基于字的分词和基于词的分词

1. 基于字的分词
- 不依赖于分词算法，不会出现分词边界切分错误
- 基本上不会出现OOV问题


2. 基于词的分词
- 序列相对于基于字的分词更短

> 基于词的分词通常可以看作一个序列标注问题，即对每个token进行分类，具体可以看[这篇文章](https://aistudio.baidu.com/projectdetail/4459155?channelType=0&channel=0),这篇文章介绍了如何使用bert来做中文分词，本质上就是一个序列分类任务

## 代码

```python
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers import TrainingArguments,Trainer,DataCollatorWithPadding
from transformers.trainer_utils import EvalPrediction
from datasets import load_dataset, Features, Value, ClassLabel

model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese",num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese",return_tensors="pt")

# 只选取text_a 和  text_a ，因为dev.tsv 中有qid列，这里我们不需要它
features = Features({
    'text_a': Value('string'),
    'text_a': Value('int32'),
    })
dataset =load_dataset("csv",data_dir="./data",data_files={"train":"train.tsv","validation":"dev.tsv"},features=features, delimiter="\t")

# 这里因为bert最高只支持512长度的序列，因此这里需要对大于512长度的序列进行裁剪
tokenized_dataset = dataset.map(
    lambda examples: tokenizer(examples["text_a"],truncation=True,max_length=512),
    batched=True,
    remove_columns=["text_a"],
)

# 对每个batch 进行padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding=True)


training_args = TrainingArguments(
    output_dir="./char_based_bert_finetune",
    num_train_epochs =6,
    eval_strategy = "epoch",
    per_device_train_batch_size =64,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps =1,
    learning_rate = 1e-5,
    lr_scheduler_type = "cosine",
    logging_strategy= "steps",
    logging_steps = 20,
    save_strategy = "epoch",
    save_total_limit = 4,
    seed = 42,
    data_seed = 42,
    load_best_model_at_end=True,
    # 指定label的字段
    label_names=["labels"],
    run_name="char_based_bert_finetune",
    report_to="wandb",
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    optim="adamw_torch",
    # eval_on_start=True, # just for test eval
    )

def compute_metrics(eval_pred:EvalPrediction):
    predictions, labels = eval_pred
    accuracy = (predictions == labels).mean()
    return {
        'accuracy': accuracy,
    }

def preprocess_logits_for_metrics(logits, labels):
    predictions = logits.argmax(axis=1)
    return predictions

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    # tokenizer = tokenizer,
    processing_class = tokenizer,
    data_collator=data_collator,
    preprocess_logits_for_metrics =preprocess_logits_for_metrics,
)
trainer.train()
```
一些需要注意的点：
1. 这里我们定义了compute_metrics来自定义评价指标，preprocess_logits_for_metrics 函数对模型的输出结果做一些处理，根据logits得到最后的预测结果（这一步是和compute_metrics紧密相关的，如果不定义preprocess_logits_for_metrics的话，trainer在进行eval的时候会把输出的logits都保存下来，然后我们在compute_metrics函数中对logits进行处理，计算acc，正常来说也确实是没什么问题，因为我们这里输出的logits的大小比较小，是一个batch_size * 2 的一个tensor，但是我以前在微调大模型的时候碰到过输出的logits很大，然后如果你的eval_dataset也比较大的话，就很容易VOOM，而preprocess_logits_for_metrics 可以在eval阶段的每个batch之后将对logits进行处理，这里我们是得到了最终的预测结果，这样的话就只需要保存预测结果，大小为 batch_size *1,相对于logits就小很多，这个在微调大模型的时候很有用）


2. **还有就是如果自定义了compute_metrics函数，需要在TrainingArguments 中设定label_names 参数来指定标签是哪一个字段,否则不会执行自定义的compute_metrics**。这里我们dataset中的标签的字段是label，但是DataCollatorWithPadding函数中会把label字段变为labels

<img src="https://img.leftover.cn/img-md/202505191203140.png" alt="image-20250518133848041" style="zoom: 67%;" />

3. 由于bert最高只支持512长度的序列，在代码中我们对长度超过512的序列进行了裁剪，这是一个比较糙的做法，这里可以参考一些这篇[文章](https://zhuanlan.zhihu.com/p/493424507),里面提到了一些对于输入长度超过了512的一些解决办法。简单来说 1. 我们可以对输入进行裁剪 2. 对bert的结构修改，消除长度限制 3. 使用滑动窗口的形式来对输入进行采样得到多个子样本，这样一方面扩大了数据集，也提高了准确率（不是每一种任务都可以这样做，但文本分类任务可以）



## 添加滑动窗口来解决bert的长度限制问题

```python
from functools import partial
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers import TrainingArguments,Trainer,DataCollatorWithPadding
from transformers.trainer_utils import EvalPrediction
from datasets import load_dataset, Features, Value, ClassLabel,concatenate_datasets,Dataset,DatasetDict
import numpy as np
import torch.nn.functional as  F
import torch
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese",num_labels=2)
features = Features({
    'text_a': Value('string'),
    'label': Value('int32'),
    })
new_features = Features({
    'text_a': Value('string'),
    'label': Value('int32'),
    'overlapping': Value('int64'),
    })
dataset =load_dataset("csv",data_dir="./data",data_files={"train":"train.tsv","validation":"dev.tsv"},features=features, delimiter="\t")

window_size = 100

def preprocess_dataset(dataset,window_size):
    overlapping = 1
    add_rows =[]
    for i in range(len(dataset)):
        text = dataset[i]["text_a"]
        if len(text) > 500:
            for j in range(0, len(text), window_size):
                end = min(j + 500, len(text))
                new_text = text[j:end]
                add_rows.append({"text_a": new_text, "label": dataset[i]["label"],"overlapping" : overlapping})
            overlapping += 1  
    return add_rows

# 对数据集进行处理，将长度大于500的样本进行拆分，添加overlapping列（<0 则表示没有拆分，>0 则表示拆分了，overlapping的值相同则表示同一个样本拆分出来的）
train_add_rows = preprocess_dataset(dataset["train"], window_size)
eval_add_rows = preprocess_dataset(dataset["validation"], window_size)

add_dataset_train = Dataset.from_list(train_add_rows,features=new_features)
add_dataset_eval = Dataset.from_list(eval_add_rows,features=new_features)

# 只保留 <=500的行
filtered_dataset = dataset.filter(
    lambda examples: len(examples["text_a"]) <= 500
)
#
train_overlapping_column = [-(i+1) for i in range(len(filtered_dataset["train"]))]
eval_overlapping_column = [-(i+1)  for i in range(len(filtered_dataset["validation"]))]

filtered_dataset = DatasetDict({"train":filtered_dataset["train"].add_column("overlapping",train_overlapping_column),
                                "validation":filtered_dataset["validation"].add_column("overlapping",eval_overlapping_column)},
                                features = new_features)
train_dataset_concat = concatenate_datasets([filtered_dataset["train"],add_dataset_train])
eval_dataset_concat = concatenate_datasets([filtered_dataset["validation"],add_dataset_eval])
# 最终的数据集
final_dataset = DatasetDict({"train":train_dataset_concat,"validation":eval_dataset_concat})


tokenized_dataset = final_dataset.map(
  # 这里由于我们上面对数据集使用了滑动窗口进行了处理，因此这里可以不需要裁剪
    # lambda examples: tokenizer(examples["text_a"],truncation=True,max_length=500),
    lambda examples: tokenizer(examples["text_a"]),
    batched=True,
    remove_columns=["text_a"],
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding=True)


training_args = TrainingArguments(
    output_dir="./bert_finetune_with_window",
    num_train_epochs =6,
    eval_strategy = "epoch",
    per_device_train_batch_size =64,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps =1,
    learning_rate = 1e-5,
    lr_scheduler_type = "cosine",
    logging_strategy= "steps",
    logging_steps = 20,
    save_strategy = "epoch",
    save_total_limit = 4,
    seed = 42,
    data_seed = 42,
    load_best_model_at_end=True,
    # 指定label的字段
    label_names=["labels"],
    run_name="bert_finetune_with_window",
    report_to="wandb",
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    optim="adamw_torch",
    # eval_on_start=True, # just for test eval
    )

eval_overlapping = final_dataset["validation"]["overlapping"]

# 这里进行eval的时候，若是拆分出来的样本，则分别对这几个样本进行预测，取概率最大的作为最终的预测结果
def compute_metrics(eval_overlapping: list,eval_pred:EvalPrediction):
    logits, labels = eval_pred
    logits = F.softmax(torch.tensor(logits),dim = 1)
    final_predictions = []
    final_labels = []
    for i ,overlapping_value in enumerate(eval_overlapping):
        if overlapping_value <= 0:
            final_predictions.append(logits[i].argmax())
        else: 
            j =i+1
            final_prediction = logits[i].argmax()
            value =logits[i][final_prediction]
            while(j < len(eval_overlapping) and eval_overlapping[j] == overlapping_value):
                pred = logits[j].argmax()
                if  logits[j][pred] > value:
                    value = logits[j][pred]
                    final_prediction = pred
                j+=1
            final_predictions.append(final_prediction)
        final_labels.append(labels[i])
             
              
    accuracy =(np.array(final_predictions) == np.array(final_labels)).mean()
    return {
        'accuracy': accuracy,
    }

# def preprocess_logits_for_metrics(logits, labels):
#     predictions = logits.argmax(axis=1)
#     return predictions

# 这里由于进行compute_metrics的时候需要用到eval_overlapping，所以需要partial函数来传入eval_overlapping
# 因为由于bert模型的输入用不到overlapping这个字段，trainer会把没有使用到的列自动删除
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=partial(compute_metrics,eval_overlapping),
    # tokenizer = tokenizer,
    processing_class = tokenizer,
    data_collator=data_collator,
    # preprocess_logits_for_metrics =preprocess_logits_for_metrics,
)


trainer.train()
trainer.evaluate()

```

一些需要注意的点：

- 这部分的代码实现的不是很优雅，应该会有更好的实现办法
- 因为在compute_metrics阶段要用到logits，因此这里删掉了preprocess_logits_for_metrics
- 在compute_metrics阶段要用到eval_overlapping，但是由于bert模型的输入用不到overlapping这个字段，trainer会把没有使用到的列自动删除，否则代码会报错，因为我们这里使用了partial来将eval_overlapping传进来
- 由于我们的验证集中超过了500的数据并不多，貌似只有10个左右，所以提升并不明显，maybe 提升0.2%

## 数据集和模型

可以在hugging face上找到数据集和微调之后的模型，正常的[数据集](https://huggingface.co/datasets/left0ver/sentiment-classification) , 滑动窗口版本的[数据集](https://huggingface.co/datasets/left0ver/sentiment-classification/tree/window_version) 。没有使用滑动窗口的方法进行微调得到的[模型](https://huggingface.co/left0ver/bert-base-chinese-finetune-sentiment-classification)

# 项目代码
代码请查看[left0ver/Sentiment-Classification](https://github.com/left0ver/Sentiment-Classification)

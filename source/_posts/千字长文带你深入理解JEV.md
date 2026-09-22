---
title: 千字长文带你深入理解JEV.md
mathjax: true
date: 2026-09-22 18:51:19
tags:
  - Agent
  - python
---
JEV是typesafe推出的一款system one模型，旨在快速做出结构化的决策。因此这个模型主要是用来做决策的（做选择），那么我们可以想到他的一些常见的应用场景：

1. rerank: 将top-n的chunk作为每个选项，用户问题作为指令/state，最后得到每个chunk和问题的相关性的概率作为rerank的结果
2. tool_search / skill search: 目前Agent领域的tool search / skill search 大多通过BM25 检索或者向量检索的方式来进行工具的搜索(亦或者二者结合起来)，而jev则相当于是用一个小的模型来进行工具的选择，既能保证一个较高的准确度，同时时间和金钱都花费较少，是一个不错的选择
3. 判断是否为危险的工具调用： 像codex / Claude code中都有使用小的分类模型来判断该操作是否为危险操作，如果为危险操作则需要人类进一步的审批，这个判断也可以使用jev来实现
4. 一些常见的分类任务：例如邮件分类，工单分类
5. 对生成的数据做质检，过滤掉低质量/不相关的数据

> 这个模型和我们常见的LLM不一样，他并不是一个生成模型，有点类似于bert那种，但他想对于bert，他的分类的问题不必要是预定义的，你可以给他任何的问题，并输出对应的选项的概率，而bert的话则只能输出预定义好的类别对应的概率

<!-- more -->

## 什么是system one
他像人类的本能反应一样，遇到情况时可以瞬间做出决策。这也表明了JEV是一个做决策的模型并且他的速度很快。

## JEV的API调用
JEV 的api输入的问题有三种形式：

1. noul：输出答案为"true"的概率，输入的格式大概是下面这样

```json
{
  "state": "I have asked three times now. Can I please just talk to a real person?",
  "questions": {
    "is_human_escalation": {
      "type": "noul",
      "instructions": "Is the customer asking for a human agent?"
    },
    "is_repeat_contact": {
      "type": "noul",
      "instructions": "Has the customer contacted support about this before?",
      "criteria": {
        "true": "Mentions a prior attempt, ticket, or that they have asked before",
        "false": "No sign of any previous contact"
      }
    }
  }
}
```

2. choice: 输出每个选项的概率以及置信度,输入的格式大概是下面这样

```json
{
  "state": "My running shoes arrived in the wrong size. Can I swap them for a size 10?",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "returns": "Exchanges, wrong or damaged items",
        "shipping": "Delivery status, delays, lost packages",
        "billing": "Charges, invoices, payment problems"
      }
    }
  }
}
```

**置信度表示的是当前做出的这个选择的一个自信程度，如果两个选项的概率都比较接近，那么置信度就低，而如果某个选项的概率值很高，其他的选择的概率都很低，则置信度就很高**

3. score：当问题的答案在一个"有顺序的连续等级"上时，使用score，用来进行程度的一个判断。例如
+ bug 严重程度：轻微 → 一般 → 严重
+ 用户满意度：很不满意 → 一般 → 很满意
+ 匹配度：低 → 中 → 高
+ 技能水平：初级 → 中级 → 高级
+ 回答质量：差 → 一般 → 好 → 很好

输入的结构大概是下面这样：

```json
{
  "state": "The export button crashes the settings page in Safari. It works in Chrome, but a few of our customers only use Safari.",
  "questions": {
    "bug_severity": {
      "type": "score",
      "instructions": "How severe is the reported issue?",
      "criteria": [
        "Cosmetic; no impact to functionality",
        "Broken or degraded feature, but workaround exists",
        "Blocking issue; no workaround exists"
      ]
    }
  }
}
```

返回的结果中会包含score字段，这个字段的值是通过级别与对应概率的加权平均得到的，例如下面的例子中，score则表示bug的严重程度介于Cosmetic-Broken 之间，倾向于Cosmetic

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "bug_severity": {
      "type": "score",
      "score": 1.43,
      "confidence": 0.35,
      "legend": {
        "0": "Cosmetic; no impact to functionality",
        "1": "Broken or degraded feature, but workaround exists",
        "2": "Blocking issue; no workaround exists"
      },
      "probabilities": {
        "0": 0.0,
        "1": 0.57,
        "2": 0.43
      }
    }
  },
  "usage": {
    "input_tokens": 332,
    "output_tokens": 18
  }
}
```

## JEV的实现原理
不难看出，上面的每种输入形式都可以统一成一种形式：输入问题还有选项，输出每一个选项的概率。

Choice：输入单个问题以及多个选项，选择概率最高的

Score：输入单个问题以及多个选项，输出每个选项的概率，再根据每个选项的概率得出最终的一个score

Noul：输入单个问题或者可选的`选项`,没有输入选项的时候，程序可以动态地添加两个选项，一个为true，一个为false，最终输出为true的概率即可。

参考laya的实现方式:

将每个问题的输入序列安卓下面这样组成一个sequence，输入到model中

```json
[CLS] 问题类型和指令 [SEP] [MASK] 选项一 [MASK] 选项二 … [SEP] state [SEP]
```

> [SEP] 为分隔符
>

经过forward之后，提取特征，得出的输出向量（s,d 维）

之后提取各个[Mask]位置的向量，输入到共享的scorer中(这个scorer可以自己设计，可以是简单的小的神经网络，也可以是复杂的、大的神经网络)，得到一个最终的输出（n，1）,n为选项个数

再对每个选项的logits进行温度缩放和softmax，得到每个选项的概率，之后再由程序组合得到返回给用户的输出结果

### 置信度的计算
置信度是用来表示模型对于当前决策的自信程度，如果两个选项的概率都比较接近，那么置信度就低，而如果某个选项的概率值很高，其他的选择的概率都很低，则置信度就很高。

laya设计了一个act_head（一个小的神经网络）来得到最终的自信度，输入是[CLS] 向量 ，以及 最大概率（当前问题的选项中最大的概率值），前两项概率差（第一候选领先第二候选多少，例如 `0.60 − 0.35 = 0.25`）、归一化熵（用来表示概率的分散程度，多个选项的概率接近时，这个数值就高）、选项数量比例（选项数量/255），输出一个0-1的值，表示自信度

### 并行多问题计算
上面的介绍中，介绍了输入到模型中的是一个问题，以及其对应的选项等，而jev中可以同时传入多个问题，**我们只需要把这些问题组合成一个batch，输入到模型中即可，最终就可以得到5个问题的选项的概率**

## 总结
上述对于jev原理的解析是基于开源的laya来分析得到的，但是我觉得jev的原理跟这个也大差不差。这也能够解释为什么jev的收费是按输入的token数来收费的，而不是像LLM一样按输入+输出的token数来计费的，因为他根本就不需要输出token，只需要进行一次forward，得到最终的问题的每个选项的概率值即可，因此他的成本主要是来源于输入的文本的长度。

其实很多看似很厉害的研究都是基于前人的基础上做了一些小小的改变而做到的，对于每个爆火的模型/架构，我觉得我们应该理性地看待他们，深入地去了解他的原理，了解他是怎么实现的，往往看似很厉害的东西，他的设计/实现往往很简单

## Reference
1. [laya](https://github.com/NandhaKishorM/laya)
2. [JEV文档](https://docs.typesafe.ai/introduction)
3. [laya架构解析](https://chatgpt.com/s/cx_6ab25cdcd47c81918de520069ccfa789)


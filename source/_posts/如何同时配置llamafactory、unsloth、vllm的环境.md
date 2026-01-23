---
title: 如何同时配置llamafactory、unsloth、vllm的环境
mathjax: true
date: 2026-01-23 20:01:36
tags:
  - AI
  - python
---
## 准备工作
1. 现在电脑上安装uv
2. 创建虚拟环境，python的话使用3.12版本
3. 我们需要先确定torch的版本，尤其是unsloth和vllm，torch的版本对这二者的安装影响很大。

## TIPS
前面两步vllm和unsloth的时候我们确定了torch的版本，我们可以先单独把torch安装完，然后再安装LlamaFactory，再安装unsloth、最后安装vllm，因为LlamaFactory的有些库用的版本比较老，安装unsloth的时候会覆盖掉，而LlamaFactory对一些库的版本的依赖并没有很严格，因此一些非核心的库的版本不符合LlamaFactory的要求也能使用，只需要使用`**DISABLE_VERSION_CHECK=1**`来跳过LlamaFactory的版本检查即可

<!-- more -->

## 安装
### 安装vllm
我们先使用`uv pip install vllm --torch-backend=cu124`

> 这里我的CUDA驱动的版本为12.8，因此我这里选择cu124，这里根据自己的CUDA版本来选，建议选cu128、cu124、cu118
>

然后看一下这里安装的torch版本是多少，我这里安装的是`torch2.6.0+cu124`


![](https://img.leftover.cn/img-md/202601231957804.png)

### 安装unsloth
我们可以先在unsloth的[手动安装部分的文档](https://unsloth.ai/docs/get-started/install/pip-install)上看到


![](https://img.leftover.cn/img-md/202601231957002.png)

上面vllm安装的pytorch版本最好和unsloth的匹配起来，上面安装的是`torch2.6.0+cu124`,刚好和unsloth这里有对应的`cu124-torch260`，可以使用官网上的这个命令来确定unsloth的安装命令

```shell
wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -
```


![](https://img.leftover.cn/img-md/202601231957376.png)

### 安装LlamaFactory
根据官网的教程安装即可

```shell
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt
```

> 这部分可能会有依赖冲突，比如datasets和peft的依赖，LlamaFactory用的是比较老的版本，而unsloth使用的是比较新的版本，这里我们以unsloth的为准安装更新的版本，**LlamaFactory在启动的时候会进行依赖的版本检查，这些不是特别核心的库，版本影响不大，我们直接使用**`**DISABLE_VERSION_CHECK=1**`**来跳过LlamaFactory的版本检查**
>

```shell
DISABLE_VERSION_CHECK=1 llamafactory-cli webui
```

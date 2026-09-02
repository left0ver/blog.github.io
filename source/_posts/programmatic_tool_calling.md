---
title: Programmatic_Tool_Calling
mathjax: true
date: 2026-09-03 01:35:10
tags:
  - Agent
---

## 什么是PTC
通常传统的工具调用的流程大概是： 

模型-> 调用工具A、B->将工具调用的结果加入上下文->模型继续推理-> 调用工具C

而PTC（programmatic_tool_calling）则是我们可以把tool看作一个函数，即我们可以用可执行的代码来组合多个工具得出一个最终的结果。这样的话模型只需要发起一次工具调用即可，即调用`run_code`这个工具，输出对应的可执行代码，并将结果保存到固定的`result`变量中。

PTC的优点在于：

<!-- more -->

1. 相比于传统的工具调用流程，PTC可以通过函数组合的方式来执行一些复杂的控制流：

例如：

- 循环和分页
  
- 条件分支

- 根据前一个结果构造下一个调用

- 并行请求

- 重试和停止条件

- 多来源数据合并
 
1. 模型只需要生成一次编排代码，而不必在每个工具调用之间重新推理。底层工具依然会被多次调用，但中间结果可以留在执行环境中处理。因此它可以减少调用模型的次数、降低延迟。
2. 因为如果是传统的工具调用，则可能一个复杂的任务，需要将多个工具结果添加到上下文中，而PTC则可能只需要一次tool，工具返回的大量原始数据可以在沙箱里先过滤、聚合、去重，模型只看到最终的小结果，避免上下文被日志、表格或 API 响应塞满。

缺点：

PTC更适合那种复杂的任务，而对于那种只需要调用一两次工具的简单的任务，使用PTC反而可能增大token消耗;而对于复杂任务，PTC则可以降低token消耗以及模型的调用次数

## 实现
知道了PTC的思路之后，实现起来并不会特别难

1. tool: 给LLM的tool实际只有一个，即`execute_python`，输入为对应的python代码，输出为字符串
2. 其次模型还需要能看到其他的需要执行的工具（工具的签名、参数、返回值结构），这些工具还是需要给到LLM 的提示词中，但是LLM不能直接调用这些工具，只能通过调用`execute_python`这个工具，在这个工具中调用其他的工具来完成复杂的控制流任务
3. 代码的执行：代码的执行需要在沙盒中，并且需要限制他可以执行的函数（即可以执行的工具，以及一些常见的python函数），python可以使用restrictedpython来实现，当然也可以使用一些付费的沙箱服务，或者自部署沙盒，最后将结果保存到`result`变量中，每次`execute_python`工具只需要返回`result`变量到结果即可。

具体的代码可看[left0ver/programmatic_tool_calling](https://github.com/left0ver/programmatic_tool_calling)

## Reference 
1. [codeAct](https://arxiv.org/abs/2402.01030)
2. [Code execution with MCP: Building more efficient agents](https://www.anthropic.com/engineering/code-execution-with-mcp)


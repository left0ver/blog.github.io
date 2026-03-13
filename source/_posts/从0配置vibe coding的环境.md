---
title: 从0配置vibe coding的环境
mathjax: true
date: 2026-03-13 21:51:09
tags:
  - AI
---

## 安装
1. 从claude code 的[官网](https://code.claude.com/docs/en/overview)下载claude code
2. `npm install -g @openai/codex`下载codex-cli工具

## 安装cc switch
[cc switch](about:blank)是一个管理claude code、codex、gemini cli、opencode的工具，可以使用它来快速地切换不同的提供商，它是一个桌面端软件，[cc-switch-cli](https://github.com/SaladDay/cc-switch-cli) 则是它的cli版本，适合在Linux等没有页面的系统中使用



## 一些推荐的中转站或者官方的使用方法
### codex
1. 可以通过购买gpt team的套餐来使用，闲鱼上可以购买，通常大概8块钱左右，不过现在openai对于gpt business的封禁比较严格了，可能一周左右就封了，因此最好买那些有质保的。

> 有了gpt business之后，既可以使用网页上的gpt，还可以使用codex，还是比较划算的
>

<!-- more -->

### Claude code
Claude code则可以通过购买官方的套餐来使用，不过需要国外的银行卡，而且很容易封禁，一般人不建议去折腾Claude code的官方套餐，同样闲鱼上也有类似的拼车等，最好选择靠谱的车队

### 中转站
这里推荐一些比较不错的中转站

1. [anyrouter](https://anyrouter.top/)：只能使用教育邮箱注册，每天登录送25额度，注册可得50额度，使用我的邀请链接注册可再得50额度（[邀请链接](https://anyrouter.top/register?aff=lTPF)）
2. [agentrouter](https://agentrouter.org/)：只能使用github 和 L站的方式注册，注册送100额度，每天登录送25额度，使用我的邀请链接注册可再得100额度（[邀请链接](https://agentrouter.org/register?aff=huKF)）

> 上面两种中转站都不支持充值，只能每天登录获取额度
>

3. [ikuncode](https://api.ikuncode.cc/)：商业的中转站，需要充值，如果上面两种不够用，也可以使用这个中转站，口碑还是不错的

选择完了之后就可以在cc switch中配置好来，之后可以快速地切换不同的提供商，cc switch中也可以配置skills，mcp可以同步到claude code，codex等工具中

#### 自动签到的方法
可以使用[anyrouter-check-in](https://github.com/millylee/anyrouter-check-in) 来实现每天自动签到，该项目使用GitHub action来实现定时的自动签到，支持多个账号，按照项目的readme 进行配置即可，cookie大概1个月过期，1个月得重新配置一下

## claude code的使用
1. 终端输入claude即可使用，vscode中还有claude code的[扩展](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code)，可以下载一下，配合claude code 一起使用
2. 可以看一下这个[everything-claude-code](https://github.com/affaan-m/everything-claude-code)仓库，里面有这个大佬常用的claude code的一些skills，rules，commands，按需取即可

### 我的一些skills 和 mcp
#### skills
1. [code-review-skill](https://github.com/awesome-skills/code-review-skill) 用来进行code review
2. everything-claude-code 中的frontend-patterns skill
3. [frontend-design](https://github.com/anthropics/skills/tree/main/skills/frontend-design) 前端设计的skill

#### mcp
1. @modelcontextprotocol/server-sequential-thinking
2. @modelcontextprotocol/server-memory
3. context7
4. github-mcp
5. @modelcontextprotocol/server-filesystem



### codex 的vscode扩展请求超时的问题
我的服务器上有梯子，但是貌似codex的vscode扩展并不会使用http_proxy等环境变量，然后导致请求超时，而codex的cli版本则可以正常使用

配置为图中的样子即可
![](https://img.leftover.cn/img-md/202603132146577.png)


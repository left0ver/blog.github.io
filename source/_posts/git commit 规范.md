---
title: git commit 规范
date: 2022-07-10 21:40:39
tags:
  - git
---

# git commit 规范

> 规范git的commit可以帮助我们清楚地了解每一个commit的作用,其次,可以使用一些工具根据commit自动生成changelog,具体的方法可以参照这篇[博客](https://leftover.cn/2022/06/20/%E8%87%AA%E5%8A%A8%E7%94%9F%E6%88%90Changelog/#more)

一般一条commit包含 

`<type>(<scope>): <subject>`


<!-- more -->

1. type一般包含以下这些

- feat: 新特性,新功能等等
- fix: 修复bug
- docs: 仅仅修改了文档，比如 README, CHANGELOG, CONTRIBUTE等等
- style: 仅仅修改了空格、格式缩进、逗号等等，不改变代码逻辑
- refactor: 代码重构，没有加新功能或者修复 bug
- perf: 优化相关，比如提升性能、体验
- test: 测试用例，包括单元测试、集成测试等
- chore: 改变构建流程、或者增加依赖库、工具等
- revert: 回滚版本

2. scope 表示此次修改作用的范围,这个不同的项目不一样
3. subject 这个commit的描述  

例如：

feat:完成首页

chore(release): v0.0.2


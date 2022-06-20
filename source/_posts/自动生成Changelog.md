---
title: 自动生成Changelog
date: 2022-06-20 12:14:36
tags:
  - 前端工程化
---

# 使用release-please生成Changelog

- 以前可能很多人使用[standard-version](https://www.npmjs.com/package/standard-version)来生成对应的Changelog，现在已经不推荐使用这个库了，这个作者推荐使用 [release-please](https://github.com/googleapis/release-please)库来自动生成Changelog。

- 使用release-please最简单的方式就是利用GitHub action，当你push的时候生成对应的Changelog，官方的仓库的文档里也有对应的说明，当然也可以使用命令行的方法来生成对应的Changelog，感兴趣的同学可以自行查阅
<!-- more -->
- release-please会根据你的commit来生成对应的Changelog，所以commit一定要规范，否则将不能起到对应的效果

## 缺点

- 我自己有尝试使用release-please配合GitHub Action来生成对应的Changelog,它是根据你的commit来进行对应的版本的更改的，一般修复bug，他会修改最后一个版本号，`fix: a bug` ，如果有一个新特性`feat: a new feat `,那么他会升级中间的版本号，如果有什么破坏性的更改，那么他会升级第一个版本号,例如`feat!: a break change`
- 虽然这样子听起来也挺合理，但是有时候我们有一些小的新特性，但这时候并不想升级中间的那个版本号，而是只想`patch`,升级最后一个版本号，使用release-please就很难满足我们的需求，而且release-please是自动升级版本号，但是我们大多数情况下还是想要将版本号控制在自己的手中，自己来决定版本的升级，于是我使用了[antfu](https://github.com/antfu)大神的[changelogithub](https://github.com/antfu/changelogithub)

# 使用changelogithub生成Changelog

- [changelogithub](https://github.com/antfu/changelogithub)使用起来也是非常滴简单,也是基于GitHub Action来实现对应的Changelog的生成，当你push一个tag的时候将触发workflow，根据你push的tag的版本号来生成对应的Changelog，这样就将版本的控制掌握在自己的手中了

  ```yaml
  # .github/workflows/release.yml
  
  name: Release
  
  on:
    push:
      tags:
        - 'v*'
  
  jobs:
    release:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
          with:
            fetch-depth: 0
  
        - uses: actions/setup-node@v3
          with:
            node-version: 16.x
  
        - run: npx changelogithub
          env:
            GITHUB_TOKEN: ${{secrets.GITHUB_TOKEN}}
  ```

# 自己的理解

- 现在一般会将fix，和feat或者Breaking Changes部分的内容生成Changelog，因为这些内容才是需要用户知道的东西，而其他的一些东西的修改，比如说`docs: xxx `,  `style:xxx`等commit则不需要生成对应的Changelog，现在大多数的库也是这样做的
- 想要自动生成Changelog，首先必须得有规范的commit，可以使用commitizen配和commitlint，husky来对规范项目的commit
- 建议使用`changelogithub`来自动生成Changelog，将版本的更改控制在自己手中


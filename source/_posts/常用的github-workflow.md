---
title: 常用的github-workflow
date: 2022-09-18 14:09:32
tags:
  - 部署
---
# npm publish 自动发布npm包
```yaml
# .github/workflows/npm_publish.yml
name: npm publish
on:
  push:
    tags:
      - "v*" # Push events to matching v*, i.e. v1.0, v20.15.10

jobs:
  publish-npm:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 14.17.5
          registry-url: https://registry.npmjs.org/
      - run: npm publish
        env:
          NODE_AUTH_TOKEN: ${{secrets.NPM_TOKEN}}
```
<!-- more -->
# 自动生成changelog

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

# 连接服务器执行命令

```yaml
name: send interview

on:
  schedule:
    - cron: "00 11 * * *"
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/setup-node@v3
      with:
        node-version: '16'
      run: |
        npm i -g yarn
        yarn -v
        eval $(ssh-agent -s)
        echo "${{secrets.SERVER_SSH_PRIV_KEY}}" > deploy.key
        mkdir -p ~/.ssh
        chmod 600 deploy.key
        ssh-add deploy.key
        echo -e "Host *\n\tStrictHostKeyChecking no\n\n" > ~/.ssh/config
        ssh root@${{secrets.SERVER_IP}} "cd /www/wwwroot/server/qqRot-interview && yarn start:prod"

```


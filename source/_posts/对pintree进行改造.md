---
title: 对pintree进行改造
date: 2024-10-24 20:25:15
tags:
  - other
---

[pintree](https://github.com/Pintree-io/pintree)是一个可以将浏览器书签转为导航网站的工具，但是其效果并不是那么地好，本文章打算对其进行一个改造。主要的一个目标是：完全地自动化更新书签

本次改造不是基于官方的pintree，而是基于
wynnforthework fork 的[项目](https://github.com/wynnforthework/pintree/tree/gh-pages)

主要原因在于该项目对原来的pintree进行了一遍改造，使其可以解析本地的google书签。（不过有些bug，例如图标不能显示）

因此主要的思路就比较明朗了：

1. 我们使用dropbox作为主要的中介，前端通过dropbox 的api 获取到保存在dropbox 的书签json文件
   
2. 本地通过一个定时任务定时地将本地的书签json文件上传到dropbox




对于第一步，这里不多赘述，主要是对wynnforthework的pintree项目进行一个改造，修复图标不能显示的bug，以及通过refresh_token 获取到access_token，再利用access_token获取到dropbox上的文件。具体代码可查看[github](https://github.com/left0ver/pintree)，具体到效果可看[网站](https://pintree.leftover.cn/)

> 获取refresh_token: 
> 
> 1. 再浏览器输入这个网址:https://www.dropbox.com/oauth2/authorize?client_id=YOUR_CLIENT_ID&response_type=code& token_access_type=offline，(client_id 为你的app key)进行授权,获取到 authorize_code
> 2. 根据获取到的authorize_code获取到一个长久的refresh_token (建议使用代码进行请求，这里我使用了curl进行请求没反应)
> ```shell
>curl -X POST https://api.dropbox.com/oauth2/token \
>    -d code=<AUTHORIZATION_CODE> \
>    -d grant_type=authorization_code \
>    -d redirect_uri=<REDIRECT_URI> \
>    -u <APP_KEY>:<APP_SECRET>
> ```

对于第二步，以下是python代码上传文件到dropbox的实现，主要就是先通过refresh_token获取到access_token，再通过access_token将本地文件上传到dropbox


```python

import requests
import json
import os

# 配置
APP_KEY = os.getenv("APP_KEY", 'xxx')
APP_SECRET = os.getenv("APP_SECRET", 'xxx')
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN", 'xxx')
access_token = None


def upload_to_dropbox(file_path, dropbox_path):
  data = {
    'grant_type': 'refresh_token',
    'refresh_token': REFRESH_TOKEN,
    'client_id': APP_KEY,
    'client_secret': APP_SECRET,
  }
  response = requests.post('https://api.dropboxapi.com/oauth2/token', data=data)
  if response.status_code == 200:
    token_data = response.json()
    # 获取access_token
    access_token = token_data.get('access_token')
    print(f"成功获取access_token 为{access_token}")
    with open(file_path, "rb") as f:
      headers = {
        "Authorization": f"Bearer {access_token}",
        "Dropbox-API-Arg": json.dumps({"path": dropbox_path, "mode": "overwrite"}),
        "Content-Type": "application/octet-stream",
      }
      response = requests.post("https://content.dropboxapi.com/2/files/upload", headers=headers, data=f)
      if response.status_code == 200:
        print("File uploaded successfully")
  else:
    print(f"Error uploading file: {response.status_code} {response.text}")


# 示例调用
upload_to_dropbox("<your local file>", "<dropbox bookmark file>")

```

对于定时任务，由于我的是mac电脑，
1. 可以使用`crontab -e` 来创建一个定时任务

2. 创建mac上的plist文件，加载这个文件来进行一个定时任务

```shell
vim /Users/<username>/Library/LaunchAgents/upload.pintree.todropbox.plist

# 添加以下内容
# 每天20:00定时执行，日志输出到/tmp/uploadtodropbox.log、/tmp/uploadtodropbox.err

<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>upload.pintree.todropbox</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>your python file path</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>20</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/tmp/uploadtodropbox.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/uploadtodropbox.err</string>
</dict>
</plist>

# 加载plist文件
launchctl load /Users/<username>/Library/LaunchAgents/upload.pintree.todropbox.plist

# 手动执行一次，查看日志文件是否有输出
launchctl start upload.pintree.todropbox
```


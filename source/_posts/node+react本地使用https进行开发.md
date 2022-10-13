---
title: node+react本地使用https进行开发
date: 2022-10-13 22:08:05
tags:
  - node
  - react
---
# 准备好证书

对于本地来说，可以使用[mkcert](https://github.com/FiloSottile/mkcert)来生成证书

安装：

- mac：

  ```shell
  brew install mkcert
  brew install nss # if you use Firefox
  ```

- windows

  ```shell
  #Chocolatey
  choco install mkcert
  
  # or use Scoop
  scoop bucket add extras
  scoop install mkcert
  ```

生成证书

```shell
mkdir -p ~/.cert
#生成证书
mkcert -key-file ~/.cert/key.pem -cert-file ~/.cert/cert.pem "localhost"
#让系统信任生成的证书，只有初次生成证书时需要运行这个命令，后续通过 mkcert -key-file 生成的证书会自动被系统信任
mkcert -install
```
<!-- more -->
# express启动https服务

```typescript
import express from 'express'
const app = express()
const httpsOption = {
  //证书地址
    key: fs.readFileSync(path.join(os.homedir(), '.cert/key.pem')),
    cert: fs.readFileSync(path.join(os.homedir(), '.cert/cert.pem'),
  }
  https.createServer(httpsOption, app).listen(PORT, '127.0.0.1', () => {
    console.log(`server start and listening on port ${Port}`)
  })
```



# react启动https

如果你是通过cra创建的项目，只需添加几个环境变量即可，将启动命令修改成如下所示，即可在本地服务启动https

```json
//package.json
{
  "script":{
    "start": "HTTPS=true SSL_CRT_FILE=$HOME/.cert/cert.pem SSL_KEY_FILE=$HOME/.cert/key.pem react-scripts start",
  }
}
```

之后你的网络请求全部改成https即可

# 反向代理

现在线上环境经常会使用反向代理，为了本地和线上环境最大限度地保持一致，本地可以使用[caddy](caddy reverse-proxy --from localhost --to localhost:3000)实现反向代理,mac下运行`brew install caddy`即可下载，其他系统可以自行查阅文档，这里附上一个中文的[文档](https://caddy2.dengxiaolong.com/docs/)(非官方)。

只需要在项目启动之后运行下面命令即可，这条命令是将localhost代理到https://localhost:3000，我们打开网页输入localhost即可访问服务。

Caddy 会自动生成证书，获取系统信任，无需另行生成证书，也无需修改项目的启动服务。

```shell
caddy reverse-proxy --from localhost --to https://localhost:3000
```

![image-20221013215353805](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/202210132153926.png)



# 线上环境

针对线上环境，只需要将证书的地址什么的改一下，然后前端使用https发送请求，配置对应的反向代理即可

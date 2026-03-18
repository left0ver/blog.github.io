---
title: 使用charles来进行抓包
mathjax: true
date: 2026-03-18 21:48:24
tags:
  - other
---

# 抓取https的包以及外网的包
1. [charles官网](https://www.charlesproxy.com/) 下载`charles`
2. 然后菜单栏`Help-> SSL proxying -> Install Charles Root Certificate` 安装根证书
3. 然后菜单栏`Proxy -> Proxy Settings`设置成下面这样


![](https://img.leftover.cn/img-md/202603182146492.png)

<!-- more -->

4. 因为我们会使用clash 来访问外网，如果需要对外网的流量进行抓包的话，我们需要配置外部代理：`Proxy->External proxies settings` 设置成如下的样子,**端口需要和你的clash的代理的端口一致**


![](https://img.leftover.cn/img-md/202603182146755.png)


![](https://img.leftover.cn/img-md/202603182146104.png)



5. 最后菜单栏`Proxy-> SSL proxying settings`设置你需要对哪些网站进行https的代理（即抓https的包）


![](https://img.leftover.cn/img-md/202603182146606.png)

6. 最后如果需要抓外网的包的话，需要先启动clash（如果不需要抓外网的流量的包的话可以不启动clash），再菜单栏`Proxy->Windows proxy` 来启动代理，进行抓包。

# 对vscode进行抓包
好像Clarles设置的系统代理对vscode并没有生效，我们需要手动在vscode中设置一下`http_proxy` 和 `https_proxy` 环境变量

```python
set http_proxy=http://127.0.0.1:8888
set https_proxy=http://127.0.0.1:8888
```

# 对Claude code进行抓包
貌似Claude code 不允许这种自签的证书，因此我们需要在Charles的菜单栏`Help-> SSL proxying -> Charles Root Certificate`将正式导出到本地,然后设置`NODE_EXTRA_CA_CERTS`环境变量为证书对应的路径即可



# Reference
1. [[原创]Charles + Clash + Postern 对外网 App Vpx 抓包](https://bbs.kanxue.com/thread-282476-1.htm)
2. [claude code 突然连不上了 Unable to connect to API: Self-signed certificate detected](https://linux.do/t/topic/1498990)


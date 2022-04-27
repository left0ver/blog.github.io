---
title: ssh密钥远程连接服务器
date: 2022-04-27 14:54:21
tags:
  - linux
---


1. `ssh-keygen` 命令可以生成密钥，密钥默认保存在 `C:/Users/[your username]/.ssh` 文件下面  ，建议`-f`生成指定名字的密钥，因为你可能会有很多ssh密钥对，这样容易区分辨别，` ssh-keygen -f xxx`   ，这时候会在当前工作目录生成密钥对，之后将其移到`C:/Users/[your username]/.ssh`即可

      `tips: 不需要的密钥对尽量及时删除哦 `
<!-- more -->
2. 之后将公钥的文件复制到服务器上对应的目录上,root用户是`/root/.ssh`，其他用户是`/home/[your username]/.ssh`下面，之后看一下.ssh文件下有没有`authorized_keys `文件，如果没有则创建一个`touch authorized_keys ` ,
3. 最后将公钥里面的内容追加到 authorized_keys中  `cat [pub files name] >> authorized_keys`，这一步一定要有，不然公钥不生效
4. 当然现在ssh远程连接也可以使用密码进行验证，不过相信你以后会用到密钥连接的场景的


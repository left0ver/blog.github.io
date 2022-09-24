---
title: ssh密钥远程连接服务器
date: 2022-04-27 14:54:21
tags:
  - Linux
---


## 配置
1. `ssh-keygen` 命令可以生成密钥，密钥默认保存在 `C:/Users/[your username]/.ssh` 文件下面  ，建议`-f`生成指定名字的密钥，因为你可能会有很多ssh密钥对，这样容易区分辨别，` ssh-keygen -f xxx`   ，这时候会在当前工作目录生成密钥对，之后将其移到`C:/Users/[your username]/.ssh`即可

      `tips: 不需要的密钥对尽量及时删除哦 `
      <!-- more -->

2. 之后将公钥的文件复制到服务器上对应的目录上,root用户是`/root/.ssh`，其他用户是`/home/[your username]/.ssh`下面

3. 之后回到自己的电脑上，看一下`.ssh`目录下有没有`config`文件，没有则创建一个`touch config`，根据具体情况修改以下内容

      ```shell
      Host github.com
      HostName github.com
      User git
      Port 22
      IdentityFile ~/.ssh/id_ed25519
      
      ```

Host：表示别名，一般填写域名或者ip即可
HostName:服务器的IP或者域名
User:登录的用户
Port:指定端口号（默认是22）
IdentityFile 指定私钥文件的位置,例如 ~/.ssh/id_ed25519

​      

4. 最后在服务器上将公钥里面的内容追加到 authorized_keys文件中，authorized_keys文件位于.ssh目录下面，如果没有的话则创建一个, `touch authorized_keys `  将公钥里面的内容追加到 authorized_keys文件中 `cat [pub files name] >> authorized_keys`，这一步一定要有，不然公钥不生效
5. 你要配置不同是主机的ssh，在config文件里面新增即可
6.  验证是否配置成功`ssh -T [用户名]@[服务器ip或者域名]`,例如 `ssh -T root@127.0.0.1`



## known_hosts文件

- 你每连接一个新的服务器时，第一次连接的时候都会提示你说这是一台新的主机，询问你是否要连接，之后则会将这台主机的一些信息添加进这个文件，之后如果这个主机的信息没有改变，则不会再次询问你，否则会再次询问你是否连接

## authorized_keys

- `AuthorizedKeysFile`指定储存用户公钥的目录，默认是用户home目录的`ssh/authorized_keys`目录（`AuthorizedKeysFile .ssh/authorized_keys`）


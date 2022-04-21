---
title: commitizen +husky 实现commit之前eslint检查代码和规范commit的提交
date: 2022-04-21 17:38:15
tags:
  - git
  - 前端工程化
---

1. 
   
   ```java
   //使用下面两个命令下载husky，不要使用手动安装，Windows下会出现很多问题
   npx husky-init
   npm install
   ```
   
   安装完了之后可以去package.json的script里面配置
   

   <!-- more -->
   
   ![image-20220226210621805](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220226210621805.png)


   之后会生成一个.husky文件夹，![image-20220226210708634](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220226210708634.png)

在箭头所示的文件夹中配置commit之前需要执行的命令，一般会配置一些eslint检查代码的命令，之后就可以了，当你commit代码的时候就会执行你配置的命令啦

![image-20220226210800988](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220226210800988.png)

2. 
   
   ```java
   //执行下面两个命令
   npm i commitizen -D
   commitizen init cz-conventional-changelog --save-dev --save-exact
   /*然后在package.json的script里面配置一下命令，因为我们使用了husky，这里的命令不能使用commit,否则执行命令的时候会提交两次,到这里就完成啦，之后commit代码时会让你选择commit的message，如下图*/
   ```
   
   ![image-20220226211100047](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220226211100047.png)

![image-20220226211416581](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220226211416581.png)
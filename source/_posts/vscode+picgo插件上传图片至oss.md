---
title: vscode+picgo插件上传图片至oss
date: 2022-09-10 16:33:19
tags:
  - interesting
---
# vscode+picgo插件上传图片至oss

以前我是使用typora写markdown的,typora确实香，所见即所得的特性对新手比较友好，也可以配合pcigo，直接粘贴图片即可上传到图床,但是typora开始收费了,虽然可以下载以前不用收费的历史版本，但是用了一段时间之后也用不了,这之后我开始使用vscode来写markdown,配合markdown all in one 插件,体验还是不错的（不得不说vscode是真滴牛）

- 写作体验好了，但是不能像typora一样自动上传图片到图床,好在vscode里面也有这么一个插件叫picgo ,可以帮助你自动上传图片至各大图床。
  

<!-- more -->

1. 下载完picgo之后,打开扩展设置
![leftover](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/20220714224737-2022-07-14.png)

![leftover](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/20220714225219-2022-07-14.png)

![leftover](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/20220714230908-2022-07-14.png)

 - Custom Output Format 是自定义显示在你的文件中的内容
 - Custom Upload Name 是自定义上传文件的名称,尽量保证每个文件唯一,别使用${mdFileName} ,因为如果你的md文件名是中文，会出现上传不上去的情况可以参考我的设置，
 - AccessKey ID 设置阿里云的AccessKey
 - Access Key Secret 设置阿里云的AccessKeySecret
 - Area :你的bucket是哪个区域的
 - Bucket : 你的Bucket名称
 - Custom Url: 如果你的oss上使用了自定义域名,则可以将此选项设置为你的域名
 - Path:要存放在Bucket的哪个文件夹下,不设置默认根路径
 - Current： 设置为aliyun 即可


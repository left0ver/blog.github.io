---
title: 手把手教你用 hexo+github page 搭建博客
date: 2022-04-21 18:06:29
tags:
  - hexo
---

# 利用GitHub pages 搭建博客以及部署

1. 首先在hexo官网下载hexo-cli，之后按官网的命令初始化博客

2. 之后依次运行`hexo clean  `，`hexo g` ,`hexo s`，这时候你就可以在你的本地查看博客了

3. 如果想要使用GitHub-page托管你的博客，则你需要新建一个GitHub仓库，仓库名命名为`xxx.github.io`,一定要以`github.io`结尾,如图![image-20220418113645763](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220418113645763.png)

<!-- more -->


 4. - 之后在你的本地博客文件李找到`_config.yml`文件，这里url填写你仓库的地址，以我的为例，就是箭头所指的那样子，如果你想要自定义选择自己的域名，则url按箭头的上一行那样子填写自己的域名，这里我选择了的是自己的域名。  ![image-20220418113839968](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220418113839968.png)

   - 在`_config.yml`文件中找到`deploy`配置，这里`type`填`git`,`repo`填你的仓库地址，这里建议使用ssh的方式，也可以使用https的方式，`branch`填你想使用哪个分支来部署，一般是用`main` 或者`master` ，这里我专门建了一个分支来部署博客，因此我这里填的是`docs`，`message`:是指commit的信息，可以自定义，也可以也不填，使用默认值

     ```yaml
     deploy:
       type: git
       repo: git@github.com:left0ver/blog.github.io.git
       branch: docs
       message: updated:{{ now("yyyy-MM-DD HH:mm:ss") }}
     ```

   - 之后在vscode命令行运行` npm install hexo-deployer-git --save`，安装`hexo-deployer-git`这个包

5. 之后找到你的仓库的setting，点击它![image-20220418114138247](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220418114138247.png)

6. 左侧找到pages来，设置你的博客的所在的分支，我这里新建了一个分支`docs`专门来存放博客的页面,你也可以直接使用`main`或者`master`分支，右侧选`root`，

   下面你可以选择发布站点时的主题，这个选不选都无所谓，最下面那个是自定义域名，如果你使用自定义域名，则填写对应的域名，如果你不是使用自己的域名，则不用配置![image-20220418114247687](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220418114247687.png)

7. - 如果你不是选择的使用自己的域名，则可以依次运行`hexo clean` ,`hexo g`,`hexo d` ，之后就会把生成的public文件夹提交并push到你上面配置的的GitHub对应的分支上了,如果提交不成功则多试几次，然后你可以使用GitHub pages页面中所提供的url访问你的网站了。
   - 如果你是选择的使用自己的域名，则需要到对应的云服务厂商中配置对应的域名解析，具体请往下看，这里我选择的是阿里云

8. 首先先打开本地命令行，ping 一下`[usename].github.io` ，username那里填写你自己的GitHub用户名，之后记住上面显示的那个ip，我这里是`185.199.109.153`

   ![image-20220418163626860](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220418163626860.png)


9. -  如果你用的不是子域名，例如`leftover.cn`这种，则找到云服务商对应的域名解析，添加两条记录，如下图，第一条中的那个记录值填写第8步得到的那个ip，第二条如下图：只需把记录值中的那个用户名那个换成自己GitHub用户名即可，完成之后等待10分钟

     ![image-20220418164206849](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220418164206849.png)

   ![image-20220418163841386](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220418163841386.png)

- 如果你用的是子域名，例如`blog.leftover.cn` 这样子的，则添加一条记录即可，方法和上面类似，如图

  ![image-20220418175123625](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220418175123625.png)

10. 之后再你的GitHub的那个分支里会多出这样一个`CNAME`文件，打开它，复制里面的内容，然后在本地的博客文件夹里面找到`source`文件夹，新建一个`CNAME`文件，粘贴复制的内容,`CNAME`文件的内容其实就是你的网站域名，例如`blog.leftover.cn`,之后重新运行`hexo clean`,`hexo g`,`hexo d`,就可以用自己的域名打开博客了![image-20220418164700340](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220418164700340.png)

    ![image-20220418164829524](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220418164829524.png)

# 直接部署到云服务器

- 上述讲的是利用GitHub Pages来实现部署，倘若你不想通过GitHub pages来部署，想直接部署到自己的云服务器，官网说的是利用`hexo-deployer-rsync` 插件来部署到云服务器，但是经过一番尝试，这个方法太麻烦了，里面的坑很多，官网也没有详细的教程，最后找到了另一种方法，详细请看[视频](https://www.bilibili.com/video/BV1qU4y1K7Hk?p=1),亲测有效。

# 鸣谢

- 感谢[**游云逸子**](https://space.bilibili.com/4525102?spm_id_from=333.788.b_765f7570696e666f.2)的[视频](https://www.bilibili.com/video/BV1qU4y1K7Hk?p=1)提供的帮助


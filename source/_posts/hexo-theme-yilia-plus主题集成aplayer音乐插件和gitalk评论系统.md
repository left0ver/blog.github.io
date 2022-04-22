---
title: hexo-theme-yilia-plus主题集成aplayer音乐插件和gitalk评论系统
date: 2022-04-22 20:47:56
tags:
  - hexo
---

本文主要介绍怎么使用hexo-theme-yilia-plus主题集成`aplayer`音乐插件和`gitalk`评论系统

# 简单方法：

- `hexo-theme-yilia-plus`这个主题现在作者已经没有维护了，我fork了这个repo，原本该主题的`gitment`和`giteement`评论系统用不了了并在他的基础上集成了`aplayer`音乐插件和`gitalk`评论系统，开箱即用，如有需要，请查看[repo](https://github.com/left0ver/hexo-theme-yilia-plus)，如果您想要使用我这个主题，您可以将该主题clone到本地theme目录下，并重命名文件夹为`yilia-plus`，具体使用方法和官方的一样。

 <!-- more -->
 
- 需要注意的是，如果您clone了我的仓库，如果您要使用aplayer音乐插件,请先在自己博客文件的根目录安装`hexo-tag-aplayer` ，`npm install --save hexo-tag-aplayer`,找到自己博客的根目录的配置文件`_config.yml`，在最后面加上这两句即可使用，如果您要使用`gitalk`评论系统，则需要看下面的集成`gitalk`评论系统的前两步即可，其他细节请具体看主题的`_config.yml`文件

```yaml
aplayer:
    meting: true
```

# 复杂方法：

- 如果您使用的官方的`hexo-theme-yilia-plus`主题，具体的集成步骤如下：

## 使用[hexo-theme-yilia-plus](https://github.com/JoeyBling/hexo-theme-yilia-plus)主题集成aplayer音乐插件

- 详细见官网 [aplayer音乐插件](https://github.com/MoePlayer/hexo-tag-aplayer/blob/master/docs/README-zh_cn.md)  

1. 因为`hexo-theme-yilia-plus` 主题使用的网易云插件只能播放一首歌，不大符合我的预期，因此自己集成了一个`analyer`音乐插件，可以播放多首歌曲，同时也支持调节音量，切换歌曲等操作。

2. 自己博客文件的根目录安装`hexo-tag-aplayer` ，`npm install --save hexo-tag-aplayer`

3. 找到自己博客的根目录的配置文件`_config.yml`，在最后面加上这两句

   ```yaml
   aplayer:
       meting: true
   ```

4. 找到`hexo-theme-yilia-plus` 主题的`_config.yml`文件，修改里面`music`这个配置项,修改成如下所示

   ```yaml
   # 网易云音乐插件
   music:
     cloudMusic:
       enable: false
     # 播放器尺寸类型(1：长尺寸、2：短尺寸)
       type: 2
     #id: 1334445174  # 网易云分享的音乐ID(更换音乐请更改此配置项)
       autoPlay: true  # 是否开启自动播放
     # 提示文本(关闭请设置为false)
       text: '学习之余来听首歌叭'
   # 添加aplayer 音乐插件
    # aplayer 音乐插件，https://github.com/MoePlayer/hexo-tag-aplayer/blob/master/docs/README-zh_cn.md
     # {% meting "60198" "netease" "playlist" "autoplay" "mutex:false" "listmaxheight:340px" "preload:none" "theme:#ad7a86" ... %}
     aplayer:                # aplayer 音乐插件
       enable: true          # 使用aplayer
       id: '8444951237'      # 必须值 歌曲 id / 播放列表 id / 相册 id / 搜索关键字
       server: 'tencent'	  # 必须值 音乐平台: netease, tencent, kugou, xiami, baidu
       type: 'playlist'	  # 必须值 song, playlist, album, search, artist
       fixed: true	          # false 开启固定模式 true 开启吸底模式
       loop: none	          # 列表循环模式：all, one,none
       order: random	      # 列表播放模式： list, random
       volume: 0.4	          # 播放器音量
       listfolded: true	  # 指定音乐播放列表是否折叠
       autoplay: false	      # 自动播放，移动端浏览器暂时不支持此功能
       mutex: true	          # 该选项开启时，如果同页面有其他 aplayer 播放，该播放器会暂停
       listmaxheight: 400px  # 播放列表的最大长度
       preload: none	      # 音乐文件预载入模式，可选项： none, metadata, auto
       theme: '#C7A3AD'	  # 播放器风格色彩设置
   ```

5. 找到`themes\yilia-plus\layout\_partial\left-col.ejs`，在`left-col.ejs`文件夹中

   ![image-20220420172738527](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220420172738527.png)

   如图，第一个箭头的地方，把这行代码改成图片所示那样

   ```ejs
   <% if (theme.music && theme.music.cloudMusic.enable){ %>
   ```

   在网易云插件的末尾，即第二个箭头处添加如下代码

   ```ejs
         <!-- 添加aplayer音乐插件-->
       <% if (theme.music.aplayer.enable){ %>
         <%# "aplayer音乐插件" %>
         <div id="aplayer-hjnObCgk" class="aplayer aplayer-tag-marker meting-tag-marker" data-id="<%=theme.music.aplayer.id%>"
           data-server="<%=theme.music.aplayer.server%>" data-type="<%=theme.music.aplayer.type%>" data-loop="<%=theme.music.aplayer.loop%>"
           data-order="<%=theme.music.aplayer.order%>" data-lrctype="0" data-listfolded="<%=theme.music.aplayer.listfolded%>"
           data-fixed="<%=theme.music.aplayer.fixed%>" data-autoplay="<%=theme.music.aplayer.autoplay%>"
           data-volume="<%=theme.music.aplayer.volume%>" data-mutex="<%=theme.music.aplayer.mutex%>"
           data-listmaxheight="<%=theme.music.aplayer.listmaxheight%>" data-preload="<%=theme.music.aplayer.preload%>"
           data-theme="<%=theme.music.aplayer.theme%>">
         </div>
        <% } %>
   ```

6. 此时已经完成了`aplayer`插件的添加，打开博客即可看到效果。

7. 修改播放列表

   - 这里我使用的是qq音乐的歌单列表播放，其他的播放平台自行寻找方法
   - 打开QQ音乐的歌单，分享歌单，复制分享的链接，链接格式像这样`https://c.y.qq.com/base/fcgi-bin/u?__=K8wodTSr4vs7`,之后将链接粘贴到浏览器中打开，url会变成类似这样`https://y.qq.com/n/ryqq/playlist/8444951237`,最后面的数字就是歌单id，将上面配置项里的`id`换成自己的即可播放自己歌单里的歌，vip歌曲播放不了

## 使用[hexo-theme-yilia-plus](https://github.com/JoeyBling/hexo-theme-yilia-plus)主题集成gitalk评论系统

- gitalk的具体配置见[gitalk官网](https://github.com/gitalk/gitalk/blob/master/readme-cn.md)

1. 因为一些原因，`Gitment`和`giteement`已经用不了了，当时我不知道，搞这两个东西搞了好久，一直出问题用不了，真是气死我了，发现用不了了之后决定自己在原本的基础上集成`gitalk`评论系统。

2. 这里需要自己去GitHub创建一个注册一个新的 OAuth Application ➡️ [OAuth Application](https://github.com/settings/applications/new)

   ![image-20220420175133675](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220420175133675.png)

   ```javascript
   //Application name     应用的名称,可以随便填
   //Homepage URL         一般是填自己的网站域名，
   //Application description  描述，随便填
   //Authorization callback URL   填写网站域名
   ```

   之后创建完成之后会给你一个`client_id`和`secret`，找到`hexo-theme-yilia-plus`主题的`_config.yml`配置文件，在`giteement`的评论系统的后面添加上如下配置项

   ```yaml
   #5、自己集成gitalk评论系统
   gitalk:
     client_id: '' #上面得到的client_id
     client_secret: '' #上面得到的secret
     repo: 'blog.github.io' #填写你存储评论的仓库名称，一般填写自己的博客仓库名即可
     owner: 'left0ver' #仓库的所有者,填写你的用户名即可
     admin: ['left0ver'] #仓库的管理员，没有特殊情况填你一个你自己的用户名即可
     id: location.href  #issue的id,具有唯一性，默认是location.href，长度小于50，下面的步骤中后面会对id做一些修改
     distraction_free_mode: true #类似Facebook评论框的全屏遮罩效果
     labels: ['Comment'] #issue的标签，默认['Gitalk']
   ```

3. 找到主题下的`article.ejs`文件夹，`themes\yilia-plus\layout\_partial\article.ejs`，搜索

   `<% if (!index && post.comments){ %>`,在它的下面一行添加如下代码,这个代码是有响应式的

   ```ejs
    <% if (theme.gitalk && theme.gitalk.client_id && theme.gitalk.client_secret){ %>
       <section id="comments" class="comments">
         <style>
           .comments{margin:30px;padding:10px;background:#fff}
           @media screen and (max-width:800px){.comments{margin:auto;padding:10px;background:#fff}}
         </style>
         <%- partial('post/gitalk', {
           key: post.slug,
           title: post.title,
           url: config.url+url_for(post.path)
           }) %>
     </section>
   <% } %>
   ```

4. 在`themes\yilia-plus\layout\_partial\post` 文件夹下新建`gitalk.ejs`文件，添加如下代码

   ```ejs
   <% if (!index && theme.gitalk.client_id && theme.gitalk.client_secret){ %>
   <div id="gitalk-container"></div>
   <link rel="stylesheet" href="https://unpkg.com/gitalk/dist/gitalk.css">
   <script src="https://unpkg.com/gitalk/dist/gitalk.min.js"></script>
   <script>
   var gitalk = new Gitalk({
     clientID: '<%=theme.gitalk.client_id%>',
     clientSecret: '<%=theme.gitalk.client_secret%>',
     repo: '<%=theme.gitalk.repo%>',
     owner: '<%=theme.gitalk.owner%>',
     admin: '<%=theme.gitalk.admin%>',
    //id 是为了保证issue的唯一性，id长度不能大于50，因此这里我们使用路径来当id，做一个截取，最好不要修改此配置项，否则可能导致评论丢失  
     id: <%=theme.gitalk.id%>.slice(<%=theme.gitalk.id%>.length-25),
     distractionFreeMode: '<%=theme.gitalk.distraction_free_mode%>',
     labels: '<%=theme.gitalk.labels%>'.split(',').filter(l => l) ,
   })
   gitalk.render('gitalk-container')
   </script>
   <% } %>
   
   ```

5. 这行面这个`labels`，在配置文件里面定义的是数组，但是在这上面使用的时候就成字符串了，因此我们需要将其转换成数组，否则会报`Error: r.concat(...).join is not a function`这种错误

6. 至此即完成了gitalk评论系统的添加。重新运行即可看到效果

## 鸣谢

- 本文大多内容来自于下面两位的博客和PR，在这里衷心感谢他们
- 感谢[yansheng836](https://github.com/yansheng836)的[博客](https://yansheng836.github.io/article/72a91df5.html)为我集成gitalk评论系统指引了方向
- 感谢[ProudRabbit](https://github.com/ProudRabbit)的[PR](https://github.com/ProudRabbit/hexo-theme-yilia-plus/commit/ce5253c8d84c48efa5124c4f5f3d886440e22060)，为我集成aplayer插件提供了很大的帮助


---
title: 使用宝塔部署vue+node 的项目
date: 2022-04-21 17:30:21
tags:
  - 部署
  - 宝塔
---

 1. **先讲讲后端部署的方法吧**

         

 - 我是用pm2来部署node项目，数据库的配置如下：

          
![在这里插入图片描述](https://img-blog.csdnimg.cn/be787e1545cc4b52bcfd562862900646.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAbGVmdDB2ZXI=,size_20,color_FFFFFF,t_70,g_se,x_16)
<!-- more -->

 - 这里数据库的ip地址改成0.0.0.0 
 - 用户名就填你自己的数据库登录时的用户名，有些人的是root
 - 密码就填数据库登陆时的密码
 - 最后一个database就填你数据库的名称       

![在这里插入图片描述](https://img-blog.csdnimg.cn/e91c83cf7f8542329fadfcac602684ab.png)
这里我后端用的是express框架，这里的监听的ip地址是0.0.0.0，端口号根据你的项目的实际端口号来填

把这些东西修改好了之后就可以了，然后用宝塔的pm2部署一下，很简单，这里我就不赘述了

 2. **这里讲一下怎么部署前端，这一部分内容需要和下面nginx配置代理相关联**
 
**前端这里我在每个请求的前面都加上了一个/api，**如下

```javascript
   const res = await axios.post("http://sczh.xyz/api/article/comment",
```
这里的/api和待会nginx配置代理有关联

**前面的地址就填你的域名，或者你的服务器的ip地址**
 

 **3. 接下来配置一个nginx代理，**
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/f8c7b0c3f05a463699fc3b157589b022.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2f6ed0bd5136421985f2cdc8a8373bd2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAbGVmdDB2ZXI=,size_20,color_FFFFFF,t_70,g_se,x_16)
宝塔进去网站的设置里，找到配置文件，这里的端口号很关键，
**我们这里先找到80端口的那个server**
在下面加上这么一段

```javascript
 location / {
     index index.php index.html index.htm default.php default.htm default.html;
     #root是你自己服务器上前端的项目的存放的文件地址
    root /www/wwwroot/sczh.xyz/dist;
    try_files $uri $uri/ @rewrites;
    }
     location @rewrites
    {
      rewrite ^(.+)$ /index.html last;
    }

当你的项目用了history模式的时候，刷新会出现404的情况，这段配置用来防止使用history模式刷新出现404的情况，一定要在80端口的那个server里加。

 - 因为我们前面请求的是
  const res = await axios.post("http://sczh.xyz/api/article/comment",
  是80端口，所以这里找到80端口的那个server，这里因为我们前面的请求的地址里的前面都加上了/api
  所以这里我们对带有/api的请求进行代理，如下，这里的端口号是你后端监听的端口号，


  location /api/ {
  #这里填的是你自己的域名，即把以/api的请求转发到下面你配置的ip或者域名上面
  	proxy_pass http://sczh.xyz:8000/;
  }
```
**这里是将请求转发到了`http://sczh.xyz:8000/`，但是这里我的后端是允许了跨域的，如果你这里按照我的这样做了的话，还是提示跨域的问题，这里你可以把proxy_pass `http://sczh.xyz:8000/`; 改成proxy_pass `http://127.0.0.1:8000/`**

这个东西也是我摸索了好几天才摸索出来，希望能帮助到你们
到这里就完成了部署了，如果你有什么问题，可以评论求助


---
title: axios + applicationx-www-form-urlencode传参的问题
date: 2022-07-01 14:04:28
tags:
 - vue
 - javaScript
---


- 使用axios的时候，在post请求的body传递数据的时候，axios会默认转换成json字符串的格式传递给后端
- 但有时候后端接收的格式是`application/x-www-form-urlencode`,axios默认的格式是`application/json` ,你可以通过设置headers来设置传递的数据类型
  
  ```json
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  ```
但是这时候axios依然会将你body里的数据转换成json字符串的形式,一般我们可以使用[qs](https://www.npmjs.com/package/qs)库来进行转换,如下：

<!-- more --> 

import QueryString from "qs"
axios.post(
    '/login',
    QueryString.stringify({ username, password }),
    {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    }
  )
```


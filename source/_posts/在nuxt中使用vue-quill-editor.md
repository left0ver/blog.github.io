---
title: 在nuxt中使用vue-quill-editor 出现document is not defined
date: 2022-04-21 17:42:01
tags:
  - vue
---


首先在plugins目录下面创建这个文件夹

![image-20220303231018712](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220303231018712.png)

添加如下内容

```javascript
import Vue from "vue";
let VueQuillEditor;
if (process.browser) {
  VueQuillEditor = require("vue-quill-editor/dist/ssr");
}

Vue.use(VueQuillEditor);
```

<!-- more -->

1. 然后在nuxt.config.js文件中

   加入这几行代码

   ```javascript
    "quill/dist/quill.snow.css",
       "quill/dist/quill.bubble.css",
       "quill/dist/quill.core.css",
        
        
       { src: "~plugins/nuxt-quill-plugin.js", ssr: false },  
   ```

   如下图所示位置加代码

![image-20220303231208982](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220303231208982.png)

![image-20220303231240589](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220303231240589.png)

3. 最后回到自己的vue文件中

   在template中像这样引入vue-quill-editor

   不用在script标签中导入组件

   ```javascript
   <div
           class="quill-editor"
           v-quill:myQuillEditor="editorOption"
         ></div>
   ```

   然后在script中的data中添加

   ```javascript
    editorOption: {
           // Some Quill options...
           theme: "snow",
           modules: {
             toolbar: [
               ["bold", "italic", "underline", "strike"],
               ["blockquote", "code-block"],
             ],
           },
         },
   ```

   

   ![image-20220303231708773](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220303231708773.png)

这样一个简单的富文本编辑器就完成了，具体其他配置请看[**官方文档**](https://github.com/surmon-china/vue-quill-editor)

![image-20220303231816364](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220303231816364.png)


---
title: 在vscode中调试ts代码,不支持esm模块
date: 2023-01-13 22:34:11
tags:
  - node
---

话不多说,直接上配置,在vscode中的setting.json文件中的`launch`下的`configurations` 中添加配置即可

```json
//setting.json
"launch": {
 "configurations": [
   // ... other
    {
      "name": "ts-node",
      "type": "pwa-node",
      "request": "launch",
      "args": [
        "${relativeFile}"
      ],
      "env": {
        "NODE_OPTIONS": "--no-warnings"
    },
      "runtimeArgs": [
        "-r",
        //我这里使用的是全局安装的ts-node,当然你也可以直接写ts-node/register,建议使用全局安装的ts-node,不然你每次调试ts代码都得装ts-node和typescript,
        "/usr/local/lib/node_modules/ts-node/register",
      ],
      "cwd": "${workspaceRoot}",
      "protocol": "inspector",
      "internalConsoleOptions": "openOnSessionStart"
    },
  ],
  
  "compounds": []
  }
```
<!-- more -->

这样的配置在 `cjs` 的项目里面是可以用的,当你把`package.json`里的type:"module" and  `tsconfig.json`里的 module:"ESNext",这样子就调试不了了.

![image-20230113221608934](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/202301132216062.png)

这里是因为没有对esm进行支持,之后我将`runtimeArgs`改成这样

```json
      "runtimeArgs": [
        "--loader",
        "/usr/local/lib/node_modules/ts-node/esm/transpile-only.mjs",
        "-r",
        "/usr/local/lib/node_modules/ts-node/register",
      ],
```

![image-20230113221907055](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/202301132219118.png)

{% markmap 400px %}
- links
- **inline** ~~text~~ *styles*
- multiline
  text
- `inline code`
- ```js
  console.log('code block');
  console.log('code block');
  ```
- KaTeX - $x = {-b \pm \sqrt{b^2-4ac} \over 2a}$
{% endmarkmap %}

这里是提示没有这个文件,但文件是存在的,因为ts里面导入文件是可以不用扩展名的,之后我将导入的文件加上`.ts`的扩展名,依旧报同样的错误,猜测是因为ts-node在esm下面模块解析的问题



不知道有哪位大佬知道原因的,麻烦指导一下

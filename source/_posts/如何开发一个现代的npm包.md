---
title: 如何开发一个现代的npm包
date: 2023-01-11 00:18:54
tags:
  - npm
---

# 模块化

1. IIFE其实就是一个立即执行函数，最开始是使用这个来进行模块化的，隔离变量作用域

2. AMD（Asynchronous Module Definition异步模块定义）
3. UMD (*Universal Module Definition*,也就是通用模块定义),UMD是AMD+cjs的兼容版，在AMD和cjs的项目中，都可以引入UMD模块
4. CJS 是nodejs采用的模块化标准，使用require引入模块，exports 或 modules.exports来导出模块
5. ESM 是es6提出的模块化方案，是当前比较流行的模块化方案，在node.js中默认是使用cjs的模块化方案，可以在package.json中设置  "type": "module"来标记为esm模块,使用import导入模块，export导出模块

<!-- more -->
# 打包成cjs和esm模块

# tsc

```json
//tsconfig.base.json
{
  "compilerOptions": {
    "strict": true,
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "skipLibCheck": true,
    "checkJs": true,
    "allowJs": true,
    "declaration": true,
    "declarationMap": true,
    "allowSyntheticDefaultImports": true
  },
  "files": ["src/**/*.ts"]
}
```

```json
//tsconfig.cjs.json
{
  "extends": "./tsconfig.base.json",
  "compilerOptions": {
    "lib": ["ES6", "DOM"],
    "target": "ES2016",
    "module": "CommonJS",
    "moduleResolution": "Node",
    "outDir": "./lib/cjs",
    "declarationDir": "./lib/cjs/types"
  }
}
```

```json
// tsconfig.esm.json
{
  "extends": "./tsconfig.base.json",
  "compilerOptions": {
    "lib": ["ES2022", "DOM"],
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "NodeNext",
    "outDir": "./lib/esm",
    "declarationDir": "./lib/esm/types"
  }
}
```

之后我们可以运行`tsc -p ./tsconfig.esm.json` ,`tsc -p ./tsconfig.cjs.json` 分别打包esm和cjs的包

之后我们可以在`package.json`里面配置script,

```json
//package.json
{
  "script":{
    "clean": "rm -rf ./lib/*",
    "build": "npm run clean && npm run build:esm && npm run build:cjs",
    "build:cjs": "tsc -p ./tsconfig.cjs.json",
    "build:esm": "tsc -p ./tsconfig.esm.json && mv ./lib/esm/index.js ./lib/esm/index.mjs",
    "prepublishOnly": "npm run build"
  }
}
```

`prepublishOnly`脚本会在你每次运行`npm publish`之前自动调用

# tsup

- 一个打包工具，开箱即用，对新手友好

  例如：我们需要打包出cjs和esm的包

  `tsup index.ts --format cjs,esm,iife  -d lib --dts --clean --global-name P -d lib`
	
	一行命令即可打包出对应的代码，相比tsc来说，更加地简单，不需要像tsc那样配置多个配置文件，但是这两个工具都只能打包根目录下的模块，不能打包子模块

# unbuild

`unbuild`和`tsup` 类似,相对tsup来说,unbuild 会更复杂一些,需要配置对应的配置文件,但他可以打包一些子模块,相比tsup则更灵活

```typescript
import { defineBuildConfig } from 'unbuild'

export default defineBuildConfig({
  entries: [
    './src/index',
    //打包子模块
    {
      builder: 'rollup',
      input: './src/package/components/',
      outDir: './build/components',
    },
  ],
  rollup: {
    emitCJS: true,
    inlineDependencies: true,
  },
  clean: true,
  declaration: true,
})

```





# package.json

1. file：发布的时候哪些文件需要上传到npm
2. main: 项目的入口文件,在浏览器环境和node都可以使用
3. browser： 浏览器端的入口文件，只针对浏览器环境
4. module: 指定 ES 模块的入口文件，这就是 module 字段的作用
> module和browser的优先级高于main

5. exports: node 在 14.13 支持在 package.json 里定义 exports 字段，拥有了条件导出的功能。exports 字段可以配置不同环境对应的模块入口文件，并且当它存在时，它的优先级最高，exportsl

   ```json
   "exports": {
     ".": {
       "types": "./dist/index.d.ts",
       "require": "./dist/index.js",
       "import": "./dist/index.mjs"
       "browser":"./dist/index.global.js"
     }
    }
   }
   ```

6. engines

   平常有些项目对node版本和包管理器的版本有要求，我们可以通过设置engines来指定版本号，如果版本号不符合，用户下载依赖的时候则会被终止，会提示对应的报错信息

   ```json
     "engines": {
       "node": ">=16 <18"
     }
   ```

   

7. os

   有些npm包可能只能在Linux上运行，不支持windows操作系统，我们可以通过os字段来设置支持的操作系统

   ```json
   {
     "os": ["linux","darwin"]
   	// or 
     "os": ["!win32"]
   }
   ```

   

8. bin

   有时候我们有一些可执行文件，下载npm包的时候希望下载到PATH里，这时候我们可以通过bin字段来指定，这时候我们全局下载的时候会在/usr/local/bin目录下面创建一个软链接，指向我们指定的那个文件

   ```json
   {
     "bin":{
       "formula": "./bin/yqn.js"
     }
   }
   ```

   ```javascript
   //bin/yqn.js
   #!/usr/bin/env node
   ```

9. types

   使用types来指定类型的入口文件

   ```json
   "types": "./lib/index.d.ts",
   ```

# 测试

对于一个正规的npm包来说，单元测试是非常有必要的

我们可以使用jest来编写测试代码

然后package.json

```json
//package.json

{
  "script":{
    //...
    "test":"jest"
  }
}
```



CI

```yaml
name: Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:

    runs-on: ubuntu-latest

    strategy:
      matrix:
        node-version: [12.x, 14.x, 16.x, 18.x]

    steps:
      - uses: actions/checkout@v3
      - name: Use Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node-version }}
      - run: npm ci
      - run: npm test
```

---
title: 自动加载项目中的svg图标,并很简单的使用它
date: 2022-07-16 21:52:43
tags:
  - 前端工程化
  - webpack
---

你的项目中是否有很多的图标呢，现在我们的图标基本都是svg文件,现在使用图标有主流的两种方法。
1. 一种是使用线上的，如阿里的字体图标库，我们可以使用在线链接，引入css，js等文件，并在我们的项目中使用这些图标
2. 第二种则是将图标下载到本地,我们可以使用img标签引用他

<!-- more -->
现在我们来讲一下第二种

1. 现在一般会通过webpack的配置处理图像文件,如下
   ```typescript
     {
            test: /\.(png|svg|jpg|jpeg|gif|webp)$/i,
            type: 'asset',
            parser: {
              dataUrlCondition: {
                maxSize: 25 * 1024, // 25kb
              },
            },

     }
   ```
  在上面的配置中，当文件大小小于25kb时,会将图片转成base64引入,大于的时候则会将图片以data-url的形式引入,这样的好处就是小图片使用base64的形式可以减少网络请求,但base64会增加打包的大小,因此大图片不适合使用base64的形式,这种方式虽然可以，但是我们依旧要使用img标签来引入我们的图标

2. 第二种方式,使用[svg-sprite-loader](https://github.com/JetBrains/svg-sprite-loader)和[svgo-loader](https://github.com/svg/svgo-loader)处理svg图标,我们的webpack如下配置，如果是图标则使用svgo-loader和svg-sprite-loader处理,如果不是则依然按上面那种方式处理,svg的字体图标存放在src/icons/svg目录下,


使用svg-sprite-loader作用是合并一组单个的svg图片为一个sprite雪碧图，并把合成好的内容，inject插入到html内，形式是添加svg标签,我们通过xlink:href来实现对某一个图标的引用，使用雪碧图可以减少请求，只需请求一张图片

和我们以前的雪碧图可能不大一样，我们以前的雪碧图是通过ui将所有的图标放在一个图片中，然后你通过background-position来引用不同的图标


xlink:href的值是一个id,可以通过设置loader的options中的symbolId来配置
这里我配置的是`symbolId: 'icon-[name]'`,如果我的文件名是`bug.svg`,xlink:href的值则是`#icon-bug`

```typescript
// webpack配置
// 需要使用include和exclude来确定需要处理的.svg图标的范围
         {
            test: /\.svg$/,
            use: [
              {
                loader: 'svg-sprite-loader',
                options: {
                  symbolId: 'icon-[name]',
                },
              },
              'svgo-loader',
            ],
            include: path.resolve(__dirname, '../src/icons/svg'),
          },

           {
            test: /\.(png|svg|jpg|jpeg|gif|webp)$/i,
            type: 'asset',
            parser: {
              dataUrlCondition: {
                maxSize: 25 * 1024, // 25kb
              },
            },
            // 排除掉icons目录下的图标，单独使用svg-sprite-loader进行处理
            exclude: path.resolve(__dirname, '../src/icons/svg'),
          },
```
1. 之后我们可以封装一个自己的Icon组件
   
```vue
<template>
  <svg class="svg-icon" aria-hidden="true" v-bind="$attrs" :class="className">
    <use :xlink:href="iconName" />
  </svg>
</template>

<script lang="ts" setup>
import { toRefs,computed,defineProps} from "vue"


interface IconProps {
  iconClass: string;
  svgClass?:string;
}
const props = defineProps<IconProps>()
const { iconClass, svgClass } = toRefs(props)
const iconName = computed(() => `#icon-${iconClass.value}`)
const className = computed(() => {
  if (svgClass ===undefined) {
    return ''
  }
  return svgClass
})
</script>

<style lang="less" scoped>
  .svg-icon {
  width: 1em;
  height: 1em;
  overflow: hidden;
  vertical-align: -0.15em;
  fill: currentColor;
}
</style>
```

这里只是封装了一个Icon组件,但是只有我们使用了某个图标，svg-sprite-loader才会帮助我们做处理，因此我们需要在main.ts或者其他的某个地方引入需要用到的图标，那么能不能引入所有的图标呢,也是可以的，利用 webpack 提供的 require.context API 引入某个目录下的所有的svg图标

它接受三个参数

- 要搜索的文件夹目录
- 是否还应该搜索它的子目录，
- 以及一个匹配文件的正则表达式。

如果使用的是ts，需要下载`@types/webpack-env`，然后再在.eslintrc.js中配置

```javascript
module.exports = {
 rules: {
  // ...
 },
//  加上下面这个，否则会eslint会报错
globals: {
    __WebpackModuleApi: 'writable',
  },
}

```
```typescript
// 自动导入的方法
  export function importAllSvg(): void {
  const importAll = (requireContext: __WebpackModuleApi.RequireContext) =>
    requireContext.keys().forEach(requireContext)

  try {
    // 导入src/icons/svg下的所有以.svg结尾的文件
    importAll(require.context('@/icons/svg', false, /\.svg$/))
  } catch (error) {
    console.log(error)
  }
}


// main.ts中
import {importAllSvg} from "@/icons/index"
importAllSvg()

```

只要是在src/icons/svg下面的图标都可以自动引入了，然后我们只要在使用icon组件的时候这样用

```vue
<!-- svg的文件名是bug.svg -->
<template>
  <svg-icon icon-class='bug'></svg-icon>
</template>

<script lang="ts" setup>
import SvgIcon from '@/components/SvgIcon'
</script>

```
进一步我们可以全局注册SvgIcon组件，之后就可以不用引入了组件了

```typescript

// main.ts
// 加上下面一行即可
  app.component('SvgIcon', SvgIcon)

```

---
title: 为VitePress添加busuanzi的访问量统计功能
date: 2024-06-21 23:50:59
tags:
  - other
---
1. 在配置文件`config.mts`中的 head 中引入busuanzi的js文件，这里使用CDN的方式引入
```typescript

import { defineConfig } from 'vitepress'
export default defineConfig({

  head: [
    ["script",{src:"https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js",defer:''}]
  ],
})    
```

2. 我们需要将展示的浏览人数的代码添加到网站中，这里我使用的是默认的VitePress主题，所以我们需要对默认主题进行扩展，具体可看[扩展默认主题](https://vitepress.dev/zh/guide/extending-default-theme#layout-slots)
   
```vue
<!-- .vitepress/theme/PageView.vue -->

<script setup>
import DefaultTheme from 'vitepress/theme'

const { Layout } = DefaultTheme
</script>

<template>
  <Layout>
    <!-- 使用对应的锚点扩展默认的主题 -->
    <template #nav-bar-content-after
>
      <div class="page-view">
      <span  id="busuanzi_container_site_uv">本站总访问量<span id="busuanzi_value_site_pv"></span>次</span>
      </div>
    </template>
  </Layout>
</template>

<style lang="less">
  .page-view {
    padding-left: 14px;
    font-size: 14px;
    font-weight: 400;
    color: #1565C0;
  }
</style>

```

```javascript
// .vitepress/theme/index.js 
import DefaultTheme from 'vitepress/theme'
import PageView from './PageView.vue'

export default {
  extends: DefaultTheme,
  // 使用注入插槽的包装组件覆盖 Layout
  Layout: PageView
}

```

3. 最后的效果
![leftover](http://img.leftover.cn/img-md/20240622000430-2024-06-22.png)

参考站点：https://note.leftover.cn/
